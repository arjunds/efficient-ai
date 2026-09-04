# A Physically-Grounded Energy Model for LLM Serving, and Energy-Optimal GPU Selection

**Status:** preliminary results + proposal. All data on NVIDIA H200 (+ one A100
cross-GPU point) via vLLM 0.10.2 continuous batching. Branch `vllm_ragged`.

---

## 1. Motivation

LLM inference energy is now a first-order cost, but the field lacks a model that
is simultaneously **(a) physically interpretable**, **(b) validated against
measured power**, and **(c) predictive for unseen models and hardware**. Existing
work sits at two extremes:

- **Analytical / roofline** models (LIMINAL; LLM-Viewer; "Tokens-to-Watt-hours")
  predict performance or estimate energy from datasheet constants but are **not
  calibrated or validated against measured power**.
- **Black-box learned** models (WattGPU: XGBoost on GPU specs) predict power on
  unseen GPUs well but are **not interpretable** and **exclude MoE**.

**The gap we target:** a model whose coefficients are *measured* yet *physically
meaningful*, so it can answer a concrete deployment question:

> **Given a workload (model + prompt/generation profile), which GPU — or which GPU
> for each *phase* of serving — minimizes energy?**

This matters because **prefill is compute-bound and decode is memory-bound**, so
the energy-optimal hardware can differ *within a single request* — motivating
disaggregated prefill/decode placement across heterogeneous GPUs.

---

## 2. Model formulation (X → Y)

We fit a per-time-bin regression on **measured** GPU energy against **analytically
computed** work. Binned over Δ≈200 ms of real ragged continuous-batching traffic.

**Y (response):** measured *dynamic* energy per bin,
`Y_i = ∫ P_NVML dt over bin i − P_static · Δt_i`, with `P_static` = measured idle.

**X (regressors), all analytic from model architecture + vLLM per-iteration logs
(prefill/decode tokens, resident KV tokens):**

| model | regressors | coefficients |
|---|---|---|
| 2-term | `bytes`, `flops` | `e_bit`, `e_flop` |
| **3-term (best)** | `weight_bytes`, `kv_bytes`, `gemm_flops` | `e_wbyte`, `e_kvbyte`, `e_gemm` |

where per iteration: `weight_bytes = params_resident · dtype`, `kv_bytes =
kv_tokens_resident · kv_bytes/token`, `gemm_flops = 2·active_params·tokens`.
`P_static` is fixed from the idle baseline, not fit. Fit by ordinary least
squares; coefficients are physically constrained ≥ 0.

**Training-space characterization (domain of validity):**
- **Hardware:** H200, **uncapped** regime (see §5 — the model breaks under power
  capping; that boundary is itself a result).
- **Models:** dense 7–8B (calibration) + a size ladder 0.5–32B + one 30B MoE (§4).
- **Load:** concurrency 1–64 and Poisson arrivals; real variable-length prompts
  (alpaca + sharegpt) → heterogeneous batches.
- **Arithmetic-intensity span ≈ 1–2000× FLOP/byte** — the identifiability
  condition that lets `e_*` separate. **Extrapolation beyond this range (fp8,
  100k-context, cap-limited) is untested and out of domain.**

---

## 3. Result A — the FLOP term is essential, coefficients are precise

On 9,274 bins across 4 dense models (H200):

- **Bytes-only R² is negative** (−0.2 to −0.9): a memory-only model is worse than
  the mean. Adding the FLOP term is what makes the model work.
- **2-term:** R² 0.62, held-out (leave-one-model-out) MAPE **8.5%**, with tight
  bootstrap CIs: `e_bit = 1.120 [1.115, 1.126] ×10⁻¹⁰ J/byte`,
  `e_flop = 0.874 [0.85, 0.91] pJ/flop`.
- **3-term (weight/KV split): R² 0.80, held-out 7.0%** — separating KV from weight
  bytes is a large, identifiable improvement (all CIs exclude 0):
  `e_wbyte = 1.077e-10`, `e_kvbyte = 3.38e-10`, `e_gemm = 0.673 pJ/flop`.
  **KV-cache bytes cost ~3× more per byte than weight bytes** — a real
  memory-hierarchy finding (scattered paged KV reads vs streaming weight reads).

*(Sanity: `e_flop ≈ 0.7–0.9 pJ/flop` is the right order for H200 fp16 peak
efficiency; `e_bit` is an effective coefficient, not the raw HBM cell energy —
see caveats §6.)*

## 3b. Result B — energy splits by phase exactly as the recommender needs

Using the fitted coefficients, dynamic-energy attribution per bin:

| phase | compute share | memory share |
|---|---:|---:|
| decode-heavy | 5.9% | **94.1%** |
| mixed | 16.9% | 83.1% |
| prefill-heavy | **35.1%** | 64.9% |

Compute share rises **6×** from decode to prefill — the memory-bound-decode /
compute-bound-prefill split, measured.

## 3c. Result C — coefficients are hardware constants (transfer across models)

- Across the 4 dense architectures the coefficients agree to <7% — they behave as
  **hardware constants**, not per-model fits.
- **Held-out prediction (fit 3 models, predict the 4th): 6.5–11% MAPE.**
- **Size independence (Qwen2.5 ladder, 0.5B→32B, 64× range, single architecture):**
  the memory coefficient is size-independent and the channel split makes it more
  so — **`e_wbyte` = 1.06–1.14 ×10⁻¹⁰ J/byte for models ≥1.5B (±7% across 64×
  size)**, tighter than the lumped `e_bit` (1.10–1.35). The 3-term fit's R²
  exceeds the 2-term's at *every* size (e.g. 0.82 vs 0.65 at 7B; 0.90 vs 0.73 at
  1.5B). The ladder's 7B point (e_bit 1.10, e_flop 0.84) independently reproduces
  the cross-family Qwen2-7B result — a consistency check across model versions.
  **Honest limitation:** the *compute* coefficient is less stable — `e_flop`
  drifts ~2× (1.16→0.61 pJ, small→large) and 2-term R² falls for the big,
  strongly memory-bound models (32B: R²=0.25, →0.51 with channels), because those
  runs have little compute-bound variance to pin `e_flop`. So *memory* energy is a
  clean hardware constant; *compute* energy needs a utilization/overhead term for
  the extremes (small models: fixed overhead; large models: identifiability).
  ⇒ the coefficients are **not** an artifact of overfitting to 7–8B.
- **Size transfer (the decisive test, data in hand):** fitting on some sizes and
  predicting a *held-out* size's per-bin energy works — leave-one-size-out MAPE
  **4.6–6.9%** for ≥1.5B (0.5B 13%), with `e_wbyte` invariant at 1.073–1.078e-10
  regardless of which size is dropped. Extrapolation holds too: fit ≤3B → predict
  14B+32B at **8.0%**; **a single 7B calibration predicts all sizes 0.5B–32B (64×
  range) at R²=0.95, 7.2% MAPE.** ⇒ calibrate once, predict any size.

---

## 4. Result D — transfer to a new architecture (MoE), with a routing correction

Fit on 4 dense models, predict a held-out **30B MoE (Qwen3-30B-A3B)**:

- With naive per-token `active_params` byte accounting: **fails, R² = −1.35**.
- **Fix (necessary + sufficient):** MoE weight-byte HBM traffic scales with the
  *expected distinct experts touched by a batch*, `E·(1−(1−k/E)^t)` for `t`
  batched tokens — not per-token active params. With this: **R² = 0.79, MAPE 21%**
  (naive baseline 115%; MoE self-fit ceiling R² 0.91).
- The fix also collapses the MoE's *own* fitted `e_bit` from 4.8e-10 onto the
  dense ~1.1e-10 — independent confirmation that `e_bit` is hardware, not model.

*(The occupancy formula is standard combinatorics and appears in prior MoE
latency/perf work; our contribution here is applying it to **energy** and showing
it is what makes cross-architecture energy transfer work.)*

---

## 5. Result E + the proposal — energy-optimal GPU selection

**Cross-GPU boundary (measured):** the same fit on an **A100-80-PCIe** gives
*negative* R² — because its **300 W cap is saturated at every operating point**
(even concurrency 1). Power is DVFS-clamped ≈ constant, so energy ≈ `P_cap·t`, not
work. A constant-power model beats the two-term there (R² +0.13..+0.42 vs
negative). ⇒ the model is valid in the **uncapped** regime, and a serving-time
model must include a cap term `P = min(e·rates + P_static, P_cap)`.

**Recommender prototype (the deliverable):** given a workload, predict per-phase
energy on candidate GPUs (3-term model + roofline latency + cap throttling) and
recommend the energy-optimal GPU. Coefficients: **H200 measured**; other GPUs
**datasheet-scaled** via physical priors `e_byte ∝ 1/HBM_BW`, `e_flop ∝
1/peak_FLOPS`. Example (Qwen2-7B, prompt 2048 / gen 256 / batch 32):

| phase | winner | 2nd | key reversal |
|---|---|---|---|
| **prefill** (compute) | H100/H200 | **L40S > A100** | high-FLOPS parts win |
| **decode** (memory) | H200 | **A100 > L40S** (2×) | high-bandwidth parts win |

**L40S and A100 swap rank between phases** — the model recommends *different*
hardware for prefill vs decode, i.e. disaggregated placement. This is the novel,
useful output; it falls directly out of the interpretable memory/compute split.

**Proposed validation (the core next experiment):** confirm the datasheet scaling
by *measuring* `e_*` on 2–3 more (uncapped) GPUs. If `e_byte ∝ 1/BW` and `e_flop ∝
1/FLOPS` hold, we get **zero-shot** energy-optimal GPU selection from datasheets —
physically grounded, unlike WattGPU's black box, and covering MoE, unlike all
prior work.

---

## 6. Honest positioning vs prior work

| piece | closest prior | our delta |
|---|---|---|
| two-term memory+compute energy form | Choi et al. 2013 (roofline energy); Horowitz 2014 | not novel as a *form* |
| analytical LLM-inference energy | "Tokens-to-Watt-hours" (2025) | they use datasheet constants, **no measured-power calibration/validation** |
| cross-GPU inference power prediction | WattGPU | they are **black-box XGBoost, exclude MoE**; we are interpretable + MoE |
| MoE expert-occupancy byte count | MoE latency/perf papers; MoE-CAP (S-MBU) | we apply it to **energy** + show it enables cross-arch transfer |
| power waveform from scheduler state | "Smoothing the Ramp" (2025) | overlaps; we tie it to the fitted energy model |

**What is genuinely ours (integration + empirical):** measured-power-calibrated,
interpretable coefficients that are shown to be **transferable hardware constants**
(across models, sizes, and one MoE), packaged into a **phase-aware energy-optimal
GPU recommender**. No single prior work does calibrated + interpretable +
MoE + phase-level GPU selection together. This is a **measurement/systems** paper
(MLSys / workshop tier), not a new-model paper.

## 7. Risks / open threats
- **Byte/FLOP counts are analytic, not HW-measured** (perf counters blocked by
  `ERR_NVGPUCTRPERM`). `e_*` are effective coefficients; a DCGM/NCU cross-check
  would ground them. *(needs admin)*
- **Cross-GPU scaling rests on 1 clean GPU (H200)** + datasheet priors; needs 2–3
  measured (uncapped) GPUs to be more than a hypothesis. *(needs unpegged A100 /
  more hardware)*
- MoE transfer (21%) assumes uniform routing; real routing has 20–31% expert
  overlap → occupancy is an upper bound.

## 8. Plan (next 3 experiments, by leverage)
1. **Size-ladder analysis** (data in hand) — lock the size-independence claim.
2. **Measure `e_*` on 2–3 GPUs** — turn the scaling law + recommender from
   proposal into result. *(hardware access is the blocker)*
3. **HW byte validation** via DCGM/NCU if perf counters get enabled *(admin)*.

---
*Artifacts: `two_term_summary.csv`, `logs/ragged*/…/binned_table.csv` (re-fittable),
`fit_channels.py`, `diagnose_fit.py`, `predict_moe.py`, `recommend_gpu.py`,
`plots_two_term/`. Full narrative: `SESSION_LOG.md`.*
