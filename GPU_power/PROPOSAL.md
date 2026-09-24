# A Measured, Interpretable Energy Model for LLM Serving — and What It Buys: GPU Selection and Power Control

**Status (v2, 2026-09-24):** preliminary results + proposal. vLLM 0.10.2 continuous
batching, fp16. Measured on NVIDIA **H200** (primary), **B200** (second session,
Penn PARCC), **RTX A5000 / GDDR6** (power-capped; microbenchmarks), and an A100-PCIe
(power-capped). Branch `vllm_ragged` on `arjunds/efficient-ai`.

> **What changed since v1 (read this first).**
> 1. **The datasheet scaling law is falsified.** v1 proposed `e_byte ∝ 1/bandwidth`
>    so a recommender could work "zero-shot from datasheets." B200 (2× bandwidth)
>    costs **1.16× H200** per byte, not 0.60×. Energy per byte is instead **set by
>    memory technology**: HBM3e ≈ 1.1–1.25, GDDR6 ≈ 2.7–3.1 (×10⁻¹⁰ J/byte).
> 2. **The recommender now uses measured coefficients + realized utilization**, and
>    is validated for the first time: **J/token within 4.1% (H200) / 3.7% (B200)** on
>    137 measured runs (6.0% leave-one-model-out on H200); v1 was off by 34–62%.
> 3. **Two measurement biases were found and corrected**: prefix-cache hits made us
>    over-count prefill FLOPs (→ `e_gemm` +17–19%), and a possible NVML power smoothing
>    (→ `e_gemm` maybe +26–30% more, not yet adopted). The **memory coefficient is
>    robust to both (±2%)**; the compute coefficient is the uncertain one.
> 4. **The "L40S prefill / A100 decode" reversal from v1 is withdrawn** (it rested on
>    the falsified law). The defensible reversal is between measured GPUs:
>    **B200 for prefill (probable), H200 for decode (confident).**
> 5. **New: a control layer.** The energy model serves as the internal model of an
>    economic MPC that allocates a shared power budget across a heterogeneous
>    H200+B200 fleet: **−12% to −22% energy vs the best feedback baseline** (simulation;
>    live validation in progress).

---

## 1. Motivation

LLM inference energy is a first-order cost, but existing models pick two of three:
**interpretable** (roofline/analytical: LIMINAL, LLM-Viewer, "Tokens-to-Watt-hours"
— datasheet constants, never validated against measured power), **measured and
predictive** (WattGPU — learned XGBoost, not interpretable, excludes MoE), or
**controlled** (POLCA, reactive power capping without a predictive model).

We target a model whose coefficients are **measured, physically meaningful, and
transferable** — and then use it for two concrete decisions:

> **(a) Which GPU — or which GPU for each serving phase — minimizes energy?**
> **(b) How should a shared, fluctuating power budget be allocated across a
> heterogeneous fleet under latency SLOs?**

Prefill is compute-bound and decode is memory-bound, so both answers depend on
getting the memory/compute split right.

---

## 2. Model

**Energy (per ~200 ms bin), fit by OLS on measured NVML power:**
```
E_bin − P_static·Δt  =  e_wbyte·weight_bytes + e_kvbyte·kv_bytes + e_gemm·gemm_flops
```
- **Y:** measured dynamic energy (∫P dt − idle·Δt). **X:** analytic, from model
  architecture (`models.py`) + vLLM per-iteration scheduler logs (resident KV tokens,
  prefill/decode tokens). **gemm_flops use *computed* prefill tokens** (prefix-cache
  hits excluded — §3d). `weight_bytes` counts all layer weights + lm_head once
  (the input-embedding gather is excluded; this convention matters, see §3c).
- **Why 3 terms, not more:** attention in decode is memory-bound — within a model
  its work is exactly proportional to resident KV tokens, collinear with `kv_bytes`.
  A separate attention-FLOP term is unidentifiable (returns an unphysical ~65
  pJ/flop). Model params as extra regressors would be collinear *and* would turn
  hardware constants into per-model fits.

**Time (per iteration) — the roofline, plus the part LIMINAL leaves out:**
```
t_iter ≈ t_overhead(host CPU launch, ~1.4–2.2 ms/step) + per-layer max(latency floor,
          weight-streaming at asymptotic MBU η) + KV + per-token terms
```
Asymptotic **η = 0.72 (H200), 0.86–0.87 (B200)** — found independently by two
analyses (the recommender and the control-plant system ID). The v1 "MBU ≈ 45%" was
an *effective* number that hid 2–5 ms of fixed per-iteration CPU overhead; that
overhead is why a 7B model runs only ~10% faster on B200 than on H200.

**Performance is a max, energy is a sum.** Roofline time is bounded by the bottleneck
resource; energy adds across resources that all draw power simultaneously. So
`power = E_iter/t_iter`, `perf/watt = tokens_s/power` — the energy layer LIMINAL lacks.

**Domain of validity:** uncapped GPUs (a GPU pinned at its power limit is in a
constant-power regime — §3c), fp16, one GPU, contexts ≲ 1.3k tokens, ≤ 64 sequences,
realized MBU ≳ 0.2 (at very low utilization per-iteration fixed energy inflates J/byte).

---

## 3. Results

### 3a. The memory coefficient is a hardware constant — within a memory technology

H200, 9,274 bins over 4 dense models (canonical fit, `gpu_coefficients.json`):
**`e_wbyte` = 1.066 [1.063, 1.069]×10⁻¹⁰ J/B, `e_kvbyte` = 3.29e-10, `e_gemm` = 0.794
[0.773, 0.822] pJ/flop; R² 0.84, held-out-by-model MAPE 6.6%.** The FLOP term is
essential (bytes-only R² is negative); KV bytes cost ~3× weight bytes per byte.
- **Size-independent:** Qwen2.5 0.5B→32B (64×), `e_wbyte` 1.06–1.14e-10 for ≥1.5B;
  **a single 7B calibration predicts every size at 7.2% MAPE** (leave-one-size-out
  4.6–6.9%).
- **Architecture transfer:** dense-calibrated coefficients predict a held-out 30B MoE
  at R² 0.79 once weight traffic is modeled by expert occupancy `E·(1−(1−k/E)^t)`
  (R² −1.35 without it).

### 3b. The datasheet scaling law is falsified; memory technology sets energy per byte

Direct c=1 measurement (dynamic energy ÷ bytes over the run — no regression):

| GPU (memory) | J/byte, Qwen2-7B c=1 | 1/BW-law prediction |
|---|---|---|
| H200 (HBM3e, 4.8 TB/s) | **1.076e-10** | anchor |
| B200 (HBM3e, 8.0 TB/s) | **1.252e-10** (flat 1.24–1.26 for 7B/32B/72B, MBU 0.31→0.62) | 0.646e-10 |
| A5000 (GDDR6, 0.77 TB/s) | **2.70–2.78e-10** (lower bound; capped) · DRAM-stream microbench 2.6–3.1e-10 | 6.7e-10 |

- B200 is **1.16×** H200 per byte on the same memory technology (law: 0.60×); a 72B
  that doubles realized bandwidth does not lower J/byte, ruling out "the 7B just
  doesn't exercise the B200."
- GDDR6 is **2.2–2.8×** HBM3e per byte but only **0.40–0.50×** the law's prediction.
  DRAM-only energy ≈ 22–28 pJ/bit; ~30% of a streamed byte's energy is on-chip.
- **Not hidden on-chip traffic:** on the same GPU and clock state, serving costs
  1.04–1.14× a pure weight-streaming GEMV per analytic byte.
- Ordering so far: **HBM3e (1.07–1.25) < HBM2e (~1.75, weak) < GDDR6 (≥2.7) ×10⁻¹⁰
  J/B**. Confounded with process node and V/f point (medium-strong evidence).
- **A convention trap we hit:** a reconstructed `models.py` counted both embedding
  tables (+7.7% bytes for Qwen2-7B); all cross-GPU numbers above use one canonical
  convention (`reconcile_gpus.py`).

### 3c. Where the linear model breaks (measured boundaries)
- **Power-capped GPUs:** an A100-PCIe at 300 W and all node-d1 A5000s at an enforced
  100 W sit at the cap at every operating point → energy ≈ `P_cap·t`, fits have
  negative R². Such GPUs are gated out of the coefficient table.
- **Saturating models:** a 72B on B200 pins power near its ceiling (CV of dynamic
  energy 0.05) → coefficients degenerate to a ratio of means (R² −0.57). Fits need a
  model big enough to use the GPU but small enough that power still swings (7B–32B).

### 3d. The compute coefficient carries real systematic uncertainty
- **Prefix-cache bias (corrected):** logged prefill tokens include prefix-cache hits;
  26–36% of prompt tokens at c=64 were never computed. Using computed tokens raises
  `e_gemm` +17% (H200) / +19% (B200), CIs disjoint, R² and held-out error improve.
- **Power-smoothing bias (not yet adopted):** per-bin NVML power on H200/B200 behaves
  like a ~0.4 s smear; correcting raises `e_gemm` a further 26–30% (R² 0.80→0.86),
  moves `e_wbyte` −1–2%. Cause (1 s averaging vs timestamp lag) is being settled on B200.
- ⇒ **Absolute prefill energy is uncertain at the tens-of-percent level; the
  B200/H200 `e_gemm` ratio is robust (~0.79–0.81 under every variant)**, so relative
  prefill conclusions hold. Decode (memory-dominated) is robust.

### 3e. Energy splits by phase
Decode-heavy bins: ~94% memory energy; prefill-heavy: ~35% compute (6× shift).

---

## 4. Application 1 — energy-optimal GPU selection (recommender v2.1)

Measured per-GPU coefficients + the realized-utilization time model + power-cap
throttling + prefix-cache fraction; unmeasured GPUs get memory-technology-class
priors (labeled LOW confidence).

**Backtest on every measured serving run** (prediction from model, GPU, measured
prompt/gen lengths, concurrency only):

| | J/token MAPE | v1 (datasheet) |
|---|---|---|
| H200 (110 runs: dense, size ladder, MoE) | **4.1%** (6.0% leave-one-model-out) | 34% |
| B200 (27 runs: 7B/32B/72B) | **3.7%** | 48% |
| A100-PCIe (capped; prior tuned on these) | 5.6% | 62% |

**H200 vs B200 win map** (prompt 1024 / gen 256):
- **Decode: H200 wins by 1.05–1.44× at ≤64 concurrent sequences — confident**
  (P ≥ 0.9 nearly everywhere). B200's +16% J/byte and 2× idle power (238 vs 116 W)
  are never recovered at realized utilization. Beyond 128 sequences: a toss-up.
- **Prefill: B200 ~13–15% cheaper for ≥3B at batch ≥4 — probable, not confident**
  (P 0.78–0.88); its compute demand approaches its 1000 W cap.
- **Phase reversal:** B200-prefill / H200-decode (the credible pair).
  *[Pending: win map with memory-technology priors for L40S (GDDR6) / A100 (HBM2e).]*

---

## 5. Application 2 — model-predictive power control (simulation; live test in progress)

**Plant (system identification).** A discrete-time model of one GPU serving one
model: queue, running batch, staged decode progress and KV occupancy as state;
admission cap and power cap as inputs; our energy model + time model as outputs.
Held-out iteration-time error 1.1–5.1% per GPU×model (pure roofline 2–41%); 1–10 s
ahead predictions beat persistence for running batch, KV and throughput. *Untested
in logs:* real queues, a binding cap, KV-full preemption.

**Controller.** Economic MPC (HiGHS MILP over per-GPU caps, routing and on/off, 10 s
horizon, Little's-law SLO, drain-aware budget, online correction); solve 8–42 ms mean
for 2–8 GPUs. Compared against uncapped, POLCA-style reactive capping, PI on
admission, PI on power cap, and a tuned static setting.

| scenario | MPC vs best feedback baseline (PI-cap) |
|---|---|
| 2×H200 + 2×B200, 7B | **−12%** energy (−9% on the calibrated plant) |
| 4×H200, 7B | **−17%** (−14%) |
| 32B | **−22%** (−18%) |

- **SLO-aware power capping is the dominant lever** — any feedback controller gets
  −48–52% vs uncapped by running at the lowest power that meets TPOT (bigger batches
  amortize weight reads).
- **MPC's extra gain comes from model-based allocation, not prediction:**
  consolidation/on-off ~6%, heterogeneity-awareness 4–8%; a 1-step horizon matches a
  10-step one; a perfect forecast adds 0.8%.
- **Our own hypothesis was wrong:** MPC's advantage is the same on flat and
  fluctuating budgets; *load variability* is what matters. On steady load a tuned
  static setting is within 2.5% — MPC is overkill there.
- Robust to ±20% coefficient error (<2% energy); online adaptation is essential.
- Heterogeneous routing depends on unmeasured parameters (B200 minimum power limit,
  parked power, wake time) — flagged, with sensitivity sweeps.
- *[Pending: disaggregated prefill/decode pools under a shared budget — the setting
  of the closest prior work; live closed-loop validation on an A5000 with admission
  control as the actuator (power limits are not settable without root).]*

---

## 6. Honest positioning vs prior work

| piece | closest prior | our delta |
|---|---|---|
| memory+compute energy decomposition | Choi et al. 2013; Horowitz 2014 | not novel as a form |
| analytical LLM-inference energy | "Tokens-to-Watt-hours" (2025) | datasheet constants; no measured calibration or validation |
| cross-GPU power prediction | WattGPU | black-box, excludes MoE; we are interpretable, MoE-aware, and show *why* datasheet scaling fails |
| MoE expert-occupancy bytes | MoE-CAP, MoE latency work | applied to energy; enables cross-arch transfer |
| MPC for phase power caps | **arXiv 2609.11133 (this month)** | they calibrate per config and punt on heterogeneous clusters; we do heterogeneous fleets with a validated plant, and find the value is model-based allocation, not forecasting |
| disaggregated prefill/decode | DistServe, Splitwise | static/heuristic placement; we supply the energy model to choose hardware per phase |

**What is ours:** a measured, interpretable, transferable energy model with a
validated time model; the measurement that falsifies datasheet scaling and points to
memory technology; a recommender validated at ~4%; and a model-based controller that
exploits fleet heterogeneity. Framing: a **measurement + systems** paper (MLSys /
workshop), with the controls result as a second contribution.

## 7. Risks / open threats
- **Byte counts are analytic, not hardware-measured.** Path: `ncu` is available on the
  B200 node (permission probe pending) — `HANDOFF_TO_B200_v2.md`.
- **Compute coefficient uncertainty** (§3d) — absolute prefill energy ±tens of %.
- **Memory-technology claim** has one GDDR6 part, power-capped, confounded with process
  node and V/f.
- **Time-model overhead is host-specific** (zero-shot transfer to a new host was 33%
  off; a 30 s calibration run fixes it).
- **All control results are simulation** so far; cap floors, parked power and wake
  time are assumptions.
- **Compute access:** H200 access was revoked on 2026-09-24 (all H200 raw data is
  backed up in the repo); the only local GPUs (A5000s) are capped at 100 W by another
  user.

## 8. Plan (by leverage)
1. **Hardware byte validation with `ncu` on B200** — DRAM/L2/L1 bytes per op and per
   decode step vs our analytic counts.
2. **Live closed-loop control** on an A5000 (in progress) — validates the plant's
   queue/saturation dynamics that logs never exercised.
3. **Settle `e_gemm`**: B200 smoothing test; decide on the smoothing correction.
4. **An uncapped GDDR6 / HBM2e measurement** (needs a power-limit change or other
   hardware) to firm up the memory-technology ordering.
5. **Disaggregated pools in the MPC** (in progress).

---
*Artifacts: `gpu_coefficients.json` (canonical), `realized_utilization.csv`,
`reconcile_gpus.py`, `recommend_gpu.py` + `recommender_backtest.py` + `RECOMMENDER.md`,
`controls/` (`plant.py`, `sysid.py`, `controllers.py`, `SYSID.md`, `RESULTS.md`),
`microbench/`, `FINDINGS_B200.md`, `FINDINGS_A5000.md`, `CONTROL_THEORY_NOTES.md`.
Full narrative: `SESSION_LOG.md`.*
