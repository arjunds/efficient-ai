# Energy Profiling — Meeting Brief

Standalone talk-ready summary of the `vllm_ragged` profiling iteration (LIMINAL
energy extension). Data collected on **NVIDIA H200** via SLURM + Apptainer.

> Status: **4 models fully swept** (Qwen2-7B, Llama-3-8B, Mistral-7B, gemma-7b) ×
> concurrency 1–64 × two workloads, plus long-context, Poisson, and waveform
> validation runs. A100 comparison queued for cross-GPU.

---

## TL;DR (say this)
- We built an **offered-load** energy profiler and calibrated a **per-iteration**
  energy model, then **validated it by predicting the measured GPU power
  waveform** — not just matching an average.
- **Headline efficiency result:** tokens/joule improves **~44×** from concurrency
  1→64 while power rises only ~11%.
- **Headline modeling result:** the energy-per-byte coefficient **`e_bit` is not
  constant** — it grows ~2.6× with concurrency and context as the workload shifts
  memory-bound → compute-bound. That's the finding that should drive the next
  model refinement.

## FOLLOW-UP RESULT: two-term (memory+compute) model on ragged traffic
The single-term `e_bit` drifted 2.6× with load/context — evidence a compute term
was missing. We added it: `E = e_bit·bytes + e_flop·FLOPs + P_static·t`, fit on
real ragged traffic (alpaca + sharegpt, variable-length → heterogeneous batches;
sharegpt's long prompts give arithmetic-intensity spread up to ~2000×), pooled
across each model's operating points.

| model | e_bit (J/byte) | e_flop (pJ/flop) | R²_dyn |
|-------|---------------:|-----------------:|-------:|
| Qwen2-7B   | 1.11e-10 | 0.86 | 0.65 |
| Llama-3-8B | 1.11e-10 | 0.86 | 0.69 |
| Mistral-7B | 1.10e-10 | 0.85 | 0.62 |
| gemma-7b   | 1.18e-10 | 0.98 | 0.59 |

- **The split flattens `e_bit`**: from a 2.6× drift down to <7% spread across four
  architectures. A constant memory coefficient + a compute term replaces the
  drifting single coefficient — the model was right, it just needed the FLOP term.
- **Both coefficients are ~constant across models → hardware constants** (`e_bit` ≈
  H200 HBM energy/byte, `e_flop` ≈ 0.86 pJ/flop compute energy). This is the key
  claim: the coefficients are properties of the GPU, not fits to a model — so the
  model should predict an *unseen* model's energy on H200 from its byte/FLOP
  counts. That elevates it from "curve fit" to "predictive model."
- Fit is well-identified (bytes↔flops r≈−0.6, AI span ~2000×), R²_dyn 0.6–0.7.
- **Rigorous checks (the evidence, not the R² alone):**
  - FLOP term is *essential*: bytes-only R² is **negative** (−0.2 to −0.9); adding
    FLOPs is what makes the model work.
  - Coefficients tightly pinned (bootstrap 95%): `e_bit = 1.120 [1.115,1.126]×10⁻¹⁰
    J/byte`, `e_flop = 0.875 [0.846,0.902] pJ/flop` — moderate per-bin R² but
    precise slopes (9k bins over 2000× intensity).
  - Held-out (leave-one-model-out) prediction: **6.5–11% MAPE** — coefficients
    transfer across models, not per-model fits.
  - Robustness: a more detailed prefill-attention FLOP term changes nothing (R²
    0.69→0.69) — the result isn't sensitive to that approximation.
- Plots: `plots_two_term/coefficients_by_model.png`, `two_term_fit_quality.png`
  (the drift plot was dropped — the held-out + bytes-only-vs-two-term + CIs above
  are the honest evidence).
- **Cross-GPU (A100 80GB PCIe): revealed the model's domain of validity.** The
  A100 PCIe (300 W cap) pegs its power limit (~296 W) at *every* operating point —
  even concurrency 1 — so power is DVFS-clamped ~constant and energy ≈ P_cap·t, not
  bytes/FLOPs. The linear two-term model breaks there (R² negative); it's valid in
  the *uncapped* regime (H200 had 700 W headroom). Implication: add a cap term
  `P = min(e_bit·byte_rate + e_flop·flop_rate + P_static, P_cap)`. A clean A100
  coefficient needs an unpegged part (SXM 400 W) or a raised limit. This maps where
  the model applies — a strengthening result, not a failure.

## Methodology (brief)
- **Offered load, not static batch.** Drive vLLM (0.10.2, V1 engine) with N
  concurrent clients (closed-loop) *or* Poisson arrivals (open-loop), so vLLM's
  continuous-batching scheduler forms real **ragged** running batches (mixed
  prefill/decode, mixed context lengths).
- **One shared clock.** NVML power/clock/util/temp (~10 ms) + per-iteration vLLM
  scheduler state (running/waiting counts, KV-cache usage → resident KV tokens,
  prefill/decode token split) via a `StatLoggerBase`, all on the same wall clock.
- **Energy model** (LIMINAL extension):
  `E_iter = e_bit·(weight_bytes + Σ_seq KV_bytes(ctx)) + P_static·t_iter`.
  `P_static` anchored by an idle baseline; `e_bit` fit by **binned** regression
  (200 ms) of dynamic energy vs bytes moved. Bytes are **analytic** (weights =
  active_params×dtype; KV from attention geometry) — exact from config.
- **Validation.** Replay the per-iteration log through the calibrated model →
  predicted `power(t)`; compare to NVML (RMSE / R² / MAPE), including a **bursty
  active↔idle run** where power actually moves.
- Baselines: idle (P_static) + sustained-prefill ceiling.

## What THIS iteration did that the previous one didn't
The prior data had two blocking problems; both are resolved.

| | Previous | This iteration |
|---|---|---|
| Operating points | **one** (concurrency=1) | concurrency sweep 1→64, Poisson arrivals, long-context 512→30k |
| Power signal | run-level **average** only | **time-resolved** (~10 ms) aligned to **per-iteration** scheduler state |
| Model | fit to averages | per-iteration model **validated against the power waveform** (R² 0.83) |
| DRAM byte counter | **"broken"** (<3% capture, 7× inconsistent) | **diagnosed as a permissions wall** (`ERR_NVGPUCTRPERM`, perf counters admin-only) — *not a code bug*. Removed the dependency: model uses NVML power + analytic bytes. |
| Coverage | single model/point | 4 models × concurrency × context × arrival pattern |

## Key results
- **Concurrency (Qwen2-7B, 512/128):** tok/J **0.45 → 19.8** (c1→c64), power
  369 → 410 W, `e_bit` 1.08 → 1.60e-10 J/byte.
- **Long context (Qwen2-7B, c=16):** power **415 → 537 W**, `e_bit` **1.45 →
  2.81e-10** as context goes 4k → 30k (KV bytes come to rival the 14 GB weights).
- **Waveform validation:** bursty active↔idle **R² 0.83, MAPE <7%**; fixed-load
  MAPE 1.6–2.4%.
- **`e_bit` spans 1.08e-10 → 2.81e-10 (2.6×)** across load and context.
- **Cross-model (4 models, 512/128), c=1 → c=64:**

  | model | tok/J c1→c64 | e_bit c1→c64 (J/byte) |
  |-------|--------------|------------------------|
  | Qwen2-7B-Instruct | 0.45 → 19.8 | 1.08e-10 → 1.60e-10 |
  | Meta-Llama-3-8B   | 0.43 → 17.5 | 1.07e-10 → 1.69e-10 |
  | Mistral-7B-v0.1   | 0.46 → 17.0 | 1.03e-10 → 1.88e-10 |
  | gemma-7b          | 0.36 → 12.6 | 1.05e-10 → 1.85e-10 |

  **Strikingly consistent across architectures:** all four start at
  `e_bit ≈ 1.03–1.08e-10` at c=1 (pure weight-movement / memory-bound) and rise
  to `1.6–1.9e-10` by c=64 (compute contribution grows). tok/J improves 12–20×.
  This universality is a strong point — the coefficient and its load-dependence
  aren't a quirk of one model.

## The take (synthesis for the analysis-side agent)
1. **`e_bit` as a single constant is insufficient.** It rises systematically with
   arithmetic intensity: higher concurrency amortizes each weight-load over more
   tokens *and* pushes toward compute-bound; longer context adds KV traffic +
   attention FLOPs. Two clean next steps:
   - make `e_bit` a function of the running batch's arithmetic intensity
     (tokens/iter, Σ ctx), or
   - split the model into a **memory term** (weights+KV bytes, ~constant
     coefficient) **+ an explicit compute term** (attention/GEMM FLOPs).
   The pure-bytes model's MAPE climbing to ~10% in the long-context/high-
   concurrency corner is exactly where the missing compute term lives.
2. **The waveform validation is the strong result** — predicting `power(t)` from
   scheduler state shows the model captures ragged-batch *dynamics*, not averages.
3. **Caveats to state plainly:** results are **H200-specific** (A100 queued);
   **hardware byte reconciliation is blocked** by perf-counter permissions (the
   real reason the old counter was "broken" — worth raising with the cluster
   admin if a hardware cross-check is required).

## Plots for the PPT
Generated by `python3 plot_meeting.py` → `plots_meeting/` (reads the result CSVs;
if matplotlib is blocked in the container, run it on the analysis side — inputs
are small CSVs). In rough priority:

| # | file | what it shows | slide use |
|---|------|---------------|-----------|
| 1 | `tokens_per_joule_vs_concurrency.png` | tok/J vs concurrency, per model | **efficiency headline** |
| 2 | `ebit_vs_concurrency.png` | e_bit rises with load | **modeling headline** |
| 3 | `waveform_overlay.png` | predicted vs measured power(t) | **validation money shot** |
| 4 | `ebit_vs_context.png` | e_bit vs context, by concurrency | KV-term story |
| 5 | `power_vs_context.png` | power vs context | supports #4 |
| 6 | `power_vs_concurrency.png` | power vs concurrency | supports #1 |
| 7 | `throughput_vs_energy_per_token.png` | Pareto, per model | efficiency framing |

## Data locations
- `sweep_summary.csv` / `sweep_summary_H200.csv` — concurrency sweep (all models)
- `logs/longctx/` — context sweep · `logs/poisson/` — arrival-rate sweep
- `logs/wave/waveform_series.csv` — the validation overlay data
- per run: `run_meta.json`, `results.json`, `calibration.json`,
  `waveform_validation.json`
- Full narrative: `SESSION_LOG.md`. Code: branch `vllm_ragged`.
