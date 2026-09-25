# A5000 (GDDR6) energy measurement — findings

Run 2026-09-24 on node-d1 (debug partition), RTX A5000 24 GB GDDR6, sm_86.
Measurement by the physics agent; written up by the lead from its report,
`microbench/a5000_summary.json`, `microbench/a5000_analysis.py`,
`microbench/nvml_smoothing_check.json`. Raw data: `logs/A5000/` (16-run protocol
sweep), `logs/A5000_samegpu/`, `logs/A5000_duty/`, `microbench/results_A5000*.json`.

## Headline

**Energy per byte is set by memory technology.** On GDDR6 it is **2.2–2.8×** the
HBM3e parts (H200/B200), and **0.40–0.50×** what the datasheet 1/bandwidth law
predicts. Both the 1/BW law and plain cross-technology invariance are rejected.

| | J/byte | vs H200 | vs B200 |
|---|---|---|---|
| H200 Qwen2-7B serving, c=1 direct | 1.076e-10 | 1.0 | — |
| B200 Qwen2-7B serving, c=1 direct | 1.252e-10 | 1.16 | 1.0 |
| **A5000 Qwen2-7B serving, c=1 direct (lower bound)** | **2.70–2.78e-10** | 2.5–2.6 | 2.2 |
| A5000 Qwen2.5-3B serving, c=1 direct (lower bound) | 2.87–3.01e-10 | 2.7–2.8 | 2.3–2.4 |
| A5000 DRAM stream, full clocks (microbench) | 2.6–3.1e-10 | | |
| A5000 cuBLAS GEMV, full clocks | 2.7–3.6e-10 | | |
| A5000 L2-resident stream | 0.83e-10 | | |
| A5000 L1-resident stream | ~0.2e-10 (3 pts, weak) | | |
| 1/BW-law prediction for A5000 | 6.7e-10 | | |

- **DRAM-side energy alone** (DRAM stream − L2 stream) ≈ **1.8–2.3e-10 J/B
  (22–28 pJ/bit)**. That alone exceeds the *entire* H200 serving J/byte.
- About **30%** of a streamed byte's energy is on-chip (L2 path).
- **GEMM** (fp16): 98–100 TFLOPS achieved = 88–90% of the 111.1 TFLOPS dense
  datasheet figure (222.2 is sparse). `e_gemm` = **2–4 pJ/flop**, against 6.1 pJ
  (1/peak law) and 0.68 pJ (H200 invariance). Neither prediction holds.

## The explanation that is NOT the main story

"Our analytic bytes hide on-chip traffic" was tested directly. On **the same GPU
in the same throttled clock state**, serving costs **1.04–1.14× (7B) / 1.14–1.29×
(3B)** per analytic byte what a pure 512 MB weight-streaming GEMV costs, and that
ratio holds for **any** assumed idle floor. So serving energy is essentially DRAM
weight-streaming energy, with no large hidden on-chip excess. The lumped
coefficient still contains the ~30% on-chip share of streaming itself.

## Evidence strength

- **Against invariance across memory technologies: strong.** The capped *lower bound*
  is already ≥2.2× B200.
- **Against the 1/BW law: fairly strong.** No choice of idle floor rescues it; even
  an impossibly low floor (21 W, no CUDA context) gives 5.4e-10 < 6.7e-10.
- **For memory technology: medium-strong.** Confounded with process node (Samsung 8N
  vs TSMC 4N) and the voltage/frequency operating point: per-byte on-chip energy
  changes with clock (L2 stream 0.83e-10 at 1.9 GHz vs ~0.3e-10 at 1.1 GHz).
- With the A100-PCIe (HBM2e) implying ~1.7–1.8e-10 (from capped runs, weak), the
  ordering is **HBM3e (1.07–1.25) < HBM2e (~1.75) < GDDR6 (≥2.7) ×10⁻¹⁰ J/byte**.

## The big caveat: every A5000 on node-d1 is capped at 100 W

The enforced power limit on **all four** A5000s is **100 W** (TDP 230 W). It was set
by another user, and we cannot change it without root. So:
- All 16 protocol serving runs sat at 99.3–102.5 W, in the constant-power regime.
  The SM clock was held at 210 MHz at c=1 (Qwen2-7B 10.5 tok/s, 95 ms steps,
  realized BW 149 GB/s, MBU 0.19–0.20).
- **The 3-term fit is invalid** (R² = −35, CV(dyn) = 0.043). `reconcile_gpus.py`
  now gates this out: A5000 is listed under `_excluded` in `gpu_coefficients.json`.
  Do not quote `e_wbyte` 2.07e-10 / `e_gemm` 0.25 pJ from that fit.
- Serving J/byte is a **lower bound**. The vLLM idle used for subtraction (60.5 W at
  SM 1695 MHz) overstates the true floor at 210 MHz. Self-consistency puts 7B at
  ~2.8–3.4e-10.
- The full-clock numbers come from **short bursts at low duty with re-measured
  floors**. The cap controller reacts within 5–20 ms, so steady full-clock serving is
  not measurable on this node.
- Per-run limits: `logs/A5000/gpu_power_limits.csv` + per-run `gpu_limit.json`. All
  A5000 sbatch files now log the enforced limit to `logs/A5000_gpu_limits/`.

## Surprises worth knowing (apply to other GPUs too)

1. **The NVML total-energy counter read only 3–26% of true board power** on this part.
   Instantaneous power (`POWER_INSTANT`, field 186) and the 1 s average agree within
   0.5%. Use power, not the energy counter.
2. **Idle depends on clock state:** 54 / 60.5 / 86 W on one GPU.
3. Duty cycles with gaps ≥ 50 ms let the floor sag, so burst energy vanishes from
   floor-subtracted power. Only short-gap points are valid.
4. DRAM-bound steady work drew 110–123 W, *over* the 100 W cap. The memory clock is
   not reduced by the cap.
5. **NVML power smoothing on the H200/B200 per-bin data.** Re-fitting the existing
   per-bin data with the regressors smeared by a moving window is best explained by
   a **~0.4 s smear** (H200 R² 0.803 → 0.859; B200 0.658 → 0.730). Correcting for it
   barely moves `e_wbyte` (−1 to −2%), lowers `e_kvbyte` 17–40%, and raises `e_gemm`
   26–30% (H200 0.68 → 0.86 pJ, B200 0.54 → 0.70 pJ; computed on the logged-token
   FLOPs, i.e. *before* the separate prefix-cache correction). Run-level ratios
   (all direct J/byte numbers) are unaffected. The cause is unsettled (NVML 1 s
   averaging vs timestamp lag); the B200 `square` test in `HANDOFF_TO_B200_v2.md`
   settles it.

## What this means for the model

- **The memory coefficient is robust**: ±2% under both the smoothing and prefix-cache
  corrections, flat within a GPU, and set by memory technology across GPUs.
- **The compute coefficient carries large systematic uncertainty.** Two independent,
  physically-motivated corrections both push `e_gemm` *up*: prefix-cache (+17–19%,
  adopted) and power smoothing (+26–30%, *not* adopted pending the B200 test).
  Absolute prefill energy is therefore uncertain at the tens-of-percent level. The
  **B200/H200 `e_gemm` ratio is robust** (~0.79–0.81 under every variant), so
  *relative* prefill conclusions hold.
- For the recommender, the GDDR6 prior should move from "unknown" to e_byte ≈
  2.7–3.6e-10 J/B (GEMV at full clocks, i.e. uncapped behavior), noting the process
  and V/f confounds.

## Next (in HANDOFF_TO_B200_v2.md)

1. `ncu` permission probe → per-op DRAM/L2/L1 bytes → serving decode bytes via
   `ncu_dram_check.py --profile_range`: hardware validation of the analytic byte
   model, which we have never been able to do.
2. The energy microbenchmark suite on B200 (incl. the smoothing `square` test).
3. dcgmi `DRAM_ACTIVE` as fallback.

A local `ncu` permission probe + end-to-end smoke of the default suite (job 84945,
`microbench/ncu_A5000/`, `microbench/smoke_full/`) was queued when this was written.
Check it before the B200 session relies on the handoff.

## Addendum — job 84945 (ncu probe + full-suite smoke)
- **`ncu` counters are blocked on node-d1** (`ERR_NVGPUCTRPERM`, same as the old
  cluster): the per-op DRAM/L2 byte check cannot run here. It needs either an admin
  (`NVreg_RestrictProfilingToAdminUsers=0`) or the B200 node, where
  `HANDOFF_TO_B200_v2.md` probes permission first and falls back to `dcgmi`.
- **The full default microbenchmark suite passed end-to-end** (~6 min, short
  windows, every section wrote output: duty, duty10, burstlen, burstscan, stream,
  grid, vendor, gemv, gemm, square, idle checks). Safe to run on B200.
- The GPU used (CA) was also capped at 100 W; its capped steady-state numbers match
  the earlier runs within ~3% (DRAM stream 2.2e-10 J/B, GEMV 2.9e-10 J/B, GEMM
  25–30 TFLOPS under cap). Limits: `logs/A5000_gpu_limits/job_84945.csv`.
