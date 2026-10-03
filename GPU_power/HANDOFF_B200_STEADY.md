# B200 handoff — steady-state energy calibration (PI notes + EnergAIzer protocol)

**Goal.** Calibrate B200 energy coefficients from long steady loops instead of
reconstructing power in 200 ms bins. Together these runs answer the PI's notes (below).
Pure microbenchmarks: no vLLM and no model downloads. About 45 minutes on one GPU.

## Run it
```bash
cd <your GPU_power checkout> && git pull          # fork arjunds/efficient-ai, branch vllm_ragged
sbatch microbench/b200_steady_calib.sbatch         # steady + analyze (~45 min)
# optional, only if this node lets you lock clocks (always auto-reset via -rgc):
STAGES=steady,clocks,analyze sbatch microbench/b200_steady_calib.sbatch
```
The job prints a summary at the end, and `microbench/steady_calib_B200.json` holds all
the numbers. **Commit back:** `microbench/results_B200_steady.json`,
`microbench/results_B200_steady_square_multi_trace.csv`,
`microbench/steady_calib_B200.json`, `microbench/b200_steady_gpu.csv`, `slurm_out/`
for this job, and `results_B200_sm*.json` plus `b200_lgc_*.txt` if the clocks stage
ran. Then add a short section to `FINDINGS_B200.md` with the printed summary.

## What each piece answers
| PI note | test | read off |
|---|---|---|
| "blast HBM with constant read requests" | `stream` (≥16×L2 working sets), `vendor` (1 GB torch sum/copy) | J/byte at the highest sustained bandwidth, and that bandwidth as a fraction of peak (EnergAIzer's A100 reached only 0.82) |
| "track BW" / are we saturating | `grid` (occupancy sweep) | J/byte against bandwidth fraction (is it flat up to saturation?); marginal J/byte = slope of P_dyn vs BW, plus the intercept |
| DRAM vs on-chip energy | `stream` at L2-resident sizes | on-chip share of a streamed byte |
| compute coefficient without the per-bin smear | `llmgemm`: Qwen2-7B layer shapes, M = 1…8192 tokens | steady 2-feature fit `P_dyn = e_byte·BW + e_flop·FLOP/s`, with bootstrap CIs; compare with the serving `e_gemm` = 0.642 pJ |
| fixed host/launch overhead energy | `launch` | J per kernel launch and W while launch-bound (maps to our ~1.4–2.2 ms/step overhead) |
| "200 ms sampling is a lot" | `square_multi` (on/off at 0.1–2 s) | for `p_avg` (GetPowerUsage) and `p_inst` (POWER_INSTANT): a fitted **boxcar window vs lag** |
| balanced or overprovisioned | everything above | energy ridge `e_byte/e_flop` vs performance ridge `peak/BW` |

## How to read the results
- **Smoothing.** On the A5000, `GetPowerUsage` fits a **1.0 s boxcar** with only a
  0.02–0.06 s lag, while `POWER_INSTANT` fits ~0.1 s. If B200 shows the same, our
  per-bin "~0.4 s smear" is NVML averaging: use POWER_INSTANT and long windows from now
  on. If B200 shows a short window but a large lag, it is a timestamp offset instead.
- **Energy per byte.** If streaming J/byte ≈ the serving `e_wbyte` (1.243e-10), the
  serving coefficient is pure weight-streaming energy. If it is lower, serving carries
  extra (attention, KV, overhead).
- **Compute.** If the steady `e_flop` is above the serving 0.642 pJ (the A5000 hinted
  +26–30% from smoothing alone), the per-bin fits under-estimated compute energy, and
  we adopt the steady value. The analysis already refits serving with `e_gemm` held
  at the steady value.
- **Clock dependence matters.** On the A5000, steady streaming at a throttled SM clock
  (210–460 MHz) gave ~1.3e-10 J/B against 2.6–3.1e-10 for full-clock bursts, a ~2×
  clock/voltage effect. That is why the opt-in `clocks` stage exists. If clocks cannot
  be locked, the per-point `sm_mhz` is recorded, so stratify by it instead.

## Gotchas
- **Do not change power limits.** The `clocks` stage only tries `nvidia-smi -lgc` and
  always runs `-rgc` on exit. If locking is not permitted, it logs that and skips.
- Check that `enforced.power.limit` in `b200_steady_gpu.csv` equals the default
  (1000 W). If a point sits at the cap, the analysis prints a warning, and that fit is
  not a calibration.
- Keep the GPU otherwise idle for the job (an exclusive allocation is the default).
- The new tests were validated end-to-end on an A5000 (job outputs parse and every
  analysis section runs). On that capped part the fits are invalid by design, so the
  B200 is where they mean something.
