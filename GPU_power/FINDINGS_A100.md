# A100 80GB PCIe: measured energy coefficients, idle states, power roof (2026-10-21)

The A100s are on Kubernetes (KAI scheduler) and launched with `runai/launch_a100_kai.sh`.

**Hardware:**
- NVIDIA A100 80GB PCIe on node8; 300 W enforced limit, which is also its maximum.
- SM clock 1410 MHz max; HBM2e at 1.935 TB/s peak; 40 MB L2.

**Software:** vLLM 0.10.2, torch 2.8, transformers 4.56, identical to the B200/H200 container.

**Readings:**
- POWER_INSTANT is available. The energy counter agrees with it (0.998–0.999).
- On GA100, `GetPowerUsage` is **not** a ~1 s average: its fitted window is 0.15 s, the same as POWER_INSTANT. So the September A100 runs weren't smeared.
- NCU is blocked (`ERR_NVGPUCTRPERM`), so byte counts come from the B200 validation (they're a workload property).

Raw data:
- `microbench/results_A100_steady.json` and `microbench/steady_calib_A100.json`
- `logs/A100_prov/idle_states.json` (from the `a100-more` job log)
- `runai/a100_job_*.log`

## Energy coefficients (duty-cycle method: bursts at full clock, mean power under the cap)
| quantity | A100 | B200 (same method) | A100 / B200 |
|---|---|---|---|
| DRAM stream, J/B | **1.368e-10** ± 0.029 (R² 0.996) | 1.283e-10 | 1.07 |
| occupancy grid, uncapped (11–70% of peak BW) | 26.7 W + **1.265e-10 J/B**·BW (R² 1.000) | 38.7 W + 1.251e-10 | 1.01 |
| L2-resident stream, J/B | 4.62e-11 | 3.31e-11 | 1.40 |
| L1-resident stream, J/B | 1.34e-11 | 1.10e-11 | 1.22 |
| batch-1 GEMV, J/B | 1.83e-10 | 1.33e-10 | 1.38 |
| fp16 GEMM 4096³, J/FLOP | **1.47 pJ** | 0.77 pJ | 1.91 |
| capped DRAM stream (SM throttled to ~1260 MHz) | 1.09–1.12e-10 at 1.83 TB/s (95% of peak) | 1.03–1.05e-10 | |
| kernel launch | 47–73 µJ | 16.7 µJ | |

**Reading:**
- **The EnergAIzer-derived prior (1.36e-10, boost clock) is confirmed to 0.6%.**
- DRAM energy per byte is nearly flat across memory generations (HBM2e 1.37, HBM3e 1.07–1.28e-10).
- The generational gain is on-chip: L2/L1 bytes are 20–40% cheaper on B200, and compute is 1.9× cheaper per FLOP (7 nm → 4 nm).
- Per-channel split, assuming B200's NCU traffic ratios (each DRAM byte crosses L2 1.5×): e_L1 ≈ 1.34e-11, e_L2 ≈ 3.3e-11, e_DRAM ≈ 0.74e-10 J/B. That puts about 46% of a streamed byte's energy on-chip, against 35% on B200. It's an estimate: NCU is blocked here.

## Idle states (`idle_states.py`, NVML at 10 ms)
| state | power | SM clock |
|---|---|---|
| no process on the GPU | **47.5 W** | 210 MHz |
| CUDA context open, idle | 97.2 W | 1410 MHz |
| vLLM loaded, idle (= the serving harness's P_static) | **100.2 W** (harness: 97.7 W) | 1410 MHz |
| after engine shutdown, process alive | 99.7 W | 1410 MHz |
| 90 s later, still alive | 99.2 W | 1410 MHz (no decay) |

- **On the A100, any live CUDA process holds the clocks at maximum and doubles idle
  power.** A provisioned GPU with nothing attached draws 47.5 W; a serving-ready one
  draws about 99 W.
- The RTX A6000 behaves differently: its P-states drop with vLLM loaded and idle
  (84–86 W). Its energy counter is broken (reads ~5 W), like the A5000's, so the
  counter check rejects it.

## The A100 is power-limited for all LLM serving we measured
- **Every LLM-shaped GEMM loop ran at the cap**, including batch-1 (M = 1) decode
  shapes (32 of 32 points, SM 953–1357 MHz).
- **Every long-context serving run so far is at the cap** (298 W, cap 100% of samples).
- **The budget:** 300 W − 99.6 W (clocks-up idle) − 26.7 W (activity) = 174 W. At
  full clock that streams 174 / 1.265e-10 ≈ 1.37 TB/s, which is 71% of peak bandwidth.
  A 7B decode step reads 14.1 GB of weights, so it can't run faster than about 11.5 ms
  under the cap (≈ 1.23 TB/s), whatever the batch.

**The power roof brackets measured decode step times** (Qwen2-7B, decode steps outside
prefill windows, long-context arm):

| run | measured step | predicted, full-clock coefficients | predicted, throttled stream (1.10e-10) |
|---|---|---|---|
| ctx 1k, c=4 | 11.43 ms | 11.91 (+4%) | 9.50 (−17%) |
| ctx 8k, c=4 | 12.51 | 13.58 (+9%) | 10.53 (−16%) |
| ctx 30k, c=4 | 15.82 | 18.83 (+19%) | 13.81 (−13%) |
| ctx 1k, c=16 | 12.44 | 14.25 (+15%) | 11.13 (−11%) |
| ctx 8k, c=16 | 16.64 | 20.91 (+26%) | 15.29 (−8%) |

So the A100 runs at the cap with partly throttled clocks. First-principles power-roof
time is within about ±15%, and calibrating an effective at-cap J/B is the next step
(LIMINAL-E). Serving-level KV and compute coefficients **can't** be fitted on A100:
under constant power, energy just tracks time.

## Still running / next
- **`a100-more`:** long context and prefill at the cap; the Qwen2.5 size ladder
  0.5B–14B in eager mode (measured-vs-measured against the H200 ladder); Qwen2-7B/14B
  eager vs CUDA graphs; up to 256 concurrent requests.
- **`a6000-energy`:** the uncapped GDDR6 point. Same microbenchmark and serving grids,
  plus idle states.
