# RTX A6000 (48 GB GDDR6): measured coefficients, idle states, serving (2026-10-21)

Launched via KAI (`runai/launch_a100_kai.sh`, `GPU_TYPE=a6000`).

**Hardware:**
- NVIDIA RTX A6000 (GA102, the same die family as the A5000).
- 300 W enforced limit, which is also its maximum.
- SM clock 2100 MHz max; GDDR6 at 8001 MHz, 0.768 TB/s peak; 48 GB.

**Readings:**
- POWER_INSTANT is available. `GetPowerUsage` is a ~1.05 s moving average, as on the
  A5000.
- **The energy counter is broken:** it reads 2–5 W at an 85 W board power. The counter
  check rejects it, so bins use POWER_INSTANT.
- NCU isn't available.

Data:
- `microbench/results_A6000_steady.json` and `microbench/results_A6000_steady_calib.json`
- `logs/A6000_prov/`
- `runai/a100_job_20261021_095725.log` (despite the name, it's this job's log)

## Coefficients (duty-cycle method, full clock) vs the HBM parts
| | A6000 (GDDR6) | A100 (HBM2e) | B200 (HBM3e) |
|---|---|---|---|
| batch-1 GEMV (weight-streaming proxy), J/B | **2.59e-10** (R² 0.999) | 1.83e-10 | 1.33e-10 |
| DRAM stream, J/B | 2.74e-10 (R² 0.72, noisy) | 1.37e-10 | 1.28e-10 |
| L2-resident / L1-resident, J/B | 6.4e-11 / 2.4e-11 | 4.6e-11 / 1.3e-11 | 3.3e-11 / 1.1e-11 |
| fp16 GEMM 4096³, J/FLOP | **2.64 pJ** | 1.47 pJ | 0.77 pJ |
| kernel launch | 470–490 µJ | 47–73 µJ | 17 µJ |

- **GDDR6 has a large activity cost.** In the occupancy sweep, 0.22 TB/s (29% of peak)
  already draws 194 W of dynamic power, and 0.45 TB/s draws 234 W.
  - The cause: memory clocks jump from 417 MHz at idle to 8001 MHz whenever there's
    traffic.
  - The marginal cost on top of that is about 1.75e-10 J/B (two uncapped points only).
  - So most of a GDDR6 byte's cost is the fixed price of keeping the memory awake. That
    is also why a kernel launch costs ~10× more energy than on the A100.
- From 0.68 TB/s up, everything is at the 300 W cap. It still reaches 0.71 TB/s
  (93% of peak) with the SM clock barely throttled (~1857 MHz).

## Idle states
| state | A6000 | A100 |
|---|---|---|
| no process | **32.5 W** (SM 228, memory 417 MHz) | 47.5 W |
| CUDA context, idle | 73 W | 97 W |
| vLLM loaded, idle | **85.8 W** | 100 W |
| after engine shutdown | 45 W (drops back on its own) | 99.7 W (stays at max clock) |

Unlike the A100, the A6000 drops P-states by itself while idle, even with a process
alive.

## Serving: power-capped and bandwidth-bound at once
Every serving run sits at the 300 W cap, and bandwidth utilization is the highest of any
GPU we've measured: 0.79–0.81 for Qwen2-7B at batch 1, 0.86 for 14B. CUDA graphs add
only ~2%.

Qwen2-7B alpaca, J/token (mJ) and tok/s:

| concurrency | A6000 | A100 | B200 | H200 (graphs-adjusted est.) | A6000 / A100 | A6000 / B200 | A6000 / H200 |
|---|---|---|---|---|---|---|---|
| 1 | 6855 (44) | 3333 (89) | 2934 (218) | 1974 | 2.06 | 2.34 | 3.47 |
| 4 | 1832 (163) | 875 (341) | 768 (840) | 508 | 2.09 | 2.39 | 3.61 |
| 16 | 480 (622) | 232 (1286) | 206 (3183) | 144 | 2.07 | 2.34 | 3.34 |
| 64 | 137 (2177) | 72 (4124) | 68 (10408) | 53 | 1.90 | 2.02 | 2.59 |
| 128 | 88 (3374) | 54 (5535) | 42 (18779) | — | 1.65 | 2.10 | — |
| 256 | 80 (3720) | 43 (6839) | 31 (27347) | — | 1.86 | 2.57 | — |

**Memory technology is the first-order factor.** Across three HBM generations
(A100 HBM2e, H200 and B200 HBM3e), DRAM energy per byte stays within 1.07–1.37e-10 J/B.
The GDDR6 A6000 costs about 2× per byte, mostly as fixed memory-activity power, and
2.0–3.6× per token.

This is the second GDDR6 part we've measured, after the 100 W-capped A5000, and it
confirms the GDDR6 prior (2.3e-10 [1.3, 3.1]) for boost-clock serving.
