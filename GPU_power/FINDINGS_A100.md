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

## Cross-GPU energy per token: A100 vs H200 vs B200 (2026-10-21)

**A100 vs H200, both measured in eager mode, 4 models × 2 tasks** (`logs/ragged_a100`
from September vs `logs/ragged`):

| concurrency | A100 / H200 J/token: min–max (median) | A100 / H200 throughput |
|---|---|---|
| 1 | 1.48–1.53 (1.50) | 0.52–0.56 |
| 4 | 1.44–1.56 (1.53) | 0.50–0.56 |
| 16 | 1.45–1.61 (1.48) | 0.45–0.54 |
| 64 | 1.16–1.66 (1.40) | 0.34–0.52 |

- Every A100 run is at or near its 300 W cap (270–299 W).
- **Reproducible across months and nodes:** Qwen2-7B alpaca c=1 was 3,382 mJ/token in
  September and 3,535 mJ/token today (eager, +4.5%).
- **Idle baselines differ between the campaigns:** 73–76 W in September vs 98–100 W
  today (clocks pinned at 1410 MHz with any CUDA process, see the idle-state table).
  This moves only the idle-vs-dynamic split, not measured J/token.

**Qwen2-7B alpaca, all three GPUs** (J/token in mJ, tok/s in parentheses):

| concurrency | A100, CUDA graphs | B200, CUDA graphs | H200, graphs-adjusted est. (eager measured) | A100/B200 |
|---|---|---|---|---|
| 1 | 3333 (89) | 2934 (218) | 1974 (2230) | 1.14 |
| 4 | 875 (341) | 768 (840) | 508 (575) | 1.14 |
| 16 | 232 (1286) | 206 (3183) | 144 (161) | 1.13 |
| 64 | 72 (4124) | 68 (10408) | 53 (57) | 1.07 |
| 128 | 54 (5535) | 42 (18779) | — | 1.27 |
| 256 | 43 (6839) | 31 (27347) | — | 1.38 |

**Ranking per token at normal load: H200 < B200 ≈ 1.3× < A100 ≈ 1.5×.**
- The A100 is within 7–14% of B200 at up to 64 concurrent requests, despite 2.4–2.6×
  lower throughput. Its idle power is 2.4× lower, and both parts spend most of their
  energy streaming weights at similar J/B.
- B200 pulls ahead only from 128 concurrent requests on.
- **CUDA graphs barely help a power-capped GPU:** A100 +5–7% throughput, −4–6% J/token,
  against B200's +40–50% / −13–16%. When power sets the step time, launch overhead
  doesn't matter.

## Tensor parallelism on PCIe A100s: a poor trade (2026-10-21)
**Topology** (`logs/A100_prov/a100_topology.txt`): the two A100 80GB PCIe cards have
**no active NVLink** ("all links are inActive"). They're connected over PCIe Gen4 x16
through the CPU's host bridges (`nvidia-smi topo`: NODE), so TP all-reduces cross
PCIe (~25 GB/s per direction).

| Qwen2.5-14B, CUDA graphs | TP=1 | TP=2 (2 GPUs, both metered) | TP=2 ÷ TP=1 |
|---|---|---|---|
| batch 1 | 47.8 tok/s, 6.25 J/tok, 20.9 ms/step | 73.0 tok/s, 7.14 J/tok, 13.6 ms/step | 1.53× speed, **1.14× energy** |
| 16 concurrent | 671, 0.445, 22.5 ms | 736, 0.620, 17.5 ms | 1.10×, **1.39×** |
| 64 concurrent | 2247, 0.132, 25.4 ms | 2199, 0.216, 27.6 ms | 0.98×, **1.64×** |

- The pair draws only 456–521 W of its combined 600 W cap, so the GPUs spend much of
  each step waiting on PCIe all-reduce.
- On B200 (NVLink), TP=2 cost 3–13% more energy for 1.4–1.6× throughput.

**32B needs TP=2 on 80 GB A100s:**
- A100 TP=2 costs **1.36–1.71× B200 (TP=1)** per token: 15.6 / 1.28 / 0.44 J/tok at
  concurrency 1/16/64.
- Against H200 (TP=1, eager, measured) it's **1.73–1.92×**.
- Two idle GPUs (199 W) plus PCIe sync make the A100 the worst choice for 32B.

**Provisioning rule:** on PCIe-only A100s, use TP > 1 only when the model doesn't fit
on one GPU. It buys latency at batch 1 and nothing beyond.

## MoE (Qwen3-30B-A3B), A100 vs H200, both eager and measured (2026-10-21)
| run | A100 J/tok (tok/s, W) | H200 J/tok (tok/s, W) | A100 / H200 |
|---|---|---|---|
| alpaca c=1 | 4.62 (27, 124 W) | 3.83 (39, 149 W) | 1.20 |
| alpaca c=4 | 1.57 (98, 154 W) | 1.23 (150, 184 W) | 1.27 |
| alpaca c=16 | 0.552 (384, 212 W) | 0.422 (601, 253 W) | 1.31 |
| alpaca c=64 | 0.191 (1528, 292 W) | 0.142 (2298, 326 W) | 1.34 |
| sharegpt c=1 | 4.74 (27, 128 W) | 3.65 (42, 152 W) | 1.30 |
| sharegpt c=4 | 1.65 (94, 156 W) | 1.25 (153, 191 W) | 1.32 |
| sharegpt c=16 | 0.616 (383, 235 W) | 0.468 (609, 285 W) | 1.32 |
| sharegpt c=64 | 0.232 (1202, 278 W) | 0.160 (2356, 376 W) | 1.45 |

- **Unlike dense models, MoE does not power-cap the A100** (124–292 W): only about 3B
  parameters are active per token.
- Both GPUs are limited by per-step overhead, not bandwidth. In eager mode the H200 does
  only 39 tok/s at batch 1.
- When both GPUs mostly wait on fixed per-step work, idle power dominates. The A100's
  idle is close to the H200's (about 98 vs 116 W), so the gap narrows to **1.20–1.45×**,
  against ~1.5× for dense 7–8B models.
- **Provisioning implication:** for MoE, the older GPU's disadvantage is smaller.
  Eager-mode overhead dominates on both GPUs, so CUDA graphs and a newer vLLM would
  change both columns, and that should be re-measured before drawing conclusions.

## Size ladder, long context, prefill (`a100-more`, 54 runs, 2026-10-21)

**A100 vs H200 over model size** (Qwen2.5 0.5B–14B, eager, both measured; mean of the
alpaca and sharegpt J/token ratios):

| model | c=1 | c=4 | c=16 | c=64 | A100 power |
|---|---|---|---|---|---|
| 0.5B | 1.55 | 1.59 | 1.58 | 1.47 | 122–140 W |
| 1.5B | 1.65 | 1.66 | 1.63 | 1.51 | 155–199 W |
| 3B | 1.72 | 1.75 | 1.62 | 1.51 | 186–235 W |
| 7B | 1.60 | 1.60 | 1.53 | 1.35 | 298–299 W (capped) |
| 14B | 1.56 | 1.56 | 1.54 | 1.29 | 298–299 W (capped) |

- **The A100 costs 1.5–1.75× H200 per token at up to 16 concurrent requests, at every
  size.**
- Small models (≤ 3B) don't reach the A100's cap, yet the A100 still delivers only
  ~0.5× H200's throughput. Those runs are limited by fixed per-step host work, so the
  comparison includes the two nodes' CPU and launch paths, not just the GPUs.

**Long context on a capped A100** (Qwen2-7B, 1k → 30k context; decode-only bins with
prefill windows excluded):
- At constant 300 W, energy just tracks time, so these are **at-cap costs**
  (300 W × time per byte), not physical energy per byte.
- A KV byte costs about 1.29–1.32e-10 J/B at the cap (slope method, R² 0.97–0.98).
  A weight byte costs about 1.59e-10 J/B at the cap, because it also carries GEMM and
  per-layer time, so the at-cap KV/weight ratio is 0.79.
- The implied e_gemm (2.6–8.9 pJ) is meaningless under the cap.
- **Physical KV energy is calibrated on B200** (1.53–1.61e-10 J/B), not here.

**Prefill-dominated serving** (unique 2k/8k-token prompts, 16 output tokens; both GPUs
at the cap):

| run | A100 J per prompt token | B200 J per prompt token | A100 / B200 | B200 prompt throughput / A100 |
|---|---|---|---|---|
| in2048 c=4 | 0.0269 | 0.0190 | 1.42 | 4.2× |
| in2048 c=16 | 0.0216 | 0.0142 | 1.52 | 4.9× |
| in8192 c=4 | 0.0234 | 0.0175 | 1.34 | 4.3× |
| in8192 c=16 | 0.0198 | 0.0158 | 1.25 | 4.1× |

B200 prefill is 1.25–1.52× cheaper per prompt token and 4–5× faster, in line with its
1.9× cheaper GEMM energy (0.77 vs 1.47 pJ/FLOP at full clock).
