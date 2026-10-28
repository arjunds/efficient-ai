# Real workloads on the energy roofline: decode-only trace replay (A100, A6000; 2026-10-28)

Design and caveats are in `TRACES.md`.

**Setup:**
- Qwen2.5-7B, vLLM 0.10.2, CUDA graphs.
- Every prompt is prefix-cache-warmed, so the measured window is the decode tier of a
  disaggregated deployment. Prefix hits are 98.5–99.9% on all non-reasoning levels.
- 59 load levels across 7 workloads × 2 GPUs.
- Data: `logs/{A100,A6000}_prov/trace_summary.json`, from `trace_analysis.py`.
- Figures: `talk_figs/fig_trace_{roofline,jtok,breakdown}.png`.
- Fleet sizing: `fleet_summary_{tight,relaxed}.json`, from `fleet_analysis.py`.
- **B200 pending** (`HANDOFF_B200_TRACE.md`).

## 1. Real decode traffic sits on the power roof, not the bandwidth roof
- **A100:** every trace × load level up to the latency knee lies on the full-clock
  power roof (P_cap − P_s − P_0)/(e_b/I + e_f). Arithmetic intensity is 5–50 FLOP/B.
  Achieved bandwidth is 41–62% of peak, while the bandwidth roof is 1.4–2.5× higher.
  - Capped steps are 66–100% of the window at most levels that meet the SLO. The
    exceptions are the lightest Mooncake levels (0.1×, 34–36%), which sit below the
    roof.
- **A6000:** the same, at 71–77% of peak bandwidth (its fixed GDDR6 activity power
  leaves less for bytes).
- **Past the knee:** points fall *below* the power roof. Steps that admit requests run
  attention without GQA packing or split-KV, as §4 shows.
- **Implication:** on 300 W parts, decode throughput at real load is set by joules per
  byte and per FLOP, not by bandwidth. Faster memory alone doesn't help; cheaper bytes
  (and higher caps) do.

## 2. LIMINAL-E holds on real traffic up to the knee
The predicted window energy is computed from coefficients on the measured step timeline,
with capped steps clipped to P_cap.

| | error at levels up to the relaxed-SLO knee |
|---|---|
| A100 | −2.5% to +2.1% (most within ±1%) |
| A6000 | −4.4% to +1.6% (excluding the 5-request reasoning point) |
| past the knee (overload) | −8% to −26% (under-prediction) |

The under-prediction past the knee is the mixed-step attention cost (§4), which isn't in
the default model yet.

## 3. Capacity and energy at the latency SLO (one GPU, Qwen2.5-7B)
Tight SLO = TPOT p99 ≤ 50 ms and TTFT p99 ≤ 1 s. Relaxed = 100 ms and 2 s.

| workload | A100 tight: req/s, J/token | A100 relaxed | A6000 tight | A6000 relaxed |
|---|---|---|---|---|
| Azure chat '23 | 10.0, 0.12 | 15.2, 0.10 | 4.9, 0.28 | 7.3, 0.22 |
| Azure chat '24 peak hour | 11.6, 0.25 | 23.3, 0.16 | 5.6, 0.47 | 11.2, 0.35 |
| BurstGPT | 71, 0.11 | 108, 0.10 | 33, 0.26 | 50, 0.19 |
| Azure code '23 | 26, 0.34 | 52, 0.19 | 12, 0.89 | 24, 0.60 |
| Mooncake chat (12k ctx) | 1.5, 0.54 | 1.5, 0.54 | — | — |
| Mooncake agent (8.6k ctx) | 1.7, 0.90 | 6.6, 0.36 | 0.5, 2.28 | 2.6, 0.93 |
| Reasoning (4.9k-token outputs) | 0.30, 0.22 | 0.62, 0.10 | 0.09, 0.72 (0.05×) | 0.22, 0.26 |

- **A100 vs A6000:** the A100 serves about 2× the requests at 2–2.5× lower J/token. That's
  2× the bandwidth at the same 300 W, plus the GDDR6 activity power the A6000 pays.
- **Long-context agent and chat traffic** costs 3–8× more per token than chat at the
  same SLO.
- **Reasoning** costs about the same per token as chat, but ~25–50× more per request.
  Decode fleets for reasoning traffic scale with output length, not request count.
- **Code completion** (short outputs, ~2k-token prompts) is admission-heavy (§4).
  - At the relaxed SLO the A100 decodes 1,552 tok/s on code vs 2,842 on chat; at the
    tight SLO, 886 vs 2,416.
  - Code's J/token stays 1.8–2.8× chat's.
  - (An earlier "~650 tok/s plateau" was whole-level throughput, dominated by ramp and
    drain in seconds-long windows.)

## 4. Admission steps cost a mixed-batch attention penalty (profiler-confirmed)
- Admitting a prefix-cached request recomputes ≤16 tokens. The mixed batch then switches
  FlashAttention-2 (Ampere) to a path with no GQA packing (28 query heads) and no
  split-KV (1 split) for *every* decode sequence.
- Attention goes from 7 to 79 ms per step (A5000 profile, `logs/diag_admission/`).
  Host time (scheduling, hashing at 0.13 µs/token) is <2%.
- This **confirms the FA2 term that TIMEMODEL_HIGHC.md had inferred** for A100/A6000 at
  high concurrency.
- It's the limiting cost for short-output traffic and the main unmodelled cost past
  every knee.

## 5. Where the energy goes, at each workload's tight-SLO capacity (`fig_trace_breakdown.png`)

| term | A100 | A6000 |
|---|---|---|
| static (idle, model loaded) | 34–39% | 36–43% |
| memory-activity power P_0 | (in static; the A100 calibrated profile has P_0 = 0) | 22–27% |
| weight streaming | 37–51% | 21–29% |
| KV | 6–25% (Mooncake chat 25%, agent 14%) | 3–7% |
| GEMM | 2–18% | 1–11% |
| unexplained | ±2% (≤4% in the code trace) | ±4% |

## 6. Fleet sizing over a real day (Azure 2024, 37 h, 17–70 req/s)
J/request, using measured P(load) per GPU from the azure24 levels:

| SLO | GPU | fleet | always on, spread | consolidated (rest idle, loaded) | consolidated + rest at no-process idle |
|---|---|---|---|---|---|
| tight | A100 | 6 | 45.4 | 36.1 (−21%) | 32.8 (−28%) |
| tight | A6000 | 13 | 97.5 | 76.6 (−21%) | 62.8 (−36%) |
| relaxed | A100 | 3 | 24.6 | 19.8 (−19%) | 18.7 (−24%) |
| relaxed | A6000 | 7 | 57.4 | 41.3 (−28%) | 33.9 (−41%) |

- Spreading load over a peak-sized fleet wastes about a fifth of the energy. A lightly
  loaded A100 or A6000 still draws ~300 W, since neither is energy-proportional.
- Relaxing TPOT from 50 to 100 ms halves energy per request: bigger batches, the same
  watts.
- Gating idle GPUs needs model reload on scale-up (minutes); not modelled.

## 7. Research levers, ranked by measured energy at the SLO
1. **Fixed power (34–63%).** Energy-proportional idle and activity power, consolidation
   plus gating (§6), and memory that idles without a 400→8000 MHz clock jump (GDDR6).
2. **Weight re-streaming at real batch sizes (21–51%).** Larger effective batches via
   routing and consolidation, weight compression/quantization, and multi-model weight
   sharing.
3. **Mixed-batch attention (§4).** Admission-aware attention kernels (separate the
   chunk's attention from decode attention), recompute-free admission (KV transfer),
   batched admissions. These lift short-output capacity (code reaches 0.55× chat's
   decode tok/s at the relaxed SLO) and the knee of every trace.
4. **KV (3–25%).** Matters mainly for long-context agents and chat, and grows with
   reasoning load at high batch (27% at A100 0.2×). KV compression is a second-order
   lever for chat.

## Caveats
- **Synthetic token ids.** Data-dependent switching power is not reproduced.
- **Small cache-limited samples:** the A6000 Mooncake and code levels at 4–8× are short
  windows (25–64 requests).
- **The A6000 reasoning 0.01× level** is 5 requests (the trace start is sparse); excluded.
- **A100 P_0 = 0** in its calibrated profile, so activity power is folded into static
  there.
- **Mooncake prompts** are clipped to 32k (~4% of requests).
