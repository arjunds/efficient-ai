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
- **B200:** §8 (data: `logs/B200_prov/trace_summary.json`; B200-side notes in `FINDINGS_B200.md` UPDATE 10 and
  `HANDOFF_FROM_B200_TRACE.md`).

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
- **Model-free check (`cap_check.py`; NVML throttle reasons and clocks, steady windows,
  2026-11-05):**

  | | SW-power-cap flag set | SM clock (median / max) | memory clock | power |
  |---|---|---|---|---|
  | A100, all 7B trace levels except the lightest | 95–100% of samples | 1380 → 1125 MHz / 1410 (−2% to −20% with load) | fixed 1512 | 296–299 W |
  | A6000, same | 94–100% | 1755 → 1575 / 1905 (−8% to −17%) | fixed 7601 | 288–299 W |
  | B200, 7B (all traces) | 0% (7% at reasoning 0.5×) | 1965 / 1965 (never throttled) | — | 355–861 W |
  | B200, 32B chat 4× / 6× | 10% / 30% | 1965 / 1965 | — | 934–955 W |

  No thermal or hardware-slowdown flags anywhere.
- **What this does and doesn't support:**
  - **Certain (direct measurement):** the 300 W parts run at their cap at real decode
    load, and the driver holds them there by lowering the SM clock. The B200 with a 7B model
    never reaches its 1000 W cap; a 32B model at high load starts to.
  - **Not shown:** that the cap explains the bandwidth gap. The memory clock never drops,
    and the SM clock drops ≤20%, so the cap costs at most ~2–20% of step speed (an upper
    bound; only part of a decode step scales with SM clock). Most of the gap between
    achieved and peak bandwidth (41–62% on A100) is not a power effect. The causal test
    (raise the cap, measure throughput) needs a power-limit change, which is out of bounds
    on shared hardware.
- **Implication:** the nameplate power is the wrong planning number in both directions.
  Small-cap parts sit at it under any real load, while a 1000 W part serving a 7B model
  draws 36–86% of it. The model's power-roof-vs-bandwidth-roof ratio predicts which
  regime a GPU/model pair is in.

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

## 8. B200 (2026-11-06; Qwen2.5-7B, 32B, Qwen3-30B-A3B)
- **Not power-capped.** The SW-power-cap flag is 0% on every 7B level and 10–30% only at the 32B's top
  rates (§1). Achieved bandwidth is 32–41% of peak (7B) and 50–56% (32B), so the B200 is below *both*
  roofs: on short prompts the step is dominated by per-layer and per-sequence overhead (`fig_thesis_time.png`).
- **Capacity rows are lower bounds** (the speedup ladder ran out first) except azure23_code
  (60 req/s tight, 115 relaxed). At the highest rates tested: 7B chat ≥87 req/s at 0.046 J/token
  (A100: 10 at 0.12), BurstGPT ≥276 at 0.066, Azure '24 peak ≥115 at 0.076, reasoning ≥2.4 at 0.080.
  Retained fractions after the KV trim: 32B azure23_conv 31.7%, azure24_peak 34.1%, code 73.4%.
- **KV, not weights, is the largest dynamic term on B200 7B chat and reasoning at capacity** (25–40% of
  energy vs weights 17–22%), because the B200 runs 160–230 sequences per step. Static is 26–44%. This
  moves KV compression up the lever list (§7) for big-memory GPUs.
- **LIMINAL-E on B200 real traffic:** −6.8% to +4.3% for 7B/32B on FlashInfer; MoE −10% to +13%.
  B200 Mooncake ran on the Triton backend (FlashInfer deadlocks on ~10k-token prompts; A100/A6000
  ran Mooncake clean on FlashAttention). Both Mooncake traces now completed on Triton (chat and agent;
  the FlashInfer hang reproduced on both), and LIMINAL-E under-predicts them by 5–18%, growing with batch
  (chat −8..−18%, agent −5..−15%) — the coefficients were measured under FlashInfer. Mooncake agent on
  B200: 3.5 req/s at 0.82 J/token tight (A100: 1.7 at 0.90), 22 req/s at 0.31 relaxed. Keep B200
  Mooncake out of cross-GPU rows.
- **MoE (`talk_figs/fig_thesis_moe.png`).** Qwen3-30B-A3B costs 0.57× a dense 7B per token at batch 1
  (A100, fixed concurrency), not the 0.43× active parameters imply, and >1× by batch 3–4 on both GPUs
  (A100 2.2× at 64; B200 1.65× at 244; the B200 reasoning trace adds 0.93× at 3.7 and 1.12× at 19). This is expert union. LIMINAL v2 models it (Monte Carlo active
  experts), and LIMINAL-E already counts E(1−(1−k/E)^t) distinct experts per step
  (`energy_model.weight_params_for_tokens`): on the B200 measured timeline it predicts the MoE/7B
  ratio within 15% at every level. So "active_params is the wrong denominator" is right but not new
  for either model. What *is* broken is LIMINAL-E's MoE **time**: on A100 graphs runs it over-predicts
  MoE J/token by 28–108%, even at c=1 where union does not apply (the fused-MoE layer floor is an
  H200 eager prior).

## Caveats
- **Synthetic token ids.** Data-dependent switching power is not reproduced.
- **Small cache-limited samples:** the A6000 Mooncake and code levels at 4–8× are short
  windows (25–64 requests).
- **The A6000 reasoning 0.01× level** is 5 requests (the trace start is sparse); excluded.
- **A100 P_0 = 0** in its calibrated profile, so activity power is folded into static
  there.
- **Mooncake prompts** are clipped to 32k (~4% of requests).
