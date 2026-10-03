# Balance and overprovisioning analysis: energy alongside performance

PI question: *"Is the system overprovisioned, or balanced once energy is considered?
Track bandwidth and inferences/s. We might not be saturating the memory."*

Everything here uses data already in hand: 110 measured H200 and B200 serving runs,
`gpu_coefficients.json`, and the recommender time model for counterfactuals.
`python3 balance_analysis.py` writes `balance_analysis.json`, `balance_runs.csv`
(per-run, including excluded runs) and `plots_proposal/fig9_balance.png`.

**Definition.** A capability (peak BW, peak FLOPs) is *overprovisioned* when its
static power costs more energy per token than it saves at the target SLO.
Equivalently, the GPU's real idle power exceeds the **break-even idle**: the idle
power at which it would tie the alternative on J/token.

## Bottom line
1. **We do not saturate memory in serving.**
   - Realized MBU is 0.39–0.67 on the H200 and only **0.27–0.32 on the B200 for 7B
     models**. Every measured operating point is far left of both ridges.
   - 7B decode steps take about 5.4–7 ms on *both* GPUs. Host and launch overhead
     plus the per-layer floor, not HBM, set the pace.
2. **Overprovisioned at the loads we measured.** At concurrency ≤ 64 the B200 costs
   **1.11–1.50× the H200's J/token** (22 matched pairs, median 1.32).
   - It turns its 1.67× bandwidth into only 1.02–1.16× throughput on 7B and 1.21–1.29×
     on 32B (conversion 0.61–0.77).
   - Its 2.04× idle power is therefore not amortized: static energy per token is
     1.6–2.2× the H200's. For 7B models, static is 36–45% of B200 energy, against 23–34% on the H200.
3. **Balanced only if it is fed much larger batches.** At a TPOT SLO with the largest
   batch each GPU can hold, the time model puts B200 decode **11–32% cheaper** per
   token than H200.
   - The break-even B200 idle is 358–660 W, against 238 W measured.
   - The catch: that needs batches of roughly 200–1900, far beyond the measured
     c ≤ 64. At the same demand (the H200's SLO batch run on a B200) the result is a
     wash, 0.91–1.09×.
   - Balance depends on load, not only on hardware. The B200 pays off only with
     enough concurrency to use its bandwidth. That is a capacity-planning and
     consolidation question as much as a choice of GPU.
4. **Bandwidth is the capability worth paying idle power for; FLOPs are not, for
   decode.** On an H200 at TPOT ≤ 50 ms with idle held fixed:

   | upgrade | J/token change | idle power it could afford |
   |---|---|---|
   | 2× FLOPs | −0.7 to −1.5% | +4 to +8 W |
   | 1.5× BW | −6 to −20% | +42 to +190 W |
   | full B200 capabilities | −11 to −22% | **+108 W (7B), +142 W (14B), +240 W (32B)** |

   The B200 actually carries +121 W of idle. It is slightly overprovisioned for 7B
   and worth it for 14B and up. Adding bandwidth without raising the 700 W TDP hits
   the power cap at large batches, so TDP has to scale with bandwidth.
5. **Ridges, in FLOP per byte; for decode this is ≈ tokens per step.**

   | | performance ridge (peak/BW) | energy ridge (e_wbyte/e_gemm) |
   |---|---|---|
   | H200 | 206 | 134 |
   | B200 | 281 | 194 |

   The energy ridge sits at 0.65–0.69× the performance ridge. Compute energy matches
   weight-streaming energy only at about 130–190 tokens per step. Measured serving sits
   at 1–87, so dynamic energy is memory-dominated everywhere we operate. GEMM is
   ≤ 27% of run energy, KV ≤ 15% (31% for MHA gemma-7b at c=64), and static is
   18–45% for 7B–72B models.
6. **Small models are the clearest case of overprovisioning.** Qwen2.5-0.5B on an H200
   runs at MBU 0.05, with **68–80% of energy static**. Qwen2.5-1.5B runs at MBU 0.13
   with 49–63% static. These belong on a smaller part, or consolidated.

## Tracking (bandwidth and inferences/s, as requested)
`balance_runs.csv` records, per run: realized TB/s, MBU, MFU, arithmetic intensity,
tok/s, **req/s**, J/token, **J/request**, median step time, and the five-way energy
split. Examples:

| GPU / model | c | TB/s | MBU | req/s | J/req | static share |
|---|---|---|---|---|---|---|
| H200 Qwen2-7B | 1 / 64 | 2.37 / 1.95 | 0.49 / 0.41 | 0.9 / 43 | 447 / 12 | 31% / 23% |
| B200 Qwen2-7B | 1 / 64 | 2.59 / 2.18 | 0.32 / 0.27 | 1.0 / 48 | 612 / 14 | 42% / 36% |
| H200 Qwen2.5-32B | 1 / 64 | 3.21 / 2.78 | 0.67 / 0.58 | 0.2 / 12 | 1883 / 57 | 26% / 18% |
| B200 Qwen2.5-32B | 1 / 64 | 4.05 / 3.36 | 0.51 / 0.42 | 0.3 / 14 | 2517 / 64 | 32% / 28% |
| B200 Qwen2-72B | 1 / 16 | 4.90 / 4.82 | 0.61 / 0.60 | 0.1 / 2.0 | 5960 / 448 | 28% / 26% |

Two B200 runs are **excluded**: Mistral-7B sharegpt_c4 and Qwen2-7B sharegpt_c16. In
both the load generator stalled: 17 requests completed, active bins cover 4–17% of the
window, and average power is at or below idle. They had depressed the earlier B200
c=4/c=16 means.

## Time resolution: a smear-free cross-check (the PI's "200 ms is a lot")
Fitting the same 3-term model on **whole-run sums** (45 s windows integrate out NVML
averaging and lag) gives:

| | e_wbyte vs per-bin | e_gemm vs per-bin (e_kv free) | e_gemm, e_kv held at per-bin | per-bin coefficients on run totals |
|---|---|---|---|---|
| H200 (32 / 80 runs) | 0.97× | 1.31–1.39× | **1.20×** [0.87, 1.03 pJ] | MAPE 3.4–4.9%, bias ≤ −1% |
| B200 (14 / 28 runs) | 0.99–1.00× | 1.31–1.45× | **1.11–1.12×** | MAPE 1.9–2.2%, bias ≤ −1.1% |

- **The memory coefficient is resolution-independent** (within 3%), and the per-bin
  model predicts whole-run energy without bias. So J/token predictions, and everything
  above, survive the sampling question.
- **The compute coefficient is 11–45% higher at run level.** The direction matches the
  NVML smoothing bias seen on the A5000 (+26–30%). The exact size is not identified,
  because KV bytes and FLOPs are correlated 0.69–0.87 across runs. The B200
  steady-state `llmgemm` calibration (`HANDOFF_B200_STEADY.md`) will settle it.
- If e_gemm is 1.2–1.4× higher, the energy ridges fall to about 95–175. Serving is
  still memory-dominated. The B200/H200 e_gemm ratio stays 0.75–0.84, so the prefill
  ranking does not move.
- This is our answer to LIMINAL's "5×": integrate over long windows, or calibrate in
  steady state as EnergAIzer does. Do not trust per-bin attribution for the
  compute/KV split.

## Caveats
- Section 4 of the script (SLO, what-ifs) uses the recommender time model. It was
  validated at c ≤ 64 (LOMO time MAPE 2.7–5.3%) and is extrapolated here to batches up
  to about 1900. It assumes pure decode, with no prefill interference or scheduler
  limits. Read the "balanced if fed" result as an **upper bound** on what the B200 can
  deliver.
- Static power is the measured GPU idle (H200 116 W, B200 238 W). Host and node power
  are not included. Including them would strengthen the case for consolidation.
- 72B does not fit on one H200, so there is no matched comparison.

## Related: recommender v2.4 (priors revised the same day; RECOMMENDER.md)
- **HBM2/2e prior:** 1.75 → **1.36e-10 J/B** [1.05, 1.60]. This is EnergAIzer's
  boost-clock point; their locked low-clock value is 1.13e-10.
  - The old value had been read off A100-PCIe runs at the 300 W default TDP, where
    J/byte cannot be identified.
  - The A100-PCIe zero-shot J/token MAPE is 13.3% at the new center, against 20.4% at
    1.13e-10. That is a two-point choice made on the backtest runs, so it is mildly
    tuned.
- **GDDR6 prior:** widened to **2.3e-10 [1.3, 3.1]** for the clock dependence.
- **Rankings:**
  - A100 beats L40S on decode by 1.54–1.70× (P ≈ 1).
  - The L40S prefill question is still open (P(L40S wins) 0.32–0.42; it depends on the
    unmeasured Ada e_gemm).
  - A100-SXM decode is now about level with B200 and 1.15× H200.
  - H200 and B200 results are unchanged (backtest 4.1% / 3.7%).
