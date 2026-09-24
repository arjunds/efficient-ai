# GPU recommender v2: measured coefficients and realized utilization

Files: `recommend_gpu.py` (the tool, CLI compatible with v1), `recommender_backtest.py`
(fit, backtest, win-map, figure), `plots_proposal/fig8_recommender_v2.png`.
Data: 177 usable serving runs (H200: 4 dense 7-8B, the Qwen2.5 0.5-32B ladder, and
Qwen3-30B-A3B MoE; B200: 7B, 32B, 72B; A100-PCIe: 4 dense 7-8B at its power cap).
Three crash-truncated B200 sharegpt runs are excluded because their bins cover less
than 95% of the window.

```
python3 recommend_gpu.py Qwen/Qwen2.5-7B-Instruct --prompt 2048 --gen 256 --batch 32
python3 recommend_gpu.py --list            # GPU table and where each number comes from
python3 recommend_gpu.py --winmap          # H200-vs-B200 map as text
# in the container:
python3 recommender_backtest.py --fit      # refit the time model and patch TIME_FIT in recommend_gpu.py
python3 recommender_backtest.py --figure   # backtest, win-maps, phase-reversal test, fig8
```

## 1. What changed and why

| | v1 | v2 |
|---|---|---|
| non-H200 energy coefficients | `scaled_gpu()`: e_byte ∝ 1/BW, e_flop ∝ 1/peak | **measured** per GPU (`gpu_coefficients.json`). Unmeasured GPUs get a **memory-technology prior** flagged `LOW (prior)` |
| time | `max(B/BW, F/peak)/0.7` at datasheet peak | **realized-utilization model** fitted to our vLLM runs (§2) |
| uncertainty | none | Monte Carlo over coefficient ranges, a 3-form × leave-one-model-out time-model ensemble, the prefill-MFU prior, and TP overheads. Every ranking comes with a P(best) |
| capacity | ignored | TP degree = smallest n ∈ {1,2,4,8} that fits the weights and the requested KV. **TP>1 is extrapolated** (all-reduce latency and a ×1.0-1.35 time penalty are drawn in the Monte Carlo) |

Why: on B200, energy per byte is **not** lower. It is 1.25e-10 J/B against H200's
1.08e-10, about 16% *higher*, where the 1/BW law predicted 40% lower. The B200 also
idles at 238 W against 116 W. And realized bandwidth sits far below peak and depends on
model size (B200 7B: 31% of peak; 72B: 62%). v1 therefore predicted that B200 wins
memory-bound decode. It loses.

**Byte-accounting fix (upstream):** the B200 bins had counted the input-embedding
table as weight traffic, which inflated weight bytes by 7.7% for Qwen2-7B. The lead
re-binned everything under canonical `~/models.py`. v2 uses that convention
throughout, and `e_gemm` multiplies dense GEMM FLOPs only, as the fit does.

**Memory-technology priors** (center [lo, hi], J/byte, weight stream):
- HBM3/3e: 1.16e-10 [1.03, 1.27]. This covers every direct c=1 J/byte measured on
  H200 and B200 for models ≥3B.
- HBM2e: 1.8e-10 [1.4, 2.4]. It comes from the A100-PCIe c=1 runs, which are pinned at
  the 300 W cap, so it is an effective value under throttling.
- GDDR6: 1.8e-10 [1.0, 3.5]. This is **unknown**. When `gpu_coefficients.json` gains a
  GDDR6 entry (the A5000), it replaces this prior (±15%) automatically, for L40S too.
- e_gemm for unmeasured GPUs spans the range between "invariant" (the H200 value) and
  "∝ 1/peak".

## 2. The utilization model

Per iteration, with inputs known before deployment (layers L, weight bytes W, KV
bytes, FLOPs F, all from the model config and the workload):

```
t_mem = t0 + L · softmax_k( τ , (W/L)/(BW·η) ) + KV/(BW·η_kv) + F/(peak·μ)
t_cmp = t0 + F/(peak·MFU_prefill)                 MFU_prefill prior 0.65 [0.50, 0.75]
t     = max(t_mem, t_cmp);   power cap: t ← E_dyn/(P_cap − P_static) if exceeded
realized MBU = (W+KV)/(BW·t)     (derived, not assumed)
```

The core idea: **each layer has a latency floor τ** (kernel launch, tail effects,
small-GEMV inefficiency), and there is a per-step framework overhead t0. So MBU rises
with *weight bytes per layer* and saturates at η (fig8a). This one mechanism explains
three observations:
- a 0.5B model runs at 5% MBU;
- a 3B model with 36 layers is slower than a 7B with 28 layers;
- a 7B reaches only 31-35% of B200's peak bandwidth but 47-50% of H200's. The B200
  must stream 1.7× faster per layer, so it hits the same floors sooner.

Fitted on dense runs. The central form is chosen per GPU by leave-one-model-out (LOMO)
error on iteration time:

| GPU | central form | t0 | τ /layer | η (asympt. MBU) | η_kv | LOMO t_iter MAPE | ensemble ranges |
|---|---|---|---|---|---|---|---|
| H200 | hard max (k=50) | 2.15 ms | 110 µs | 0.72 | 0.22 | 5.3% | η 0.65-0.95, τ 1-146 µs, t0 1.0-3.8 ms |
| B200 | additive (k=1) | 1.38 ms | 80 µs | 0.87 | 0.37 | 2.5% | η 0.61-0.89, τ 30-197 µs, t0 0-3.5 ms |

- **MoE layer floor:** τ_moe = 484 µs on H200. It is fitted on the single MoE model, so
  it is in-sample. The fused-MoE layer is about 4× slower than a dense layer. For B200
  it is scaled by the τ ratio, which is an assumption.
- **The parameters are individually poorly identified** (H200 and B200 were measured
  on different hosts and model sets, and trade-offs between t0, τ and η appear across
  forms). Their *predictions* inside the measured domain agree. That is why the
  ensemble, not the point fit, drives the win-map uncertainty.
- μ is a per-token marginal cost fitted at small batch (≤~110 tokens per iteration),
  not an MFU. B200's μ (0.69) implies about the same µs per token as H200, so its
  extra FLOPS do not help small-batch decode.
- **Prefill MFU is a prior, not a measurement.** The iteration-level timestamps for
  large prefill chunks imply more than 100% MFU on H200, which is a logger timing
  artifact. Only the B200 cuBLAS benchmark (68% of dense peak) supports the prior.

Measured vs modeled c=1 MBU (alpaca):

| | 0.5B | 1.5B | 3B | 7B (Qwen2.5) | 14B | 32B | 72B | gemma-7b |
|---|---|---|---|---|---|---|---|---|
| H200 measured | 4.6 | 12.8 | 20.6 | 50.6 | 58.1 | 67.0 | – | 41.4 |
| H200 model | 4.3 | 12.3 | 21.1 | 47.3 | 57.0 | 64.7 | – | 50.1 |
| B200 measured | | | | 32.5 (Qwen2) | | 50.5 | 61.5 | |
| B200 model | | | | 31.2 | | 50.7 | 62.7 | |

## 3. Backtest (the credibility test; first time done)

Inputs per run: model, GPU, the mean prompt and generation lengths measured from the
request stream, and the load. Load is closed-loop concurrency c, or for Poisson runs
the *realized* request rate: the harness delivers only 0.83-0.95× its nominal λ, and
the batch size then comes from a Little's-law fixed point. Outputs: tok/s, average
power and J/token, compared against `results.json`. MAPE in %:

| group | n | tok/s v2 | tok/s LOMO | tok/s v1 | power v2 | power LOMO | power v1 | **J/tok v2** | J/tok LOMO | J/tok v1 |
|---|---|---|---|---|---|---|---|---|---|---|
| H200 dense 7-8B | 40 | 4.9 | 5.4 | 45.7 | 4.5 | 4.8 | 38.3 | **3.7** | 3.8 | 17.3 |
| H200 ladder 0.5-32B | 60 | 3.0 | 4.3 | 294 | 3.0 | 4.1 | 102 | **4.0** | 4.3 | 45.1 |
| H200 MoE 30B-A3B | 10 | 2.5 | 112* | 320 | 10.9 | 57* | 114 | **9.8** | 27.6* | 35.2 |
| B200 7B | 13 | 4.2 | 4.6 | 145 | 1.9 | 1.8 | 16.3 | **3.2** | 3.4 | 52.5 |
| B200 32B | 8 | 3.2 | 2.6 | 54.1 | 2.3 | 2.9 | 16.4 | **5.3** | 5.0 | 45.5 |
| B200 72B | 6 | 3.5 | 3.1 | 19.0 | 1.7 | 1.8 | 29.9 | **3.7** | 3.6 | 41.0 |
| **all H200** | 110 | 3.7 | 14.5 | 206 | 4.3 | 9.2 | 80.2 | **4.4** | 6.3 | 34.1 |
| **all B200** | 27 | 3.7 | 3.7 | 90.3 | 2.0 | 2.1 | 19.4 | **3.9** | 3.9 | 47.9 |
| A100-PCIe (capped) | 40 | 4.5 | – | 35.8 | 3.0 | – | 3.0 | 4.1 | – | 62.5 |

- **LOMO** means the time model is refit without that model, which is the honest
  number for a new dense model.
- `*` MoE LOMO is zero-shot: dense layer floor and no MoE calibration. The time model
  **fails** there, with tok/s 2× too high. τ_moe is required, and it rests on one model.
- **Energy coefficients are partly in-sample:** the H200 dense and B200 7B runs are the
  fit sets. B200 32B and 72B are out-of-sample for energy and land at 5.3% and 3.7%.
  The ladder is out-of-sample for H200 energy.
- **A100 is not a clean zero-shot test.** Its HBM2e prior was set from these same runs,
  and at the cap, power ≈ P_cap by construction. With the HBM3e band instead (a truly
  zero-shot case), J/token is **−21%** biased. Older HBM2e parts cost measurably more
  per byte, so "invariant J/byte" holds within HBM3/3e, not across generations.
- Closed-loop vs Poisson (H200+B200): J/token 4.4% vs 3.9%.
- Signed J/token bias is −2 to −5% on most groups: v2 slightly under-predicts.
- Worst cases:
  - gemma-7b on H200 runs at 41% MBU against 50% predicted (architecture-specific
    slowness);
  - A100 gemma c=64 is +33% (throttled time);
  - 32B Poisson on H200 is −18%.

## 4. Recomputed recommendations

**H200 vs B200 win-map** (fig8c,d; prompt 1024 / gen 256; cell = E(B200)/E(H200) per
token; bold = P(winner) ≥ 0.9):

- **Decode: H200 wins everywhere in the measured domain (≤64 sequences), by
  1.05-1.44×. It is confident (P ≥ 0.9) in every cell except 14B at 64 sequences.** The margin is largest for small models and small
  batches, where B200's 2× idle power is paid on overhead-dominated steps. It narrows
  with model size (72B: 1.10-1.22×) as B200's utilization climbs to 62%. The
  mechanism, exactly as framed:
  - dynamic J/byte is about the same (B200 16% higher);
  - B200 is only 1.1-1.3× faster (realized, not 1.67×);
  - so B200's 238 W idle is never paid back.
  At batch ≥128 (extrapolated past the data) the ratio crosses 1 (0.92-1.00).
  Compute energy starts to matter there, and B200's lower e_gemm helps, but
  P(B200) is only 30-85%: **not confident**.
- **Prefill: B200 favored by about 17% (ratio 0.80-0.86) for models ≥3B at batch
  ≥4, with P(B200 best) of 0.82-0.94. Mostly below the 0.9 bar.** B200 is predicted
  to run near its 1000 W cap on compute-bound prefill. The advantage comes from its
  lower measured e_gemm (0.54 vs 0.68 pJ) plus 2.3× peak FLOPS amortizing idle. It
  rests on the unmeasured prefill-MFU prior and on e_gemm ranges that drift with model
  size. Tiny prefills (0.5B, batch 1-2) go to H200 (overhead-bound).
- 72B does not fit on one H200, so the H200 side is TP=2 and extrapolated. On that
  basis H200×2 still wins decode (P≈1 within the model), but NVLink/NCCL energy is not
  modeled.

**The phase reversal survives, but only between H200 and B200, not in its old form.**
- **Among the measured GPUs:** B200 for prefill, H200 for decode. This is a genuine
  disaggregation recommendation. The decode half is confident; the prefill half is
  about 85-90%.
- **The v1 claim ("L40S for prefill, A100 for decode") is not supported.** Both GPUs
  are prior-only. The joint Monte Carlo gives P(reversal) = 44-53%, a coin flip. The
  central L40S/A100 decode ratio is 0.93-1.08, and the decision hinges on GDDR6 J/byte,
  which nobody has measured yet. The A5000 run will decide it.

**A5000:** `load_gpus()` picks up an `A5000` entry in `gpu_coefficients.json`
automatically, and it then also replaces the GDDR6 prior. `recommender_backtest.py`
discovers `logs/*A5000*` runs, and `--fit` fits a time model for them. With a single
model, only t0 and η are fitted; τ, η_kv and μ are fixed at the measured-GPU medians,
and it falls back to the prior if the fit is worse than 15%. After those two steps,
`--winmap` also prints A5000 vs H200. None of this is in the current figures: the
A5000 data is not in yet.

## 5. Limitations

- **Software-specific:** t0 and τ belong to vLLM 0.10.2 on these hosts (CUDA graphs,
  sampler, scheduler), not only to the silicon. A different engine version or host CPU
  changes them. The H200 and B200 hosts differ.
- **Domain:** fp16, TP=1, context ≤ ~1.3k tokens, ≤64 concurrent sequences, and ≤~110
  tokens per iteration on average. Long context, fp8, large-batch decode and TP are
  extrapolations. The KV term (η_kv 0.2-0.4) is fitted on short contexts.
- **Prefill MFU** is a prior. So is **e_gemm** outside the 7B fit sets. The "B200 wins
  prefill" result is the least certain claim here.
- **MoE timing** rests on one model on one GPU.
- **Unmeasured GPUs** (H100, A100-SXM, L40S, A5000 until measured) are prior-only.
  Idle power for them is an estimate. Treat their rankings as hypotheses; the tool
  labels them `LOW (prior)`.
- **Power cap:** the throttling model holds dynamic energy fixed and stretches time.
  It fits the A100 behaviour (power 3% MAPE), but real DVFS lowers J/op somewhat.
