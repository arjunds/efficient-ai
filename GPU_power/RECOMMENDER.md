# GPU recommender v2.2: measured coefficients and realized utilization

*v2.1 (2026-09-24): prefix-cache-corrected coefficients, a `cached_frac` workload
parameter (§1a), and the power-limited A5000 excluded. v2.0 was commit 84fea1a.*

*v2.2: memory-technology-class priors for unmeasured GPUs, with GDDR6 and Ampere
GA10x e_gemm taken from the A5000 microbenchmarks (`FINDINGS_A5000.md`). The L40S/A100
question is re-answered in §4.*

Files: `recommend_gpu.py` (the tool, CLI compatible with v1), `recommender_backtest.py`
(fit, backtest, win-map, figure), `plots_proposal/fig8_recommender_v2.png`.
Data: 177 usable serving runs (H200: 4 dense 7-8B, the Qwen2.5 0.5-32B ladder, and
Qwen3-30B-A3B MoE; B200: 7B, 32B, 72B; A100-PCIe: 4 dense 7-8B at its power cap).
Three crash-truncated B200 sharegpt runs are excluded because their bins cover less
than 95% of the window.

```
python3 recommend_gpu.py Qwen/Qwen2.5-7B-Instruct --prompt 2048 --gen 256 --batch 32 [--cached-frac 0.3]
python3 recommend_gpu.py --list            # GPU table and where each number comes from
python3 recommend_gpu.py --winmap          # H200-vs-B200 map as text
# in the container:
python3 recommender_backtest.py --fit      # refit the time model and patch TIME_FIT in recommend_gpu.py
python3 recommender_backtest.py --figure   # backtest, win-maps, phase-reversal test, fig8
#   [--old-fit <file>]  to also score v2.0 with its own time fit (old-vs-new table)
```

## 1. What changed and why

| | v1 | v2 |
|---|---|---|
| non-H200 energy coefficients | `scaled_gpu()`: e_byte ∝ 1/BW, e_flop ∝ 1/peak | **measured** per GPU (`gpu_coefficients.json`). Unmeasured GPUs get a **memory-technology prior** flagged `LOW (prior)` |
| time | `max(B/BW, F/peak)/0.7` at datasheet peak | **realized-utilization model** fitted to our vLLM runs (§2) |
| uncertainty | none | Monte Carlo over coefficient ranges, a 3-form × leave-one-model-out time-model ensemble, the prefill-MFU prior, and TP overheads. Every ranking comes with a P(best) |
| capacity | ignored | TP degree = smallest n ∈ {1,2,4,8} that fits the weights and the requested KV. **TP>1 is extrapolated** (all-reduce latency and a ×1.0-1.35 time penalty are drawn in the Monte Carlo) |

Why: on B200, energy per byte is **not** lower. It is 1.24e-10 J/B against H200's
1.07e-10, about 16% *higher*, where the 1/BW law predicted 40% lower. The B200 also
idles at 238 W against 116 W. And realized bandwidth sits far below peak and depends on
model size (B200 7B: 31% of peak; 72B: 62%). v1 therefore predicted that B200 wins
memory-bound decode. It loses.

**Byte-accounting fix (upstream):** the B200 bins had counted the input-embedding
table as weight traffic, which inflated weight bytes by 7.7% for Qwen2-7B. The lead
re-binned everything under canonical `~/models.py`. v2 uses that convention
throughout, and `e_gemm` multiplies dense GEMM FLOPs only, as the fit does.

**Memory-technology-class priors for unmeasured GPUs.** Energy per byte is set by
memory technology (`FINDINGS_A5000.md`). Weight stream, J/byte, center [lo, hi]:

| class | e_wbyte | e_kvbyte | evidence |
|---|---|---|---|
| HBM3/3e (H100) | 1.16e-10 [1.07, 1.25] | 3.1e-10 [2.7, 3.5] | H200 and B200 measured band |
| HBM2e (A100) | 1.75e-10 [1.4, 2.3] | 4.5e-10 [3.3, 6.5] | A100-PCIe c=1 runs pinned at the 300 W cap. Weak |
| GDDR6 (L40S, A5000) | **3.2e-10 [2.5, 4.1]** | 6.0e-10 [3.7, 11] | A5000 microbenchmark at full clocks: GEMV 2.7-3.6e-10, and serving costs 1.04-1.14× GEMV per byte. Confounded with process node (Samsung 8N) and the V/f point. **"microbench prior"** |

- **e_gemm priors:**

  | GPU | e_gemm prior | basis |
  |---|---|---|
  | A5000 (Ampere GA10x) | 3.0 [2.0, 4.0] pJ | microbench prior: cuBLAS 2-4 pJ at 98-100 TFLOPS, 88-90% of the 111.1 dense peak |
  | H100 | H200 value [×0.72, ×1.25] | same die as H200 |
  | L40S (Ada) | 1.6 [0.64, 4.0] pJ | unmeasured, wide |
  | A100 (GA100) | 1.4 [0.7, 3.0] pJ | unmeasured, wide |

- **Idle:** A5000 idle is **measured** at 60.5 W [54, 86], depending on clock state.
  L40S idle is widened to 70 W [35, 110], because by analogy with the A5000 a card
  with a vLLM context loaded idles well above the 35 W v2.0 guess.
- **The A5000's serving runs remain excluded:** they were power-limited to 100 W. Only
  its full-clock microbenchmarks feed these priors.

## 1a. Prefix cache (v2.1)

The sysid agent found that logged `prefill_tokens` include prefix-cache hits that
vLLM never computes. In c=64 sharegpt runs, 26-36% of prompt tokens were hits. The
coefficients are therefore now fit on **computed** GEMM FLOPs (`controls/SYSID.md` §7):

| | e_wbyte | e_kvbyte | e_gemm |
|---|---|---|---|
| H200 | 1.076 → **1.066**e-10 | 3.376 → **3.288**e-10 | 0.680 → **0.794 pJ** [0.773, 0.822] |
| B200 | 1.252 → **1.243**e-10 | 3.043 → **2.852**e-10 | 0.539 → **0.642 pJ** [0.600, 0.694] |

To stay self-consistent, the recommender takes a workload parameter
`cached_frac` (`--cached-frac`, default **0** for deployment). The effect:
- Prefill tokens and FLOPs are scaled by (1 − cached_frac).
- Cached prefixes still count as resident KV and are still read by attention.
- Prefill energy is still reported per *ingested* prompt token.

The backtest uses each run's **measured** cached fraction, 1 − Σ`prefill_tokens_computed`
/ Σ`prefill_tokens` from `binned_table.csv`. It is 0 to 0.37 across runs. MoE and A100
runs have no such columns, so they use 0. The time model was refit on computed tokens;
its parameters barely moved.

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
| H200 | hard max (k=50) | 2.15 ms | 110 µs | 0.73 | 0.22 | 5.3% | η 0.65-0.95, τ 1-146 µs, t0 1.0-3.8 ms |
| B200 | additive (k=1) | 1.42 ms | 78 µs | 0.86 | 0.33 | 2.7% | η 0.61-0.89, τ 30-197 µs, t0 0-3.5 ms |

- **MoE layer floor:** τ_moe = 484 µs on H200. It is fitted on the single MoE model, so
  it is in-sample. The fused-MoE layer is about 4× slower than a dense layer. For B200
  it is scaled by the τ ratio, which is an assumption.
- **The parameters are individually poorly identified** (H200 and B200 were measured
  on different hosts and model sets, and trade-offs between t0, τ and η appear across
  forms). Their *predictions* inside the measured domain agree. That is why the
  ensemble, not the point fit, drives the win-map uncertainty.
- μ is a per-token marginal cost fitted at small batch (≤~110 tokens per iteration),
  not an MFU. B200's μ (0.66) implies about the same µs per token as H200, so its
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
| H200 dense 7-8B | 40 | 4.9 | 5.4 | 45.7 | 4.2 | 4.5 | 38.3 | **3.4** | 3.6 | 17.3 |
| H200 ladder 0.5-32B | 60 | 3.0 | 4.3 | 294 | 2.7 | 3.8 | 102 | **3.6** | 3.9 | 45.1 |
| H200 MoE 30B-A3B | 10 | 2.5 | 112* | 320 | 10.7 | 57* | 114 | **9.6** | 27.6* | 35.2 |
| B200 7B | 13 | 4.1 | 4.6 | 145 | 1.9 | 1.8 | 16.3 | **2.8** | 3.0 | 52.5 |
| B200 32B | 8 | 3.2 | 2.9 | 54.1 | 1.9 | 2.5 | 16.4 | **4.9** | 4.7 | 45.5 |
| B200 72B | 6 | 3.5 | 2.7 | 19.0 | 1.5 | 1.7 | 29.9 | **3.7** | 3.5 | 41.0 |
| **all H200** | 110 | 3.6 | 14.5 | 206 | 4.0 | 8.9 | 80.2 | **4.1** | 6.0 | 34.1 |
| **all B200** | 27 | 3.7 | 3.7 | 90.3 | 1.8 | 2.0 | 19.4 | **3.7** | 3.6 | 47.9 |
| A100-PCIe (capped) | 40 | 5.3 | – | 35.8 | 3.0 | – | 3.0 | 5.6 | – | 62.5 |

**Old vs new (prefix-cache fix).** J/token MAPE, v2.0 → v2.1. v2.0 = uncorrected
coefficients, logged prefill FLOPs, v2.0 time fit. v2.1 = corrected coefficients,
measured cached_frac, refit time model. The columns show the measured cached_frac range
and J/token (power in parentheses):

| group | cached_frac | J/tok | (power) |
|---|---|---|---|
| H200 dense 7-8B | 0-0.32 | 3.7 → **3.4** | (4.5 → 4.2) |
| H200 ladder | 0-0.37 | 4.0 → **3.6** | (3.0 → 2.7) |
| H200 MoE | n/a (0) | 9.8 → 9.6 | (10.9 → 10.7) |
| B200 7B | 0-0.30 | 3.2 → **2.8** | (1.9 → 1.9) |
| B200 32B | 0-0.18 | 5.3 → **4.9** | (2.3 → 1.9) |
| B200 72B | 0-0.05 | 3.7 → 3.7 | (1.7 → 1.5) |
| all H200 / all B200 | | 4.4 → **4.1** / 3.9 → **3.7** | |
| A100-PCIe | n/a (0) | 4.1 → 5.6 | (3.0 → 3.0) |

- Every measured group improves, modestly. The runs have short prompts, so compute is
  12-13% of dynamic energy.
- A100 gets worse. Its prior e_gemm is anchored on H200's now-larger value, and its
  runs carry no computed-token columns, so real cache hits are counted as compute.

- **LOMO** means the time model is refit without that model, which is the honest
  number for a new dense model.
- `*` MoE LOMO is zero-shot: dense layer floor and no MoE calibration. The time model
  **fails** there, with tok/s 2× too high. τ_moe is required, and it rests on one model.
- **Energy coefficients are partly in-sample:** the H200 dense and B200 7B runs are the
  fit sets. B200 32B and 72B are out-of-sample for energy and land at 4.9% and 3.7%.
  The ladder is out-of-sample for H200 energy.
- **A100 is not a clean zero-shot test.** Its HBM2e prior was set from these same runs,
  and at the cap, power ≈ P_cap by construction. With the HBM3e band instead (a truly
  zero-shot case), J/token is **−19%** biased. Older HBM2e parts cost measurably more
  per byte, so "invariant J/byte" holds within HBM3/3e, not across generations.
- Closed-loop vs Poisson (H200+B200): J/token 4.1% vs 3.7%.
- Signed J/token bias is −2 to −5% on most groups: v2 slightly under-predicts.
- Worst cases:
  - gemma-7b on H200 runs at 41% MBU against 50% predicted (architecture-specific
    slowness);
  - A100 c=64 runs are +20 to +38% (throttled time; cache hits not removed);
  - 32B Poisson on H200 is −18%.

## 4. Recomputed recommendations

**H200 vs B200 win-map** (fig8c,d; prompt 1024 / gen 256; cell = E(B200)/E(H200) per
token; bold = P(winner) ≥ 0.9; cached_frac = 0):

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
  P(B200) is only 30-85%: **not confident**. The prefix-cache fix leaves decode
  unchanged to ±0.01.
- **Prefill: B200 favored by about 13-15% (ratio 0.85-0.88) for models ≥3B at batch
  ≥4, with P(B200 best) of 0.78-0.88. No cell reaches 0.9.** In v2.0 this was 17%
  (ratio 0.80-0.86, P up to 0.94).
  - With corrected e_gemm, absolute prefill energy rose on both GPUs. For 7B, 2048
    prompt, batch 32: H200 12.5 → 14.1 mJ/token, B200 10.3 → 12.2.
  - The per-FLOP e_gemm ratio barely moved (0.79 → 0.81), but B200 is now predicted to be more
    **power-cap-limited** on compute-bound prefill (demand > 1000 W, so it throttles),
    which eats part of its advantage.
  - The advantage comes from lower measured e_gemm (0.64 vs 0.79 pJ) plus 2.3× peak
    FLOPS amortizing idle power. It
  rests on the unmeasured prefill-MFU prior and on e_gemm ranges that drift with model
  size. Tiny prefills (0.5B, batch 1-2) go to H200 (overhead-bound).
- 72B does not fit on one H200, so the H200 side is TP=2 and extrapolated. On that
  basis H200×2 still wins decode (P≈1 within the model), but NVLink/NCCL energy is not
  modeled.

**The phase reversal survives, but only between H200 and B200, not in its old form.**
- **Among the measured GPUs:** B200 for prefill, H200 for decode. This is a genuine
  disaggregation recommendation. The decode half is confident; the prefill half is
  about 80-88%.
- **The v1 claim ("L40S for prefill, A100 for decode"), re-tested with the GDDR6
  prior:** at prompt 2048 / gen 256 / batch 32, for 7B and 14B:

  | phase | E(L40S)/E(A100-SXM) | P(A100 wins) | driven by |
  |---|---|---|---|
  | decode | **1.72-1.76** (vs PCIe 1.56-1.58) | ≈1.00 | robust |
  | prefill | 1.19 (central) | P(L40S wins) = 0.32-0.42 | unmeasured Ada e_gemm |

  - **Decode:** A100 now wins **confidently**. In v2.1 this was a coin flip; it
    changed because L40S's GDDR6 bytes cost about 1.8× the HBM2e prior (about 2.7×
    HBM3e) and L40S streams 2.3× slower. The result is robust to L40S idle (35-70 W)
    and to e_gemm: the ratio stays 1.44-1.82× across the sensitivity sweep.
  - **Prefill:** the answer is set entirely by the unmeasured Ada e_gemm. The ratio
    is 0.58× at 0.8 pJ, 1.06-1.19× at 1.6 pJ, and 1.95-2.2× at 3 pJ; at a
    GA102-like 3 pJ it is ≈2×.
  - **Joint P(reversal)** is 32-42%.
  - **Verdict:** the decode half of the old story (A100 over L40S) now holds firmly,
    though for the opposite reason from v1: technology-class energy per byte, not
    1/bandwidth. The prefill half ("L40S wins prefill") is **unsupported**; it needs
    an Ada e_gemm measurement. The only credible phase reversal is still B200 for
    prefill, H200 for decode.

**A5000:** the current `logs/A5000` sweep is power-limited to 100 W and **excluded**
(`_excluded` in `gpu_coefficients.json`). Both tools skip `_excluded` GPUs, so it is in
neither the backtest, the time fit nor the figures. GDDR6 remains a *prior*, now built
from the A5000 full-clock microbenchmarks (labelled "microbench prior").

Once a valid sweep lands and the entry moves out of `_excluded`, the pipeline takes it
automatically:
- `load_gpus()` uses the entry, and it replaces the GDDR6 prior.
- The backtest discovers `logs/*A5000*`.
- `--fit` fits a time model for it. With one model it fits only t0 and η, fixing τ,
  η_kv and μ at the measured-GPU medians, and falls back to the prior if the fit is
  worse than 15%.
- `--winmap` then also prints A5000 vs H200.

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
- **Prefix cache:** win-maps assume cached_frac = 0. With shared prefixes, pass
  `--cached-frac`. Prefill energy then scales about linearly, and the GPU ranking does
  not change.
- **Unmeasured GPUs** (H100, A100-SXM, L40S, A5000 until a valid sweep) are prior-only.
  Their time model is the pooled H200/B200 ensemble, which is untested on GDDR6 parts.
  The GDDR6 J/byte prior is confounded with node and clock.
  Idle power for them is an estimate. Treat their rankings as hypotheses; the tool
  labels them `LOW (prior)`.
- **Power cap:** the throttling model holds dynamic energy fixed and stretches time.
  It fits the A100 behaviour (power 3% MAPE), but real DVFS lowers J/op somewhat.
