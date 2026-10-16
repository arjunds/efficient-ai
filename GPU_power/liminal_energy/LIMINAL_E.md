# LIMINAL-E: LIMINAL's decode roofline with measured energy and a power roof

LIMINAL (arXiv 2507.14397) models decode time from bandwidth, compute, synchronization and
capacity. For power it divides throughput by TDP, so power never depends on activity
(`LIT_LIMINAL.md`). LIMINAL-E keeps LIMINAL's workload side and adds four things:

- a fitted time model, with utilization measured rather than forced to peak;
- a measured, activity-dependent energy model;
- a power roof that can bind before the bandwidth and compute roofs do;
- a provisioning layer that chooses GPU type, TP degree and replica count to minimize J/token.

**Code.** `liminal_e.py` is the model. It reuses `recommend_gpu.py` for the GPU table and
priors, model resolution, serving work and the time model.
- `validate.py` scores it against measured runs and writes `validation.json` and
  `validation_runs.csv`.
- `kv_longctx.py` runs the KV test in §7.
- `figdata.py` computes the figure data and `plot_figs.py` draws `figs/`.
- Run everything in the container. `plot_figs.py` needs `PYTHONPATH=/shared_data0/adsampat/pydeps`.

**Labels used throughout.** **[M]** = measured, **[F]** = fitted to our runs,
**[P]** = prior or prediction, **[O]** = open question.

---

## 1. Workload per step (GPU-independent)

Take one decode step of a dense model. The model has N_act active parameters, L layers,
hidden size d, and κ KV bytes per token (all layers, fp16). The step has B running
sequences at mean resident context c, plus T_pf prefill tokens mixed in. Then T = B + T_pf
tokens pass through the GEMMs.

| quantity | expression | note |
|---|---|---|
| weight bytes | W = 2·N_w(T) | canonical `~/models.py`: no input-embedding gather, lm_head counted once. For MoE, N_w(T) counts the experts touched (`energy_model.weight_params_for_tokens`) |
| KV bytes read | K = B·c·κ, with κ = 2·L·n_kv·d_head·2 | each sequence's attention reads its own context |
| dense GEMM FLOPs | F_g = 2·N_act·T | |
| attention FLOPs | F_a = 4·n_h·d_h·L·(B·c + T_pf·c_pf/2) | in decode, attention is KV-byte-bound and its energy sits in e_kv. Prefill attention (causal, ~c²) is FLOP-bound and is charged at e_g |
| TP all-reduce bytes per GPU | A = 2·L·T·d·2·2(n−1)/n | two all-reduces per layer, ring algorithm |
| arithmetic intensity | I = F/(W+K) | about T for short-context decode |

In steady continuous batching (`rg.serving_iter_work`), with prompt P, generation G and
prefix-cache hit fraction f: T = B·(1 + (1−f)·P/G) and c = P + G/2.

- **[M] The byte model is validated.** NCU DRAM bytes / analytic bytes = **1.004** and
  **1.008** on B200 decode, in two independent captures (FINDINGS_B200.md UPDATE 3 and UPDATE 4).
- **[M] TP communication is negligible in energy.** All-reduce bytes are ≤0.3% of weight
  bytes for 72B TP2 at c ≤ 64, so e_link (prior: equal to e_wbyte) barely matters.

## 2. Time: LIMINAL's form, with fitted utilization

LIMINAL v2 (§5):
```
t_LIMINAL = max( (W+K)/(n·BW_peak), F/(n·FLOPS_peak) ) + T_exp·L,   T_exp ≈ 114 µs (8×H100 fit)
```
The memory slope is fixed at peak bandwidth, so all inefficiency lands in the per-layer intercept.

LIMINAL-E uses the recommender's realized-utilization model (`rg.iter_time`, `RECOMMENDER.md` §2):
```
t_mem = t0 + L·softmax_k( τ + τ_ar , W/(L·n·BW·η) ) + K/(n·BW·η_kv) + F/(n·FLOPS·μ)
t_cmp = t0 + F/(n·FLOPS·MFU_pf)
t     = max(t_mem, t_cmp)
```
It differs from LIMINAL in three ways:
1. η (the asymptotic MBU) and η_kv are **fitted**, not set to 1.
2. There is a per-step framework overhead t0 in addition to the per-layer floor τ.
3. The floor composes with the stream through softmax_k. k = 1 gives LIMINAL's additive form.

| | H200 [F] | B200 [F] |
|---|---|---|
| form | hard max (k=50) | additive (k=1), i.e. LIMINAL's shape |
| t0, τ | 2.15 ms, 110 µs/layer | 1.42 ms, 78 µs/layer |
| η, η_kv | 0.73, 0.22 | 0.86, 0.33 |
| LOMO step-time MAPE | 5.3% | 2.7% |

**New in LIMINAL-E: two time terms fitted on the B200_prov runs** (`validation.json`).

**CUDA graphs: τ_graphs = τ − 34.5 µs/layer [F].**
- The time model above was fitted on eager-mode runs.
- Δτ was fitted on 7B graphs runs at c ≤ 64 and tested on held-out 32B and 72B graphs runs at c ≤ 64:

  | variant | held-out step-time MAPE |
  |---|---|
  | eager model, unchanged | 12.8% |
  | **per-layer correction** | **3.5%** |
  | per-step alternative (Δt0 = 0.97 ms) | 7.5% |

- A model-free check gives the same answer. The Sept-eager minus graphs step time at c=1 is
  0.88, 1.89 and 2.45 ms for 28, 64 and 80 layers, i.e. **31.6, 29.6 and 30.7 µs per layer**.
  The saving is per layer, not per step.
- Caveat: the Oct eager "relog" runs are ~1.3 ms per step slower than the Sept eager runs,
  probably because of the heavier logger (UPDATE 5 review). The graphs saving is therefore
  measured against Sept eager, not relog eager.

**TP synchronization: τ_ar = 15.5 µs/layer, with no time-inflation factor [F] (B200 NVLink, TP=2).**
- Fitted on the 72B TP2 runs. On held-out 32B TP2 the step-time MAPE is **1.3%**:

  | sync model | 32B held-out step-time MAPE |
  |---|---|
  | **τ_ar = 15.5 µs/layer (fitted)** | **1.3%** |
  | no sync term | 10.2% |
  | recommender prior (25 µs, ×1.15) | 20.5% |

- This is the first measured value of LIMINAL's sync term on our hardware.
- [P] For other GPUs, both terms fall back to the recommender priors. **[O] H200 with CUDA
  graphs has never been run.**

## 3. Energy

Per step, on n GPUs:
```
E = n·P_s·t  +  e_w·W  +  e_kv·K  +  e_g·(F_g + F_a,prefill)  +  e_link·A
J/token = E / B
        = n·P_s·t/B  +  e_w·W/B  +  e_kv·c·κ  +  e_g·2N_act·(1 + (1−f)P/G)  (+ prefill-attention term)
```
Each term has its own slope in B:
- weights: slope −1 on log-log;
- static: slope −1 while overhead-bound, flattening as t grows;
- KV and compute: flat per token.

That is what fig (b) plots.

| coefficient | H200 | B200 | source |
|---|---|---|---|
| P_s (idle) | 116.4 W [M] | 237.8 W [M] (246.6 W on the Oct node) | gpu_coefficients.json; provisioning_summary.json |
| e_wbyte | 1.066e-10 J/B [M] | **1.25–1.26e-10, robust** [M] | gpu_coefficients.json; UPDATE 5 review |
| e_kvbyte | 3.29e-10 (per-bin) | **[O] undetermined:** 1.0–4.0e-10 depending on fit | UPDATE 5 review |
| e_gemm | 0.79 pJ per-bin; 0.96–1.10 run-level | **[O] 0.71–0.99 pJ**; steady uncapped 0.943 | BALANCE.md; steady_calib_B200.json |
| per-channel (optional) | — | e_DRAM 8.37e-11, e_L2 2.29e-11, e_L1 1.00e-11 J/B [M]. With serving traffic DRAM 1.008 / L2 1.567 / L1 0.0013 per analytic byte, they rebuild e_wbyte at 1.203e-10 (−3%) | UPDATE 4b |

**Profiles.**
- `canonical` (gpu_coefficients.json, as in the recommender) is the default.
- `kv_high`, `kv_low` and `runlevel` bracket the undetermined KV/compute split on B200 (UPDATE 5
  relog free fit, relog with e_gemm pinned at 0.943, and the whole-run fit).
- `kv_as_weight` (e_kv := e_w) is a sensitivity case for any GPU.
- On decode J/token the profiles differ by ≤1 MAPE point (§6): they are collinear on our
  data. They diverge for long context (fig b shading, fig c panels 2 vs 3).

**Two balance points** (FLOP per byte; for decode, about tokens per step):

| | performance ridge FLOPS/BW | energy ridge e_w/e_g (canonical) | e_w/e_g (smear-free alternatives) |
|---|---|---|---|
| H200 | 206 | 134 | 93 (run-level) |
| B200 | 281 | 194 | 133 (steady / kv_low), 177 (kv_high) |

- The energy ridge is 0.47–0.69× the performance ridge. Weight-streaming energy equals GEMM
  energy only at about 95–195 tokens per step.
- In batch terms, with prefill FLOPs charged to the tokens they serve, the crossover is
  **B\* = e_w·W / (2·e_g·N_act·(1+P/G))**. For Qwen2-7B on B200: 166 at P/G = 0.17, but
  **11 at P=8192, G=512**. With long prompts, prefill compute dominates per-token energy
  almost immediately.
- **Static-equivalent bandwidth** BW_s = P_s/e_w: 1.09 TB/s on H200 and 1.91 TB/s on B200,
  both 23–24% of peak.
  - The effective cost of a streamed byte is e_w + P_s/BW_realized.
  - On B200 at the measured graphs-mode realized 3.1 TB/s (7B, c=1), idle adds **+61%** to
    every weight byte.
  - This is the idle-amortization term, and it is what makes batching and consolidation pay.

## 4. Power roofline

Dynamic power is linear in the byte rate β and FLOP rate φ:
```
P = P_s + P_0 + e_b·β + e_f·φ  ≤  P_cap          (P_0: activity intercept, 38.7 W on B200)
```
At intensity I = φ/β, with Δ = P_cap − P_s − P_0:
```
φ_P(I) = Δ / (e_b/I + e_f)                   power roof (harmonic)
→ I·BW_P for small I, with BW_P = Δ/e_b      power-limited bandwidth
→ Φ_P for large I, with Φ_P = Δ/e_f          power-limited FLOP rate
attainable φ(I) = min( I·BW, FLOPS_peak, φ_P(I) )
```
- The knee of the power roof sits at **I = e_b/e_f**, which is the energy ridge.
- The power roof binds the memory side iff Δ < e_b·BW. It binds the compute side iff Δ < e_f·FLOPS_peak.

| | BW_P (share of BW roof) | Φ_P (share of peak) | φ_P at the perf. ridge | source |
|---|---|---|---|---|
| B200, full clock | **5.78 TB/s (72%)** | **768 TF (34%)** | 521 TF (23% of peak) | steady_calib_B200.json: e_b 1.251e-10 + 38.7 W, e_f 0.943 pJ [M] |
| B200, throttled (DVFS) | 6.79 TB/s (85%) | 1.23 PF (55%) | — | capped streams 1.122e-10 J/B, capped GEMMs 0.57–0.67 pJ [M] |
| B200, observed at cap | 7.39 TB/s (92%) at SM 1578 MHz | 1.1–1.3 PF at SM 1.1–1.7 GHz | — | [M] |
| H200 | 5.48 TB/s (114%): **never binds streaming** | 610 TF (62%); 735 TF at e_g 0.79 | — | **[P]**: serving e_w + run-level e_g, no steady calibration |

**B200 is a power-roof machine.** At full clock its power roof lies below the classical
roofline at every intensity. **H200 is a roofline machine for decode**: power binds only
compute-heavy work. In fig (a), the measured B200 GEMM loops fall between the full-clock and
DVFS roofs: uncapped points sit under the full-clock roof, capped points near the DVFS roof.

**Time stretch at the cap.** If the demanded power E_dyn/t + n·P_s exceeds n·P_cap, then
```
t = max( t_uncapped , E'_dyn / (n·(P_cap − P_s)) ),   power pinned at the cap
```
- Default: E' uses the full-clock coefficients (the recommender's convention).
- `cap_regime=True` uses the measured lower J/B and J/FLOP at the cap. This represents the
  measured drop in J/FLOP, **0.57–0.67 pJ capped against 0.94 pJ uncapped** (UPDATE 4 review),
  as a capped-regime coefficient.
- **Validation favours the default for serving.** On the 5 capped serving runs:

  | run | measured | default (full-clock J) | DVFS coefficients |
  |---|---|---|---|
  | 32B c=512, J/token | 0.0908 | 0.0905 | 0.0855 (−5.8%) |
  | prefill, J per prompt token (4 runs) | — | MAPE 11.9% | MAPE 14.5% |

  The steady-loop capped coefficients describe kernels, not a whole serving step at the cap.
- [M] **Prefill at the cap behaves like e_g ≈ 0.7 pJ.** Profile fits to the 4 prefill runs:

  | e_g | MAPE (bias) |
  |---|---|
  | 0.713 (kv_high profile) | **5.1% (−1.5%)** |
  | 0.642 (canonical) | 10.5% (−10.5%) |
  | 0.943 (steady uncapped) | 23.7% (+23.7%) |

  The effective J/FLOP at the cap lies between the uncapped and loop-capped values. This
  matches the 0.66–0.84 pJ/FLOP reported in UPDATE 5.

## 5. Provisioning

Given a model, a context mix (P, G, f), a load (concurrency C, or Poisson rate λ) and a TPOT
SLO, LIMINAL-E enumerates GPU type × TP degree n ∈ {1, 2, 4, 8} × replica count R:
```
feasible: weights + per-replica KV ≤ 0.9·n·mem;   b ≤ max_seqs;   t(b) ≤ SLO
closed loop: b = C/R          open loop: b = (λ/R)·G·t(b)   (Little's law, fixed point)
J/token = R·[ n·P_s·t + E_dyn(b) ] / (b·R)  →  minimized over (GPU, n, R)
```
- `provision_conc` and `provision_rate` in `liminal_e.py`.
- Every provisioned GPU is charged its idle power. So the objective favours the fewest,
  busiest GPUs that meet the SLO, and more bandwidth pays only when it is used.

**Example: open loop, Qwen2.5-32B, P=1024, G=256, TPOT ≤ 50 ms** (`figs_data.json`; [P] predictions):

| λ (req/s) | best | runner-ups | note |
|---|---|---|---|
| 0.5 | A100-SXM ×1, 2.62 J/token [low-confidence prior] | H100 3.48; H200 3.73; B200 5.94 | B200 costs 1.6× H200: idle is not amortized |
| 5 | **B200 TP1 ×1, 0.71 J/token** | H100 TP2 0.80; H200 TP2 0.83 | |
| 50 | **B200 TP2 × 3 replicas, 0.45 J/token** | H100 TP2 × 5, 0.52 | |

**Provisioning map, fig (c)** (measured GPUs only, eager time model, TPOT ≤ 50 ms):
- **Short context:** H200 wins everywhere up to c=256 for ≤7B models. B200 wins from c ≥ 256
  for 14–32B and at c=1024 for every model from 1.5B up. This agrees with the measured crossover in UPDATE 5
  (B200 wins only at c ≳ 256).
- **Long context:** B200 wins from c ≥ 64 for ≤7B models and from c ≥ 16 for 14–32B, through
  more KV capacity per GPU and its lower canonical e_kv. **Under the `kv_as_weight` hypothesis
  the crossover moves out to c=64 for 14–72B and to c=256 (1024 for 0.5B) for small models.**
  The long-context ranking depends on the open KV coefficient.

| status | items |
|---|---|
| measured | H200 and B200: P_s, e_w, time model. B200: CUDA graphs, TP2 sync, power roof, capped regime |
| priors | A100, H100, L40S, A5000: memory-technology e_w priors and e_g priors (RECOMMENDER.md); time model = pooled ensemble prior; idle estimated, except A100-PCIe and A5000 |
| open | **KV vs compute split** (both GPUs); H200 with CUDA graphs; H200 steady power roof; TP > 2; arrival-rate (open-loop) validation at high load |

## 6. Validation (`validate.py` → `validation.json`, `validation_runs.csv`)

Columns of the table below:
- **LIMINAL-E**: canonical coefficients, time model matched to the run's mode (eager fit, or
  graphs = eager fit with τ − 34.5 µs), closed-loop at the nominal concurrency. Inputs are only
  the model, P, G, f, c and the GPU.
- **Energy-only**: the energy model evaluated at the *measured* running batch and step time.
  It isolates the energy coefficients from the time model.
- **LIMINAL time + E**: LIMINAL v2's time model with our energy model.
- **LIMINAL (TDP)**: LIMINAL as published, with power = TDP.

J/token MAPE (bias); the last column is step-time MAPE:

| set (n) | LIMINAL-E | energy-only | LIMINAL time + E | LIMINAL (TDP) | t MAPE |
|---|---|---|---|---|---|
| Sept H200 eager (80) — *in-sample* | 3.8% (−2.8) | 2.6% | 9.8% | 87.5% | 4.1% |
| Sept B200 eager (28) — *in-sample* | 3.9% (−3.3) | 1.6% | 7.6% | 28.4% | 2.2% |
| Oct B200 graphs, c ≤ 64 (11) | 5.3% (−5.3) | 3.5% | 5.3% | 39.4% | 3.0% |
| Oct B200 highconc, c 128–512 (9) | 10.4% (−9.6) | 6.9% | 13.5% | 12.2% | 12.1% |
| Oct B200 relog, eager (8) | 13.4% (−13.4) | 3.7% | 19.2% | 26.7% | 18.4% |
| Oct B200 72B TP1 (2) | 4.3% | 1.9% | 3.9% | 5.2% | 3.5% |
| Oct B200 TP2 (6)† | 4.7% (−4.5) | 2.2% | 5.5% | 37.2% | 0.9% |
| Oct B200 prefill-heavy (4), per generated token | 10.5% (−10.5) | 13.4% | 11.2% | 11.2% | 4.8% |
| **all decode runs (144)** | **4.9%** | **2.8%** | **9.5%** | **61.0%** | |
| **Oct decode, out of sample (36)** | **8.2%** (6.7% without relog) | 4.1% | 10.4% | 27.5% | |

† τ_ar was fitted on the 3 72B TP2 runs.

**B200 / H200 J/token ratio**, 22 matched Sept pairs (balance_analysis.json `matched`):
- LIMINAL-E MAPE **2.5%**; LIMINAL-TDP MAPE 13.8%. Both get the sign right in 22 of 22.

**TP2 energy premium** (J/token TP2 ÷ TP1):

| | predicted | measured |
|---|---|---|
| energy premium | 1.07–1.13 | 1.03–1.13 |
| speedup | 1.37–1.58 | 1.37–1.60 |

The premium is idle power of the second GPU plus sync time. Link energy is not material.

**Where it misses:**
1. **Oct graphs runs: −5% bias.** About 1.4 points is the Oct node's higher idle (246.6 W
   against the 237.8 W coefficient). The remaining ~4% is dynamic power under-predicted by
   ~17 W. Note that the eager time model scores *better* on these runs (3.7%) only by
   cancellation: it over-predicts time by 18% and under-predicts power.
2. **Relog eager: −13%.** This is a time miss (t −18%; energy-only 3.7%), consistent with the
   logger slowing eager runs (UPDATE 5 review).
3. **7B at c = 256–512: time under-predicted 12–27%.** The μ and MFU terms were fitted at
   ≤110 tokens per step, and scheduler limits are not modelled. This is the extrapolation
   UPDATE 5 flagged: B200 7B alpaca c=512 throughput over-predicted by 28%.
4. **Long-prompt prefill at the cap: −15 to −16% at P=8192, −3 to −7% at P=2048.** Even with
   prefill attention charged, canonical e_g is too low at the cap (§4).
5. **Cap onset.** 32B at c=128/256 sits at the cap 95–98% of the time; predicted 945 W and
   999 W, i.e. at the edge rather than capped. 32B c=512 is predicted capped (P 1000 W,
   J −0.4%). 72B c=16 is 72% capped; predicted 948 W.

## 7. A KV test that half-works: the H200 long-context sweep (`kv_longctx.py`)

`logs/longctx` (H200, Qwen2-7B, July 2026; prompts 4k/16k/30k × c = 1/8/16) varies KV bytes
about 7× at fixed concurrency. That breaks the KV-FLOP collinearity.

Run-level power at the measured n and step time:

| e_kv hypothesis | power MAPE (bias) |
|---|---|
| canonical, 3.29e-10 | 38.9% (+38.9) |
| run-level, 2.42e-10 | 22.5% (+22.3) |
| **e_kv = e_w, 1.07e-10** | **1.9% (−1.9)** |
| free fit | **e_kv = 1.12e-10**, 1.8% |

**This does not settle the split.**
- The synthetic prompts are identical, so every sequence reads the same prefix-cached KV blocks.
- The incremental KV bytes per incremental step time at c=16 come to **5.87 TB/s, above the
  4.8 TB/s peak**. Some of the counted KV reads are therefore served from L2, so the fitted
  e_kv per logical byte is biased low as a DRAM coefficient.

What it does show: for **shared-prefix** long-context traffic, the canonical e_kv
over-predicts power by about 40%, and up to 2× at c=16 and 30k context.

**The decisive run** is the same sweep with `unique_prompts=True` (the harness already has
the flag), on H200 or B200.

## 8. What LIMINAL-E adds over LIMINAL

| | LIMINAL (v2) | LIMINAL-E |
|---|---|---|
| bandwidth utilization | forced to 1.0 | fitted η: 0.73 (H200), 0.86 (B200), plus η_kv |
| overhead | 114 µs/layer, fitted on H100 | t0 plus τ/layer, fitted per GPU. **CUDA graphs = −34.5 µs/layer**, measured as a per-layer effect |
| sync | modelled, not validated here | **τ_ar = 15.5 µs/layer measured** (B200 TP2), 1.3% held-out time |
| power | TDP, no activity dependence | P_s + e_w·W + e_kv·K + e_g·F, measured. J/token MAPE 4.9% against 61% for TDP on the same 144 runs |
| power limit | none | **power roof** φ_P(I) = Δ/(e_b/I + e_f), plus a capped regime (time stretch, optional DVFS coefficients) |
| balance | qualitative | energy ridge e_w/e_g against perf ridge; static-equivalent bandwidth P_s/e_w |
| bound labels | memory / compute | bandwidth / compute / **overhead** / **power** (per step) |
| provisioning | STPS/W at TDP | GPU × TP × replicas minimizing J/token under a TPOT SLO, with idle charged |

## 9. Limitations (read before quoting)

- **The KV/compute split is undetermined** on both GPUs (UPDATE 5 review; §7 is suggestive
  but confounded). Short-context J/token is insensitive to it. Long-context J/token and the
  long-context provisioning map are not.
- **The time model is fitted on eager vLLM 0.10.2.**
  - The graphs correction is B200-only and fitted on one model.
  - H200 with graphs is a prior.
  - Above ~110 tokens per step the time model is extrapolated (miss 3).
- **The power roof is measured only on B200.** The H200 roof uses serving coefficients and has
  no P_0. Priors for A100/H100/L40S scale B200's capped/uncapped ratios.
- **Capping is binary per step.** Real serving moves in and out of the cap within a step and
  across steps. Average power at "capped" runs is 96–98% of the cap.
- **The prefill-attention energy convention (e_g)** is untested beyond the 4 prefill runs.
  Attention may cost more per FLOP than GEMMs.
- **The Sept sets are in-sample** for both the energy coefficients and the time model. The
  out-of-sample numbers are the Oct B200 rows, which come from one GPU type, three models and
  one node.
- **Static power is GPU idle only.** Host and node power are excluded, and including them
  would strengthen the consolidation result. L40S/H100/A100-SXM idle values are estimates.
- **The open-loop provisioning has not been validated** against Poisson runs at high load. The
  provisioning maps are predictions, and cells at c=1024 extrapolate the time model.

## Figures (`figs/`)
- `fig_a_power_roofline.png`: H200 and B200 rooflines with the bandwidth, compute and power
  roofs (B200: full clock and throttled), measured serving points (per GPU), prefill runs and
  B200 steady GEMM microbenchmarks (capped vs uncapped).
- `fig_b_energy_roofline.png`: J/token against batch, split into static, weight, KV and
  compute, for short and long context, with the energy ridge, the power-capped region, the B200
  kv_low..kv_high band and measured points.
- `fig_c_provisioning_map.png`: best GPU (H200 vs B200), TP degree, replicas and J/token over
  model size × concurrency. Panels: short context, long context, and long context under KV = weight.
- `fig_d_validation_parity.png`: predicted against measured J/token.
