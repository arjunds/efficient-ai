# LIMINAL-E: LIMINAL's decode roofline with measured energy and a power roof

*v2, 2026-10-21. The model now covers four measured GPUs: RTX A6000, A100 80GB PCIe, H200 and
B200. The B200 KV coefficient is settled. A100 and A6000 have a zero-shot test. v1 (H200/B200 only) is
backed up in the scratchpad, and its validation is in `validation.json`.*

LIMINAL (arXiv 2507.14397) models decode time from bandwidth, compute, synchronization and
capacity. For power it divides throughput by TDP, so power never depends on activity
(`LIT_LIMINAL.md`). LIMINAL-E keeps LIMINAL's workload side and adds four things:
- a fitted time model;
- a measured, activity-dependent energy model;
- a power roof, plus a measured **at-cap roofline**;
- a provisioning layer that chooses GPU type, TP degree and replica count to minimize J/token.

**Labels used throughout.** **[M]** = measured, **[F]** = fitted to serving runs,
**[P]** = prior, assumption or prediction, **[O]** = open question.

**Code** (everything runs in the container; plotting adds `PYTHONPATH=pydeps`):

| file | what it does |
|---|---|
| `liminal_e.py` | the model. Reuses `recommend_gpu.py` and overlays `gpus_measured.json` |
| `build_measured.py` | builds `gpus_measured.json` from the microbenchmarks only (the zero-shot block) |
| `validate_v2.py` | zero-shot and calibrated scoring on 4 GPUs; writes `validation_v2.json`, `validation_v2_runs.csv` and the calibrated blocks |
| `figdata.py` | figure data, `ranking_table.csv` |
| `plot_figs.py` | draws `figs/` |
| `validate.py`, `kv_longctx.py` | v1 scripts |

`gpu_coefficients.json` is not modified.

---

## 1. Workload per step (GPU-independent)

Take one decode step of a dense model. The model has N_act active parameters, L layers, hidden
size d, and κ KV bytes per token. The step has B sequences at mean context c, plus T_pf
chunked-prefill tokens, so T = B + T_pf tokens pass through the GEMMs.

| quantity | expression |
|---|---|
| weight bytes | W = 2·N_w(T) (canonical `~/models.py`; for MoE, the experts touched) |
| KV bytes | K = B·c·κ |
| GEMM FLOPs | F_g = 2·N_act·T |
| prefill attention FLOPs (FLOP-bound, charged at e_g) | F_a,pf = 4·n_h·d_h·L·T_pf·c_pf/2 |
| TP all-reduce bytes per GPU | A = 2·L·T·d·2·2(n−1)/n |
| arithmetic intensity | I = F/(W+K) |

In steady serving, T = B·(1 + (1−f)·P/G) and c = P + G/2.

[M] **The byte model is validated by NCU.** DRAM bytes / analytic bytes = 1.004 and 1.008 at
short context, and 1.012 at 8k and 30k context (FINDINGS_B200 UPDATE 3, 4 and 6). This
validates both the weight and the KV byte counts.

## 2. Time

LIMINAL v2: `t = max((W+K)/(n·BW_peak), F/(n·FLOPS_peak)) + 114 µs·L`, with bandwidth forced to peak.

LIMINAL-E (`rg.iter_time`):
```
t_unc = max( t0 + L·softmax_k(τ+τ_ar, W/(L·n·BW·η)) + K/(n·BW·η_kv) + F/(n·FLOPS·μ),  t0 + F/(n·FLOPS·MFU) )
```
η is fitted, not forced to 1. t0 is the per-step overhead and τ the per-layer floor.

| term | value | source |
|---|---|---|
| H200 / B200 time model | [F] fitted (RECOMMENDER.md §2) | |
| A100 / A6000 time model | **[P] the pooled H200/B200 prior** | no A100/A6000 serving data used |
| CUDA graphs | [F] τ − 34.5 µs/layer, B200. Held-out 32B/72B step-time MAPE 3.5%, against 12.8% without the correction | `validation.json` |
| B200 TP sync (NVLink) | [F] τ_ar = 15.5 µs/layer. Held-out 32B 1.3% | `validation.json` |
| **A100 TP sync (PCIe Gen4, no NVLink)** | **[F] τ_ar = 185 µs/layer + all-reduce bytes at 5 GB/s effective**, fitted on 14B TP2 | `validation_v2.json tp_a100` |
| **A100-node host overhead** | **[F] t0 and τ ×1.92**, eager mode only, fitted on 27 uncapped eager runs (0.5–3B). It is a node/CPU property, so the provisioning map resets it to 1 | `validation_v2.json` |

- The A100 PCIe sync costs ~12× B200's NVLink per layer, and adds a bandwidth term.
- Held-out 32B TP2 throughput MAPE is 16.3% (no sync term: 27.4%). The 32B TP2 runs are
  power-capped, so they test the sync term only weakly.
- Measured: TP2 costs 1.14–1.64× the J/token of TP1 (FINDINGS_A100).

## 3. Energy

```
E = n·(P_s + P_0)·t + e_w·W + e_kv·K + e_g·(F_g + F_a,pf) + e_link·A        J/token = E/B
```
- **P_0** is the activity intercept above clocks-up idle. It matters on GDDR6: the memory
  clock jumps from 417 to 8001 MHz whenever there is traffic.
- e_link = e_w [P]. It is negligible (all-reduce bytes are ≤0.3% of weight bytes at TP2).

| GPU | P_s (W) | P_0 (W) | e_w (J/B) | e_kv (J/B) | e_g (pJ) | source |
|---|---|---|---|---|---|---|
| H200 | 116.4 [M] | 0 | 1.066e-10 [F] | **1.33e-10 [P]** (= 1.25·e_w, the B200 ratio) | **1.20 [F]** (refit with that KV) | gpu_coefficients.json; `validation_v2.json h200_refit` |
| B200 | 237.8 [M] | 0 | 1.310e-10 [F] | **1.548e-10 [F], settled**: 1.18× e_w; slopes 1.60–1.61e-10 | 0.70 [F] (steady uncapped 0.943 [M]) | longctx_summary.json (UPDATE 6 review) |
| A100 PCIe | 102.5 [M] (vLLM loaded; 47.5 with no process) | 0 | 1.83e-10 [M] (GEMV duty-cycle) | 2.29e-10 [P] (1.25·e_w) | 1.47 [M] | steady_calib_A100 / results_A100_steady.json |
| A6000 | 106.7 [M] (clocks up; 85.8 vLLM idle; 32.5 no process) | **80.5 [M]** | 1.75e-10 [M] (grid marginal) | 2.19e-10 [P] | 2.64 [M] | results_A6000_steady.json grid and duty |

**Zero-shot rule, fixed before scoring** (`build_measured.py`):
- P_s = clocks-up idle; P_0 and e_w = the uncapped occupancy-grid intercept and slope;
- e_kv = 1.25·e_w; e_g = the duty-cycle GEMM J/FLOP.

The alternative, e_w = GEMV duty-cycle J/B with P_s = vLLM idle and no P_0, is scored as a
sensitivity case.
- On A100 it fits measured uncapped power slightly better: 4.3% against 5.7% MAPE at the
  measured operating point. The calibrated A100 uses it.
- On A6000 it fails (J −17%) because it omits the memory-activity term.

**Ridges:**

| GPU | performance ridge (FLOPS/BW) | energy ridge e_w/e_g |
|---|---|---|
| A6000 | 202 | 66 |
| A100 | 161 | 124 |
| H200 | 206 | 89 |
| B200 | 281 | 188 |

Measured decode serving sits at 1–90 FLOP/B, left of every ridge.

## 4. Power roof and the at-cap roofline

**Full clock.** `P_s + P_0 + e_b·β + e_f·φ ≤ P_cap`, which gives `φ_P(I) = (P_cap−P_s−P_0)/(e_b/I + e_f)`.
The knee is at e_b/e_f.

**At the cap.** The GPU throttles its clocks, so full-clock coefficients badly underpredict
what it delivers. The A6000's duty-cycle GEMM costs 2.64 pJ/FLOP at full clock but about 1.8
pJ at the cap. LIMINAL-E therefore fits an **at-cap roofline** directly on capped, LLM-shaped
microbench kernels (llmgemm M = 1…8192, all at P_cap):
```
seconds per DRAM byte at the cap = softmax_k( τ_b , I·τ_f )
→ capped streaming rate 1/τ_b,  capped FLOP rate 1/τ_f,  at-cap J/B ε_b = (P_cap − P_s)·τ_b
```

| GPU | full-clock power roof (BW / peak) | **at-cap roofline [M]** (BW / peak, k, rmsle) | at-cap ε_b |
|---|---|---|---|
| A6000 | 84% / 28% | **0.66 TB/s (86%) / 107 TF (69%)**, k=2, 0.065 | 2.92e-10 J/B |
| A100 | 71% / 38% | **1.31 TB/s (68%) / 192 TF (62%)**, k=1.5, 0.12 | **1.53e-10 J/B**: serving at-cap weight byte 1.59e-10 [M], −4% |
| B200 | 73% / 34% | 5.94 TB/s (74%) / 1.42 PF (63%), k=1, 0.09 | 1.29e-10 J/B |
| H200 | 114% / 62% [P] (power never binds streaming) | none (no microbench) | — |

**At-cap step model.** A step is capped when `n(P_s+P_0) + E_dyn/t_unc > n·P_cap`. Then:
```
t = max( t_unc , s·t_cap(W+K, F) + t_overhead ),   power = n·P_cap,   J/token = n·P_cap·t / B
```
- t_overhead is the host/launch part of the time model.
- s = 1 zero-shot. The calibrated values are A100 1.17 (without the overhead term) and A6000 1.19.

How the at-cap model transfers:
- **Zero-shot, the parameter-free form already reaches 7–8% on capped runs** (§6).
- **Calibration does not improve it** (A100 capped runs: 7.0% zero-shot, 7.8% calibrated).
- **Scales do not transfer across GPUs.**

  | applied to | zero-shot | with A100's s | with A6000's s |
  |---|---|---|---|
  | A6000 | 8.2% | 7.3% | — |
  | A100 | 7.2% | — | 23.3% |
  | B200 | 4.2% | 4.2% | — |

  So the transferable at-cap model is the parameter-free one: the microbench at-cap roofline
  plus the overhead term.
- **For B200 the default is the stretch rule**: serving coefficients, with time stretched at the
  cap. On B200's 10 capped runs it gives 4.8% J/token MAPE, against 8.9% for the microbench
  roof. B200 decode sits at the edge of the cap rather than deep in it.
- The A100 longctx bracket in FINDINGS_A100 (full-clock +4 to +26%, throttled stream −8 to
  −17%) is narrowed to **−5% J / +7% tok/s zero-shot**.

**At the cap, LIMINAL's TDP power assumption becomes correct.** LIMINAL-TDP gets J/token to
9.5% on A6000 and 10.4% on A100, against 87.5% on the uncapped H200. Its remaining error is time.

## 5. Provisioning

Inputs: model, context (P, G, f), load (concurrency C or rate λ), and a TPOT SLO.
- LIMINAL-E enumerates GPU × TP ∈ {1, 2, 4, 8} × replicas R.
- Every provisioned GPU is charged its idle power.
- Constraints: KV must fit in memory, `b ≤ 512`, and `t(b) ≤ SLO`.
- Code: `provision_conc` and `provision_rate`.

Settings for fig (c) and `ranking_table.csv`:
- TPOT ≤ 50 ms, eager time model, a common host (A100 overhead scale reset to 1).
- A6000 and L40S use the A100 PCIe sync as a prior.
- **H100 and L40S are priors** (memory-technology e_w, unmeasured idle). Every number is
  **[P] predicted**.

Per-token ranking among the measured GPUs. Each cell lists the winner, then each other GPU's
J/token as a multiple of the winner's.

| short ctx (P=G=256) | c=1 | c=16 | c=64 | c=256 |
|---|---|---|---|---|
| 0.5B | H200 < A100 1.02 < B200 1.43 < A6000 1.62 | H200 < A100 1.04 < B200 1.38 < A6000 1.72 | H200 < A100 1.11 < B200 1.27 < A6000 1.96 | H200 < B200 1.06 < A100 1.23 < A6000 2.02 |
| 3B | H200 < A100 1.28 < B200 1.47 < A6000 2.36 | H200 < A100 1.20 < B200 1.35 < A6000 2.48 | A100 < H200 1.02 < B200 1.17 < A6000 2.01 | B200 < A100 1.09 < H200 1.09 < A6000 1.94 |
| 7B | H200 < B200 1.42 < A100 1.67 < A6000 3.07 | H200 < B200 1.28 < A100 1.42 < A6000 2.97 | H200 < B200 1.06 < A100 1.16 < A6000 2.17 | B200 < H200 1.21 < A100 1.26 < A6000 2.64 |
| 14B | H200 < B200 1.39 < A100 1.77 < A6000 3.29 | H200 < B200 1.24 < A100 1.47 < A6000 3.76 | H200 < B200 1.04 < A100 1.18 < A6000 5.31 | B200 < H200 1.19 < A100 1.74 < A6000 7.70 |
| 32B | H200 < B200 1.30 < A100 1.83 < A6000 3.40 | H200 < B200 1.18 < A100 1.59 < A6000 3.97 | B200 < H200 1.01 < A100 2.49 < A6000 7.80 | B200 < H200 1.28 < A100 4.15 < A6000 13.2 |
| 72B | H200 < B200 1.19 < A100 1.77 < A6000 3.34 | H200 < B200 1.09 < A100 3.34 < A6000 7.17 | B200 < H200 1.07 < A100 5.96 < A6000 16.7 | B200 < H200 1.25 < A100 9.19 < A6000 25.7 |

| long ctx (P=8192, G=1024) | c=1 | c=16 | c=64 | c=256 |
|---|---|---|---|---|
| 0.5B | H200 < A100 1.04 < B200 1.40 < A6000 1.72 | H200 < B200 1.17 < A100 1.25 < A6000 2.51 | H200 < B200 1.01 < A100 1.39 < A6000 3.05 | B200 < H200 1.07 < A100 1.63 < A6000 3.77 |
| 7B | H200 < B200 1.36 < A100 1.57 < A6000 3.35 | H200 < B200 1.02 < A100 1.10 < A6000 3.60 | B200 < H200 1.12 < A100 1.30 < A6000 4.58 | B200 < H200 1.24 < A100 1.41 < A6000 4.94 |
| 32B | H200 < B200 1.25 < A100 1.70 < A6000 3.57 | B200 < H200 1.07 < A100 2.76 < A6000 8.86 | B200 < H200 1.21 < A100 3.56 < A6000 10.9 | B200 < H200 1.32 < A100 3.93 < A6000 12.3 |
| 72B | H200 < B200 1.15 < A100 1.73 < A6000 3.70 | B200 < H200 1.06 < A100 5.51 < A6000 14.8 | B200 < H200 1.22 < A100 8.09 < A6000 21.9 | B200 < H200 1.31 |

**Conclusions** (all [P], with the ratios checked against measurement):

1. **H200 is cheapest per token up to about c=16–64.** **B200 is cheapest from c ≥ 64–256.**
   - The crossover comes earlier for bigger models and for long context.
   - At c ≥ 256, B200 beats H200 by 1.2–1.3×.
2. **The A100 is a 1.4–1.8× H200 part for ≥7B models**, and does worse once it needs TP over
   PCIe (32B and up at high load: 2.5–9×).
   - It is competitive (≤1.1× the best) only for ≤3B models on a common host.
   - On the measured A100 node, host overhead makes small models 1.5–1.75× H200 (FINDINGS_A100 ladder).
3. **The A6000 is never cheapest.** It costs 1.6–3.4× the best GPU at short context and c ≤ 16 (measured
   A6000/H200 for Qwen2-7B alpaca: 3.3–3.6), and much more once its 48 GB forces TP over PCIe.
4. **Model vs measurement for Qwen2-7B alpaca** (FINDINGS_A100, FINDINGS_A6000):

   | ratio | model | measured |
   |---|---|---|
   | A100/H200, c ≤ 16 | 1.42–1.67 | ~1.5 |
   | A100/B200, c = 1 | 1.18 | 1.14 |
   | A100/B200, c = 16 | 1.11 | 1.13 |
   | A6000/H200 | 3.0–3.3 | 3.3–3.6 |

5. **With priors, H100 is predicted to win many mid-load cells**: H200's bandwidth at lower
   idle, same die. The L40S wins tiny models at low load. Both are unmeasured.

## 6. Validation (`validate_v2.py` → `validation_v2.json`, `validation_v2_runs.csv`)

**Zero-shot** = the microbench zero-shot rule (§3), datasheet values, idle states, the at-cap
roofline, and the H200/B200 time-model prior. **No A100 or A6000 serving data is used.**

J/token and tok/s MAPE, with bias in parentheses:

| set (n) | J zero-shot | tok/s zero-shot | J calibrated | tok/s calibrated | J, LIMINAL-TDP |
|---|---|---|---|---|---|
| **A6000, all (25; all at the cap)** | **8.2% (−7.6)** | **10.4% (+9.6)** | 7.5% (−1.4) | 8.4% | 9.5% |
| A6000, c ≤ 64 short ctx (19) | 6.4% | 7.6% | 5.2% | 5.3% | 7.3% |
| A6000 long ctx (3) / c = 128–256 (3) | 0.7% / 27% | 0.9% / 38% | 6.3% / 23% | | |
| **A100, measured-capped dense TP1 (82)** | **7.0% (−1.9)** | **7.5% (+4.0)** | 7.8% | 7.9% | |
| A100, uncapped ≤3B (27) | 27% (−25) | 75% (+75): host overhead | 4.3% | 5.7% | |
| A100, all dense TP1 (109) | 12.0% | 24.2% | 7.0% | 7.4% | 10.4% |
| A100 by arm: Sept eager 40 / graphs ≤c64 14 / long ctx 8 / prefill 4 / c=128–256 4 | 6.2 / 6.3 / 10.2 / 3.5 / 18.2% | | 6.7 / 8.0 / 9.5 / 3.5 / 20.1% | | |
| A100 TP2 PCIe (6) | 15.9% (no PCIe term) | 36% | 9.1% | 13.1% | |
| A100 MoE Qwen3-30B-A3B (13) | 34% (+27) | 38% | 40% | 36% | |
| **B200, microbench zero-shot (77 dense TP1)** | **5.3% (−1.2)** | 9.4% | 5.6% (−4.2): serving fit | 8.5% | 28% |
| B200 long ctx (13) / capped (8) | 5.6 / 11.0% | | 6.4 / 4.2% | | |
| H200 Sept (80), KV prior + e_g refit | — | — | 2.7% (−1.3) | 4.7% | 87.5% |

**Cap detection, zero-shot:** A100 95 of 115 runs right, A6000 18 of 25. The A6000 misses are
low-batch runs that the full-clock model places just under the cap. J/token there is still
within 8%.

**What the calibration fixes:** only host overhead (A100 small models).
- It fits three things: the A100 eager overhead scale, the at-cap scales, and the A100 PCIe TP sync.
- It does not improve the capped regime.

**Misses:**
- **c = 128–256 on A100 and A6000:** throughput over-predicted by 24–38%. The same pattern
  appears on B200 at c=512. The time model is extrapolated past ~110 tokens per step, and
  chunked-prefill steps that vLLM hides are not modelled.
  **Diagnosed in `TIMEMODEL_HIGHC.md` (2026-10-23):** on A100/A6000 the cause is FlashAttention-2
  losing GQA packing on steps that mix prefill and decode; on B200 it is a ~4.8 µs per-sequence
  host cost plus the 512-token CUDA-graph capture limit. Opt-in fix: `highc=True`.
- **MoE on A100:** J/token +27%. The fused-MoE layer floor is an in-sample H200 prior, and
  graphs mode makes MoE 4.5× faster at c=1. This is not modelled.
- **B200 long-context J/token −6% with the settled KV.** The Sept per-bin coefficients get
  −2%, by error cancellation (they over-count KV). This should be checked with the
  prefill-spread binning fix (UPDATE 6 review "next fix").

## 7. KV coefficient: status

- **B200: settled [M].** e_kv = 1.53–1.61e-10 J/B = 1.18–1.29× e_w (UPDATE 6 review: clean
  decode bins, slope R² ≥ 0.998). It is now the canonical B200 value.
- **H200: open [O].** The prior is e_kv = 1.25·e_w; with it, e_g refits to 1.20 pJ and Sept
  J/token MAPE improves from 3.8% (per-bin 3.29e-10) to 2.7%.
  - The v1 shared-prefix long-context sweep (`kv_longctx.json`) gave ~1.1e-10 per logical
    byte, but it is L2-confounded.
  - A unique-prompt H200 sweep would settle it.
- **A100: not identifiable at the cap.** Its at-cap KV/weight ratio of 0.79 is time, not
  energy (FINDINGS_A100). A6000: unmeasured; the prior is 1.25·e_w.

## 8. What LIMINAL-E adds over LIMINAL

| | LIMINAL v2 | LIMINAL-E |
|---|---|---|
| bandwidth utilization | forced to 1.0 | fitted η |
| overhead | 114 µs/layer, H100 | t0 + τ per GPU. CUDA graphs = −34.5 µs/layer. Node host overhead (A100 ×1.92) separated from hardware |
| sync | modelled | measured: 15.5 µs/layer on NVLink; 185 µs/layer + 5 GB/s on PCIe |
| power | TDP | P_s + P_0 + e_w·W + e_kv·K + e_g·F, measured on 4 GPUs. Zero-shot from microbenchmarks to 5–8% J/token at the cap and on B200 |
| power limit | none | full-clock power roof **and a measured at-cap roofline**: A100 68%/62%, A6000 86%/69%, B200 74%/63% of BW/peak |
| balance | qualitative | energy ridge against performance ridge; static-equivalent bandwidth; the activity intercept (GDDR6) |
| provisioning | STPS/W at TDP | GPU × TP × replicas at a TPOT SLO, with idle charged and NVLink vs PCIe TP |

## 9. Limitations

- **The zero-shot uncapped time uses the H200/B200 time-model prior**, not A100/A6000 data. Its
  errors are host overhead and appear only in the uncapped small-model regime.
- **At-cap scales are GPU-specific.** The parameter-free form is the one to use on a new GPU.
  The A6000 calibration rests on 25 runs and one model family.
- **PCIe TP sync was fitted on 3 runs** (2 parameters). TP4 and TP8 over PCIe, and the A6000 and
  L40S use of it, are extrapolations.
- **Priors:** H100 and L40S (all coefficients, idle), the A6000 KV value, and the H200 KV value.
  Host and node power are excluded.
- **MoE and very high concurrency (c ≥ 128) are not modelled well** (§6).

## Figures (`figs/`)
- `fig_a_power_roofline.png`: four panels. Bandwidth and compute roofs, the full-clock power
  roof, the at-cap roofline, capped and uncapped microbench kernels, and measured serving points.
- `fig_b_energy_roofline.png`: J/token against batch by component, for the 4 GPUs at short and
  long context, with measured points.
- `fig_c_provisioning_map.png`: the winner, J/token, and GPUs × TP. Measured GPUs only, and with
  the H100/L40S priors (hatched).
- `fig_d_validation_parity.png`: zero-shot and calibrated predictions against measured J/token.
