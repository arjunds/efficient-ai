# High-concurrency time model, and prefill spreading in the binning (2026-10-23)

Scripts: `timemodel_highc.py` → `timemodel_highc.json`; `prefill_spread_compare.py` →
`prefill_spread_compare.json`. Both are CPU-only and run in the container. No GPU jobs were run.

**Labels.** [M] measured, [F] fitted, [P] prior, [I] inferred.

## Bottom line

1. **The per-sequence host-cost hypothesis is right for B200 and wrong for A100/A6000.**
   - **B200 (FlashInfer backend).** Pure-decode steps carry **4.8 µs per running sequence** that the
     model lacks [F]. It does not depend on context length.
   - **B200, c=512 only.** **85% of steps exceed vLLM's 512-token CUDA-graph capture size** and run
     without graphs: +1.5–1.6 ms per step, about 58 µs per layer [M].
   - **A100/A6000.** Pure-decode steps show **no** per-sequence excess. The whole miss sits on steps
     that **mix prefill and decode**: +7 to +53 ms per mixed step, growing with resident KV.
   - **Cause on A100/A6000 [I].** FlashAttention-2 (vLLM's backend on SM8x) packs a KV head's query
     heads into one tile only on pure-decode steps. On mixed steps each query head re-streams its KV
     head, which adds (g−1)·K_decode bytes (g = query heads per KV head, 7 for Qwen2-7B).
   - That byte term has **no fitted constant**, and it predicts the measured penalties from the
     at-cap streaming rate.
   - **Confirmed 2026-10-28 [M]** (torch profiler, A5000, FA2; TRACES.md §Method 2):
     - Mixed-step decode attention launches with grid `[1, b, 28]`: 28 query heads, so no GQA packing, and one split.
     - Pure decode launches with `[1, 32, 4]`: 4 KV heads, 32 splits.
     - Attention per step is 79 vs 7 ms. Besides the extra bytes, mixed steps also lose split-KV parallelism.
2. **Throughput fix (zero-shot).** A100 and A6000 use microbench-only coefficients plus the
   B200-fitted host terms plus the parameter-free FA2 term:
   - A100 c=128–256: tok/s MAPE **23.9% → 4.9%**, J/token **18.2% → 4.2%**.
   - A6000 c=128–256: tok/s **38.1% → 10.5%**, J/token **27.0% → 8.7%**.
   - B200 7B c=128–512 (the fit set): tok/s **21.7% → 2.1%**.
3. **Warning: on B200 7B the "power within 2%" was error cancellation.** Once the time is right:
   - J/token is still **−10%** and power is −10% (base model: −15% and 4%).
   - With the measured step times, the energy model under-predicts power by 7–13% at c ≥ 128.
   - Swapping e_g 0.70 → 0.943 pJ (the steady microbench value) closes this (J/token 10.3% → 2.1%)
     but **breaks the capped runs** (32B high-c, prefill arm).
   - The canonical coefficients are unchanged. This is open issue 1.
4. **Prefill spreading clearly helps long context.**
   - Long-context 7B bins give **e_g 0.42 → 0.95 pJ** (steady value: 0.943).
   - Fit R² goes from −4.9 to +0.14. The mean |residual| of prefill bins drops from 86 J to 26 J.
   - Long-context 32B bins: e_g 0.08 → 0.76 pJ.
   - ShareGPT and alpaca barely move (few prompts exceed the chunk budget).
   - Run totals are unchanged (largest relative change 1e-16). Old runs bin bit-identically.
5. **Every new behaviour is opt-in or automatic only when new data exists.**
   - LIMINAL-E with the default `highc=None`: identical on all 574 run×profile predictions.
   - `recommend_gpu.py` CLI and `--winmap` output: byte-identical.

## 1. Per-step evidence

**Method.** Each step (iter_log row) is predicted by LIMINAL-E at its own n_running, scheduled
tokens and resident KV.
- A request's prefill is mapped to the row whose t_end is the last before its t_first_token. That
  reproduces the logged `prefill_tokens` on more than 99% of rows.
- Steps inside the window of a prompt over 512 tokens are dropped (hidden chunks).

The table gives median residuals (measured − model, ms) for three step types:
- **pure:** no admission, ≤ 512 tokens;
- **mixed:** at least one request admitted (prefill in the step);
- **eager:** more than 512 scheduled tokens.

"Mixed penalty" is mixed minus pure at the same n. "FA2 pred" is (g−1)·K_decode divided by the
streaming rate (the at-cap 1/τ_b on A100/A6000), with no fitted constant.

| run | pure | mixed | eager (>512) | mixed penalty | FA2 pred |
|---|---|---|---|---|---|
| B200 7B alpaca c128 / c256 / c512 | +0.04 / +0.68 / +1.08 | +0.44 / +1.09 / +2.52 | – / – / **+4.08 (85% of steps)** | 0.37 / 0.25 / 0.15 | (FlashInfer: none) |
| B200 7B sharegpt c128 / c256 / c512 | +0.47 / +0.87 / +1.70 | +0.92 / +1.36 / +2.02 | +2.2 / +3.3 / +3.8 | 0.42 / 0.33 / −0.18 | (none) |
| A100 7B alpaca c64 / c128 / c256 | −1.15 / −0.23 / +0.43 | +1.12 / +6.95 / +11.37 | – | **2.3 / 7.1 / 10.4** | 2.3 / 4.6 / 9.2 |
| A100 7B sharegpt c64 / c128 / c256 | +0.33 / +1.37 / +1.99 | +4.2 / +11.2 / +21.2 | | **4.0 / 9.7 / 18.8** | 5.5 / 10.3 / 20.0 |
| A100 14B alpaca c64 | −4.78 | −0.09 | | 4.7 | 5.0 |
| A6000 7B alpaca c64 / c128 / c256 | −4.2 / −4.8 / −2.5 | −0.5 / +5.0 / +25.8 | | **3.7 / 9.8 / 27.4** | 4.5 / 9.0 / 16.9 |
| A6000 7B sharegpt c64 / c128 / c256 | −2.9 / −3.9 / −1.2 | +5.6 / +18.3 / +52.0 | | **8.5 / 22.4 / 52.9** | 9.1 / 20.7 / 44.1 |
| A6000 14B alpaca c16 / c64 (50 / 102 mixed steps) | | | | 10.6 / **−1.5** | 2.6 / 8.2 |

The A6000 14B rows do not follow the FA2 term (few mixed steps, KV-limited at 78k tokens). They are
listed as a counter-example.

Example of the effect: on A100 c=256, pure-decode steps take 27.0 ms (model: 26.5 ms). Any step
that admits a request takes about 40 ms, even when it adds only 2 computed prompt tokens.

### Hypothesis tests (B200 7B, 77,620 pure-decode steps, LAD regression)

| hypothesis | test | result |
|---|---|---|
| **per-sequence host cost** | pure residual against n and resident KV | **+4.18 µs per sequence**. The KV coefficient is −1.6 µs per 1k tokens, i.e. none. ShareGPT has 2× the context of alpaca and the same slope: 4.07 vs 3.34 µs per sequence. → **supported** |
| attention / KV growth | KV coefficient above; ShareGPT vs alpaca slopes | **rejected on B200**. On A100/A6000 an attention effect exists, but only on mixed steps (FA2, above) |
| GEMM compute growth with M | extra cost per computed prefill token beyond the model's GEMM term (model charges 9.5 µs/token) | **−0.8 µs**. Also, B200 llmgemm microbenchmarks at M=512 (capped, then clock-scaled) imply ≤ 4.4–7.2 ms of GEMM per step, against the model's additive 6.9 ms. → **rejected** |
| sampling over the 152k vocabulary | decoding is greedy (temperature 0), so the sampler is a cast to fp32 plus an argmax of the logits, about 1.5 MB per sequence | **≤ 0.4 µs per sequence** at half of peak bandwidth. → **rejected** as the main term |
| hidden chunked prefill | 0% of alpaca high-c steps and 6–27% of ShareGPT high-c steps sit in the window of a prompt over 512 tokens | excluded from the fit; alpaca has the same deficit → **not the cause of the time miss**. It does matter for energy binning (§5) |
| **CUDA-graph capture limit** | run.log: `max_capture_size: 512`, `cudagraph_mode: PIECEWISE` | residual jumps **+1.54 ms (alpaca) and +1.64 ms (ShareGPT)** between T ∈ (504, 512] and T ∈ (512, 528], i.e. 55–59 µs per layer → **supported** (c=512 only) |
| **FA2 mixed-batch attention** | mixed-step penalty against (g−1)·K_decode at the at-cap stream rate | matches A100 within about ±30%. A6000 7B is 0.8–1.6× the prediction. B200 (FlashInfer) shows a flat 0.15–0.4 ms with no KV scaling → **supported** for SM8x |

## 2. Fitted terms (B200 Qwen2-7B c=128–512 only: 6 runs, 26,566 steps)

Regression: residual = c + τ_seq·n + t_mix·[admits a request] + Δτ_eager·L·[T > 512], fitted by LAD
(robust to logger jitter).

| term | value [F] | leave-one-run-out range |
|---|---|---|
| **τ_seq** (host time per running sequence per step) | **4.80 µs** | 3.79–5.47 |
| t_mix (a step that admits ≥ 1 request; steady state f_mix = 1 − e^(−n/G)) | 0.280 ms | 0.273–0.344 |
| Δτ_eager (a step over capture_max = 512 tokens) | 58.5 µs/layer | 52.8–64.4 (the discontinuity alone gives 55–59) |
| intercept c (**not applied**; it is the base model's level offset) | −0.34 ms | |
| τ_seq as the only term (the original hypothesis) | 8.15 µs | |

- **FA2 term:** extra KV bytes = f_mix·(g−1)·K_decode. Its uncapped time uses η (sequential
  streaming); at the cap the bytes go into the at-cap roofline; the energy is charged at e_kv.
- **Steady-state eager fraction:** P(n + K·P_c > 512), with the number of admissions K ~ Poisson(n/G).

**Step level** (mean step at the logged operating points, base → full model; A100/A6000 use the
zero-shot coefficients):

| set | base | full |
|---|---|---|
| B200 7B c=128–512 | −4% to −27% | **−0.3% to +7%** |
| A100 7B c=128–256 | −10% to −19% | **−3.5% to +0.6%** |
| A6000 7B c=128–256 | −0.3% to −24% | −11% to +9% |
| B200 32B c=128–512 | +2% to +8% | +8% to +12% |

## 3. Run-level accuracy before and after

The runs and measured values are the same as in `validate_v2.py`. Each cell is MAPE % (bias %).

- **τ-only** = τ_seq at 8.15 µs only.
- **full** = τ_seq + t_mix + capture limit + FA2.
- Full numbers, including the B200-only and FA2-only variants, are in `timemodel_highc.json`.

**Canonical (calibrated) coefficients:**

| group (n) | tok/s base | tok/s τ-only | **tok/s full** | J base | J τ-only | **J full** |
|---|---|---|---|---|---|---|
| **B200 7B c128–512, the fit set (6)** | 21.7 (+21.7) | 4.2 (−2.5) | **2.1 (+0.6)** | 15.0 (−15.0) | 9.3 | **10.3 (−10.3)** |
| **B200 32B c128–512, held out (3)** | 2.4 (−0.2) | 5.1 (−4.0) | **4.3 (−4.3)** | 2.4 (+2.4) | 3.4 | **3.4 (+3.4)** |
| B200 graphs c ≤ 64 (11) | 3.3 | 2.5 | 2.4 | 3.2 | 2.4 | 2.6 |
| B200 long context (13) | 8.0 | 8.3 | 8.3 | 6.4 | 6.4 | 6.4 |
| B200 prefill arm (4) | 10.6 | 10.5 | 9.4 | 6.2 | 6.2 | 6.0 |
| B200 TP2 + TP1 reference (8) | 3.1 | 1.9 | 2.2 | 2.9 | 2.6 | 2.6 |
| B200 Sept eager (28) | 4.1 | 3.7 | 3.5 | 3.3 | 2.9 | 3.0 |
| H200 Sept eager (80; a different host node) | 4.7 | 4.0 | 3.9 | 2.7 | 2.3 | 2.4 |
| **A100 c128–256 (4)** | 26.7 (+26.7) | 26.7 | **7.5 (+6.6)** | 20.1 (−20.1) | 20.1 | **6.5 (−4.8)** |
| A100 graphs c ≤ 64 (11) / other dense TP1 (94) | 8.1 / 6.4 | 8.1 / 6.5 | 7.7 / 5.7 | 8.9 / 6.2 | 8.9 / 6.1 | 9.0 / 5.9 |
| A100 TP2 (6) / MoE (14) | 13.1 / 35.5 | 13.0 / 35.5 | 12.9 / 36.1 | 9.1 / 39.5 | 9.0 / 39.5 | 9.2 / 40.5 |
| **A6000 c128–256 (3)** | 30.8 (+30.8) | 26.6 | **6.4 (+4.0)** | 22.8 (−22.8) | 20.3 | **5.8 (−2.8)** |
| A6000 c ≤ 64 (19) / long context (3) | 5.3 / 5.5 | 5.4 / 5.6 | 4.8 / 5.5 | 5.2 / 6.3 | 5.4 / 6.3 | 5.0 / 6.3 |

**Zero-shot A100/A6000** (microbench-only coefficients, no A100/A6000 serving data, and the host
terms from B200):

| group (n) | tok/s base | tok/s τ-only | **tok/s full** | J base | J τ-only | **J full** |
|---|---|---|---|---|---|---|
| **A100 c128–256 (4)** | 23.9 (+23.9) | 17.3 (+17.3) | **4.9 (+3.6)** | 18.2 (−18.2) | 13.7 | **4.2 (−2.2)** |
| A100 graphs c ≤ 64 (11) | 5.8 | 6.2 | 6.3 | 6.5 | 7.1 | 7.2 |
| A100 other dense TP1 c ≤ 64 (94) | 26.3 | 24.9 | 24.3 | 12.4 | 11.9 | 11.4 |
| **A6000 c128–256 (3)** | 38.1 (+38.1) | 33.5 (+33.5) | **10.5 (+9.2)** | 27.0 (−27.0) | 24.5 | **8.7 (−7.0)** |
| A6000 c ≤ 64 (19) | 7.6 | 6.9 | 5.4 | 6.4 | 5.8 | 4.6 |

**Which term does the work.**
- On A100/A6000, almost all of the gain comes from the FA2 term. FA2 alone gives A100 7.5 / 5.7 and
  A6000 11.6 / 9.2 (zero-shot tok/s / J).
- The B200 host terms alone give 18.8 / 14.7 (A100) and 34.5 / 25.1 (A6000).

**Regressions at c ≤ 64:**
- A100 graphs, zero-shot: tok/s +0.5 points, J +0.7 points.
- B200 long context: tok/s +0.3 points.
- A100 MoE: J +1 point (FA2 g=8).

**Remaining misses:**
- A6000 alpaca c256: still +25% tok/s at run level (−11% at step level).
- B200 7B high-c J/token (open issue 1).

**Power MAPE.** Power error at high c stays ≤ 1.1% on A100/A6000; they sit at the cap. On B200 7B
high-c it goes 4.0 → 9.8% (open issue 1), and to 2.7% with e_g = 0.943.

## 4. Code changes

### `liminal_energy/liminal_e.py` (opt-in; `highc=None` leaves outputs unchanged, verified on 574 predictions)

New pieces:
- `HIGHC` holds the fitted constants above, `capture_max=512`, and the attention backend per GPU:
  - `fa2`: A100, A6000, A5000, L40S;
  - `flashinfer`: B200;
  - `fa3`: H100, H200 (no penalty [P]).
- `highc_params()`, `steady_mix()` (f_mix, f_eager), `highc_extra()`.

Changes to existing functions:
- `step_time(..., highc=)` adds:
  - host time τ_seq·n + t_mix·f_mix;
  - f_eager·L·Δτ_eager in graphs mode;
  - FA2 extra-KV streaming time.
- **How the host and eager terms are treated.** They enter the uncapped time and the
  "overhead" part exactly as t0 and τ already do: `roof_ovh` keeps them serial at the cap, and
  `stretch` does not. When the GPU is uncapped, it idles at P_s + P_0 during them.
- `step_work(..., highc=)` passes the FA2 extra KV bytes into energy (at e_kv) and into the at-cap
  roofline (`at_cap_time(..., KV_extra)`).
- `serving(..., highc=True)` fills n_seq, f_mix and f_eager from P, G and the cached fraction.
- `work()` adds `n_seq`, `KV_dec` and `f_mix` keys (unused unless `highc` is set).

Usage: `le.serving(g, m, P, G, conc=c, highc=True)`, or pass a dict to override terms per GPU
(`{"tau_seq": {"A100-80-PCIe": x}}`).

### `recommend_gpu.py`

- `iter_time(..., tau_seq=0.0)` adds `tau_seq * work["n_seq"]` after the max. The default is 0, and
  the CLI and `--winmap` output are byte-identical.
- To adopt it: set `w["n_seq"] = n` in `predict_serving` and pass
  `tau_seq=liminal_e.HIGHC["tau_seq"]["_default"]`.
- It is not wired in, because the FA2 and capture-limit terms (the ones that matter on Ampere) live
  only in `liminal_e`.

### `liminal_energy/LIMINAL_E.md`

One pointer under §6 "Misses".

## 5. Task 2: prefill spreading in `energy_model.py`

**What it does.**
- `bin_run(..., spread_prefill=None, spread_mode="budget")` calls `spread_prefill_rows()` before
  binning. It moves each request's prompt tokens (raw, hits and computed) from the step that
  reported them back over the steps of its [t_submit, t_first_token] window.
- GEMM FLOPs, v2 attention FLOPs and MoE weight bytes follow, because they are computed per row
  from the moved tokens.

**When it runs.**
- **Auto-on** only when `requests.csv` has `t_submit`, `t_first_token` and `prompt_tokens` *and* all
  phases are binned (`phase=None`).
- It is always off for runs without that file (all pre-October runs), for phase-filtered binning
  (the logged phase tags would no longer match, so `validate_waveform.py` is unaffected), with
  `--no_spread_prefill`, or with `ENERGY_SPREAD_PREFILL=0`.
- `--spread_mode budget|duration`, `--binned_out NAME`.
- Spread tables write `prefill_tokens` with one decimal and add a `prefill_spread` column.

**Why budget mode is the default.**
- vLLM's chunked prefill gives each step min(remaining, budget − decode tokens), first come first
  served. The budget is `max_num_batched_tokens_resolved` from `run_meta`: 2048, or 8192 on the
  prefill arm.
- Budget mode reconstructs that schedule: it fills backward from the reporting step, places
  requests in finish order, lets short prompts claim the reporting step first, and puts whatever
  does not fit on the reporting step (≤ 1% of tokens).
- Duration-proportional spreading also gives tokens to queued time and to decode-only steps in the
  window.
- Hits are logged on the same row as the prompt (checked on 4 runs: never on a row with
  `prefill_tokens = 0`).
  Each request's computed tokens are therefore P × (1 − hits/pref) of its reporting row, and raw
  tokens and hits move with the same shares. Per-row computed tokens therefore stay ≥ 0, and the
  per-bin `max(0, ·)` clip never triggers.

**Verification:**
- **(a) Bit-identity.** 15 runs without `requests.csv` (Sept B200, ragged H200, ragged MoE, H200
  long context, B200 32B) plus 2 B200_prov runs with spreading forced off.
  - Bins, the exported table and decode-phase binning are all identical to the pre-change
    `energy_model.py`.
  - Both B200_prov tables also match the committed `binned_table.csv` byte for byte.
- **(b) Totals.** Over 37 runs, prompt tokens, computed tokens and computed GEMM FLOPs change by at
  most 1.1e-16 (relative).
- **(c) Per-bin fits.** Energy counter, Y = E − P_idle·dt; "fixed" means e_w and e_kv held at the
  settled 1.310e-10 / 1.548e-10.

| group (runs, bins) | prefill bins (off → budget) | free e_g (pJ): off / budget / duration | free R²: off / budget / duration | e_g, fixed W/K: off → budget | mean \|res\| of prefill bins at steady coefficients |
|---|---|---|---|---|---|
| **long context 7B (10, 3093)** | 660 → 1240 | **0.424 / 0.949 / 0.882** | **−4.93 / 0.14 / −0.47** | 0.434 → 0.947 | **86.0 → 26.1 J** (duration 33.0) |
| **long context 32B (3, 884)** | 47 → 214 | 0.080 / 0.761 / 0.679 | −20.6 / −2.20 / −4.08 | 0.081 → 0.758 | 524 → 28.5 J |
| prefill arm 7B (4, 906; budget 8192 ≥ prompt) | 874 → 904 | 0.646 / 0.646 / 0.631 | −8.19 / −8.19 / −5.44 | 0.677 → 0.677 | 50.6 → 53.0 J |
| ShareGPT graphs c1–64 (4, 919) | 584 → 584 | 0.727 / 0.739 / 0.736 | 0.267 / 0.269 / 0.266 | 0.773 → 0.777 | 9.1 → 9.0 J |
| ShareGPT high-c c128–512 (3, 706) | 675 → 675 | 0.592 / 0.607 / 0.547 | 0.070 / 0.121 / 0.014 | 0.813 → 0.816 | 11.7 → 11.2 J |
| alpaca control (13) | unchanged | 0.754 (all three) | 0.869 | 0.870 | unchanged |

**Reading the table:**
- Pooled long-context R² is low because prefill bins sit at the power cap and the fit has no
  intercept. The comparison is like for like.
- The prefill arm does not move: its prompts fit in one 8192-token step.

**Where the files went.**
- `binned_table_spread.csv` (budget mode) is written next to `binned_table.csv` in all 37 compared
  B200_prov run directories. **No `binned_table.csv` was overwritten.**
- For long-context work the spread table is clearly better and should replace the original.
- Because spreading is now the default, re-running `energy_model.py --two_term` on any run with
  `requests.csv` will regenerate a spread `binned_table.csv`.

## Open issues

1. **B200 e_g depends on the regime.**
   - Several uncapped data sets now point to about 0.94 pJ:
     - 7B high-c serving power;
     - spread long-context bins: 0.949;
     - the steady microbench: 0.943.
   - Every uncapped B200 group improves with it (graphs J 3.2 → 1.5%, long context 6.4 → 4.6%,
     TP 2.9 → 1.5%).
   - Capped groups break with it under every cap rule tried:

     | cap rule | 32B high-c J | prefill arm J |
     |---|---|---|
     | stretch | +22% | +24% |
     | dvfs (0.62 pJ / 1.122e-10 J/B at the cap) | +8% | −10% |
     | roof_ovh | +10% | −10% |

   - The canonical 0.70 pJ works as a blend that suits the stretch rule. It needs a regime-aware
     e_g, which is not done here.
2. **Held-out B200 32B high-c tok/s is slightly worse:** 2.4 → 4.3% (bias −4.3). At the cap, host
   time may overlap with power-limited GPU work. It stays inside 5%.
3. **Host terms belong to the node.**
   - A100, A6000 and H200 use the B200 node's τ_seq, t_mix and Δτ_eager as a prior.
   - The A100 node's eager overhead scale of 1.92 suggests a slower host, but that cannot be
     identified at the cap.
   - Δτ_eager (58.5 µs/layer at batch ~500) is larger than the c ≤ 64 graphs–eager gap of 34.5.
4. **The FA2 mechanism is inferred [I], not traced.** The evidence is the mixed-step penalty, its
   KV scaling, its absence on FlashInfer, and the parameter-free fit.
   - It also fails on A6000 14B (few mixed steps).
   - One nsys or NCU capture of a mixed step on A100 would settle it.
   - **Practical implication [I]:** about 15–25% of step time at c=128–256 on A100/A6000 goes to
     this. A backend that keeps a separate decode kernel (FlashInfer on Ampere) could recover most
     of it.
5. **Version-specific.**
   - The capture limit (`max_capture_size = min(2·max_num_seqs, 512)`) and the choice of attention
     backend are vLLM 0.10.2 defaults.
   - MoE (A100 J +35%) is not addressed.
