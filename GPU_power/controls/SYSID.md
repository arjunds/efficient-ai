# SYSID — state dynamics of a GPU serving an LLM (the plant `f(x,u,d)`)

The energy/roofline work gave us the *output map* `h(x,u)` (power, energy per
iteration). A controller also needs the *state dynamics* — the queue, the
running batch and the KV accumulators, and how fast they move. This note
covers what was identified from the logs we already have, how it was
validated, and where it breaks. Code: `controls/sysid.py` (fitting and
validation, re-runnable), `controls/plant.py` (the plant), and
`controls/plant_params.json` (fitted parameters, written by sysid).

Reproduce (in the container, from `GPU_power/`):
```
python3 controls/sysid.py tokenize   # one-off: prompt-pool token lengths -> prompt_token_lens.json
python3 controls/sysid.py fit        # iteration-time model -> plant_params.json   (~15 min, 8 CPU)
python3 controls/sysid.py validate   # k-step validation    -> sysid_results.json  (~5 min)
python3 controls/sysid.py smoke      # A5000 requests.csv latency check
python3 controls/sysid.py report     # re-print all tables from the JSONs
```

## 1. Data and three findings about it

Runs used: `logs/ragged` (H200, 4 dense 7-8B models x {alpaca, sharegpt} x c{1,4,16,64}
+ Poisson 8/s), `logs/ragged_ladder` (H200 Qwen2.5 0.5-32B), `logs/ragged_moe`
(H200 Qwen3-30B-A3B), `logs/B200*` (Mistral-7B, Qwen2-7B, Qwen2.5-32B, Qwen2-72B),
plus synthetic-prompt H200 runs that were never used for fitting (`logs/load_sweep`,
`logs/poisson`, `logs/wave`), which serve as out-of-distribution tests. That is
198 runs in total. Two B200 runs crashed mid-window (the EngineCore error dump in
`run.log`) and are excluded: `B200/Qwen2-7B-Instruct/sharegpt_c16` and
`B200/Mistral-7B-v0.1/sharegpt_c4`.

**(a) `prefill_tokens` is not the prefill work.** vLLM's
`IterationStats.num_prompt_tokens` adds the **full prompt length once, at the
iteration that emits the first token**. It ignores chunking and it ignores
prefix-cache hits. Prefix caching is on (V1 default), and the generator draws
prompts *with replacement* (`random.Random(1234).choice(pool)`, pool = 2000). So
at c=64, **26-36% of logged prompt tokens were cache hits and were never
computed**. Synthetic runs send one identical prompt, so about 97% of their
prompt tokens are hits. We replay the RNG, tokenize the pools with each model's
tokenizer, and match every logged prefill event to its draw(s). The match rate
is **100% on every real-prompt run** except two old H200 Qwen2-7B runs
(`alpaca_c1`, `sharegpt_c16`, which have no run.log and came from an older
harness). Those two get cache fraction 0. The result is the *computed* prefill
tokens per row, and it matters a lot: on the 60 synthetic runs the fitted
iteration-time model is off by **3.6% (cache-aware) vs 44% (naive logged count)**,
and by up to 280% at c=64 with 2k prompts. *Implication for the energy model:
`binned_table.csv` `gemm_flops_bin` over-counts prefill FLOPs by the cached
fraction (up to ~30x on synthetic runs, up to ~35% of prompt FLOPs on c=64
real-prompt runs).*

**(b) Resident KV != KV read when prompts share a prefix.** `kv_tokens_resident`
counts unique blocks. With identical prompts, all sequences share the 512- or
2048-token prefix once, but every decode step still reads it once per sequence.
This is why the time model under-predicts synthetic c>=32 / 2k-prompt runs by
6-21%. The same bias hits the KV-energy term. It is negligible for the real
prompt pools.

**(c) Queues basically never form in these logs.** `n_waiting` is 0 in more than
99.9% of rows. Its only non-zero values are brief (<=55 requests, sub-second)
waits when sharegpt c16/c64 prefill bursts exhaust the 2048-token chunked-prefill
budget. Closed-loop c<=64 is far below `max_num_seqs` (vLLM 0.10.2 default 128;
it is not logged, but nothing is ever held back at 64), and KV capacity
(0.25-10.8 M tokens, parsed from run.log) is never approached. The queue
dynamics are implemented, but **the data only validates them in the trivial
regime**.

Other facts used: every run has `max_num_batched_tokens=2048`, chunked prefill
on, `enforce_eager=True`. The closed-loop client resubmit latency ("think time",
measured as `(c - n_run - n_wait)/completion rate`) is **Z = 14 ms (H200) and
16 ms (B200)**. The generation-length CV^2 implied by closed-loop KV
(`kv/n = p + g(1+CV^2)/2`) has a median of 0.15 (alpaca) and 0.04 (sharegpt;
most requests hit the 256 cap). The Erlang stage count is m = round(1/CV^2).

## 2. Iteration-time model (sysid)

**Target.** Stat-logger rows are async-jittery (0.07 ms to tens of ms), so no
single row is trusted. Per 1 s bin, the target is the measured busy time: the
sum of row intervals, excluding rows whose interval contains an engine-idle gap
(previous row had n_run + n_wait = 0) or a stall of 1 s or more. The prediction
is the sum over those rows of the model time for each row's work: tokens =
decode + *computed* prefill, `W(tok)` (MoE occupancy-aware), `KV_bytes =
kv_resident * kv_bytes/token`, `F = 2*P_active*tok`. The fit uses
`scipy.least_squares` on the relative residual (soft-L1), on the real-prompt
runs of each GPU x model, with all utilisations bounded to <= 1.

**Forms compared** (u = fraction of datasheet BW/peak):

| form | t_iter |
|---|---|
| roofline (LIMINAL) | `max((W+KV)/(BW u_m), F/(peak u_f))` |
| roofline+oh | `t_oh + max(...)` |
| additive | `t_oh + tau n + W/(BW u_w) + KV/(BW u_kv) + F/(peak u_f)` |
| phys | `t_oh + tau n + max(W/(BW u_w) + KV/(BW u_kv), F/(peak u_f))` |
| **phys_p (selected)** | `t_ser + tau n + pnorm_p(t_cpu, max(W/(BW u_w) + KV/(BW u_kv), F/(peak u_f)))` |

`pnorm_p(a,b) = (a^p + b^p)^(1/p)`: p=1 means CPU launch and GPU work are
serial, and p -> inf means they fully overlap. `u_w` and `p` are pinned per GPU
by a **cross-model law** fitted jointly over all dense models on that GPU:
`t = a + b*L_layers + tau*n + comb(b*L, GPU)`. Within a single model the
weight-streaming time and the fixed overhead are collinear, because W is
constant. They can only be separated across models.

**GPU-level law, held out by model (LOMO):**

| GPU | best comb | a | b /layer | tau /seq | u_w (weight MBU) | u_kv | u_f (MFU) | in-sample | **LOMO** | alternatives' LOMO |
|---|---|---|---|---|---|---|---|---|---|---|
| H200 (10 dense) | max (overlap) | 1.97 ms | 113 us | 10.5 us | **0.72** | 0.30 | 0.63 | 3.9% | **5.5%** | sum 16.2%, p-norm(4.5) 5.6% |
| B200 (4 dense) | sum (serial) | 0.96 ms | 91 us | 13.5 us | **0.86** | 0.31 | 0.46 | 2.3% | **3.8%** | max 9.1%, p 7.5% |

H200 LOMO per model: Llama-3-8B -1.3%, Mistral -0.7%, Qwen2-7B +1.0%,
Qwen2.5-{0.5,1.5,3,7,14,32}B +6.0/+1.0/-7.1/+3.1/-0.8/+10.1%, and gemma-7b
-16.4% (256k vocab / head_dim 256: large per-sequence and lm_head costs). B200
LOMO: -2.8% to +4.6%. **Interpretation.** In eager mode vLLM's per-iteration
cost is about 2-5 ms of CPU (Python + about 100 us/layer of kernel launches)
plus about 10-25 us per running sequence. On the H200 host this CPU cost
*overlaps* the GPU work. The "45% MBU" in PROPOSAL section 3a is therefore an
*effective* number. The weight-streaming MBU is 0.72 (H200) / 0.86 (B200) once
the overhead is separated out. This is also why B200 7B is only about 10% faster
than H200 7B despite 1.67x the bandwidth: both are overhead-dominated.

**Per GPU x model, 1 s-bin busy-time MAPE, in-sample / leave-one-run-out**
(`const` = constant iteration time per model, LORO):

| GPU\|model | bins | const | roofline | roofline+oh | additive | phys | **phys_p** | law LOMO | t_ser ms | t_cpu ms | tau us | u_kv | u_f | mean t_iter ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H200 Llama-3-8B | 457 | 8.2 | 2.9/3.2 | 2.9/3.2 | 1.2/1.4 | 1.1/1.2 | **1.09/1.22** | 2.1 | 2.09 | 0.23 | 14.7 | 0.56 | 0.60 | 6.95 |
| H200 Mistral-7B | 459 | 8.8 | 2.4/2.6 | 2.4/2.6 | 1.5/1.9 | 1.0/1.1 | **1.02/1.08** | 2.4 | 2.09 | 3.29 | 10.6 | 0.46 | 0.55 | 6.78 |
| H200 Qwen2-7B | 449 | 8.0 | 3.8/4.4 | 3.2/4.4 | 1.6/1.7 | 1.5/1.6 | **1.52/1.58** | 2.2 | 1.88 | 0.33 | 16.1 | 0.47 | 0.62 | 6.44 |
| H200 gemma-7b | 465 | 11.8 | 1.8/2.0 | 1.8/2.0 | 1.2/1.4 | 1.2/1.4 | **1.18/1.41** | 16.4 | 3.65 | 4.14 | 22.3 | 0.61 | 0.56 | 9.73 |
| H200 Qwen2.5-0.5B | 457 | 5.6 | 2.4/2.6 | 2.4/3.0 | 2.0/2.6 | 2.0/2.7 | **2.03/2.66** | 6.2 | 4.18 | 0.08 | 9.3 | 0.26 | 0.23 | 4.69 |
| H200 Qwen2.5-1.5B | 457 | 5.2 | 2.2/2.3 | 2.2/2.7 | 1.6/1.7 | 1.6/1.7 | **1.57/1.72** | 2.4 | 4.17 | 0.07 | 10.7 | 0.80 | 0.48 | 5.31 |
| H200 Qwen2.5-3B | 458 | 4.3 | 1.9/2.0 | 1.9/2.0 | 1.4/1.5 | 1.4/1.6 | **1.38/1.65** | 7.2 | 4.49 | 0.00 | 9.0 | 0.95 | 0.53 | 6.53 |
| H200 Qwen2.5-7B | 457 | 8.0 | 3.7/4.2 | 3.2/3.8 | 1.9/2.6 | 1.6/2.2 | **1.60/2.18** | 3.7 | 1.79 | 0.45 | 8.5 | 0.22 | 0.54 | 6.35 |
| H200 Qwen2.5-14B | 469 | 8.9 | 2.9/3.2 | 2.9/3.2 | 1.9/2.3 | 1.4/1.5 | **1.36/1.54** | 2.1 | 2.12 | 0.87 | 22.3 | 0.71 | 0.58 | 11.12 |
| H200 Qwen2.5-32B | 490 | 8.7 | 3.0/3.2 | 3.0/3.2 | 3.5/3.7 | 1.4/1.6 | **1.44/1.61** | 10.4 | 1.50 | 0.71 | 26.7 | 0.65 | 0.61 | 21.65 |
| H200 Qwen3-30B-A3B (MoE) | 474 | 4.8 | 39.7/41.1 | 14.1/16.5 | 20.7/23.3 | 20.1/22.7 | **3.23/3.79** | - | 0.28 | **24.5** | 29.1 | 1.00 | 0.25 | 25.91 |
| B200 Mistral-7B | 320 | 10.6 | 4.5/5.8 | 4.5/5.8 | 2.9/4.4 | 3.0/4.7 | **2.95/4.66** | 4.2 | 2.87 | 0.96 | 18.9 | 1.00 | 0.42 | 6.48 |
| B200 Qwen2-7B | 314 | 8.8 | 4.8/9.6 | 4.3/6.6 | 2.6/3.5 | 2.7/5.1 | **2.68/5.05** | 3.4 | 1.43 | 2.13 | 16.2 | 1.00 | 0.54 | 6.01 |
| B200 Qwen2.5-32B | 385 | 9.4 | 4.1/4.6 | 4.1/4.6 | 1.6/2.6 | 1.1/1.8 | **1.14/1.82** | 3.2 | 0.86 | 5.72 | 24.9 | 0.25 | 0.46 | 17.16 |
| B200 Qwen2-72B | 302 | 2.0 | 1.0/1.1 | 1.0/1.1 | 1.5/1.9 | 0.9/1.1 | **0.89/1.12** | 4.6 | 4.97 | 3.23 | 18.2 | 0.65 | 0.47 | 29.33 |

(u_w pinned: H200 0.718 with p=inf (numerically 12); B200 0.862 with p=1.)

- **Held out by run (LORO): 1.1-2.7% on H200 dense and 3.8% on the MoE; B200 1.1-5.1%.**
  The worst single runs are the B200 7B sharegpt_c64 runs (+9 to +13%
  over-prediction: prefill cheaper than fitted) and the H200 Qwen2.5-7B /
  0.5B c64 runs (+6%).
- The pure roofline is 2-10% (and 41% on the MoE). The per-iteration CPU term is
  what makes it work. The MoE is entirely CPU-launch-bound (t_cpu = 24.5 ms;
  expert weight traffic of 6-60 GB/iter is hidden under it), which only the
  overlap form captures.
- **Out-of-distribution** (60 synthetic-prompt H200 runs, never fitted: load
  sweeps c1-c64 x 512/2048-token prompts, Poisson 1/4/16, the concurrency wave):
  **3.6% mean MAPE with cache-aware prefill vs 44% naive**. The worst cases are
  the 2048-token prompts at c>=32 (-7 to -21%, the shared-prefix KV read of
  finding 1b).
- Separate `u_kv` is poorly identified on some B200 models (it hits the bound of
  1.0) because KV reads are nearly collinear with `tau*n`. The two are
  interchangeable for prediction inside the data range.

## 3. The plant (`controls/plant.py`)

Discrete time, Delta = `dt` (default 1 s), integrated internally in `substeps`
(20) fluid substeps. It is deterministic given d and is pure numpy (json +
math + numpy).

**State** `x` (`Plant.state_names`; fixed dimension 7 + 2m):
`q_wait` (queued, not admitted), `n_run` (admitted = prefilling + decoding,
matches vLLM `num_running_reqs`), `kv_res` (resident KV tokens as vLLM reports
them, including half-block rounding), `q_tok` (prompt tokens in the queue),
`pf_tok` / `n_pf` (admitted prompt tokens / requests whose prefill is not done),
`n_think` (closed-loop clients in their resubmit latency), and
`n_dec_1..m`, `kv_1..m` (decoding requests and their KV in m Erlang progress
stages). The age structure is what makes completions and KV drain *lag*
admissions correctly. After a load step nothing completes for about gen_len
iterations. A memoryless m=1 fluid gets this badly wrong at Delta = 1 s, because
the request lifetime is 1-7 s.

**Input** `u` (both optional): `max_running` (admission/concurrency cap enforced
by a gateway in front of vLLM; clipped to `max_num_seqs`) and `power_cap_w`.
**Disturbance** `d`: `arrivals` (open loop, per interval) and/or `clients`
(closed-loop population, which is what our load generator is), `prompt_len`,
`gen_len`, `cached_frac` (prefix-cache hit fraction: skipped compute), and
`kv_shared_frac` (prompt KV shared with resident blocks: 0 for real pools,
about `cached_frac` for identical prompts).

**Dynamics per substep h:**
1. Arrivals are added to q; closed-loop returners come back from `n_think` at
   rate 1/Z. Admission is `a = min(q, max_running - n_run, token-budget
   headroom, KV headroom)`.
2. Service. Every decoding request advances 1 token per iteration. Prefill is
   chunked into iterations carrying `n_dec + X/k_c` tokens, where
   `k_c = max(X/(2048 - n_dec), n_pf)` means at least one iteration per prefill.
   The remaining time is spent on pure decode iterations. Iteration time comes
   from the identified `t_iter(n_run, kv, tokens)` (Section 2). A fluid
   `n_run < 1` means the GPU is busy a fraction `min(1, n_run)` of the time.
3. Power: `E = sum over iterations of (e_wbyte*W(tok) + e_kvbyte*KV_bytes +
   e_gemm*2P*tok)`, and `power = P_static + E/h`. If this exceeds `power_cap_w`,
   the iterations are scaled by `s = (P_cap - P_static)*h/E`, i.e. `t_iter`
   stretches to `E_dyn/(P_cap - P_static)` (`throttled = True`).
4. Decode progress uses **exact** fluid transport through the Erlang chain.
   With tau iterations per substep, the fraction advancing j stages is
   Poisson(j; tau*m/gen_len), and completions leave the last stage carrying their
   KV. Completed prefills enter stage 1 with their prompt KV. A second admission
   pass runs at the end of the substep, so boundary states match vLLM's
   post-schedule `n_running`.

**Outputs** `y`: `power_w`, `energy_j`, `tokens_out`, `completions`, `admitted`,
`arrivals`, `iterations`, `busy_frac`, `t_iter_s`, `prefill_tokens_computed`,
`queue_delay_s = q/admission rate` (Little's law), `e2e_latency_s =
(q + n_run)/completion rate` (Little's law), `ttft_s` (queue delay + prefill
iteration), `tpot_s` (= mean t_iter), `throttled`, plus `n_waiting`,
`n_running`, `kv_tokens`.

**API**
```python
from controls.plant import Plant
p = Plant("H200", "Qwen2-7B-Instruct", dt=1.0, **overrides)   # overrides: any gpu_params key
p.state_names, p.gpu_params          # e_wbyte,e_kvbyte,e_gemm,p_static,p_cap,bw,peak_flops,
                                     # mbu(=u_w),mfu(=u_f),overhead(=t_ser),t_cpu,tau_seq,u_kv,
                                     # p_overlap,kv_capacity,max_num_seqs,token_budget,think_s,...
x = p.reset()                         # empty system; reset(x0) accepts a full vector,
x = p.state_from_obs(q_wait, n_run, kv_res, prompt_len=, gen_len=, clients=)  # observer init
x, y = p.step(x, {"max_running": 32, "power_cap_w": 450},
                 {"arrivals": 8.0, "prompt_len": 150, "gen_len": 230})
X, Ys = p.rollout(x0, us, ds)         # convenience
p.t_iter(n_run, kv_tokens, tokens), p.e_iter(kv_tokens, tokens)   # the identified maps
```
One step costs about 2.3 ms (20 substeps), so a 10-step horizon is about 25 ms
per rollout. Parameters are resolved in this order: `(GPU, model)` fit, then
the GPU-level law (needs `weight_bytes, kv_bytes_per_token,
gemm_flops_per_token, n_layers` for unknown models), then generic defaults. The
energy constants come from the lead's `gpu_coefficients.json`.

## 4. k-step-ahead validation (Delta = 1 s, k = 1..10)

**Protocol.** For every held-out run and every 1 s boundary s:
1. Initialise the full state from the three observables `(n_waiting, n_running,
   kv_tokens_resident)`, averaged over the last 100 ms of rows
   (`state_from_obs`: stationary age split, plus the closed-loop in-transit
   count).
2. Roll the plant k steps with the disturbance of that run and zero input:
   - closed loop: `clients = c`, dropped to 0 at the generator deadline (the
     last admission), after which the window drains;
   - Poisson: the exact replayed arrival counts, with per-interval prompt length
     and cached fraction of the actual arriving draws;
   - wave: the concurrency schedule.
3. Compare against the measured state at s+k and the measured mean power /
   decode-token rate over interval s+k.

**Held-out honesty.** The iteration-time parameters for each run are refit
*without that run* (LORO). `gen_len` (not knowable at arrival) is the mean of
the *other* runs of that model x task. Synthetic runs were never used in any
fit. The only in-sample parts are the 3 energy coefficients per GPU (the
lead's) and the per-GPU think time / CV^2 medians.

**Baseline.** Persistence: `x(s+k) = x(s)`, and power/tokens equal to those of
the last completed interval. `power_bc` = plant + the power residual of the
last interval (an offset/disturbance state, as used in offset-free MPC).

Cells: RMSE / NRMSE (= RMSE / mean|meas|), with persistence in brackets.
n_waiting is omitted: it is ~0 in all data (model RMSE 0.1-0.6 requests vs
persistence 0.1-0.8).

| category (starts) | var | k=1 | k=2 | k=5 | k=10 |
|---|---|---|---|---|---|
| **H200 all** (7655) | n_running | 2.59/14% [3.33/17%] | 2.69/14% [3.91/21%] | 2.94/15% [4.85/25%] | 2.91/15% [5.32/28%] |
| | kv_tokens | 1.57k/39% [1.89k/47%] | 1.68k/42% [2.24k/56%] | 1.67k/42% [2.57k/64%] | 1.60k/40% [2.70k/67%] |
| | power W (MAPE) | 22.9 (4.1%) [13.2 (1.6%)] | 23.0 (4.0%) [17.8 (2.1%)] | 24.1 (4.1%) [24.7 (2.9%)] | 24.1 (4.1%) [30.5 (3.6%)] |
| | power_bc | 15.2 (1.8%) | 16.8 (2.1%) | 18.3 (2.3%) | 19.1 (2.4%) |
| | tokens/s | 307/14% [348/16%] | 312/14% [370/17%] | 309/14% [428/19%] | 305/14% [488/22%] |
| **B200 all** (1298) | n_running | 1.52/8% [2.77/15%] | 1.65/9% [3.42/19%] | 1.73/9% [4.52/25%] | 1.79/10% [4.78/26%] |
| | kv_tokens | 1.70k/41% [2.10k/51%] | 1.77k/43% [2.24k/54%] | 1.78k/43% [2.59k/62%] | 1.70k/41% [2.68k/65%] |
| | power W (MAPE) | 35.3 (3.2%) [14.0 (1.0%)] | 36.2 (3.1%) [19.0 (1.3%)] | 47.8 (3.3%) [24.9 (1.7%)] | 52.3 (3.4%) [25.7 (1.8%)] |
| | power_bc | 16.6 (1.1%) | 23.2 (1.5%) | 40.0 (1.8%) | 46.7 (2.0%) |
| | tokens/s | 304/16% [364/19%] | 312/16% [291/15%] | 317/16% [381/20%] | 313/16% [341/18%] |
| H200 closed c>=8 (2007) | n_running | 3.58/9% [4.96/13%] | 3.86/10% [5.51/14%] | 4.43/11% [6.14/16%] | 4.34/11% [6.27/16%] |
| | kv_tokens | 2.67k/30% [3.22k/36%] | 2.89k/32% [3.74k/42%] | 2.89k/32% [4.01k/45%] | 2.77k/31% [4.15k/46%] |
| | power (MAPE) / bc | 5.5% / 2.1% [2.1%] | 5.5% / 2.5% [2.5%] | 5.6% / 2.8% [2.8%] | 5.6% / 3.0% [3.0%] |
| H200 closed c<=4 (1973) | n_running | 0.19/8% [0.31/13%] | 0.21/9% [0.33/13%] | 0.25/10% [0.35/14%] | 0.25/10% [0.35/14%] |
| | kv_tokens | 404/75% [512/96%] | 428/80% [601/112%] | 442/82% [596/110%] | 446/83% [591/110%] |
| **H200 Poisson 8/s, real prompts** (1010) | n_running | 2.93/17% [4.23/24%] | 2.97/17% [5.89/34%] | 2.97/17% [8.52/47%] | 2.67/15% [9.53/53%] |
| | kv_tokens | 1.33k/32% [1.71k/41%] | 1.44k/34% [2.36k/55%] | 1.35k/31% [3.67k/84%] | 1.14k/26% [3.97k/89%] |
| | power (MAPE) | 3.6% [2.3%] | 3.5% [3.2%] | 3.5% [4.2%] | 3.5% [4.9%] |
| | tokens/s | 274/16% [480/29%] | 255/15% [614/37%] | 239/14% [676/40%] | 235/14% [696/41%] |
| **H200 wave** (concurrency 8/0/16/0/4/0/32/0, synthetic) (80) | n_running | 3.69/41% [5.68/63%] | 3.71/41% [8.14/90%] | 3.78/42% [12.8/141%] | 3.91/43% [17.5/191%] |
| | kv_tokens | 308/39% [575/73%] | 311/39% [757/96%] | 316/40% [1.06k/135%] | 325/42% [1.33k/170%] |
| | power W (MAPE) | 43.5 (7.3%) [59.5 (12.3%)] | 41.8 (6.8%) [99.5 (26.3%)] | 42.6 (7.0%) [174 (71%)] | 43.8 (7.2%) [231 (117%)] |
| | tokens/s | 226/17% [751/55%] | 239/17% [1150/84%] | 243/18% [1920/140%] | 251/18% [2650/192%] |
| H200 Poisson 1/4/16, synthetic (118) | n_running | 1.68/31% [3.56/67%] | 1.66/31% [3.59/67%] | 1.72/32% [3.49/65%] | 1.80/33% [3.76/69%] |
| | power (MAPE) | 11.5% [12.7%] | 10.8% [17.8%] | 11.1% [18.0%] | 11.4% [17.1%] |

**Reading.**
- Queue/batch/KV states beat persistence at every k in every load class. The
  exception is n_running at B200 c<=4, where persistence wins at 10-12% vs
  13-20% NRMSE (the n=1-4 integer effect). The advantage grows with k and is
  largest exactly where control matters: transients (the wave, 3-5x better) and
  open-loop arrivals (Poisson, 2-3x better at k>=2).
- **Power.** The raw plant has a model-/GPU-specific static bias of 3-6% (the
  energy model's per-interval error). In steady closed-loop runs, persistence is
  therefore better at k=1-2. With a bias state (`power_bc`) the plant ties
  persistence at k=1-2 and beats it from k=5 on H200. In transients the raw
  plant is the best predictor by far (wave: 7% vs 12-117%).
- Throughput (tokens/s) error is 9-18% NRMSE. It consists of per-interval
  jitter of the 1 s token count plus a systematic -4% on some H200 c16 runs.

## 5. A5000 smoke run: per-request latency vs the Little's-law proxy

The run is `logs/A5000_smoke/Qwen2.5-1.5B-Instruct/{idle,alpaca_c8}`: one job,
15 s idle baseline + 30 s load, alpaca, c=8, cap 256 tokens, using the new
`requests.csv` (99 requests).

| quantity | measured (requests.csv / iter_log) | plant, zero-shot (H200 law + A5000 datasheet 768 GB/s, 111 TF) | plant, t_iter calibrated on this run |
|---|---|---|---|
| mean t_iter / TPOT | 12.81 ms / 12.80 ms | 8.59 ms (-33%) | 12.80 ms |
| throughput | 3.22 req/s | 4.56 (+42%) | 3.11 (-3%) |
| e2e latency | 2.50 s mean (Little on measured state: 7.92/3.22 = 2.46 s) | 1.73 s | **2.55 s (+2%)** |
| TTFT | 24.4 ms mean, 26.0 p90 | 8.6 ms | 12.8 ms (**-48%**) |
| power | 99 W | 100 W | 86 W (-13%, H200 energy coeffs) |

- **Little's law is an accurate latency proxy.** On the measured state
  (`n_run`/throughput) it gives mean e2e within 2%, and the calibrated plant's
  proxy is within 2%. The think time on this host is 25 ms.
- **The TTFT proxy is low by about 1 iteration plus the front-end hop** (a request
  waits on average for the in-flight iteration, then for its prefill iteration,
  then for detokenization). A better proxy is `W_q + 1.5*t_iter + Z` (about
  22 ms here). It is not yet applied in `plant.py`; treat `ttft_s` as a lower
  bound.
- **Zero-shot time transfer to a new host/GPU fails** (-33% iteration time). The
  CPU overhead is a property of the host, not of the GPU datasheet. This is
  consistent with the lead's B200 finding that datasheet scaling is falsified.
  A single short calibration run fixes it. The energy prior (H200 coefficients
  on GDDR6) happens to be within 13%. There is no A5000 energy calibration.

## 6. What the plant does not capture, and notes for controller design

**Actuators, as the data supports them.**
- `max_running` (gateway admission cap) is the only input with a strong,
  validated lever. Iteration time is almost flat in batch size (H200 7B:
  6.0 ms at n=1 vs 7.3 ms at n=64). So throughput is about n/t_iter, **power
  rises only about 30% from c=1 to c=64 (e.g. 373 -> 489 W), and energy/token
  falls about 40x** (2.3 -> 0.06 J/token). The cap acts on the queue only when
  it binds, and that regime is **not in the data**. It is simulated from the
  same service model.
- `power_cap_w` has **never been binding** in any H200/B200 run (peak about
  500 W vs a 700 W cap; about 680 W vs 1000 W). The throttle branch is a
  first-order extrapolation and **unvalidated**. The capped A100 set
  (`logs/ragged_a100`, 300 W) is the natural test once an A100 time model exists.
  Caps below about `P_static + E_dyn/t_iter` (roughly 370-500 W for H200 7B)
  bind and trade throughput linearly for power.
- `max_num_seqs`, `token_budget` (2048), prefix caching and `enforce_eager` are
  restart-time settings, not inputs. They are `gpu_params` overrides.

**Timescales.** Iteration 5-30 ms. Client resubmit about 15 ms. Prefill of 2k
tokens about 30-50 ms. Request lifetime = gen_len x t_iter, about 1-1.6 s (7B),
4 s (32B), 7 s (72B), 6.6 s (MoE). NVML power is reported at about 10 ms but
lags idle by a few hundred ms. Delta = 1 s is comparable to the request
lifetime, so the queue/KV state carries real memory across 1-10 steps.

**Nonlinearities an MPC must respect:** the roofline `max` and the CPU/GPU
overlap p-norm in `t_iter` (small models and the MoE are CPU-launch-bound, so
extra batch/KV is free until GPU time exceeds CPU time); the occupancy
`min(1, n_run)`; the saturations `min()` in admission (cap, token budget, KV);
and the power-cap saturation. Energy is linear in work; time is not.

**Known failures / limits.**
- **Single-request granularity.** At c<=4 the KV of 1-4 requests is a sawtooth
  (each request grows about 250 tokens and then vanishes). The fluid predicts
  its mean: KV NRMSE is about 75-80% at c<=4, although it still beats
  persistence. The end of a c=1 drain on slow models (32B/72B, 7 s requests) is
  smeared over seconds, giving 60-140 W power errors on those few intervals.
- **Power bias of about 4-6%** is the energy model's own per-interval error,
  model- and GPU-specific: gemma and Mistral run high, B200 7B about 5%. Use an
  offset/bias state (the `power_bc` rows) as offset-free MPC does. That removes
  it at k>=2.
- **Queueing / overload, KV-full preemption and the power-cap throttle are not
  validated** (Section 1c). Preemption (recompute) is not modeled at all:
  admission is simply blocked at 98% KV.
- **gen_len is not known ahead of time.** The plant uses a per-(model, task)
  prior. Per-request gen lengths are not in the logs (now available via
  requests.csv). A shifting prompt mix makes the Erlang stage speed approximate:
  the population-mean `gen_len` applies to all stages.
- **The iteration-time overhead is a host property.** The H200 host overlaps
  CPU launch with GPU work (max form); the B200 host looks serial (sum form).
  The A5000 zero-shot test (Section 5) is off because of this. Recalibrate
  `t_ser`, `t_cpu` and `tau_seq` per deployment host (a 30 s run is enough).
- **The MoE time model is per-model only.** Qwen3-30B-A3B is CPU-bound at about
  24 ms per iteration; its GPU work is completely hidden, so none of the
  roofline-only forms fit it.
- **Poisson runs are the hardest case** (n_run NRMSE about 17%, tokens about
  15%): the fluid cannot resolve which individual requests finish. It still beats
  persistence by 2-3x.

**Harness change (additive, backward-compatible).** `engine_compat.drain_generate`
now also returns `t_first` (first-token wall time). `energy_profile_load.py`
additionally writes `requests.csv` (`request_id, t_submit, t_first_token,
t_finish, prompt_tokens, gen_tokens, ttft_s, e2e_s`) in load mode. No other
behavior changes. A suggested follow-up (not done) is to also log
`scheduler_stats.prefix_cache_stats` (queries/hits) in `iter_logger.py`. That
would replace the RNG replay of finding (a) with a direct measurement.
