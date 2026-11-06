# Handoff: B200 trace-replay results → the synthesizing agent

You are picking up the B200 side of the real-trace work (`HANDOFF_B200_TRACE.md`)
to synthesize across GPUs. Science is in `FINDINGS_B200.md` **UPDATE 10**; this
is what you need to not misread it.

Everything below is committed on `vllm_ragged` (`a8334ff` and the commits around
it). Two jobs were still running when this was written — see §6.

---

## 1. READ THIS BEFORE QUOTING ANY NUMBER

**Use steady-window values, not raw per-level ones.** `trace_analysis.py`
excludes ramp and drain; the raw per-level figures in `run.log` do not, and the
two diverge increasingly with arrival rate:

| 7B azure23_conv | raw | steady |
|---|---|---|
| x16 J/token | 0.0553 | **0.0462** |
| x16 tok/s | 14,492 | **18,163** |

At x1 they agree to 0.5%. Anything I reported conversationally before
`trace_summary.json` existed was raw and understated the effect — the headline
azure23_conv reduction is **11.3x, not 9.5x**. `logs/B200_prov/trace_summary.json`
is the source of truth (39 levels, 16 capacity rows). This is the same artifact
your side hit on A100 (the "650 tok/s plateau" correction in `FINDINGS_TRACES.md`).

**Capacity rows are lower bounds.** "tight" and "relaxed" SLO are identical for
every trace except `azure23_code`, which means the speedup ladder ran out before
the SLO bound — those rows record the top rung tested, not a measured limit.
Only azure23_code binds (60.2 req/s tight vs 114.6 relaxed). Write them as ≥.

**Some levels are subsamples.** The harness trims the request set to fit 80% of
KV. Retained fractions: 32B azure23_conv **191/603 (31.7%)**, azure24_peak
1156/3390 (34.1%), azure23_code 1054/1435 (73.4%), burstgpt 3763/4125 (91.2%),
7B azure23_conv 1274/1274 (100%). Quote the fraction next to the number.

---

## 2. What landed

| arm | model | content |
|---|---|---|
| `b200_trace` | Qwen2.5-7B | azure23_conv, azure23_code, azure24_conv, burstgpt (FlashInfer) + mooncake_conv (**Triton**) |
| `b200_reasoning` | Qwen2.5-7B | 4 reasoning levels (x0.05–0.5) |
| `b200_trace_32b` | Qwen2.5-32B | azure23_conv + 4 reasoning levels |
| `b200_trace_moe` | Qwen3-30B-A3B | azure23_conv + reasoning (2 of 4 at write time) |

Plus `logs/B200_prov/_hang_evidence/` — run logs and iteration tails from the
three deadlocked attempts, kept because the hang is itself a result.

---

## 3. The result worth leading with: `active_params()` fails for MoE

Qwen3-30B-A3B touches 3.04 B active params vs the 7B's 7.07 B, ratio **0.430**.
If J/token tracks streamed weight bytes, the MoE should cost 0.43x per token.
Measured MoE/7B **rises monotonically with batch size** and crosses 1.0 between
c=1 and c=4:

| batch | A100 (exact c) | B200 (by mean_running) |
|---|---|---|
| 1 | **0.58** | — |
| ~4 | 1.16 | 0.93 |
| ~16 | **1.88** | **1.12** |
| ~64 | **2.22** | 1.26 |
| ~105 | — | 1.24 |
| ~244 | — | 1.65 |
| ~266 | — | 1.64 |

**Corrected 2026-11-06:** UPDATE 10 first called these "one curve" — wrong, and
the synthesis pass caught it. The crossover agrees (batch 1→4 on both) but the
slope does not: A100 is at 1.88 by batch 16 where B200 is still 1.12, and B200
needs batch 244 to reach 1.65. The rate at which union erodes the advantage is
hardware-dependent. UPDATE 10 §2 now carries both this and a second correction:
LIMINAL-E **already** models expert union and predicts the B200 MoE within
−10.2%/+13.4% over all 9 levels (the `liminal_err` field in
`trace_summary.json`, which my first pass ignored), so the failure is specific
to the naive `active_params()` denominator, not a missing model.

**Added 2026-11-06 (job 10143269, the two re-run levels):** within one GPU the
ratio is a function of batch alone, **not of the trace**. Despite very different
prompt-length distributions, azure23_conv and reasoning agree once indexed by
`mean_running` — 1.261 vs 1.236 at batch ~74/~105, and 1.647 vs 1.641 at batch
~244/~266. Two independent traces agreeing to 0.4% is the strongest evidence
that batch size (union) is the mechanism rather than prompt shape, and it means
the curve can be calibrated on whichever trace is cheapest to run.

This is **expert union**: with top-8 of 128, a batch of B tokens activates far
more than 8, so streamed weights drift from 3.04 B toward the full 30.53 B. On
B200 the MoE moves *more* bytes/s than the dense 7B (`bw_util` 0.45–0.60 vs
0.33–0.38) and spends 2.35x the weight energy per token at x12.

> An MoE's active-parameter advantage is a **batch-1 phenomenon**. It does not
> survive serving concurrency, and at realistic batches the MoE is *worse* per
> token than a dense model with 2.3x its active count.

**I used your A100 data to remove a confound, so check my read of it.** The B200
traces match the models on arrival rate, not batch size, and the MoE sits at
higher concurrency for the same rate — self-reinforcing. `a100_graphs` /
`a100_moe_graphs` run both at *fixed* concurrency, which matches batch exactly,
on sm_80 with FlashAttention rather than FlashInfer. Same curve, so the confound
is not driving it. Those A100 `results.json` use the older schema with no
`j_per_token`, so I computed `energy_window_j / generated_tokens_window`; if that
is not what your A100 numbers use, my absolute values will differ from yours
even though the ratio should not.

Caveat I could not remove: J/token includes static power, which does not scale
with weights, so inverting the ratio to an effective active-parameter count is
crude (~4.0 B at batch 1 rising to ~15.7 B at c=64). Direction sound, magnitudes
indicative. **Pinning the shape of effective-active(B) is the obvious next
experiment** — a fixed-concurrency MoE ladder with the energy decomposition the
newer schema records would do it in about an hour on one GPU.

---

## 4. The arrival-rate result reproduces on real traffic

UPDATE 7's synthetic-Poisson collapse holds on every production trace: J/token
falls **4.3–15.4x**, largest where outputs are longest (reasoning 15.4x),
smallest where prompts are longest (mooncake 4.3x). Full table in UPDATE 10 §1.
`prefix_hit` is 0 on reasoning by construction; every non-reasoning trace clears
the ≥0.98 bar (0.983–0.999).

---

## 5. Blackwell-only: FlashInfer deadlocks on long prompts

**Both Mooncake traces deadlock the engine on B200 under FlashInfer.** ~42k
iterations with no inter-iteration gap above 1 s, then a dead stop mid-decode at
`n_running=11, n_waiting=0, kv_cache_pct=0.0622` — no error, no traceback, no
slowdown. The 120 s drain timeout ends the level but never aborts the stuck
requests, so the next level never starts and the job sits at SLURM RUNNING to
its wall clock. Two hypotheses were tested and **rejected**: KV starvation from
the warm set (KV was 6% full with an empty queue) and clipped-to-limit prompts
(`load_trace` reserves room). `VLLM_USE_TRTLLM_ATTENTION=0` does not avoid it.

The same trace completes all 5 levels on `TRITON_ATTN_VLLM_V1`, identical
request set and warm tokens — so it is FlashInfer's long-sequence path. The two
Mooncake traces are the only ones with ~10k-token prompts (10,449 and 7,858
tok/req vs ~1,300 for azure23_conv) and the only ones with requests clipped at
`max_model_len`.

**This is Blackwell-specific and it matters for your cross-GPU tables.** A100
ran `mooncake_conv` *and* `mooncake_tool` clean, and A6000 ran `mooncake_tool` —
both on FlashAttention. So:

> **Do not put B200 Mooncake numbers in the same row as A100/A6000 Mooncake
> numbers.** B200's are Triton-backend; the others are FlashAttention. There is
> no completed B200 FlashInfer Mooncake run to calibrate against, and the only
> Triton-vs-FlashInfer A/B we have is ~27% slower on *sharegpt*, a different
> workload. Treat B200 mooncake as a Triton measurement and probably an
> overestimate of its energy per token.

Second Blackwell attention bug this project has found, both in variable- or
long-sequence handling; the first was TRTLLM switching kernels per batch.

---

## 6. State when this was written

All jobs finished. `trace_summary.json` carries **53 levels**, and everything
described here is committed — there is nothing left for you to pick up.

* **10141163 — done.** `mooncake_tool` completes all 5 levels on Triton at hit
  0.999, the same trace that deadlocked under FlashInfer. **The backend A/B is
  now two-for-two across both Mooncake traces**, so §5 does not rest on one
  trace. Triton numbers, so still not poolable (see §5).
* **10141162 — partial, 2 of 4 MoE reasoning levels.** It died on the third with
  `OSError(116, 'Stale file handle')` — a transient VAST/NFS fault, not a code
  bug. It reached the arm-level handler, so x0.5 never started and the x0.25 run
  dir did not survive. The two levels that landed (x0.05, x0.1) are the ones that
  bracket the crossover, which is why §3 is unaffected. Resubmitted as
  **10143269**; if that is also missing, the gap is the x0.25/x0.5
  higher-concurrency points, a regime the A100 matched-batch ladder covers better
  anyway (c=16 1.88, c=64 2.22).
* **10143269 — done (2026-11-06 00:55, 26 min, rc 0).** The re-run of those two
  levels completed with the watchdog armed. x0.25: 3229 tok/s, 0.2596 J/tok,
  batch 105.3. x0.5: 4900 tok/s, 0.1889 J/tok, batch 265.5. Committed, and
  §3 above now carries the trace-independence result they made possible.

If either is missing from `trace_summary.json`, regenerate:
`python3 -W ignore trace_analysis.py --root logs/B200_prov` (runs fine on the
login node; no container needed).

**Those two runs are now in git** (the contingency below is kept only as the
recipe for any future straggler, since the repo `.gitignore` excludes `logs/`
and `*.csv` and a plain `git add` silently drops the data):

```bash
git add -f logs/B200_prov/b200_trace_moe/Qwen3-30B-A3B/reasoning_x0.25 \
           logs/B200_prov/b200_trace_moe/Qwen3-30B-A3B/reasoning_x0.5 \
           logs/B200_prov/b200_trace/Qwen2.5-7B-Instruct/mooncake_tool \
           logs/B200_prov/trace_summary.json
```

Add only directories that contain a `results.json` — one without it is an
incomplete or deadlocked run, and committing a live one shows up dirty
immediately. **Never commit `prompts_*.json`.**

**Operational, if you run anything here:** use `./hang_watchdog.sh 15 <jobid>...`.
It kills a job whose `power_trace.csv` is still growing while `iter_log.csv` has
been frozen 15 min — the live-logger/dead-engine divergence. Polling SLURM state
cannot see this, which is why the first deadlock burned 7.8 GPU-h unnoticed; the
watchdog caught the second in 15 min. `PROV_SKIP_TAGS=mooncake` skips the
poisoned traces, and it must be propagated as `APPTAINERENV_PROV_SKIP_TAGS`
(the sbatch does this now — setting it in the submitting shell alone silently
does nothing inside the container). Same trap bit the backend override: exporting
`VLLM_ATTENTION_BACKEND` *empty* makes vLLM fall back to V0 and fail every run,
so it is exported only when non-empty.

Total B200 spend on this trace work: ~12 GPU-h, of which **~8.2 went to the two
deadlocks** before the watchdog existed.
