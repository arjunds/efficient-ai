# B200 handoff: real-trace decode-only replay

**Why.** Every serving result so far uses synthetic load (fixed concurrency, alpaca or
sharegpt, Poisson). The next paper step is real traffic:
- public production traces (Azure 2023/2024, BurstGPT, Mooncake chat + tool-agent);
- a reasoning workload;
- replayed open-loop, **decode only**: prompts are prefix-cache-warmed before the
  window, so the GPU runs the decode tier of a disaggregated deployment.

The A100 and A6000 run the same arms. The design is in `TRACES.md`.

## Run it
```bash
cd <GPU_power checkout> && git pull                  # fork arjunds/efficient-ai, branch vllm_ragged
# models: Qwen2.5-7B-Instruct and Qwen3-30B-A3B may not be in the B200 cache yet
MODELS="Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen3-30B-A3B,Qwen/Qwen2.5-32B-Instruct" sbatch b200_prefetch_models.sbatch
sbatch b200_trace.sbatch                              # arms b200_trace,b200_reasoning: 7B, ~4-5 h
export PROV_ARMS=b200_trace_moe,b200_trace_32b        # then a second job (set in the env; --export splits commas)
sbatch b200_trace.sbatch                              # ~4 h
```
Both are 1-GPU jobs and resume-safe. `trace_analysis.py` prints a table at the end of
`slurm_out/b200-trace-<job>.out`.

**Commit back:**
- `logs/B200_prov/{b200_trace,b200_reasoning,b200_trace_moe,b200_trace_32b}/`
- `logs/B200_prov/trace_summary.json`
- the slurm logs
- a short "UPDATE 7" in `FINDINGS_B200.md`

Never commit `prompts_*.json`.

## What runs
One run per trace. Each run replays the same request set at several arrival-rate
multipliers ("levels") in one engine:

| arm | model | traces (speedups) |
|---|---|---|
| `b200_trace` | Qwen2.5-7B | azure23_conv (1–16), azure23_code (2–32), azure24_peak (0.25–2), burstgpt (2–32), mooncake_tool (0.25–3), mooncake_conv (0.25–3) |
| `b200_reasoning` | Qwen2.5-7B | reasoning ×0.05, 0.1, 0.25, 0.5 (one 10-min run each) |
| `b200_trace_moe` | Qwen3-30B-A3B | azure23_conv, mooncake_tool, reasoning |
| `b200_trace_32b` | Qwen2.5-32B | azure23_conv, mooncake_tool, reasoning |

## Sanity checks (in each run's log / `results.json` → `levels`)
- **`hit` (prefix hit fraction)** should be ≥0.98 for every non-reasoning level. Lower
  means the warm set was evicted. `[trace]` shows `trace_dropped_for_cache`; the
  harness already cuts the request set to fit 80% of the KV cache.
- **`TPOT p99`** rising steeply with speedup is the saturation knee. That's expected,
  and it's what the SLO capacity is read from.
- **The level window (`[trace] level xN: … s`)** should be about the trace span ÷
  speedup. Much longer means overload: the queue built up.

## Gotchas
- `max_model_len=32768`; Mooncake prompts are clipped to fit.
- Keep `VLLM_USE_TRTLLM_ATTENTION=0` (set in the sbatch).
- Don't change power limits or clocks.
