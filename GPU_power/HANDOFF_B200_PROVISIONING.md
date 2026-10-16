# B200 handoff: provisioning runs ("LIMINAL extended")

**Goal.** Measure four things the provisioning model currently only predicts. All four
come from serving runs (vLLM 0.10.2), logged without the 0.85 s NVML smoothing.

## Run it
```bash
cd <GPU_power checkout> && git pull                 # fork arjunds/efficient-ai, branch vllm_ragged
sbatch b200_provisioning.sbatch                    # 1 GPU, ~2-3 h: relog, graphs, highconc, prefill
sbatch b200_provisioning_tp2.sbatch                # 2 GPUs, ~1 h: tp2, tp1ref (can run at the same time)
```
- Both jobs are resume-safe: resubmitting skips finished runs.
- The analysis prints at the end of each job's `slurm_out/b200-prov*-<job>.out` and
  writes `logs/B200_prov/provisioning_summary.json`.
- If the 72B isn't cached, the TP job runs only the 32B and prints the prefetch
  command (CPU job, free).

**Commit back:**
- `logs/B200_prov/` (excluding any `prompts_*.json`);
- both `slurm_out/b200-prov*` files;
- a short section in `FINDINGS_B200.md` with the printed summary.

## What each arm answers
| arm | runs | question |
|---|---|---|
| `relog` | Qwen2-7B, eager (as all old runs), alpaca + sharegpt, c = 1/4/16/64 | Same grid as `logs/B200`, now logged with the energy counter, `POWER_INSTANT` and per-step prefix-cache hits. **Smear-free per-bin refit on serving data:** does it reproduce e_gemm ≈ 0.94 pJ and e_kv ≈ e_wbyte (FINDINGS_B200 "UPDATE 4 review")? |
| `graphs` | Qwen2-7B c = 1–64 (both tasks), Qwen2.5-32B alpaca c = 1/16/64, **CUDA graphs on** | **Every serving run so far used eager mode** (`enforce_eager=True`). How much of the "fixed per-step overhead" (3.6 ms of a 5.4 ms 7B step), and so of the low bandwidth use, is eager-mode launch cost? A5000 smoke: graphs were 2.1× faster. |
| `highconc` | graphs, `max_num_seqs=512`, c = 128/256/512 (7B both tasks, 32B alpaca) | The model predicts B200 beats H200 per token only at very large batch. This measures B200 there (H200 stays a prediction). 32B at c = 512 approaches the KV limit (capacity, preemption). |
| `prefill` | graphs, unique synthetic prompts of 2048 / 8192 tokens, 16 output tokens, c = 4/16, 8192-token step budget | Prefill-dominated serving: how often it runs at the 1000 W cap, J per prompt token, and effective J/FLOP (steady loops: 0.94 uncapped, 0.57–0.67 at cap). Settles "is prefill cheaper on B200?" |
| `tp2` / `tp1ref` | TP = 2 on 2 GPUs, both metered: Qwen2.5-32B and Qwen2-72B, alpaca c = 1/16/64; TP = 1 72B comparator | One big GPU vs two: J/token and throughput, and the first measurement of LIMINAL's sync limit (all-reduce). Earlier TP runs would have metered only one GPU. |

## What changed in the code (validated on A5000 smoke runs, 2026-10-16)
- **`nvml_logger.py`** adds three trailing columns:
  - `energy_mj`: the cumulative NVML energy counter;
  - `clk_reasons`: power-cap throttle flags;
  - multi-GPU summing for TP > 1.

  Existing columns are unchanged.
- **`energy_model.py`** bins from the best validated reading: energy counter →
  POWER_INSTANT → GetPowerUsage.
  - The counter is used only if it agrees with POWER_INSTANT within 5%. It does on
    B200 (0.9998 in the steady microbench) but reads ~0.30× on the A5000, where it
    is rejected.
  - Old runs have only `power_w`, so they bin exactly as before.
  - New binned tables carry `power_source`, plus `prefill_tokens_computed` /
    `gemm_flops_bin_computed` from logged prefix-cache hits.
- **`iter_logger.py`** adds per-step `prefix_queries` and `prefix_hits`.
- **`energy_profile_load.py`** adds `--max_num_seqs`, `--max_num_batched_tokens`,
  `--unique_prompts` and `--meter_gpus`. It records `energy_window_j` with its source,
  plus the resolved scheduler settings and `enforce_eager` in `run_meta.json`.
- **`engine_compat.py`**: torch.compile is no longer force-disabled when
  `VLLM_ALLOW_TORCH_COMPILE=1`. The sweep sets that for the CUDA-graph arms.
  - Without it, vLLM's piecewise CUDA graphs don't run the model, and the A5000 smoke
    run was 2× *slower* than eager.
  - Eager arms keep the historical environment exactly.

## Gotchas
- Don't change power limits or clocks (denied anyway).
- The first CUDA-graph run per model spends 1–3 min in torch.compile; later runs
  reuse the cache.
- 72B with TP = 1 leaves little KV room; c = 16 may queue. That is expected and shows
  up as `wait%`.
- `highconc` 32B at c = 512 may preempt. That is the capacity data point, not a
  failure; check `KV%` and `wait%`.
