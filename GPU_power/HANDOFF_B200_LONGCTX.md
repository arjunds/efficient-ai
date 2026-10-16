# B200 handoff: long-context sweep (KV vs compute energy), NCU at long context, GPU probe

**Why.** Every serving sweep so far changed concurrency, and KV bytes and GEMM FLOPs
grow together with concurrency (per-bin correlation 0.74–0.76). So the fit can't tell
KV-byte energy from compute energy (FINDINGS_B200 "UPDATE 5 review"). Changing
**context length at fixed concurrency** changes KV bytes per decode step ~30× while
weight bytes and GEMM FLOPs per step stay fixed. That separates them.

## Run it
```bash
cd <GPU_power checkout> && git pull                 # fork arjunds/efficient-ai, branch vllm_ragged
sbatch b200_longctx.sbatch                          # 1 GPU, ~2.5-3 h, resume-safe
```
The analysis prints at the end of `slurm_out/b200-longctx-<job>.out`.

**Commit back:**
- `logs/B200_prov/{longctx,longctx32,loggerab}/`
- `logs/B200_prov/longctx_summary.json`
- `logs/B200_prov/gpu_partitions_probe.txt`
- `microbench/ncu_B200/serve_ctx*/`
- the slurm log
- a short "UPDATE 6" in `FINDINGS_B200.md`

Never commit `prompts_*.json`.

## Stages
| stage | what | answers |
|---|---|---|
| `probe_gpus` | `sinfo` + `sbatch --test-only` on every PARCC partition with A100/H100/H200/L40S GPUs (read-only; submits nothing) | Can this account use another GPU type here? The UPenn cluster's A100 node rejects our account by partition ACL. |
| `sweep` → arm `longctx` | Qwen2-7B, CUDA graphs, unique synthetic prompts, context 1k/4k/8k/16k/30k × concurrency 4/16, 512 output tokens, 60 s windows | KV energy per byte, identified from **decode-only bins** (no prefill in the bin) |
| `sweep` → arm `longctx32` | Qwen2.5-32B, context 1k/8k/30k, concurrency 4 | Same, for a second model (different KV bytes per parameter) |
| `sweep` → arm `loggerab` | Qwen2-7B eager, alpaca concurrency 1/64, **slim logger** (power, POWER_INSTANT, energy counter only) | Are the new logger's extra NVML calls slowing serving? relog was ~20% slower than September (7B c=1: 148 vs 182 tok/s). |
| `ncu` | NCU, Qwen2-7B batch-1 decode at context 8192 and 30720 (prompt prefix-cached before profiling, so this measures decode, not prefill) | Does the analytic KV byte count hold at long context? At 30k, KV is ~11% of decode bytes, so a 2× KV miscount would move DRAM/analytic by ~10%. |
| `analyze` | `longctx_analysis.py` + the run table | e_kv/e_w with CI, prefill e_gemm with the memory terms fixed, logger A/B |

## How to read the results
- **`A. decode-only bins … e_kv / e_w = X [lo, hi]`** is the headline.
  - X ≈ 1 means a KV byte costs about the same as a weight byte.
  - X ≈ 2.3–3.2 means the September per-bin premium was real.

  Decode attention FLOPs grow with KV, so e_kv here *includes* attention compute.
  That's the quantity the energy model needs.
- **`B. prefill bins … e_gemm`**: compare with the steady uncapped 0.943 pJ and the
  capped 0.57–0.67 pJ. Long prompts will push prefill to the power cap, so expect
  the capped range.
- **NCU `DRAM/analytic`**: should stay ≈ 1.00 (it was 1.004–1.008 at short context).
  If it drifts at 30k, KV bytes are miscounted; then note by how much before trusting
  e_kv.
- **Logger A/B**: if slim tok/s ≈ relog, the slowdown is the node or the environment,
  not the logger. If slim tok/s is near September's 182, make the slim logger the
  default.

## Gotchas
- 30k-token prefills at concurrency 16 run at the 1000 W cap; that's expected and
  shows up in `cap%`. Decode-only bins are what identify e_kv.
- Qwen2-7B and Qwen2.5-32B both support 32k context; `max_model_len=32768`.
- 32B at 30k with concurrency 4 needs ~34 GB of KV; it fits.
- Don't change power limits or clocks (denied anyway). The probe stage only reads and
  `--test-only`s.
