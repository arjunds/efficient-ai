# B200 handoff: 70B-class scale-out (TP4/TP8) and realistic load

**Why.** LIMINAL-E predicts power to about 2% at every load, but over-predicts throughput
for small models at high concurrency (TIMEMODEL_HIGHC.md). Large models are already
accurate (32B at c=128–512: ±3%; 72B TP2: 4.7%).

What's missing for provisioning large models is **multi-GPU beyond TP2**, and serving
specified as an **arrival rate** (requests/s at an SLO) rather than fixed concurrency.
The A100 side is running now via KAI: Poisson, long outputs, and 72B TP4 over PCIe.

## Run (three independent jobs; each resume-safe)
```bash
git pull
sbatch --gres=gpu:1 --export=ALL,PROV_ARMS=b200_poisson,longout b200_scale.sbatch
sbatch --gres=gpu:4 --export=ALL,PROV_ARMS=b200_tp4 b200_scale.sbatch
sbatch --gres=gpu:8 --export=ALL,PROV_ARMS=b200_tp8 b200_scale.sbatch      # whole node; skip if not grantable
```
- Keep `--cpus-per-task=28` as written; SLURM multiplies it by the GPU count.
- **Weights:** Qwen2-72B must be in the group cache. If it isn't, run
  `MODELS=Qwen/Qwen2-72B-Instruct sbatch b200_prefetch_models.sbatch` first (CPU, free).
- **Commit back:** `logs/B200_prov/{b200_poisson,longout,b200_tp4,b200_tp8}/`,
  `logs/B200_prov/provisioning_summary.json`, and the slurm logs. Never commit
  `prompts_*.json`.

## Arms
| arm | GPUs | runs | answers |
|---|---|---|---|
| `b200_poisson` | 1 | Qwen2-7B, CUDA graphs, Poisson arrivals: alpaca 2/8/16/32/48/64 req/s, sharegpt 2/8/16/24/32 req/s | J/token and TTFT/TPOT tails vs offered load up to saturation (~70 req/s alpaca at c=64), including idle gaps and queueing |
| `longout` | 1 | Qwen2-7B, 256-token prompts, **2048 / 8192 generated tokens**, c=4/16 | KV growth during each request (all earlier long-context runs had long prompts and short outputs) |
| `b200_tp4` | 4 | Qwen2-72B TP4, alpaca + sharegpt c=1/16/64/256 | NVLink sync cost at TP4 (TP2: 15.5 µs/layer) and the J/token premium of 4 GPUs vs 2 |
| `b200_tp8` | 8 | Qwen2-72B TP8, same grid | Same at TP8 |

The A100 counterparts (Poisson, longout, 72B TP4 over PCIe) are running via KAI, so
NVLink vs PCIe at TP4 will be directly comparable.

## Notes
- The new logger samples slow NVML fields every 10th sample (the full logger cost ~20%
  throughput). The energy counter is validated against POWER_INSTANT per run.
- Open-loop runs drain in-flight requests after the window. `requests.csv` has
  per-request TTFT and TPOT.
