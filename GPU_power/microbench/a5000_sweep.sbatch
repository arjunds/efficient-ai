#!/bin/bash
#SBATCH --job-name=a5k-sweep
#SBATCH --partition=debug
#SBATCH --account=exwong
#SBATCH --gres=gpu:a5000:1
#SBATCH --time=01:50:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --output=microbench/slurm-%x-%j.out
#
# RTX A5000 (GDDR6) serving sweep, same protocol as H200/B200 (HANDOFF_CROSSGPU.md):
# float16, enforce_eager, alpaca+sharegpt, closed-loop concurrency, idle baseline
# per model, 45 s windows, output cap 256, no Poisson points.
# Submit from ~/efficient-ai/GPU_power:
#   RAGGED_MODELS=Qwen/Qwen2-7B-Instruct sbatch microbench/a5000_sweep.sbatch
#   RAGGED_MODELS=Qwen/Qwen2.5-3B-Instruct sbatch microbench/a5000_sweep.sbatch
# Grid overrides pass straight through (RAGGED_TASKS, RAGGED_CONCURRENCY, ...).
# Resume-safe (run_ragged_sweep skips completed runs). ~20-25 min per model.
set -u
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs/A5000_gpu_limits
nvidia-smi --query-gpu=index,pci.bus_id,power.limit,enforced.power.limit,power.default_limit --format=csv > logs/A5000_gpu_limits/job_${SLURM_JOB_ID}.csv  # record the ENFORCED limit per job
SIF=/shared_data0/fwzhang/vllm.sif
LOGROOT="${RAGGED_LOG_ROOT:-logs/A5000}"
mkdir -p "$LOGROOT"

echo "==== A5000 sweep start $(date) on $(hostname) ===="
nvidia-smi --query-gpu=name,pci.bus_id,power.limit,power.default_limit,memory.total,clocks.max.sm,clocks.max.mem,driver_version --format=csv

# specs.json for pool_gpus.py / reconcile: datasheet BW + DENSE fp16 tensor peak,
# p_cap read from hardware (the ENFORCED limit, which on node-d1 is 100 W, not the
# 230 W TDP).
nvidia-smi --query-gpu=name,power.limit,power.default_limit --format=csv,noheader,nounits | head -1 | \
python3 -c "
import json,sys
f=[x.strip() for x in sys.stdin.read().split(',')]
json.dump({'bw':768e9,'peak_flops':111.1e12,'p_cap':float(f[1]),'tdp_w':float(f[2]),
  'gpu_name':f[0],'mem_tech':'GDDR6 384-bit 16 Gbps',
  'note':'peak_flops = DENSE fp16 tensor w/ fp32 acc (222.2 sparse / 2); '
         'p_cap = ENFORCED limit (admin-set), not TDP'},
  open('$LOGROOT/specs.json','w'),indent=2)
print(open('$LOGROOT/specs.json').read())"

export APPTAINERENV_HF_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null)"
export APPTAINERENV_HUGGINGFACE_HUB_TOKEN="$APPTAINERENV_HF_TOKEN"
export APPTAINERENV_HF_HOME="$HOME/.cache/huggingface"
export APPTAINERENV_HF_XET_CACHE="/shared_data0/adsampat/hf_cache/xet"
export APPTAINERENV_PYTHONNOUSERSITE=1
export APPTAINERENV_TOKENIZERS_PARALLELISM=false
export APPTAINERENV_RAGGED_LOG_ROOT="$LOGROOT"
export APPTAINERENV_RAGGED_MODELS="${RAGGED_MODELS:-Qwen/Qwen2-7B-Instruct}"
export APPTAINERENV_RAGGED_TASKS="${RAGGED_TASKS:-alpaca,sharegpt}"
export APPTAINERENV_RAGGED_CONCURRENCY="${RAGGED_CONCURRENCY:-1,4,16,64}"
export APPTAINERENV_RAGGED_POISSON=""
export APPTAINERENV_RAGGED_DURATION_S="${RAGGED_DURATION_S:-45}"
export APPTAINERENV_RAGGED_MAX_MODEL_LEN="${RAGGED_MAX_MODEL_LEN:-8192}"
export APPTAINERENV_RAGGED_GPU_MEM_UTIL="${RAGGED_GPU_MEM_UTIL:-0.90}"

apptainer exec --nv -B "$SLURM_SUBMIT_DIR:/workspace" \
  -B "$HOME/models.py:/workspace/models.py" -B /shared_data0 "$SIF" bash -lc '
    set -u; cd /workspace
    test -s prompts_alpaca.json   || { echo "FATAL: prompts_alpaca.json missing"; exit 1; }
    test -s prompts_sharegpt.json || { echo "FATAL: prompts_sharegpt.json missing"; exit 1; }
    python3 -u run_ragged_sweep.py 2>&1
  '
echo
echo "=== avg power vs ENFORCED cap per run ==="
for r in "$LOGROOT"/*/*/results.json; do
  python3 -c "
import json,sys,os
r=json.load(open('$r')); d=os.path.dirname('$r')
try: m=json.load(open(d+'/run_meta.json'))
except Exception: m={}
print(f\"{os.path.relpath(d,'$LOGROOT'):<40} avgW={r.get('avg_power_window_w') or 0:7.1f} tok/s={r.get('aggregate_tokens_per_sec') or 0:8.1f} idleW={m.get('idle_power_w') or 0:6.1f} limitW={m.get('power_limit_w') or 0}\")"
done
echo "==== A5000 sweep end $(date) ===="
