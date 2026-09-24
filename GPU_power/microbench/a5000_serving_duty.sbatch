#!/bin/bash
#SBATCH --job-name=a5k-sduty
#SBATCH --partition=debug
#SBATCH --account=exwong
#SBATCH --gres=gpu:a5000:1
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --output=microbench/slurm-%x-%j.out
#
# Duty-cycled serving (see serving_duty.py docstring). Submit from GPU_power/:
#   MODEL=Qwen/Qwen2-7B-Instruct sbatch microbench/a5000_serving_duty.sbatch
#   MODEL=Qwen/Qwen2.5-3B-Instruct DUTY_ARGS="--gaps 3,2,1.2,0.8" sbatch ...
set -u
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs/A5000_gpu_limits
nvidia-smi --query-gpu=index,pci.bus_id,power.limit,enforced.power.limit,power.default_limit --format=csv > logs/A5000_gpu_limits/job_${SLURM_JOB_ID}.csv  # record the ENFORCED limit per job
SIF=/shared_data0/fwzhang/vllm.sif
export APPTAINERENV_HF_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null)"
export APPTAINERENV_HF_HOME="$HOME/.cache/huggingface"
export APPTAINERENV_HF_XET_CACHE="/shared_data0/adsampat/hf_cache/xet"
export APPTAINERENV_PYTHONNOUSERSITE=1
export APPTAINERENV_TOKENIZERS_PARALLELISM=false
nvidia-smi --query-gpu=name,pci.bus_id,power.limit --format=csv,noheader
apptainer exec --nv -B "$SLURM_SUBMIT_DIR:/workspace" -B "$HOME/models.py:/workspace/models.py" \
  -B /shared_data0 "$SIF" bash -lc "
  cd /workspace && python3 -u microbench/serving_duty.py run --model ${MODEL:-Qwen/Qwen2-7B-Instruct} ${DUTY_ARGS:-}
"
