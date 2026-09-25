#!/bin/bash
#SBATCH --job-name=a5k-mbench
#SBATCH --partition=debug
#SBATCH --account=exwong
#SBATCH --gres=gpu:a5000:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=microbench/slurm-%x-%j.out
#
# Energy microbenchmarks on one RTX A5000. Submit from ~/efficient-ai/GPU_power:
#   sbatch microbench/a5000_microbench.sbatch            # full suite, ~25 min
#   MB_ARGS="--tests duty --duty_repeats 3" MB_OUT=microbench/results_A5000_duty.json sbatch ...
set -u
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs/A5000_gpu_limits
nvidia-smi --query-gpu=index,pci.bus_id,power.limit,enforced.power.limit,power.default_limit --format=csv > logs/A5000_gpu_limits/job_${SLURM_JOB_ID}.csv  # record the ENFORCED limit per job
SIF=/shared_data0/fwzhang/vllm.sif
OUT="${MB_OUT:-microbench/results_A5000.json}"
echo "==== microbench start $(date) on $(hostname) ===="
nvidia-smi --query-gpu=name,pci.bus_id,power.limit,power.default_limit,temperature.gpu --format=csv
export APPTAINERENV_PYTHONNOUSERSITE=1
apptainer exec --nv -B "$SLURM_SUBMIT_DIR:/workspace" -B /shared_data0 "$SIF" bash -lc "
  cd /workspace && python3 -W ignore -u microbench/energy_microbench.py --out $OUT --repeats 2 ${MB_ARGS:-}
"
echo "==== microbench end $(date) ===="
