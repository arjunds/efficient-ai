#!/bin/bash
#SBATCH --job-name=a5k-same
#SBATCH --partition=debug
#SBATCH --account=exwong
#SBATCH --gres=gpu:a5000:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --output=microbench/slurm-%x-%j.out
#
# Like-for-like on ONE GPU in ONE job: kernel microbenchmarks (steady, in the same
# power-capped DVFS state as serving) + sensor-aliasing + low-grid tests, then vLLM
# idle + c=1 serving for Qwen2-7B and Qwen2.5-3B. Removes GPU-to-GPU and
# idle-state differences from the serving-vs-kernel comparison.
set -u
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs/A5000_gpu_limits
nvidia-smi --query-gpu=index,pci.bus_id,power.limit,enforced.power.limit,power.default_limit --format=csv > logs/A5000_gpu_limits/job_${SLURM_JOB_ID}.csv  # record the ENFORCED limit per job
SIF=/shared_data0/fwzhang/vllm.sif
BUS=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | head -1 | tr -d ' ' | sed 's/00000000://;s/\.0$//;s/:/-/')
OUTROOT=logs/A5000_samegpu/gpu_$BUS
mkdir -p $OUTROOT
nvidia-smi --query-gpu=name,pci.bus_id,power.limit --format=csv,noheader | tee $OUTROOT/gpu.txt
export APPTAINERENV_HF_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null)"
export APPTAINERENV_HF_HOME="$HOME/.cache/huggingface"
export APPTAINERENV_HF_XET_CACHE="/shared_data0/adsampat/hf_cache/xet"
export APPTAINERENV_PYTHONNOUSERSITE=1
export APPTAINERENV_TOKENIZERS_PARALLELISM=false
apptainer exec --nv -B "$SLURM_SUBMIT_DIR:/workspace" -B "$HOME/models.py:/workspace/models.py" \
  -B /shared_data0 "$SIF" bash -lc "
  cd /workspace
  python3 -W ignore -u microbench/energy_microbench.py --tests lowgrid,aliasing,stream,vendor,gemv \
      --repeats 2 --out $OUTROOT/microbench.json
  for M in Qwen/Qwen2-7B-Instruct Qwen/Qwen2.5-3B-Instruct; do
    S=\${M##*/}; D=$OUTROOT/\$S
    python3 -u energy_profile_load.py --mode idle --model \$M --dtype float16 --run_dir \$D/idle \
        --max_model_len 8192 --gpu_memory_utilization 0.9 --duration_s 20 2>&1 | tail -2
    python3 -u energy_profile_load.py --mode load --model \$M --dtype float16 --run_dir \$D/alpaca_c1 \
        --max_model_len 8192 --gpu_memory_utilization 0.9 --output_len 256 --task alpaca \
        --concurrency 1 --duration_s 45 2>&1 | tail -2
    python3 -u energy_model.py --two_term --run_dir \$D/alpaca_c1 2>&1 | tail -1
  done
"
