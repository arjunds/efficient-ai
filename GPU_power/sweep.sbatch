#!/bin/bash
#SBATCH --job-name=egy-sweep
#SBATCH --partition=standby
#SBATCH --account=exwong
#SBATCH --gres=gpu:h200:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=%x-%j.out
#
# Full offered-load sweep: MODELS x CONCURRENCY x WORKLOADS with idle +
# prefill-ceiling baselines, per-point calibration + waveform validation, and a
# rolled-up sweep_summary.csv. Resume-safe (re-submit to continue if preempted).
set +e
SIF="${SIF:-/shared_data0/fwzhang/vllm.sif}"
HARNESS="$HOME/efficient-ai/GPU_power"
export APPTAINERENV_HUGGINGFACE_HUB_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null)"
export APPTAINERENV_HF_TOKEN="$APPTAINERENV_HUGGINGFACE_HUB_TOKEN"
export APPTAINERENV_HF_HOME="$HOME/.cache/huggingface"
# forward optional sweep overrides into the container
export APPTAINERENV_SWEEP_MODELS="${SWEEP_MODELS:-}"
export APPTAINERENV_SWEEP_LOG_ROOT="${SWEEP_LOG_ROOT:-}"

echo "==== sweep start $(date) on $(hostname) ===="
nvidia-smi --query-gpu=name,power.limit,memory.total --format=csv,noheader

apptainer exec --nv -B "${HARNESS}:/workspace" \
  -B "$HOME/models.py:/workspace/models.py" -B /shared_data0 "$SIF" bash -lc "
    cd /workspace
    mkdir -p logs/load_sweep
    python3 -u run_load_sweep.py 2>&1 | tee logs/load_sweep/sweep_console.log
    echo '==== SWEEP SUMMARY ===='
    cat logs/load_sweep/sweep_summary.csv 2>/dev/null
  "
echo "==== sweep end $(date) ===="
