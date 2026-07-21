#!/bin/bash
#SBATCH --job-name=egy-ragged
#SBATCH --partition=standby
#SBATCH --account=exwong
#SBATCH --gres=gpu:h200:1
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=%x-%j.out
#
# Full ragged-workload + two-term sweep. datasets is expected pre-installed in the
# shared pydeps (from smoke_ragged); reused via PYTHONPATH (NFS install is slow).
# Override GPU via CLI --gres and log dir via RAGGED_LOG_ROOT (e.g. for A100).
set +e
SIF=/shared_data0/fwzhang/vllm.sif
export APPTAINERENV_HUGGINGFACE_HUB_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null)"
export APPTAINERENV_HF_TOKEN="$APPTAINERENV_HUGGINGFACE_HUB_TOKEN"
export APPTAINERENV_HF_HOME="$HOME/.cache/huggingface"
export APPTAINERENV_HF_XET_CACHE="/shared_data0/adsampat/hf_cache/xet"
export APPTAINERENV_HF_DATASETS_CACHE="/shared_data0/adsampat/hf_cache/datasets"
export APPTAINERENV_PYTHONPATH="/shared_data0/adsampat/pydeps"
export APPTAINERENV_RAGGED_LOG_ROOT="${RAGGED_LOG_ROOT:-}"
export APPTAINERENV_RAGGED_MODELS="${RAGGED_MODELS:-}"

echo "==== ragged sweep start $(date) on $(hostname) ===="
nvidia-smi --query-gpu=name,power.limit,memory.total --format=csv,noheader

apptainer exec --nv -B "$HOME/efficient-ai/GPU_power:/workspace" \
  -B "$HOME/models.py:/workspace/models.py" -B /shared_data0 "$SIF" bash -lc '
    cd /workspace
    mkdir -p logs/ragged
    python3 -c "import datasets" 2>/dev/null || \
      pip install --target=/shared_data0/adsampat/pydeps "datasets>=2.0" 2>&1 | tail -2
    python3 -u run_ragged_sweep.py 2>&1 | tee logs/ragged/ragged_console.log
    echo "==== TWO-TERM SUMMARY ===="
    cat logs/ragged/two_term_summary.csv 2>/dev/null
  '
echo "==== ragged sweep end $(date) ===="
