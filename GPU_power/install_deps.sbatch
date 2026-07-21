#!/bin/bash
#SBATCH --job-name=egy-install
#SBATCH --partition=standby
#SBATCH --account=exwong
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/shared_data0/adsampat/install-%j.out
#
# One-time (slow, NFS) install of `datasets` into shared pydeps + pre-warm the
# alpaca/sharegpt dataset caches, so every subsequent ragged run is fast. CPU only.
set +e
SIF=/shared_data0/fwzhang/vllm.sif
export APPTAINERENV_HF_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null)"
export APPTAINERENV_HF_HOME="$HOME/.cache/huggingface"
export APPTAINERENV_HF_XET_CACHE="/shared_data0/adsampat/hf_cache/xet"
export APPTAINERENV_HF_DATASETS_CACHE="/shared_data0/adsampat/hf_cache/datasets"

apptainer exec -B "$HOME/efficient-ai/GPU_power:/workspace" -B /shared_data0 "$SIF" bash -lc '
  cd /workspace
  PD=/shared_data0/adsampat/pydeps
  echo "=== remove partial/bleeding-edge datasets, reinstall stable 3.x $(date) ==="
  rm -rf $PD/datasets $PD/datasets-*.dist-info
  # deps (pyarrow, dill, multiprocess, xxhash, fsspec, ...) already present -> only
  # datasets reinstalled -> fast + complete.
  pip install --target=$PD "datasets>=2.19,<4" 2>&1 | tail -4
  export PYTHONPATH=$PD:$PYTHONPATH
  echo "=== import check $(date) ==="
  python3 -c "import datasets; print(\"datasets\", datasets.__version__)" 2>&1
  echo "=== pre-warm alpaca $(date) ==="
  python3 prompts.py alpaca 2>&1 | tail -2
  echo "=== pre-warm sharegpt (large; may be slow) $(date) ==="
  python3 prompts.py sharegpt 2>&1 | tail -3
  echo "=== done $(date) ==="'
