#!/bin/bash
# KAI pod job: reasoning-model output lengths (traces/gen_reasoning_lengths.py). Copy, don't edit while running.
set -u
cd "$(dirname "$0")/.."
[ -w "${HOME:-/nonexistent}" ] || { export HOME=/tmp/home; mkdir -p $HOME; }
export VLLM_CACHE_ROOT=/tmp/vllm_cache TRITON_CACHE_DIR=/tmp/triton_cache VLLM_ALLOW_TORCH_COMPILE=1
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
LOG=runai/job_$(hostname)_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
echo "==== reasoning-length job $(date) host=$(hostname) ===="
nvidia-smi --query-gpu=name,power.limit --format=csv
python3 -W ignore -u traces/gen_reasoning_lengths.py --out traces/reasoning_lengths_qwen3-30b-a3b.json \
    --power_dir "${PROV_LOG_ROOT:-logs/A100_prov}/reasoning_gen" ${GEN_LIMIT:+--limit $GEN_LIMIT}
echo "==== done $(date) ===="
