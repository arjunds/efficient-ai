#!/bin/bash
# Thermal / leakage experiment (microbench/thermal_leakage.py) inside a KAI pod.
#   JOB_SCRIPT=thermal_job.sh STAGES=thermal NAME=a100-thermal GPU_TAG=A100 bash runai/launch_a100_kai.sh
#   JOB_SCRIPT=thermal_job.sh STAGES=thermal NAME=a6000-thermal GPU_TAG=A6000 GPU_TYPE=a6000 bash runai/launch_a100_kai.sh
# Output: microbench/thermal_<GPU_TAG>/{samples.csv,phases.json,summary.json}. No clock or power-limit change.
set -u
cd "$(dirname "$0")/.."
[ -w "${HOME:-/nonexistent}" ] || { export HOME=/tmp/home; mkdir -p $HOME; }
export TRITON_CACHE_DIR=/tmp/triton_cache PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
TAG="${GPU_TAG:-A100}"
OUT="microbench/thermal_${TAG}"
mkdir -p runai
LOG=runai/thermal_${TAG}_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
echo "==== thermal $(date) host=$(hostname) tag=$TAG ===="
python3 -c "import pynvml" 2>/dev/null || { pip install -q --target /tmp/pyx nvidia-ml-py && export PYTHONPATH=/tmp/pyx:${PYTHONPATH:-}; }
nvidia-smi --query-gpu=name,pci.bus_id,power.limit,enforced.power.limit,clocks.sm,clocks.max.sm,temperature.gpu,fan.speed --format=csv
python3 -W ignore -u microbench/thermal_leakage.py --out "$OUT" ${THERMAL_ARGS:-}
echo "==== done $(date) ===="
