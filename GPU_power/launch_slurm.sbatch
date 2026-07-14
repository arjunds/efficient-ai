#!/bin/bash
#SBATCH --job-name=egy-load
#SBATCH --partition=standby
#SBATCH --account=exwong
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out
#
# SLURM launcher for the offered-load energy sweep via Apptainer/Singularity.
# The probe (job 62598) confirmed: apptainer + singularity ARE on compute nodes,
# PyPI is reachable, GPU is an L40 (300W, 46GB). Missing: dcgmi (see NOTE).
#
# FIRST-RUN SETUP (do once, interactively, then this script just runs):
#   salloc -p standby -A exwong --gres=gpu:1 --time=01:00:00
#   apptainer pull ~/pytorch_2401.sif docker://nvcr.io/nvidia/pytorch:24.01-py3
#   apptainer exec --nv ~/pytorch_2401.sif bash -lc '
#       pip install --target=$HOME/efficient-ai/GPU_power/pydeps \
#           vllm nvidia-ml-py matplotlib "pandas<3"'   # + pin your known-good vllm
#   # (installing to a bind-mounted target dir keeps the .sif read-only & reusable)
#
# Then:  sbatch launch_slurm.sbatch
set -e
SIF="${SIF:-$HOME/pytorch_2401.sif}"
HARNESS="${HARNESS:-$HOME/efficient-ai/GPU_power}"
export APPTAINERENV_HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-}"
export APPTAINERENV_PYTHONPATH="/workspace/pydeps"

apptainer exec --nv \
  -B "${HARNESS}:/workspace" \
  -B "$HOME/models.py:/workspace/models.py" \
  -B "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  "$SIF" bash -lc '
    cd /workspace
    python probe_env.py --dump-vllm-api
    # Prove the math with no GPU first (fast, always safe):
    python energy_profile_load.py --mode self_test --run_dir logs/selftest
    python energy_model.py       --run_dir logs/selftest
    python validate_waveform.py  --run_dir logs/selftest
    # The real sweep (edit MODELS/WORKLOADS in run_load_sweep.py first):
    python run_load_sweep.py
  '
#
# NOTE on dcgmi (DRAM gate #1): not present on the node or in the stock pytorch
# image. Options, in order of preference:
#   1. apptainer pull a DCGM-enabled image, or `apt-get install datacenter-gpu-manager`
#      in a writable overlay, then start nv-hostengine before the sweep.
#   2. Ship dcgmi via a second bind-mount if it exists elsewhere on the cluster.
# Without dcgmi, iter_log.dram_bytes stays blank and gate #1 cannot pass — energy
# calibration still runs (it integrates NVML power), but the byte-reconciliation
# check that caught the previous broken counter won't be available.
