#!/bin/bash
#SBATCH --job-name=egy-probe
#SBATCH --partition=standby
#SBATCH --account=exwong
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out
#
# Environment probe for the offered-load energy harness. Reports what a SLURM GPU
# node can actually do, so we know whether/how vLLM can run under SLURM here.
# Submit from the login node:   sbatch probe.sbatch
set +e
echo "================ NODE ================"
hostname; date
echo "SLURM_JOB_ID=$SLURM_JOB_ID  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

echo "================ GPU ================"
nvidia-smi --query-gpu=name,power.limit,memory.total,driver_version --format=csv 2>&1 | head

echo "================ CONTAINER RUNTIMES ================"
for c in apptainer singularity enroot podman docker runai; do
  printf "%-12s " "$c"; command -v "$c" || echo "(not found)"
done

echo "================ MODULES (cuda/python/apptainer/nsight) ================"
module avail 2>&1 | grep -iE "cuda|python|apptainer|singularity|nsight|anaconda|conda" | head -40
echo "--- module spider (names only) ---"
timeout 30 module spider 2>&1 | grep -iE "cuda|python|apptainer|singularity|nsight" | head -20

echo "================ PROFILING TOOLS ================"
for c in dcgmi ncu nsys; do
  printf "%-8s " "$c"; command -v "$c" || echo "(not found)"
done
command -v dcgmi >/dev/null && { echo "--- dcgmi DRAM_ACTIVE sample ---"; dcgmi dmon -e 1005 -c 3 2>&1 | head; }

echo "================ SYSTEM PYTHON ================"
python3 --version 2>&1; command -v python3

cd "$SLURM_SUBMIT_DIR" 2>/dev/null || cd ~/efficient-ai/GPU_power

echo "================ SYNTAX CHECK (py_compile all modules) ================"
python3 -m py_compile *.py && echo "OK: all modules compile" || echo "FAILED: syntax error above"

echo "================ SELF-TEST (no GPU: math end-to-end) ================"
# Pure-stdlib: synth run -> calibrate -> validate. Recovered coeffs should match
# selftest_truth.json and waveform R^2 should be ~0.99.
python3 energy_profile_load.py --mode self_test --run_dir logs/selftest 2>&1 | tail -5
python3 energy_model.py       --run_dir logs/selftest 2>&1 | tail -20
python3 validate_waveform.py  --run_dir logs/selftest 2>&1 | tail -25
echo "--- truth vs recovered ---"
cat logs/selftest/selftest_truth.json 2>&1

echo "================ PyPI REACHABILITY ================"
python3 -c "import urllib.request; urllib.request.urlopen('https://pypi.org', timeout=10); print('PyPI reachable')" 2>&1

echo "================ probe_env.py (system python; vLLM expected MISSING) ================"
cd "$SLURM_SUBMIT_DIR" 2>/dev/null || cd ~/efficient-ai/GPU_power
python3 probe_env.py 2>&1 | tail -60

echo "================ DONE ================"
date
echo "If apptainer/singularity exists above, next step is to exec the harness"
echo "inside nvcr.io/nvidia/pytorch:24.01-py3 (see README_HANDOFF.md)."
