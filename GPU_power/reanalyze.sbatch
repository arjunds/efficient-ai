#!/bin/bash
#SBATCH --job-name=egy-reanalyze
#SBATCH --partition=standby
#SBATCH --account=exwong
#SBATCH --time=00:12:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out
#
# Re-run ONLY the analysis (calibration + waveform) on existing run dirs. Pure
# Python, no GPU — but needs the container's python3.12 (node system python is 3.6,
# lacks dataclasses). Pass run dirs as args; default to first_run c1/c8.
set +e
SIF="${SIF:-/shared_data0/fwzhang/vllm.sif}"
HARNESS="$HOME/efficient-ai/GPU_power"
DIRS="${@:-logs/first_run/c1 logs/first_run/c8}"

apptainer exec \
  -B "${HARNESS}:/workspace" \
  -B "$HOME/models.py:/workspace/models.py" -B /shared_data0 \
  "$SIF" bash -lc "
    cd /workspace
    for d in ${DIRS}; do
      echo \"================ \$d ================\"
      python3 energy_model.py --run_dir \$d 2>&1
      echo '---- waveform ----'
      python3 validate_waveform.py --run_dir \$d 2>&1
    done
  "
