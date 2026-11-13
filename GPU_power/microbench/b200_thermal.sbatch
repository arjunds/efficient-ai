#!/bin/bash
#SBATCH --job-name=b200-thermal
#SBATCH --partition=dgx-b200
#SBATCH --qos=dgx
#SBATCH --account=leebcc-systems-architecture-int
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28   # PARCC dgx-b200 enforces DefCpuPerGPU=28
#SBATCH --time=02:30:00
#SBATCH --output=slurm_out/%x-%j.out
#
# Thermal / leakage experiment on one B200 (microbench/thermal_leakage.py): idle power vs
# die temperature during cool-downs after a DRAM-stream and a GEMM heat soak, then an
# occupancy ramp up and back down. No clock or power-limit change; ~1-1.5 h.
# Submit from the GPU_power/ checkout:  sbatch microbench/b200_thermal.sbatch
# Commit microbench/thermal_B200/ with `git add -f` (logs/ and *.csv are gitignored).
set -u
cd "$SLURM_SUBMIT_DIR"
BASE=/vast/projects/leebcc/systems-architecture-int/adsampat
SIF="${SIF:-$BASE/containers/vllm-0.10.2.sif}"
[ -s "$SIF" ] || { echo "FATAL: no $SIF"; exit 1; }
mkdir -p slurm_out
source /vast/parcc/sw/etc/profile.d/Z98-lmod.sh
module load apptainer/1.4.1
export APPTAINERENV_PYTHONNOUSERSITE=1
echo "==== B200 thermal $(date) node=$(hostname) args=${THERMAL_ARGS:-none} ===="
nvidia-smi --query-gpu=index,name,pci.bus_id,power.limit,enforced.power.limit,clocks.sm,clocks.max.sm,temperature.gpu --format=csv
apptainer exec --nv -B "$SLURM_SUBMIT_DIR:/workspace" "$SIF" bash -lc \
  "cd /workspace && python3 -W ignore -u microbench/thermal_leakage.py --out microbench/thermal_B200 ${THERMAL_ARGS:-}"
echo "==== done $(date) ===="
