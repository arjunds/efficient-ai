#!/bin/bash
#SBATCH --job-name=b200-steady
#SBATCH --partition=dgx-b200
#SBATCH --qos=dgx
#SBATCH --account=leebcc-systems-architecture-int
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28   # PARCC dgx-b200 enforces DefCpuPerGPU=28;
#                              anything else is rejected by the cli_filter plugin
#SBATCH --time=02:00:00
#SBATCH --output=slurm_out/%x-%j.out
#
# Steady-state energy calibration on B200 (EnergAIzer protocol: every point is a
# >=10 s back-to-back loop, POWER_INSTANT sampled at 10 ms). See
# HANDOFF_B200_STEADY.md. Submit from the GPU_power/ checkout:
#   sbatch microbench/b200_steady_calib.sbatch                      # steady + analyze
#   STAGES=steady,clocks,analyze sbatch microbench/b200_steady_calib.sbatch
# Stages:
#   steady   stream, grid, vendor, gemv, llmgemm, launch, square_multi   ~40 min
#   clocks   OPT-IN: try `nvidia-smi -lgc` at several SM clocks and rerun grid +
#            vendor + llmgemm at each; needs permission; always resets clocks
#   analyze  microbench/steady_calib_analysis.py -> steady_calib_B200.json
set -u
cd "$SLURM_SUBMIT_DIR"
STAGES="${STAGES:-steady,analyze}"
CLOCKS="${CLOCKS:-1000,1500,1965}"          # SM MHz for the opt-in clocks stage
BASE=/vast/projects/leebcc/systems-architecture-int/adsampat
CTR=$BASE/containers
SIF="${SIF:-}"
[ -z "$SIF" ] && [ -s "$CTR/vllm-0.10.2.sif" ] && SIF="$CTR/vllm-0.10.2.sif"
[ -z "$SIF" ] && { echo "FATAL: no .sif in $CTR"; exit 1; }
mkdir -p slurm_out microbench
source /vast/parcc/sw/etc/profile.d/Z98-lmod.sh
module load apptainer/1.4.1
export APPTAINERENV_PYTHONNOUSERSITE=1

echo "==== B200 steady calibration $(date) node=$(hostname) stages=$STAGES ===="
nvidia-smi --query-gpu=index,name,pci.bus_id,power.limit,enforced.power.limit,power.default_limit,clocks.sm,clocks.max.sm,clocks.max.mem,driver_version --format=csv \
  | tee microbench/b200_steady_gpu.csv

RUN() { apptainer exec --nv -B "$SLURM_SUBMIT_DIR:/workspace" "$SIF" bash -lc "cd /workspace && $1"; }
TESTS="stream,grid,vendor,gemv,llmgemm,launch,square_multi"

case ",$STAGES," in *,steady,*)
  echo "=== [steady] $TESTS (>=10 s per point, 3 repeats) ==="
  RUN "python3 -W ignore -u microbench/energy_microbench.py --tests $TESTS --steady_s 10 --settle_s 3 --repeats 3 --out microbench/results_B200_steady.json"
esac

case ",$STAGES," in *,clocks,*)
  echo "=== [clocks] opt-in SM clock sweep: $CLOCKS ==="
  trap 'nvidia-smi -rgc >/dev/null 2>&1; echo "[clocks] reset (-rgc)"' EXIT
  for F in ${CLOCKS//,/ }; do
    if nvidia-smi -lgc "$F,$F" 2>&1 | tee microbench/b200_lgc_$F.txt | grep -qiE "insufficient|not permitted|permission|error"; then
      echo ">>> clock locking NOT permitted (see microbench/b200_lgc_$F.txt) - skipping clocks stage"
      break
    fi
    nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv | tee -a microbench/b200_lgc_$F.txt
    RUN "python3 -W ignore -u microbench/energy_microbench.py --tests grid,vendor,llmgemm --steady_s 10 --settle_s 3 --repeats 2 --out microbench/results_B200_sm${F}.json"
  done
  nvidia-smi -rgc >/dev/null 2>&1; trap - EXIT
esac

case ",$STAGES," in *,analyze,*)
  echo "=== [analyze] ==="
  RUN "python3 microbench/steady_calib_analysis.py --results microbench/results_B200_steady.json --gpu B200 --serving_root logs/B200 --out microbench/steady_calib_B200.json"
  for f in microbench/results_B200_sm*.json; do
    [ -s "$f" ] && RUN "python3 microbench/steady_calib_analysis.py --results $f --gpu B200 --out ${f%.json}_calib.json"
  done
esac
echo "==== done $(date) ===="
