#!/bin/bash
#SBATCH --job-name=b200-trace
#SBATCH --partition=dgx-b200
#SBATCH --qos=dgx
#SBATCH --account=leebcc-systems-architecture-int
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=08:00:00
#SBATCH --output=slurm_out/%x-%j.out
#
# Real-trace decode-only replay on B200 (HANDOFF_B200_TRACE.md, TRACES.md). Resume-safe.
#   sbatch b200_trace.sbatch                                   # 7B traces + reasoning, ~4-5 h
#   PROV_ARMS=b200_trace_moe,b200_trace_32b sbatch b200_trace.sbatch   # second job, ~4 h
# (set PROV_ARMS in the environment before sbatch; --export splits on commas)
set -u
cd "$SLURM_SUBMIT_DIR"
BASE=/vast/projects/leebcc/systems-architecture-int/adsampat
CTR=$BASE/containers
SIF="${SIF:-}"
[ -z "$SIF" ] && [ -s "$CTR/vllm-0.10.2.sif" ] && SIF="$CTR/vllm-0.10.2.sif"
[ -z "$SIF" ] && { echo "FATAL: no vllm-0.10.2.sif in $CTR"; exit 1; }
export PROV_LOG_ROOT="${PROV_LOG_ROOT:-logs/B200_prov}"
export PROV_ARMS="${PROV_ARMS:-b200_trace,b200_reasoning}"
mkdir -p "$PROV_LOG_ROOT" slurm_out
echo "==== B200 trace replay  $(date)  node=$(hostname)  arms=$PROV_ARMS ===="
nvidia-smi --query-gpu=name,power.limit,enforced.power.limit,memory.total,clocks.max.sm,driver_version --format=csv
for f in traces/azure23_conv.jsonl.gz traces/reasoning_math.jsonl.gz; do
  [ -s "$f" ] || { echo "FATAL: $f missing (git pull)"; exit 1; }; done

export APPTAINERENV_HF_HOME=$BASE/hf_cache
export APPTAINERENV_HF_HUB_CACHE=$BASE/hf_cache/hub
export APPTAINERENV_HF_XET_CACHE=$BASE/hf_cache/xet
export APPTAINERENV_PYTHONNOUSERSITE=1
export APPTAINERENV_VLLM_USE_TRTLLM_ATTENTION=0        # Blackwell long-sequence crash fix
export APPTAINERENV_TOKENIZERS_PARALLELISM=false
export APPTAINERENV_PROV_LOG_ROOT="$PROV_LOG_ROOT"
export APPTAINERENV_PROV_ARMS="$PROV_ARMS"
[ -n "${HF_TOKEN:-}" ] && export APPTAINERENV_HF_TOKEN="$HF_TOKEN"
source /vast/parcc/sw/etc/profile.d/Z98-lmod.sh
module load apptainer/1.4.1
RUN() { apptainer exec --nv -B "$SLURM_SUBMIT_DIR:/workspace" -B "$BASE/hf_cache:$BASE/hf_cache" "$SIF" bash -lc "cd /workspace && $1"; }

RUN "python3 -u run_provisioning_sweep.py 2>&1"
RUN "python3 -W ignore trace_analysis.py --root $PROV_LOG_ROOT 2>&1 | tail -60" || true
echo "==== done $(date) ===="
