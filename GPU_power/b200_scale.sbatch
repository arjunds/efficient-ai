#!/bin/bash
#SBATCH --job-name=b200-scale
#SBATCH --partition=dgx-b200
#SBATCH --qos=dgx
#SBATCH --account=leebcc-systems-architecture-int
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=04:00:00
#SBATCH --output=slurm_out/%x-%j.out
#
# Provisioning / "LIMINAL extended" runs (HANDOFF_B200_PROVISIONING.md).
# Scale-out + realistic-load runs (HANDOFF_B200_SCALE.md). GPU count is set at submit time:
#   sbatch --gres=gpu:1 --export=ALL,PROV_ARMS=b200_poisson,longout b200_scale.sbatch   (~1.5 h)
#   sbatch --gres=gpu:4 --export=ALL,PROV_ARMS=b200_tp4 b200_scale.sbatch               (~1 h)
#   sbatch --gres=gpu:8 --export=ALL,PROV_ARMS=b200_tp8 b200_scale.sbatch               (~1 h)
# Keep --cpus-per-task=28: PARCC cli_filter requires exactly DefCpuPerGPU and multiplies it.
set -u
cd "$SLURM_SUBMIT_DIR"
BASE=/vast/projects/leebcc/systems-architecture-int/adsampat
CTR=$BASE/containers
SIF="${SIF:-}"
[ -z "$SIF" ] && [ -s "$CTR/vllm-0.10.2.sif" ] && SIF="$CTR/vllm-0.10.2.sif"
[ -z "$SIF" ] && { echo "FATAL: no vllm-0.10.2.sif in $CTR"; exit 1; }
export PROV_LOG_ROOT="${PROV_LOG_ROOT:-logs/B200_prov}"
export PROV_ARMS="${PROV_ARMS:-b200_poisson,longout}"
mkdir -p "$PROV_LOG_ROOT" slurm_out

echo "==== B200 provisioning  $(date)  node=$(hostname)  arms=$PROV_ARMS ===="
nvidia-smi --query-gpu=name,power.limit,enforced.power.limit,power.default_limit,memory.total,clocks.max.sm,driver_version --format=csv \
  | tee "$PROV_LOG_ROOT/gpu_$SLURM_JOB_ID.csv"
for M in Qwen/Qwen2-7B-Instruct Qwen/Qwen2-72B-Instruct; do
  D="$BASE/hf_cache/hub/models--$(echo "$M" | sed 's|/|--|')"
  [ -d "$D" ] && echo "  cached: $M" || echo "  NOT cached (will download on GPU time): $M"
done

export APPTAINERENV_HF_HOME=$BASE/hf_cache
export APPTAINERENV_HF_HUB_CACHE=$BASE/hf_cache/hub
export APPTAINERENV_HF_XET_CACHE=$BASE/hf_cache/xet
export APPTAINERENV_PYTHONNOUSERSITE=1                 # host ~/.local numpy breaks vLLM
export APPTAINERENV_VLLM_USE_TRTLLM_ATTENTION=0        # Blackwell sharegpt crash fix (b200_attnfix)
export APPTAINERENV_TOKENIZERS_PARALLELISM=false
export APPTAINERENV_PROV_LOG_ROOT="$PROV_LOG_ROOT"
export APPTAINERENV_PROV_ARMS="$PROV_ARMS"
[ -n "${PROV_DURATION_S:-}" ] && export APPTAINERENV_PROV_DURATION_S="$PROV_DURATION_S"
[ -n "${HF_TOKEN:-}" ] && export APPTAINERENV_HF_TOKEN="$HF_TOKEN"

source /vast/parcc/sw/etc/profile.d/Z98-lmod.sh
module load apptainer/1.4.1
apptainer exec --nv -B "$SLURM_SUBMIT_DIR:/workspace" -B "$BASE/hf_cache:$BASE/hf_cache" "$SIF" bash -lc '
  set -u; cd /workspace
  test -s prompts_alpaca.json && test -s prompts_sharegpt.json || { echo "FATAL: prompts_*.json missing"; exit 1; }
  python3 -u run_provisioning_sweep.py 2>&1
  echo; echo "==== analysis ===="
  python3 -W ignore provisioning_analysis.py --root "$PROV_LOG_ROOT" 2>&1
'
echo "==== done $(date) ===="
