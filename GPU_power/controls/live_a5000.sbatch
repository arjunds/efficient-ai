#!/bin/bash
#SBATCH --job-name=a5k-live
#SBATCH --partition=debug
#SBATCH --account=exwong
#SBATCH --gres=gpu:a5000:1
#SBATCH --time=02:30:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --output=controls/slurm-%x-%j.out
#
# Live closed-loop admission-control experiment on one (100-W-capped) RTX A5000.
# Submit from ~/efficient-ai/GPU_power (ONLY when node-d1's A5000s are otherwise idle):
#   sbatch controls/live_a5000.sbatch
# Overrides: LIVE_MODEL, LIVE_ROOT, LIVE_REPS, LIVE_DURATION, LIVE_CTRLS, LIVE_BUDGET,
#            LIVE_SLO_TTFT, SKIP_SYSID=1 (reuse an existing a5000_table.json)
# Steps: (0) record enforced power limit  (1) idle baseline  (2) concurrency
# staircase -> (3) throughput/power table + sysid fit  (4) live episodes
# (4 controllers x reps, shuffled order, one engine load)  (5) analysis.
# Wall time: ~2 min idle + ~6 min staircase + (reps*4*(duration+drain+gap)) ~ 75 min.
set -u
cd "$SLURM_SUBMIT_DIR"
MODEL="${LIVE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
ROOT="${LIVE_ROOT:-logs/A5000_live}"
mkdir -p "$ROOT"
SIF=/shared_data0/fwzhang/vllm.sif

echo "==== A5000 live start $(date) on $(hostname) job $SLURM_JOB_ID ===="
nvidia-smi --query-gpu=index,pci.bus_id,power.limit,enforced.power.limit,power.default_limit,power.min_limit,power.max_limit --format=csv \
  | tee "$ROOT/gpu_query_job_${SLURM_JOB_ID}.csv"
# refuse to run if another process is on our GPU (shared node)
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tee "$ROOT/gpu_apps_job_${SLURM_JOB_ID}.txt"

export APPTAINERENV_HF_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null)"
export APPTAINERENV_HUGGINGFACE_HUB_TOKEN="$APPTAINERENV_HF_TOKEN"
export APPTAINERENV_HF_HOME="$HOME/.cache/huggingface"
export APPTAINERENV_HF_XET_CACHE="/shared_data0/adsampat/hf_cache/xet"
export APPTAINERENV_PYTHONNOUSERSITE=1
export APPTAINERENV_TOKENIZERS_PARALLELISM=false
# NOTE: no pydeps on PYTHONPATH (it breaks transformers under vLLM)

run() {
  apptainer exec --nv -B "$SLURM_SUBMIT_DIR:/workspace" \
    -B "$HOME/models.py:/workspace/models.py" -B /shared_data0 "$SIF" bash -lc "cd /workspace && $*"
}

if [ "${SKIP_SYSID:-0}" != "1" ]; then
  echo "---- (1) idle baseline"
  run python3 -u energy_profile_load.py --mode idle --model "$MODEL" --run_dir "$ROOT/idle" --duration_s 30
  echo "---- (2) staircase"
  CMD=$(run python3 -m controls.a5000_sysid staircase_cmd --model "$MODEL" --run_dir "$ROOT/staircase")
  echo "$CMD"
  cp "$ROOT/idle/baselines.json" "$ROOT/staircase_baselines.json" 2>/dev/null || true
  mkdir -p "$ROOT/staircase" && cp "$ROOT/idle/baselines.json" "$ROOT/staircase/baselines.json" 2>/dev/null || true
  run $CMD
  echo "---- (3) table + sysid fit"
  run python3 -m controls.a5000_sysid table "$ROOT/staircase" --out "$ROOT/a5000_table.json"
  run python3 -m controls.a5000_sysid fit \
      --glob "logs/A5000/*/alpaca_c*,$ROOT/staircase" --out "$ROOT/plant_params_a5000.json" || \
      echo "[warn] sysid fit failed (table is the primary internal model; continuing)"
fi

echo "---- (4) live episodes"
run python3 -u -m controls.live_load --model "$MODEL" --task alpaca \
    --table_json "$ROOT/a5000_table.json" --run_dir "$ROOT/live" \
    --controllers "${LIVE_CTRLS:-static64,pi,polca,mpc}" --reps "${LIVE_REPS:-3}" \
    --duration_s "${LIVE_DURATION:-300}" --drain_s 90 --gap_s 20 \
    --budget_schedule "${LIVE_BUDGET:-100:90,88:60,78:60,100:90}" \
    --slo_ttft "${LIVE_SLO_TTFT:-30}" --slo_tpot 0.25 --budget_window 20 --horizon 30

nvidia-smi --query-gpu=index,pci.bus_id,power.limit,enforced.power.limit --format=csv \
  | tee "$ROOT/gpu_query_end_job_${SLURM_JOB_ID}.csv"
echo "---- (5) analysis"
run python3 -m controls.live_load --analyze "$ROOT/live"
echo "==== A5000 live end $(date) ===="
