#!/bin/bash
#SBATCH --job-name=egy-wave
#SBATCH --partition=standby
#SBATCH --account=exwong
#SBATCH --gres=gpu:h200:1
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out
#
# Bursty-load run: alternating load levels and idle gaps so the NVML power
# waveform actually moves (active ~370W <-> idle ~118W). Calibrate on active
# bins, validate over ALL bins (idle bins predict P_static). This is the strong
# form of the handoff's "predict the power waveform" validation.
set +e
SIF="${SIF:-/shared_data0/fwzhang/vllm.sif}"
HARNESS="$HOME/efficient-ai/GPU_power"
MODEL="${MODEL:-Qwen/Qwen2-7B-Instruct}"
RD="logs/wave"
SCHED="8:12,0:8,16:12,0:8,4:12,0:8,32:12,0:8"

export APPTAINERENV_HUGGINGFACE_HUB_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null)"
export APPTAINERENV_HF_TOKEN="$APPTAINERENV_HUGGINGFACE_HUB_TOKEN"
export APPTAINERENV_HF_HOME="$HOME/.cache/huggingface"
export APPTAINERENV_HF_XET_CACHE="/shared_data0/adsampat/hf_cache/xet"

apptainer exec --nv -B "${HARNESS}:/workspace" \
  -B "$HOME/models.py:/workspace/models.py" -B /shared_data0 "$SIF" bash -lc "
    cd /workspace
    # seed idle baseline for this run dir's parent (reuse first_run's if present)
    mkdir -p logs
    [ -f logs/first_run/baselines.json ] && cp logs/first_run/baselines.json logs/baselines.json
    python3 energy_profile_load.py --mode load --model ${MODEL} \
      --concurrency_schedule '${SCHED}' --input_len 512 --output_len 64 \
      --run_dir ${RD} --max_model_len 4096 2>&1 | tail -20
    echo '==== calibrate (active bins) + validate (all bins incl idle) ===='
    python3 energy_model.py      --run_dir ${RD} 2>&1
    python3 validate_waveform.py --run_dir ${RD} 2>&1
  "
echo '==== segment power summary ===='
echo "done: $(date)"
