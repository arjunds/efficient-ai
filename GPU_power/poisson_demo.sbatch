#!/bin/bash
#SBATCH --job-name=egy-poisson
#SBATCH --partition=standby
#SBATCH --account=exwong
#SBATCH --gres=gpu:h200:1
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out
#
# Open-loop Poisson-arrival demo (vs the closed-loop concurrency sweep): sweep a
# few arrival rates (req/s) on Qwen and calibrate/validate each. Shows the model
# under a fluctuating-queue regime.
set +e
SIF="${SIF:-/shared_data0/fwzhang/vllm.sif}"
HARNESS="$HOME/efficient-ai/GPU_power"
MODEL="${MODEL:-Qwen/Qwen2-7B-Instruct}"
export APPTAINERENV_HUGGINGFACE_HUB_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null)"
export APPTAINERENV_HF_TOKEN="$APPTAINERENV_HUGGINGFACE_HUB_TOKEN"
export APPTAINERENV_HF_HOME="$HOME/.cache/huggingface"

apptainer exec --nv -B "${HARNESS}:/workspace" \
  -B "$HOME/models.py:/workspace/models.py" "$SIF" bash -lc "
    cd /workspace
    [ -f logs/load_sweep/Qwen2-7B-Instruct/baselines.json ] && \
      cp logs/load_sweep/Qwen2-7B-Instruct/baselines.json logs/poisson_baselines.json
    for R in 1 4 16; do
      RD=logs/poisson/${MODEL##*/}/rate\${R}
      echo \"================ arrival_rate=\${R} req/s ================\"
      mkdir -p \$(dirname \$RD)
      [ -f logs/poisson_baselines.json ] && mkdir -p \$RD && cp logs/poisson_baselines.json \$RD/baselines.json
      python3 energy_profile_load.py --mode load --model ${MODEL} \
        --arrival_rate \$R --input_len 512 --output_len 128 --duration_s 40 \
        --run_dir \$RD --max_model_len 4096 2>&1 | tail -6
      python3 energy_model.py      --run_dir \$RD 2>&1 | grep -E 'e_bit_j|p_static_w|r2_energy|n_bins|error'
      python3 validate_waveform.py --run_dir \$RD 2>&1 | grep -E 'rmse_w|mean_abs_pct|r2|n_bins|error'
    done
  "
echo "done: $(date)"
