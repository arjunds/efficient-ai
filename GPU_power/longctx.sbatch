#!/bin/bash
#SBATCH --job-name=egy-longctx
#SBATCH --partition=standby
#SBATCH --account=exwong
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=%x-%j.out
#
# Long-context sweep: exercise the Σ KV_bytes(ctx) term (invisible at 512-2048
# ctx vs 14 GB weights). Qwen2-7B on H200; contexts 4k/16k/32k x concurrency.
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
    BL=logs/load_sweep/Qwen2-7B-Instruct/baselines.json
    for CTX in 4096 16384 30720; do
      for C in 1 8 16; do
        RD=logs/longctx/${MODEL##*/}/ctx\${CTX}_c\${C}
        echo \"================ ctx=\${CTX} concurrency=\${C} ================\"
        mkdir -p \$RD
        [ -f \$BL ] && cp \$BL \$RD/baselines.json
        python3 energy_profile_load.py --mode load --model ${MODEL} \
          --concurrency \$C --input_len \$CTX --output_len 128 --duration_s 55 \
          --run_dir \$RD --max_model_len 32768 --gpu_memory_utilization 0.9 2>&1 | tail -5
        python3 energy_model.py      --run_dir \$RD 2>&1 | grep -E 'e_bit_j|e_bit_agg|p_static_w|r2_energy|mean_bytes|n_bins|error'
        python3 validate_waveform.py --run_dir \$RD 2>&1 | grep -E 'rmse_w|mean_abs_pct|error'
      done
    done
    echo '==== KV-fraction check (Σ KV bytes vs weight bytes per iter) ===='
    python3 - <<'PY'
import json, glob, os
for d in sorted(glob.glob('logs/longctx/*/ctx*_c*')):
    cal = os.path.join(d,'calibration.json')
    if not os.path.exists(cal): continue
    c = json.load(open(cal))
    wb = c.get('weight_bytes'); mb = c.get('mean_bytes_per_bin'); nb = c.get('n_bins')
    kvfrac = None
    if wb and mb and nb:
        # mean bytes/iter ~ mean_bytes_per_bin / (iters per bin); use ratio to weight
        kvfrac = (mb) # informational
    print(os.path.basename(d), 'e_bit=%.3e'%(c.get('e_bit_j_per_byte') or 0),
          'weight_bytes=%.2e'%(wb or 0))
PY
  "
echo "done: $(date)"
