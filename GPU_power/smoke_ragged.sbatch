#!/bin/bash
#SBATCH --job-name=egy-smoke-ragged
#SBATCH --partition=standby
#SBATCH --account=exwong
#SBATCH --gres=gpu:h200:1
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out
#
# Validate the ragged/two-term path before the full sweep: datasets load, ragged
# prompts populate iter_log fields, two-term fit + identifiability work.
set +e
SIF=/shared_data0/fwzhang/vllm.sif
export APPTAINERENV_HUGGINGFACE_HUB_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null)"
export APPTAINERENV_HF_TOKEN="$APPTAINERENV_HUGGINGFACE_HUB_TOKEN"
export APPTAINERENV_HF_HOME="$HOME/.cache/huggingface"
export APPTAINERENV_HF_XET_CACHE="/shared_data0/adsampat/hf_cache/xet"
export APPTAINERENV_HF_DATASETS_CACHE="/shared_data0/adsampat/hf_cache/datasets"
export APPTAINERENV_PYTHONPATH="/shared_data0/adsampat/pydeps"

apptainer exec --nv -B "$HOME/efficient-ai/GPU_power:/workspace" \
  -B "$HOME/models.py:/workspace/models.py" -B /shared_data0 "$SIF" bash -lc '
    cd /workspace
    echo "=== dataset load + length spread (datasets preinstalled in pydeps) ==="
    python3 prompts.py alpaca 2>&1 | tail -2
    python3 prompts.py sharegpt 2>&1 | tail -2

    M=Qwen/Qwen2-7B-Instruct
    mkdir -p logs/ragged/Qwen2-7B-Instruct
    cp logs/load_sweep/Qwen2-7B-Instruct/baselines.json logs/ragged/Qwen2-7B-Instruct/ 2>/dev/null

    for spec in "alpaca c1 --concurrency 1" "sharegpt c16 --concurrency 16"; do
      set -- $spec; TASK=$1; TAG=$2; shift 2
      RD=logs/ragged/Qwen2-7B-Instruct/${TASK}_${TAG}
      mkdir -p $RD; cp logs/ragged/Qwen2-7B-Instruct/baselines.json $RD/ 2>/dev/null
      echo "================ $TASK $TAG ================"
      python3 energy_profile_load.py --mode load --model $M --task $TASK "$@" \
        --output_len 256 --duration_s 40 --run_dir $RD --max_model_len 8192 2>&1 | tail -6
      echo "--- iter_log head (verify prefill/decode/kv_tokens_resident populated) ---"
      head -4 $RD/iter_log.csv
      echo "--- phase distribution ---"
      awk -F, "NR>1{print \$3}" $RD/iter_log.csv | sort | uniq -c
      python3 energy_model.py --two_term --run_dir $RD 2>&1 | grep -E "e_bit|e_flop|identifiable|pearson|arithmetic|n_bins|error"
    done

    echo "================ POOLED two-term fit (2 points) ================"
    python3 fit_two_term.py --glob "logs/ragged/Qwen2-7B-Instruct/*" \
      --out logs/ragged/Qwen2-7B-Instruct/two_term_fit.json 2>&1
  '
echo "done: $(date)"
