#!/bin/bash
#SBATCH --job-name=egy-ncudram
#SBATCH --partition=standby
#SBATCH --account=exwong
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out
#
# Offline NCU DRAM reconciliation (handoff gate #1 without dcgmi). Profiles a
# short batch=1 decode, sums dram read+write bytes, reconciles vs analytic
# weight+KV bytes. ncu perf-counter access can be locked on shared nodes -> retry.
set +e
SIF="${SIF:-/shared_data0/fwzhang/vllm.sif}"
HARNESS="$HOME/efficient-ai/GPU_power"
MODEL="${MODEL:-Qwen/Qwen2-7B-Instruct}"
RD="logs/ncu_dram/${MODEL##*/}"
export APPTAINERENV_HUGGINGFACE_HUB_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null)"
export APPTAINERENV_HF_TOKEN="$APPTAINERENV_HUGGINGFACE_HUB_TOKEN"
export APPTAINERENV_HF_HOME="$HOME/.cache/huggingface"

apptainer exec --nv -B "${HARNESS}:/workspace" \
  -B "$HOME/models.py:/workspace/models.py" "$SIF" bash -lc "
    cd /workspace
    mkdir -p ${RD}
    which ncu && ncu --version | head -1
    echo '==== profiling short decode under ncu (retry on perf-counter lock) ===='
    ok=0
    for attempt in 1 2 3 4 5; do
      echo \"-- attempt \$attempt --\"
      ncu --target-processes all --replay-mode kernel \
        --metrics dram__bytes_read.sum,dram__bytes_write.sum \
        --profile-from-start yes -f -o ${RD}/prof \
        python3 ncu_dram_check.py --mode generate --model ${MODEL} \
          --run_dir ${RD} --prompt_len 8 --gen_tokens 24 --max_model_len 2048 \
          --gpu_memory_utilization 0.6 2>&1 | tail -20
      if [ -f ${RD}/prof.ncu-rep ]; then ok=1; break; fi
      echo 'no report (likely counter lock); retrying in 15s...'; sleep 15
    done
    if [ \$ok -ne 1 ]; then echo 'NCU FAILED after retries (perf-counter access?)'; exit 0; fi
    echo '==== export csv + reconcile ===='
    ncu --import ${RD}/prof.ncu-rep --csv --page raw > ${RD}/prof.csv 2>/dev/null
    python3 ncu_dram_check.py --mode parse --model ${MODEL} --run_dir ${RD} \
      --csv ${RD}/prof.csv --prompt_len 8 --gen_tokens 24
  "
echo "done: $(date)"
