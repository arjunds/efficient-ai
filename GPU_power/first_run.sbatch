#!/bin/bash
#SBATCH --job-name=egy-first
#SBATCH --partition=standby
#SBATCH --account=exwong
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out
#
# First real vLLM run: shared vllm.sif on an H200, open Qwen2-7B-Instruct.
# Diagnostic-heavy and non-fatal per step (set +e) so one job surfaces the real
# vLLM API + any harness bug. Secrets are READ FROM ~ (never hardcoded here).
set +e
echo "==== HOST ===="; hostname; date
nvidia-smi --query-gpu=name,power.limit,memory.total --format=csv,noheader

SIF="${SIF:-/shared_data0/fwzhang/vllm.sif}"
HARNESS="$HOME/efficient-ai/GPU_power"
MODEL="${MODEL:-Qwen/Qwen2-7B-Instruct}"
RD="logs/first_run"

# secrets from files, exported into the container via APPTAINERENV_*
export APPTAINERENV_HUGGINGFACE_HUB_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null)"
export APPTAINERENV_HF_TOKEN="$APPTAINERENV_HUGGINGFACE_HUB_TOKEN"
export APPTAINERENV_HF_HOME="$HOME/.cache/huggingface"

echo "==== SIF ===="; ls -laL "$SIF"

apptainer exec --nv \
  -B "${HARNESS}:/workspace" \
  -B "$HOME/models.py:/workspace/models.py" -B /shared_data0 \
  "$SIF" bash -lc "
    set +e
    cd /workspace
    echo '==== PYTHON / VLLM ===='
    python3 --version
    python3 -c 'import torch; print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda)'
    echo '==== PROBE (vLLM version + scheduler API) ===='
    mkdir -p ${RD}
    python3 probe_env.py --dump-vllm-api 2>&1 | tee ${RD}/probe.txt

    echo '==== deps (pynvml + matplotlib already in sif; load path decoupled from pandas/datasets) ===='

    echo '==== SELF-TEST (no GPU math) ===='
    python3 energy_profile_load.py --mode self_test --run_dir ${RD}/selftest
    python3 energy_model.py       --run_dir ${RD}/selftest 2>&1 | tail -8
    python3 validate_waveform.py  --run_dir ${RD}/selftest 2>&1 | tail -8

    echo '==== IDLE BASELINE (${MODEL}) ===='
    python3 energy_profile_load.py --mode idle --model ${MODEL} \
      --run_dir ${RD}/idle --duration_s 15 --max_model_len 4096 2>&1 | tail -15

    echo '==== LOAD c=1 ===='
    python3 energy_profile_load.py --mode load --model ${MODEL} \
      --concurrency 1 --input_len 512 --output_len 128 --duration_s 30 \
      --run_dir ${RD}/c1 --max_model_len 4096 2>&1 | tail -25
    echo '---- dram gate (expected: no dcgmi) ----'
    python3 gate_dram.py --run_dir ${RD}/c1 2>&1 | tail -20
    echo '---- calibrate + validate c=1 ----'
    python3 energy_model.py      --run_dir ${RD}/c1 2>&1 | tail -20
    python3 validate_waveform.py --run_dir ${RD}/c1 2>&1 | tail -15

    echo '==== LOAD c=8 (ragged) ===='
    python3 energy_profile_load.py --mode load --model ${MODEL} \
      --concurrency 8 --input_len 512 --output_len 128 --duration_s 30 \
      --run_dir ${RD}/c8 --max_model_len 4096 2>&1 | tail -25
    python3 energy_model.py      --run_dir ${RD}/c8 2>&1 | tail -20
    python3 validate_waveform.py --run_dir ${RD}/c8 2>&1 | tail -15

    echo '==== ARTIFACT SUMMARY ===='
    for d in ${RD}/c1 ${RD}/c8; do
      echo \"--- \$d ---\"
      wc -l \$d/iter_log.csv \$d/power_trace.csv 2>/dev/null
      head -3 \$d/iter_log.csv 2>/dev/null
    done
  "
echo "==== DONE ===="; date
