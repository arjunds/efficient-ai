#!/bin/bash
#SBATCH --job-name=b200-ncu
#SBATCH --partition=dgx-b200
#SBATCH --qos=dgx
#SBATCH --account=leebcc-systems-architecture-int
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=01:15:00
#SBATCH --output=slurm_out/%x-%j.out
#
# Hardware-counter validation of the analytic byte model (HANDOFF gotcha #5),
# plus the memory-hierarchy decomposition that would explain the B200 result.
#
# NOTE: an earlier probe wrongly reported ncu as missing -- it checked the HOST
# PATH. nsight-compute ships INSIDE the vLLM container at /usr/local/cuda/bin/ncu.
#
# NCU uses kernel REPLAY, which serializes execution and destroys the power
# waveform, so this cannot ride along with an energy run. It is a separate
# profiling pass joined to the energy data by workload -- the same split the
# group's GEMM dataset used.
#
# Three steps:
#   0. permission probe   -- ERR_NVGPUCTRPERM is what blocked the old cluster
#   1. profile a batch=1 decode collecting DRAM + L2 + L1 + TMA + shared bytes
#   2. compare against the analytic model, and test the hierarchy hypothesis
set -u
cd "$SLURM_SUBMIT_DIR"
BASE=/vast/projects/leebcc/systems-architecture-int/adsampat
SIF=$BASE/containers/vllm-0.10.2.sif
MODEL="${MODEL:-Qwen/Qwen2-7B-Instruct}"
OUT=logs/B200_ncu
GEN_TOKENS="${GEN_TOKENS:-8}"
PROMPT_LEN="${PROMPT_LEN:-8}"
rm -rf "$OUT"; mkdir -p "$OUT" slurm_out

export APPTAINERENV_HF_HOME=$BASE/hf_cache
export APPTAINERENV_HF_HUB_CACHE=$BASE/hf_cache/hub
export APPTAINERENV_HF_HUB_OFFLINE=1
export APPTAINERENV_PYTHONNOUSERSITE=1
export APPTAINERENV_TOKENIZERS_PARALLELISM=false
export APPTAINERENV_VLLM_USE_TRTLLM_ATTENTION=0
# FlashInfer JIT autotuning benchmarks many kernel variants; under NCU kernel
# replay that ran for 30+ min and timed the last attempt out. Triton attention
# has no JIT autotune step and was verified working (1825 tok/s in b200_attnfix).
export APPTAINERENV_VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1
# Keep kernels in-process so ncu sees them (the tool sets this too).
export APPTAINERENV_VLLM_ENABLE_V1_MULTIPROCESSING=0

source /vast/parcc/sw/etc/profile.d/Z98-lmod.sh
module load apptainer/1.4.1
echo "=== $(date) node=$(hostname) model=$MODEL ==="
nvidia-smi --query-gpu=name,memory.total,power.limit --format=csv,noheader

# Metrics verified to exist on sm_100 (taken from a colleague's B200 capture).
METRICS=dram__bytes.sum,lts__t_bytes.sum,l1tex__t_bytes.sum,l1tex__m_xbar2l1tex_read_bytes_pipe_tma.sum,sm__sass_data_bytes_mem_shared.sum

apptainer exec --nv \
  -B "$SLURM_SUBMIT_DIR:/workspace" -B "$BASE/hf_cache:$BASE/hf_cache" \
  "$SIF" bash -lc "
set -u; cd /workspace
NCU=/usr/local/cuda/bin/ncu
echo '--- ncu: '\$(\$NCU --version 2>&1 | head -2 | tail -1)

echo
echo '=== [0] counter permission: already confirmed GRANTED on 2026-09-24 ==='
if false; then
cat > /tmp/probe.py <<'PY'
import torch
x = torch.randn(512, 512, device='cuda', dtype=torch.float16)
for _ in range(3):
    y = x @ x
torch.cuda.synchronize()
print('probe kernel ran')
PY
probe=\$(\$NCU --metrics dram__bytes.sum --csv --target-processes all \
          python3 /tmp/probe.py 2>&1)
if echo \"\$probe\" | grep -q ERR_NVGPUCTRPERM; then
  echo 'PERMISSION: DENIED (ERR_NVGPUCTRPERM) -- same wall as the old cluster.'
  echo \"\$probe\" | grep -A3 ERR_NVGPUCTRPERM | head -8
  exit 42
elif echo \"\$probe\" | grep -q 'dram__bytes'; then
  echo 'PERMISSION: GRANTED -- counters readable. This is a first for the project.'
  echo \"\$probe\" | grep 'dram__bytes.sum' | head -2
else
  echo 'PERMISSION: inconclusive'; exit 43
fi
fi

echo
echo '=== [1] profiling a batch=1 decode (kernel replay: slow by design) ==='
# --replay-mode kernel is the default; --target-processes all follows any child.
timeout 3000 \$NCU --metrics $METRICS --csv --target-processes all \
  --replay-mode application --profile-from-start off \
  python3 ncu_decode_probe.py --model '$MODEL' \
    --run_dir '$OUT' --prompt_len $PROMPT_LEN --gen_tokens $GEN_TOKENS \
    --max_model_len 2048 --gpu_memory_utilization 0.6 \
  > '$OUT'/ncu_metrics.csv 2> '$OUT'/ncu_stderr.log
rc=\$?
echo \"ncu exit=\$rc  csv=\$(wc -l < '$OUT'/ncu_metrics.csv) lines\"
tail -5 '$OUT'/ncu_stderr.log

echo
echo '=== [2] analysis ==='
python3 ncu_hierarchy.py --csv '$OUT'/ncu_metrics.csv --run_dir '$OUT' \
    --model '$MODEL' --prompt_len $PROMPT_LEN --gen_tokens $GEN_TOKENS \
    --ncu_rc \$rc 2>&1
"
echo "==== ncu run end $(date) ===="
