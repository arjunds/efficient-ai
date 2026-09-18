#!/bin/bash
#SBATCH --job-name=b200-dbg
#SBATCH --partition=dgx-b200
#SBATCH --qos=dgx
#SBATCH --account=leebcc-systems-architecture-int
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:20:00
#SBATCH --output=slurm_out/%x-%j.out
#
# Targeted debug: the smoke test proved sm_100 works (arch_list has sm_100,
# capability sm_100, vLLM 0.10.2 imports) but the V1 engine core died during
# startup with an empty "Failed core proc(s): {}". The real traceback is emitted
# by the engine-core SUBPROCESS and was lost to a `tail` in the smoke script.
#
# This job captures EVERYTHING untruncated, plus the environment facts that
# usually explain a dead V1 core under a container:
#   - /dev/shm size (V1 uses shared memory + ZMQ between client and core)
#   - whether a plain in-process LLM() works when the async MP client does not
# Also redoes the matmul benchmark WITH warmup (the smoke run reported 23 TFLOPS,
# which was cuBLAS init being timed, not a real ceiling).
set -u
cd "$SLURM_SUBMIT_DIR"
BASE=/vast/projects/leebcc/systems-architecture-int/adsampat
SIF=$BASE/containers/vllm-0.10.2.sif
mkdir -p slurm_out logs/B200_debug

echo "=== host-side facts ==="
nvidia-smi --query-gpu=name,memory.total,power.limit --format=csv,noheader
df -h /dev/shm /tmp | tail -3
echo "ulimit -l (locked mem): $(ulimit -l)"

export APPTAINERENV_HF_HOME=$BASE/hf_cache
export APPTAINERENV_HF_HUB_CACHE=$BASE/hf_cache/hub
export APPTAINERENV_HF_XET_CACHE=$BASE/hf_cache/xet
# The host $HOME is bind-mounted into the container, so a "pip install --user"
# numpy on the host SHADOWS the container's numpy and breaks vLLM (numba needs
# numpy<=2.2; the host had 2.5.3). Ignore host user-site packages entirely.
export APPTAINERENV_PYTHONNOUSERSITE=1
export APPTAINERENV_TOKENIZERS_PARALLELISM=false
export APPTAINERENV_VLLM_LOGGING_LEVEL=DEBUG

source /vast/parcc/sw/etc/profile.d/Z98-lmod.sh
module load apptainer/1.4.1

apptainer exec --nv \
  -B "$SLURM_SUBMIT_DIR:/workspace" \
  -B "$BASE/hf_cache:$BASE/hf_cache" \
  "$SIF" bash -lc '
set -u; cd /workspace
echo "=== container-side facts ==="
df -h /dev/shm /tmp | tail -3
python3 -c "import os; print(\"cpu count:\", os.cpu_count())"

echo
echo "=== [1] matmul WITH warmup (real achieved fp16 TFLOPS) ==="
python3 - <<PY
import torch, time
x = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
for _ in range(10): y = x @ x          # warmup: cuBLAS init + autotune
torch.cuda.synchronize()
t0 = time.time()
for _ in range(50): y = x @ x
torch.cuda.synchronize()
dt = time.time() - t0
print("achieved fp16 dense: %.0f TFLOPS" % (50*2*8192**3/dt/1e12))
print("(B200 datasheet dense fp16 = 2250 TFLOPS; expect ~60-80%% of that)")
PY

echo
echo "=== [2] plain in-process LLM (no async MP client) ==="
timeout 600 python3 - <<PY 2>&1 | tail -25
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen2-7B-Instruct", dtype="float16", max_model_len=2048,
          gpu_memory_utilization=0.85, enforce_eager=True)
out = llm.generate(["Hello, my name is"], SamplingParams(max_tokens=8))
print("IN_PROCESS_LLM_OK:", out[0].outputs[0].text[:40].replace("\n"," "))
PY

echo
echo "=== [3] the async MP path the harness actually uses (FULL output) ==="
timeout 600 python3 - <<PY 2>&1 | tail -60
import asyncio, sys
sys.argv = ["x"]
class A:  # mimic the harness args
    model="Qwen/Qwen2-7B-Instruct"; dtype="float16"; tensor_parallel_size=1
    gpu_memory_utilization=0.85; max_model_len=2048; enforce_eager=True
    kv_cache_dtype="auto"
import engine_compat
eng, meta = engine_compat.build_async_engine(A())
print("ASYNC_ENGINE_OK:", meta)
PY
'
echo "==== debug end $(date) ===="
