#!/bin/bash
#SBATCH --job-name=egy-diag
#SBATCH --partition=standby
#SBATCH --account=exwong
#SBATCH --gres=gpu:h200:1
#SBATCH --time=00:25:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=%x-%j.out
set +e
SIF="${SIF:-/shared_data0/fwzhang/vllm.sif}"
HARNESS="$HOME/efficient-ai/GPU_power"
export APPTAINERENV_HUGGINGFACE_HUB_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null)"
export APPTAINERENV_HF_TOKEN="$APPTAINERENV_HUGGINGFACE_HUB_TOKEN"
echo "==== /dev/shm on host ===="; df -h /dev/shm

apptainer exec --nv -B "${HARNESS}:/workspace" "$SIF" bash -lc '
  set +e
  cd /workspace
  echo "==== /dev/shm in container ===="; df -h /dev/shm
  echo "==== V1 STATS API DUMP ===="
  python3 - <<PY
import inspect, dataclasses
try:
    from vllm.v1.metrics.stats import SchedulerStats, IterationStats
    print("SchedulerStats fields:", [f.name for f in dataclasses.fields(SchedulerStats)])
    try:
        print("IterationStats fields:", [f.name for f in dataclasses.fields(IterationStats)])
    except Exception as e:
        print("IterationStats not a dataclass:", [a for a in dir(IterationStats) if not a.startswith("_")])
except Exception as e:
    print("stats import err:", e)
try:
    from vllm.v1.metrics.loggers import StatLoggerBase
    print("StatLoggerBase methods:", [m for m in dir(StatLoggerBase) if not m.startswith("__")])
    print("record sig:", inspect.signature(StatLoggerBase.record))
    print("init sig:", inspect.signature(StatLoggerBase.__init__))
except Exception as e:
    print("StatLoggerBase err:", e)
try:
    from vllm.v1.engine.async_llm import AsyncLLM
    print("AsyncLLM.from_engine_args:", inspect.signature(AsyncLLM.from_engine_args))
    print("AsyncLLM.__init__:", inspect.signature(AsyncLLM.__init__))
except Exception as e:
    print("AsyncLLM sig err:", e)
PY
  echo "==== ENGINE STARTUP TEST (opt-125m, FULL traceback) ===="
  python3 - <<PY
import os, traceback
print("VLLM_ENABLE_V1_MULTIPROCESSING =", os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING"))
from vllm import AsyncLLMEngine, AsyncEngineArgs
try:
    e = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model="facebook/opt-125m", enforce_eager=True,
        max_model_len=1024, gpu_memory_utilization=0.3))
    print("ASYNC ENGINE OK")
except Exception:
    traceback.print_exc()
PY
  echo "==== OFFLINE LLM IN-PROC TEST (VLLM_ENABLE_V1_MULTIPROCESSING=0) ===="
  VLLM_ENABLE_V1_MULTIPROCESSING=0 python3 - <<PY
import traceback
from vllm import LLM, SamplingParams
try:
    llm = LLM(model="facebook/opt-125m", enforce_eager=True, max_model_len=1024,
              gpu_memory_utilization=0.3)
    o = llm.generate(["Hello"], SamplingParams(max_tokens=8))
    print("OFFLINE LLM OK; sample:", o[0].outputs[0].text[:40])
except Exception:
    traceback.print_exc()
PY
'
echo "==== DONE ===="; date
