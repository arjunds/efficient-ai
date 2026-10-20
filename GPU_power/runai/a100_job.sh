#!/bin/bash
# Runs INSIDE the RunAI/KAI A100 pod (submitted by runai/launch_a100.sh), from /workspace
# (= GPU_power). Stages (env STAGES): probe,microbench,serve,analyze. Resume-safe.
# Output: logs/A100_prov/ (serving), microbench/results_A100_steady.json,
#         runai/a100_job_<date>.log (everything printed).
set -u
cd /workspace
STAGES="${STAGES:-probe,microbench,serve,analyze}"
export PROV_LOG_ROOT="${PROV_LOG_ROOT:-logs/A100_prov}"
export PROV_ARMS="${PROV_ARMS:-a100_graphs,a100_eager,a100_highconc}"
export HOME=/tmp/home; mkdir -p $HOME
export HF_HOME="${HF_HOME:-/hfcache}"                 # host ~/.cache/huggingface mounted here
export HF_XET_CACHE=/shared_data0/adsampat/hf_cache/xet
export VLLM_CACHE_ROOT=/tmp/vllm_cache TRITON_CACHE_DIR=/tmp/triton_cache
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
mkdir -p "$PROV_LOG_ROOT" runai
LOG=runai/a100_job_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
echo "==== A100 job $(date) host=$(hostname) stages=$STAGES arms=$PROV_ARMS ===="

# pynvml: the vLLM image normally ships nvidia-ml-py; install to /tmp if not.
python3 -c "import pynvml" 2>/dev/null || { pip install -q --target /tmp/pyx nvidia-ml-py && export PYTHONPATH=/tmp/pyx:${PYTHONPATH:-}; }

case ",$STAGES," in *,probe,*)
  echo "=== [probe] ==="
  nvidia-smi --query-gpu=name,pci.bus_id,power.limit,enforced.power.limit,power.default_limit,power.max_limit,clocks.max.sm,clocks.max.mem,memory.total,driver_version --format=csv \
    | tee "$PROV_LOG_ROOT/gpu_probe.csv"
  python3 - <<'PY'
import pynvml, time
pynvml.nvmlInit(); h = pynvml.nvmlDeviceGetHandleByIndex(0)
def tr(f):
    try: return f()
    except Exception as e: return f"ERR {e}"
print("name", pynvml.nvmlDeviceGetName(h))
print("POWER_INSTANT field 186:", tr(lambda: pynvml.nvmlDeviceGetFieldValues(h, [186])[0].nvmlReturn))
e0 = tr(lambda: pynvml.nvmlDeviceGetTotalEnergyConsumption(h)); time.sleep(2)
e1 = tr(lambda: pynvml.nvmlDeviceGetTotalEnergyConsumption(h))
print("energy counter mJ over 2 s:", e0, e1, "-> W", (e1 - e0) / 2000 if isinstance(e0, int) else "n/a",
      "| GetPowerUsage W", tr(lambda: pynvml.nvmlDeviceGetPowerUsage(h)) / 1000)
print("power limit settable? (expect no):", tr(lambda: pynvml.nvmlDeviceGetPowerManagementLimitConstraints(h)))
PY
  python3 -c "import vllm, torch, transformers; print('vllm', vllm.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)"
  if command -v ncu >/dev/null; then
    echo "--- ncu permission probe"
    ncu --metrics dram__bytes_read.sum --target-processes all python3 -c "import torch; x=torch.randn(4096,4096,device='cuda'); y=x@x; torch.cuda.synchronize()" 2>&1 | grep -E "ERR_NVGPUCTRPERM|dram__bytes_read" | head -3
  else echo "ncu: not in this image"; fi
;; esac

case ",$STAGES," in *,microbench,*)
  # The duty-cycle tests keep mean power under the 300 W cap so bursts run at full
  # clock: the only way to get full-clock J/byte on a part that is capped under load.
  echo "=== [microbench] steady + duty-cycle (A100 HBM2e J/byte, power roof) ==="
  python3 -W ignore -u microbench/energy_microbench.py \
    --tests duty,stream,grid,vendor,gemv,llmgemm,launch,square_multi \
    --duty_ops dram_stream,l2_stream_half,l1_stream_8KBperprog,gemv_b1,gemm_fp16_4096 \
    --steady_s 10 --settle_s 3 --repeats 2 --out microbench/results_A100_steady.json
;; esac

case ",$STAGES," in *,serve,*)
  echo "=== [serve] $PROV_ARMS ==="
  test -s prompts_alpaca.json && test -s prompts_sharegpt.json || { echo "FATAL: prompts_*.json missing"; exit 1; }
  python3 -u run_provisioning_sweep.py
;; esac

case ",$STAGES," in *,analyze,*)
  echo "=== [analyze] ==="
  [ -s microbench/results_A100_steady.json ] && python3 -W ignore microbench/steady_calib_analysis.py \
      --results microbench/results_A100_steady.json --gpu A100 --out microbench/steady_calib_A100.json
  python3 -W ignore provisioning_analysis.py --root "$PROV_LOG_ROOT"
;; esac
echo "==== A100 job done $(date) ===="
