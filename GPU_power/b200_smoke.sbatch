#!/bin/bash
#SBATCH --job-name=b200-smoke
#SBATCH --partition=dgx-b200
#SBATCH --qos=dgx
#SBATCH --account=leebcc-systems-architecture-int
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:35:00
#SBATCH --output=slurm_out/%x-%j.out
#
# TIER 1 -- the go/no-go. ~25 min on ONE B200. Everything expensive is gated
# behind this. It answers, in order:
#   A. What GPU is this really, and what is its power cap?      (record for handoff)
#   B. Does torch/vLLM actually support sm_100 (Blackwell)?     (the main risk)
#   C. Can we lock clocks?  (enables the frequency-scaling experiment)
#   D. Measured idle P_static.
#   E. At concurrency 1, does average power hit the cap?        (GOTCHA #1)
#      If it does, the linear energy model degenerates and the sweep is pointless.
#   F. Do we get a sane binned_table + coefficients out the far end?
#
# Deliberately ONE short load run, not a sweep. Costs ~0.5 GPU-hour.
set -u
cd "$SLURM_SUBMIT_DIR"
BASE=/vast/projects/leebcc/systems-architecture-int/adsampat
CTR=$BASE/containers
SIF="${SIF:-}"
[ -z "$SIF" ] && [ -s "$CTR/vllm-0.10.2.sif" ] && SIF="$CTR/vllm-0.10.2.sif"
[ -z "$SIF" ] && [ -s "$CTR/vllm-0.11.0.sif" ] && SIF="$CTR/vllm-0.11.0.sif"
if [ -z "$SIF" ]; then echo "FATAL: no .sif in $CTR"; exit 1; fi

MODEL="${MODEL:-Qwen/Qwen2-7B-Instruct}"
RUNROOT="${RUNROOT:-logs/B200_smoke}"
mkdir -p "$RUNROOT" slurm_out

echo "==================================================================="
echo " B200 SMOKE  $(date)  node=$(hostname)  sif=$(basename "$SIF")"
echo "==================================================================="

echo "--- [A] GPU identity + power cap ---"
nvidia-smi --query-gpu=index,name,uuid,memory.total,power.limit,power.max_limit,power.default_limit,clocks.max.sm,clocks.max.mem,driver_version --format=csv
echo
nvidia-smi -q -d POWER | sed -n '1,40p'

echo "--- [C] clock/power-limit permissions (for the frequency-scaling test) ---"
# Probe only: read the supported clocks, then try a lock and immediately reset.
nvidia-smi -q -d SUPPORTED_CLOCKS 2>/dev/null | head -12
if nvidia-smi -i 0 -lgc 1000,1000 >/dev/null 2>&1; then
    echo "CLOCK_LOCK=yes  (nvidia-smi -lgc permitted)"; nvidia-smi -i 0 -rgc >/dev/null 2>&1
else
    echo "CLOCK_LOCK=no   (needs root/admin -- frequency-scaling test not possible)"
fi
if nvidia-smi -i 0 -pl "$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits | head -1 | cut -d. -f1)" >/dev/null 2>&1; then
    echo "POWER_LIMIT_SET=yes"
else
    echo "POWER_LIMIT_SET=no"
fi

export APPTAINERENV_HF_HOME=$BASE/hf_cache
export APPTAINERENV_HF_HUB_CACHE=$BASE/hf_cache/hub
export APPTAINERENV_HF_XET_CACHE=$BASE/hf_cache/xet
export APPTAINERENV_HF_HUB_OFFLINE=0
# The host $HOME is bind-mounted into the container, so a "pip install --user"
# numpy on the host SHADOWS the container's numpy and breaks vLLM (numba needs
# numpy<=2.2; the host had 2.5.3). Ignore host user-site packages entirely.
export APPTAINERENV_PYTHONNOUSERSITE=1
export APPTAINERENV_TOKENIZERS_PARALLELISM=false
[ -n "${HF_TOKEN:-}" ] && export APPTAINERENV_HF_TOKEN="$HF_TOKEN"

source /vast/parcc/sw/etc/profile.d/Z98-lmod.sh
module load apptainer/1.4.1

apptainer exec --nv \
  -B "$SLURM_SUBMIT_DIR:/workspace" \
  -B "$BASE/hf_cache:$BASE/hf_cache" \
  "$SIF" bash -lc "
set -u
cd /workspace
echo '--- [B] does this build support Blackwell (sm_100)? ---'
python3 - <<'PY'
import torch
print('torch      :', torch.__version__)
print('cuda build :', torch.version.cuda)
print('arch_list  :', torch.cuda.get_arch_list())
print('device     :', torch.cuda.get_device_name(0))
cap = torch.cuda.get_device_capability(0)
print('capability : sm_%d%d' % cap)
ok = any(f'sm_{cap[0]}{cap[1]}' in a for a in torch.cuda.get_arch_list())
print('SM_SUPPORTED:', 'yes' if ok else 'NO -- kernels not built for this arch')
x = torch.randn(8192, 8192, device='cuda', dtype=torch.float16)
for _ in range(10): y = x @ x          # warmup: cuBLAS init/autotune, else you
torch.cuda.synchronize()               # time initialization, not the GPU
import time; t0=time.time()
for _ in range(50): y = x @ x
torch.cuda.synchronize()
dt = time.time()-t0
print('achieved fp16 dense: %.0f TFLOPS (datasheet 2250 -> expect 60-80%%)'
      % (50*2*8192**3/dt/1e12))
import vllm; print('vllm       :', vllm.__version__)
PY
rc=\$?
if [ \$rc -ne 0 ]; then echo 'FATAL: torch/vLLM import or sm check failed'; exit 1; fi

echo
echo '--- [D] idle baseline (P_static) ---'
python3 -u energy_profile_load.py --mode idle --model '$MODEL' --dtype float16 \
    --run_dir '$RUNROOT/idle' --duration_s 15 --max_model_len 8192 \
    --gpu_memory_utilization 0.90 2>&1 | tee $RUNROOT/idle_full.log | tail -40

echo
echo '--- [E] ONE load run, concurrency 1 (the power-cap check) ---'
python3 -u energy_profile_load.py --mode load --model '$MODEL' --dtype float16 \
    --task alpaca --concurrency 1 --output_len 128 --duration_s 30 \
    --max_model_len 8192 --gpu_memory_utilization 0.90 \
    --run_dir '$RUNROOT/alpaca_c1' 2>&1 | tee $RUNROOT/load_full.log | tail -40

echo
echo '--- [F] analysis: binned table + coefficients ---'
python3 -u energy_model.py --two_term --run_dir '$RUNROOT/alpaca_c1' 2>&1 | tee $RUNROOT/model_full.log | tail -40
"

echo
echo "=== VERDICT ==="
python3 - "$RUNROOT" <<'PY'
import json, os, sys
root = sys.argv[1]
def j(p):
    try:    return json.load(open(p))
    except Exception: return {}
meta = j(os.path.join(root, "alpaca_c1", "run_meta.json"))
res  = j(os.path.join(root, "alpaca_c1", "results.json"))
idle = j(os.path.join(root, "idle", "results.json"))
cap  = meta.get("power_limit_w")
avg  = res.get("avg_power_window_w")
print(f"  GPU            : {meta.get('gpu_name')}")
print(f"  vLLM           : {meta.get('vllm_version')}  api={meta.get('engine_api')}")
print(f"  power cap      : {cap} W")
print(f"  idle P_static  : {meta.get('idle_power_w')} W")
print(f"  avg power @c=1 : {avg} W")
print(f"  tokens/s       : {res.get('aggregate_tokens_per_sec')}")
if cap and avg:
    frac = avg / cap
    print(f"  power / cap    : {frac:.1%}")
    if frac > 0.97:
        print("  >> GOTCHA #1 TRIPPED: saturating the cap at c=1. The linear model")
        print("     degenerates to E = P_cap*t. Do NOT run the sweep as-is.")
    else:
        print("  >> OK: headroom under the cap at c=1. Safe to sweep.")
else:
    print("  >> could not read power/cap -- inspect the log above.")
PY
echo "==== smoke end $(date) ===="
