#!/bin/bash
#SBATCH --job-name=b200-large
#SBATCH --partition=dgx-b200
#SBATCH --qos=dgx
#SBATCH --account=leebcc-systems-architecture-int
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=02:30:00
#SBATCH --output=slurm_out/%x-%j.out
#
# THE DISCRIMINATING EXPERIMENT.
#
# The 7B sweep found the datasheet scaling law fails by +64..84%, but could not
# say whether the law is WRONG or whether the WORKLOAD simply never exercises a
# B200: realized bandwidth was ~31% of peak at every concurrency, and B200 was
# only 10% faster than H200 despite 1.67x the peak bandwidth.
#
# Qwen2-72B-Instruct separates those. Its weights are 145 GB, so ONE c=1 decode
# token sweeps 145 GB of HBM -- the roofline at 8 TB/s is 55 tok/s. If we now
# measure high bandwidth utilization AND the coefficients fall toward the
# scaling-law prediction, the law holds and the 7B result was a utilization
# artifact. If utilization rises but the coefficients DON'T move, the law is
# genuinely wrong and e_byte is ~invariant across GPU generations.
#
# Same family as the 7B we measured (Qwen2), ungated (no HF token), and TP=1 so
# the single-GPU NVML power reading stays valid -- tensor-parallel would
# undercount energy because the harness meters one --gpu_id.
#
# STORAGE: 145 GB does not fit the ~50 GB home quota. Weights live in the GROUP
# project cache ($BASE/hf_cache), pre-downloaded by b200_prefetch_models.sbatch
# on a CPU node so no B200 time is spent fetching them. That space is SHARED --
# delete the 72B from the cache once the measurement is done.
set -u
cd "$SLURM_SUBMIT_DIR"
BASE=/vast/projects/leebcc/systems-architecture-int/adsampat
SIF=$BASE/containers/vllm-0.10.2.sif
MODEL="${MODEL:-Qwen/Qwen2-72B-Instruct}"
LOGROOT="${RAGGED_LOG_ROOT:-logs/B200_large}"
mkdir -p "$LOGROOT" slurm_out

echo "=== $(date) node=$(hostname) model=$MODEL ==="
nvidia-smi --query-gpu=name,memory.total,power.limit --format=csv,noheader
df -h /tmp | tail -1

# ---------------------------------------------------------------------------
# Free piggy-backed check: are hardware perf counters usable HERE?
# HANDOFF gotcha #5 says DRAM counters were admin-blocked (ERR_NVGPUCTRPERM) on
# the old cluster, so the analytic byte model underpinning EVERY coefficient in
# this project has never been validated against hardware. A colleague's B200
# data in the group project space contains dram__bytes.sum, so it worked for
# them in July -- but that may have needed admin enablement. Costs ~5 seconds.
# ---------------------------------------------------------------------------
echo "=== [counters] availability (gotcha #5) ==="
for t in ncu dcgmi nsys; do
  p=$(command -v $t 2>/dev/null || true)
  echo "  $t: ${p:-NOT FOUND on this node}"
done
if command -v ncu >/dev/null 2>&1; then
  # The real question is not whether ncu exists but whether we are PERMITTED to
  # read the counters. Profile a trivial kernel and look for ERR_NVGPUCTRPERM.
  cat > /tmp/ncu_probe_$$.py <<'PY'
import torch
x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
for _ in range(3):
    y = x @ x
torch.cuda.synchronize()
PY
  source /vast/parcc/sw/etc/profile.d/Z98-lmod.sh; module load apptainer/1.4.1
  out=$(PYTHONNOUSERSITE=1 timeout 300 ncu --metrics dram__bytes_read.sum \
          --target-processes all --csv \
          apptainer exec --nv -B /tmp:/tmp "$SIF" python3 /tmp/ncu_probe_$$.py 2>&1 || true)
  rm -f /tmp/ncu_probe_$$.py
  if echo "$out" | grep -q "ERR_NVGPUCTRPERM"; then
    echo "  COUNTERS: BLOCKED (ERR_NVGPUCTRPERM) -- same as the old cluster"
  elif echo "$out" | grep -q "dram__bytes_read"; then
    echo "  COUNTERS: AVAILABLE -- ncu_dram_check.py is runnable here (a first)"
    echo "$out" | grep -E "dram__bytes_read" | head -3
  else
    echo "  COUNTERS: inconclusive; first lines of ncu output:"
    echo "$out" | head -5
  fi
else
  echo "  COUNTERS: ncu not installed on compute nodes"
fi

echo "=== weights ==="
# Weights are pre-staged into the GROUP PROJECT cache by b200_prefetch_models
# (a CPU job -- no GPU charge for a 145 GB download). Fail fast rather than
# silently downloading 145 GB while holding a B200.
MDIR="$BASE/hf_cache/hub/models--$(echo "$MODEL" | sed 's|/|--|')"
if [ ! -d "$MDIR" ]; then
  echo "FATAL: $MODEL not in the cache at $MDIR"
  echo "Run first (CPU, free):  MODELS=$MODEL sbatch b200_prefetch_models.sbatch"
  exit 1
fi
echo "  cached: $MDIR  ($(du -shL "$MDIR" 2>/dev/null | cut -f1))"
nsafet=$(find "$MDIR" -name "*.safetensors" | wc -l)
echo "  safetensors shards: $nsafet"
[ "$nsafet" -lt 1 ] && { echo "FATAL: no safetensors in cache"; exit 1; }

# Record hardware specs next to the logs for pool_gpus/scaling_check.
python3 - "$LOGROOT" <<'PY'
import json, subprocess, sys
o = subprocess.run(["nvidia-smi", "--query-gpu=name,power.limit",
                    "--format=csv,noheader,nounits"], capture_output=True,
                   text=True).stdout.strip().split("\n")[0].split(",")
json.dump({"bw": 8.0e12, "peak_flops": 2250e12, "p_cap": float(o[1]),
           "gpu_name": o[0].strip(),
           "note": "peak_flops is DENSE fp16 (NVIDIA sparse figure / 2)"},
          open(sys.argv[1] + "/specs.json", "w"), indent=2)
PY

export APPTAINERENV_HF_HOME=$BASE/hf_cache
export APPTAINERENV_HF_HUB_CACHE=$BASE/hf_cache/hub
export APPTAINERENV_HF_HUB_OFFLINE=1          # weights are already local
export APPTAINERENV_PYTHONNOUSERSITE=1        # host ~/.local numpy breaks vLLM
export APPTAINERENV_TOKENIZERS_PARALLELISM=false
# Attention fix for long sequences on Blackwell -- set by b200_attnfix results.
# Empty => container default (the config that crashed 5/8 sharegpt runs).
[ -n "${ATTN_FIX_VAR:-}" ] && export APPTAINERENV_${ATTN_FIX_VAR}="${ATTN_FIX_VAL:-0}"

# 72B leaves only ~28 GB for KV after 145 GB of weights, so cap context at 4096
# (c=16 x 4096 = 65k tokens < the ~84k the KV cache holds) and stop at c=16.
export APPTAINERENV_RAGGED_LOG_ROOT="$LOGROOT"
export APPTAINERENV_RAGGED_MODELS="$MODEL"
export APPTAINERENV_RAGGED_TASKS="${RAGGED_TASKS:-alpaca,sharegpt}"
export APPTAINERENV_RAGGED_CONCURRENCY="${RAGGED_CONCURRENCY:-1,4,16}"
export APPTAINERENV_RAGGED_POISSON=""
export APPTAINERENV_RAGGED_DURATION_S="${RAGGED_DURATION_S:-45}"
export APPTAINERENV_RAGGED_MAX_MODEL_LEN="${RAGGED_MAX_MODEL_LEN:-4096}"

source /vast/parcc/sw/etc/profile.d/Z98-lmod.sh
module load apptainer/1.4.1

apptainer exec --nv \
  -B "$SLURM_SUBMIT_DIR:/workspace" -B "$BASE/hf_cache:$BASE/hf_cache" \
  "$SIF" bash -lc '
    set -u; cd /workspace
    test -s prompts_alpaca.json && test -s prompts_sharegpt.json \
      || { echo "FATAL: prompt pools missing"; exit 1; }
    python3 -u run_ragged_sweep.py 2>&1
  '

echo
echo "=== utilization check: did the 72B actually saturate the B200? ==="
python3 - "$LOGROOT" "$MODEL" <<'PY'
import glob, json, os, sys
root, mid = sys.argv[1], sys.argv[2]
sys.path.insert(0, ".")
from models import model_from_hf_id
W = model_from_hf_id(mid).active_params() * 2
print(f"  weights = {W/1e9:.1f} GB -> c=1 roofline at 8 TB/s = {8e12/W:.1f} tok/s")
print(f"{'run':<34}{'tok/s':>9}{'realBW TB/s':>13}{'%peak':>8}{'avgW':>8}{'%cap':>7}")
for rd in sorted(glob.glob(os.path.join(root, "*", "*"))):
    rp = os.path.join(rd, "results.json")
    if not os.path.exists(rp):
        continue
    r = json.load(open(rp))
    t = r.get("aggregate_tokens_per_sec") or 0
    p = r.get("avg_power_window_w") or 0
    bw = t * W                      # decode-dominated approximation
    print(f"{os.path.relpath(rd, root):<34}{t:>9.1f}{bw/1e12:>13.2f}"
          f"{100*bw/8e12:>7.0f}%{p:>8.0f}{100*p/1000:>6.0f}%")
print("\n  (7B reference: ~31% of peak bandwidth, B200 only 10% faster than H200)")
PY
echo "==== large run end $(date) ===="
