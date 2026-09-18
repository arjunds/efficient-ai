#!/bin/bash
#SBATCH --job-name=b200-sweep
#SBATCH --partition=dgx-b200
#SBATCH --qos=dgx
#SBATCH --account=leebcc-systems-architecture-int
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=02:00:00
#SBATCH --output=slurm_out/%x-%j.out
#
# TIER 2 -- the actual measurement. ONE B200, ~2 h.
# ONLY RUN THIS AFTER b200_smoke.sbatch REPORTS "OK: headroom under the cap".
#
# Scope (agreed): 2 models x 2 tasks x c={1,4,16,64} = 16 load runs + 2 idle.
# Both models are from the original H200 set AND ungated (no HF token), so the
# B200 coefficients are directly comparable to the H200 ones:
#     Qwen/Qwen2-7B-Instruct   mistralai/Mistral-7B-v0.1
# Poisson/open-loop points are skipped -- redundant for coefficient fitting.
#
# BOTH tasks are mandatory: alpaca (short, decode-heavy, low arithmetic
# intensity) + sharegpt (long, prefill-heavy, high AI). With only one, bytes and
# flops are collinear and e_bit/e_flop cannot separate (HANDOFF gotcha #2).
#
# Resume-safe: re-submitting skips runs that already completed.
set -u
cd "$SLURM_SUBMIT_DIR"
BASE=/vast/projects/leebcc/systems-architecture-int/adsampat
CTR=$BASE/containers
SIF="${SIF:-}"
[ -z "$SIF" ] && [ -s "$CTR/vllm-0.10.2.sif" ] && SIF="$CTR/vllm-0.10.2.sif"
[ -z "$SIF" ] && [ -s "$CTR/vllm-0.11.0.sif" ] && SIF="$CTR/vllm-0.11.0.sif"
[ -z "$SIF" ] && { echo "FATAL: no .sif in $CTR"; exit 1; }

LOGROOT="${RAGGED_LOG_ROOT:-logs/B200}"
mkdir -p "$LOGROOT" slurm_out

echo "==================================================================="
echo " B200 SWEEP  $(date)  node=$(hostname)  sif=$(basename "$SIF")"
echo "==================================================================="
nvidia-smi --query-gpu=name,power.limit,memory.total,driver_version --format=csv
# Record the true datasheet-relevant facts alongside the logs, so pool_gpus.py
# can pick up specs without relying on its built-in table.
python3 - "$LOGROOT" <<'PY'
import json, subprocess, sys
q = "name,power.limit,power.max_limit,memory.total,clocks.max.sm,clocks.max.mem"
out = subprocess.run(["nvidia-smi", f"--query-gpu={q}",
                      "--format=csv,noheader,nounits"],
                     capture_output=True, text=True).stdout.strip().split("\n")[0]
f = [x.strip() for x in out.split(",")]
# B200 SXM datasheet: 8.0 TB/s HBM3e, 2.25 PFLOPS DENSE fp16 (NVIDIA quotes 4.5
# PFLOPS *with 2:1 sparsity* -- halved here to match the H200 990 TFLOPS dense
# anchor). p_cap is read from the hardware, not assumed.
json.dump({"bw": 8.0e12, "peak_flops": 2250e12, "p_cap": float(f[1]),
           "gpu_name": f[0], "power_max_limit_w": float(f[2]),
           "memory_total_mib": float(f[3]),
           "note": "peak_flops is DENSE fp16 (sparse/2)"},
          open(sys.argv[1] + "/specs.json", "w"), indent=2)
print("wrote", sys.argv[1] + "/specs.json:", f)
PY

export APPTAINERENV_HF_HOME=$BASE/hf_cache
export APPTAINERENV_HF_HUB_CACHE=$BASE/hf_cache/hub
export APPTAINERENV_HF_XET_CACHE=$BASE/hf_cache/xet
# The host $HOME is bind-mounted into the container, so a "pip install --user"
# numpy on the host SHADOWS the container's numpy and breaks vLLM (numba needs
# numpy<=2.2; the host had 2.5.3). Ignore host user-site packages entirely.
export APPTAINERENV_PYTHONNOUSERSITE=1
# Blackwell long-sequence fix. vllm/utils/flashinfer.py::use_trtllm_attention()
# picks the TRTLLM kernel PER BATCH via `num_tokens <= 256`, so variable-length
# sharegpt traffic makes vLLM switch kernels mid-run -> "illegal memory access"
# (killed 5 of 8 sharegpt runs on 2026-09-17). Pinning it either way fixes it;
# =0 is both correct and fastest. Verified: 0 crashes in the 72B run.
export APPTAINERENV_VLLM_USE_TRTLLM_ATTENTION=0
export APPTAINERENV_TOKENIZERS_PARALLELISM=false
[ -n "${HF_TOKEN:-}" ] && export APPTAINERENV_HF_TOKEN="$HF_TOKEN"

# Sweep grid -> run_ragged_sweep.py env overrides.
export APPTAINERENV_RAGGED_LOG_ROOT="$LOGROOT"
export APPTAINERENV_RAGGED_MODELS="${RAGGED_MODELS:-Qwen/Qwen2-7B-Instruct,mistralai/Mistral-7B-v0.1}"
export APPTAINERENV_RAGGED_TASKS="${RAGGED_TASKS:-alpaca,sharegpt}"
export APPTAINERENV_RAGGED_CONCURRENCY="${RAGGED_CONCURRENCY:-1,4,16,64}"
export APPTAINERENV_RAGGED_POISSON=""          # empty => skip open-loop points
export APPTAINERENV_RAGGED_DURATION_S="${RAGGED_DURATION_S:-45}"

source /vast/parcc/sw/etc/profile.d/Z98-lmod.sh
module load apptainer/1.4.1

apptainer exec --nv \
  -B "$SLURM_SUBMIT_DIR:/workspace" \
  -B "$BASE/hf_cache:$BASE/hf_cache" \
  "$SIF" bash -lc "
    set -u; cd /workspace
    # NOTE: no pydeps on PYTHONPATH here -- 'datasets' drags huggingface-hub 1.x,
    # which breaks transformers/vLLM (gotcha #3). Prompts are read from the
    # pre-dumped prompts_*.json with stdlib json.
    test -s prompts_alpaca.json   || { echo 'FATAL: prompts_alpaca.json missing';   exit 1; }
    test -s prompts_sharegpt.json || { echo 'FATAL: prompts_sharegpt.json missing'; exit 1; }
    python3 -u run_ragged_sweep.py 2>&1
  "

echo
echo "=== per-run coefficient fits ==="
python3 - "$LOGROOT" <<'PY'
import csv, glob, json, os, sys
root = sys.argv[1]
print(f"{'run':<46}{'bins':>6}{'idle W':>8}")
n = 0
for t in sorted(glob.glob(os.path.join(root, "*", "*", "binned_table.csv"))):
    rd = os.path.dirname(t)
    meta = json.load(open(os.path.join(rd, "run_meta.json")))
    rows = sum(1 for _ in csv.DictReader(open(t)))
    print(f"{os.path.relpath(rd, root):<46}{rows:>6}{meta.get('idle_power_w', 0):>8.1f}")
    n += 1
print(f"\n{n} runs with binned_table.csv")
PY
echo "==== sweep end $(date) ===="
