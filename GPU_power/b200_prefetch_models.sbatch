#!/bin/bash
#SBATCH --job-name=hf-prefetch
#SBATCH --partition=genoa-std-mem
#SBATCH --qos=genoa-std
#SBATCH --account=leebcc-systems-architecture-int
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5632M
#SBATCH --output=slurm_out/%x-%j.out
#
# CPU-only: pre-download model weights into the shared HF cache so the GPU job
# never spends B200 time pulling ~15 GB per model off the network. No GPU charge.
#
# Models are the two from the H200 set that are UNGATED (no HF token needed):
#   Qwen/Qwen2-7B-Instruct, mistralai/Mistral-7B-v0.1
# Override with: MODELS="a,b" sbatch b200_prefetch_models.sbatch
# Gated ones (Meta-Llama-3-8B, google/gemma-7b) need HF_TOKEN exported first.
set -u
export HF_HOME=/vast/projects/leebcc/systems-architecture-int/adsampat/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export HF_XET_CACHE=$HF_HOME/xet
export HF_HUB_ENABLE_HF_TRANSFER=0
MODELS="${MODELS:-Qwen/Qwen2-7B-Instruct,mistralai/Mistral-7B-v0.1}"

echo "==== prefetch start $(date) on $(hostname) ===="
echo "HF_HOME=$HF_HOME"
echo "models: $MODELS"

PYTHONPATH=/vast/projects/leebcc/systems-architecture-int/adsampat/pydeps python3 - "$MODELS" <<'PY'
import os, sys, time
from huggingface_hub import snapshot_download
# Weights only: skip the duplicate .bin/.pth copies when safetensors exist, and
# skip anything not needed to load the model.
IGNORE = ["*.pth", "*.msgpack", "*.h5", "original/*", "*.gguf", "*.bin"]  # *.bin: all
# four H200-set models ship safetensors, so the .bin copies are pure duplication
# (Mistral-7B-v0.1 is 28 GB with them, 15 GB without).
for mid in sys.argv[1].split(","):
    mid = mid.strip()
    if not mid:
        continue
    t0 = time.time()
    try:
        p = snapshot_download(mid, ignore_patterns=IGNORE,
                              max_workers=8, token=os.environ.get("HF_TOKEN"))
        n = sum(os.path.getsize(os.path.join(r, f))
                for r, _, fs in os.walk(p) for f in fs
                )  # follows symlinks into blobs/
        print(f"[ok] {mid}: {n/1e9:.1f} GB in {time.time()-t0:.0f}s -> {p}", flush=True)
    except Exception as e:
        print(f"[FAIL] {mid}: {type(e).__name__}: {e}", flush=True)
PY

echo "=== cache size ==="
du -sh "$HF_HOME" 2>/dev/null
echo "==== end $(date) ===="
