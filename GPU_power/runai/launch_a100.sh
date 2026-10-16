#!/bin/bash
# Submit the A100 energy job to RunAI (KAI scheduler; node3 A100s). Requires a live
# login: `runai login` (token expires; symptom: oauth2 "invalid_grant").
#   bash runai/launch_a100.sh                       # full job (probe, microbench, serve, analyze)
#   STAGES=probe bash runai/launch_a100.sh          # 2-minute check of the pod/image/NVML
#   IMAGE=nvcr.io/nvidia/pytorch:24.01-py3 ...      # fallback image (then pip installs vllm==0.10.2)
# Watch: runai describe job $NAME ; runai logs $NAME -f ; delete: runai delete job $NAME
set -eu
REPO="$HOME/efficient-ai/GPU_power"
NAME="${NAME:-adsampat-a100-energy}"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.10.2}"
STAGES="${STAGES:-probe,microbench,serve,analyze}"
SETUP=""
case "$IMAGE" in nvcr.io/*) SETUP="pip install -q vllm==0.10.2 nvidia-ml-py && ";; esac
runai submit "$NAME" \
  -i "$IMAGE" \
  -g 1 --node-type a100 \
  --large-shm --memory 100G --cpu 16 \
  --run-as-user \
  -v "$REPO:/workspace" \
  -v "$HOME/models.py:/workspace/models.py" \
  -v "$HOME/.cache/huggingface:/hfcache" \
  -v /shared_data0:/shared_data0 \
  -e STAGES="$STAGES" -e PROV_ARMS="${PROV_ARMS:-a100_graphs,a100_eager,a100_highconc}" \
  --working-dir /workspace \
  --command -- bash -lc "${SETUP}bash /workspace/runai/a100_job.sh"
echo "submitted $NAME ($IMAGE). Follow: runai logs $NAME -f"
