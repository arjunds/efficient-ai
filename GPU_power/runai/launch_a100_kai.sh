#!/bin/bash
# Submit the A100 energy job through the KAI CLI (brachiolab/kai; ~/.kai/bin/kai).
#   bash runai/launch_a100_kai.sh                     # full job (probe, microbench, serve, analyze)
#   STAGES=probe bash runai/launch_a100_kai.sh        # ~2-5 min: image, GPU, NVML fields
# Follow: kai logs <name> ; status: kai list ; delete: kai delete <name>
# /shared_data0 is mounted because ~/.cache/huggingface/hub is a symlink into it.
# Queue: brachiolab-collaborators (1 guaranteed GPU). QUEUE=preemptible borrows idle GPUs
# but can be reclaimed (the serving sweep resumes; the microbench restarts).
set -eu
export PATH="$HOME/.kai/bin:$PATH"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
STAGES="${STAGES:-probe,microbench,serve,analyze}"
NAME="${NAME:-a100-energy-$(echo "$STAGES" | tr ',' '-' | cut -c1-20)}"
kai submit --name "$NAME" \
  -i "${IMAGE:-vllm/vllm-openai:v0.10.2}" \
  -g "${GPUS:-1}" --gpu-type "${GPU_TYPE:-a100}" -q "${QUEUE:-brachiolab-collaborators}" \
  --cpu "${CPU:-16}" --mem "${MEM:-100Gi}" --large-shm --mount-home --keep \
  -v /shared_data0:/shared_data0 \
  --workdir "$REPO" \
  -e STAGES="$STAGES" -e PROV_ARMS="${PROV_ARMS:-a100_graphs,a100_eager,a100_highconc}" \
  -e PROV_LOG_ROOT="${PROV_LOG_ROOT:-logs/A100_prov}" -e MB_OUT="${MB_OUT:-microbench/results_A100_steady.json}" \
  -e GPU_TAG="${GPU_TAG:-A100}" \
  -- bash "$REPO/runai/${JOB_SCRIPT:-gpu_job.sh}"
echo "submitted $NAME. Follow: kai logs $NAME   (also tee'd to $REPO/runai/a100_job_*.log)"
