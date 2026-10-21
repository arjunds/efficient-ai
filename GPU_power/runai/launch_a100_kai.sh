#!/bin/bash
# Submit the A100 energy job through the KAI CLI (brachiolab/kai; ~/.kai/bin/kai).
#   bash runai/launch_a100_kai.sh                     # full job (probe, microbench, serve, analyze)
#   STAGES=probe bash runai/launch_a100_kai.sh        # ~2-5 min: image, GPU, NVML fields
# Follow: kai logs <name> ; status: kai list ; delete: kai delete <name>
# Queue: brachiolab-collaborators (1 guaranteed GPU). QUEUE=preemptible borrows idle GPUs
# but can be reclaimed (the serving sweep resumes; the microbench restarts).
set -eu
export PATH="$HOME/.kai/bin:$PATH"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
STAGES="${STAGES:-probe,microbench,serve,analyze}"
NAME="${NAME:-a100-energy-$(echo "$STAGES" | tr ',' '-' | cut -c1-20)}"
kai submit --name "$NAME" \
  -i "${IMAGE:-vllm/vllm-openai:v0.10.2}" \
  -g 1 --gpu-type "${GPU_TYPE:-a100}" -q "${QUEUE:-brachiolab-collaborators}" \
  --cpu 16 --mem 100Gi --large-shm --mount-home --keep \
  --workdir "$REPO" \
  -e STAGES="$STAGES" -e PROV_ARMS="${PROV_ARMS:-a100_graphs,a100_eager,a100_highconc}" \
  -- bash "$REPO/runai/a100_job.sh"
echo "submitted $NAME. Follow: kai logs $NAME   (also tee'd to $REPO/runai/a100_job_*.log)"
