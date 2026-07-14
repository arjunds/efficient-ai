#!/bin/bash
# Launch the offered-load energy harness under RunAI (the known-good path).
# Mirrors ~/sweep_job but mounts the version-controlled harness dir.
#
# Usage:
#   ./launch_runai.sh                 # submit an interactive pod (sleep infinity)
# then:
#   runai exec -it adsampat-energy-load -- bash
#   cd /workspace && bash setup_load.sh && python run_load_sweep.py
set -e

HARNESS_DIR="${HARNESS_DIR:-$HOME/efficient-ai/GPU_power}"
NAME="${NAME:-adsampat-energy-load}"

runai submit --name "$NAME" \
  -i nvcr.io/nvidia/pytorch:24.01-py3 \
  -g 1 \
  --node-type a100 \
  --large-shm \
  --memory 100G \
  -v "${HARNESS_DIR}:/workspace/" \
  -v "$HOME/models.py:/workspace/models.py" \
  -v /shared_data0/:/shared_data0 \
  -e TORCHINDUCTOR_CACHE_DIR="/workspace/torch_cache" \
  -e HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-}" \
  --working-dir /workspace \
  --interactive \
  -- sleep infinity

cat <<EOF

Pod '$NAME' submitted. Next:
  runai exec -it $NAME -- bash
  cd /workspace
  bash setup_load.sh                     # install deps (+ your known-good vLLM)
  python probe_env.py --dump-vllm-api    # confirm GPU/vLLM/dcgmi + scheduler API
  python energy_profile_load.py --mode self_test --run_dir logs/selftest
  python energy_model.py --run_dir logs/selftest
  python validate_waveform.py --run_dir logs/selftest   # expect high R^2
  python run_load_sweep.py               # the real sweep
EOF
