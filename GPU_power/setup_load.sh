#!/bin/bash
# Install the offered-load harness deps INSIDE the GPU container (RunAI pod or
# SLURM+Apptainer shell). Run from the harness directory.
#
# vLLM itself is intentionally NOT pinned here: install it exactly the way you
# already do for energy_profile_vllm.py (that's the known-good build for this
# cluster's CUDA). This script only adds the extra bits the load harness needs.
set -e
python -m pip install --upgrade pip

# Light deps used by the loggers / model / plots.
python -m pip install "nvidia-ml-py>=12.535.0" matplotlib "pandas>=1.5,<3" || true

# If vLLM is not yet present, install your known-good version. Uncomment and pin
# to whatever worked for energy_profile_vllm.py on this node:
# python -m pip install vllm==0.10.0

echo "--- versions ---"
python - <<'PY'
for m in ("torch","vllm","transformers","pynvml","matplotlib","pandas"):
    try:
        mod=__import__(m); print(m, getattr(mod,"__version__","?"))
    except Exception as e:
        print(m, "MISSING:", e)
PY
command -v dcgmi >/dev/null && echo "dcgmi: $(command -v dcgmi)" || \
  echo "dcgmi: NOT FOUND (DRAM counter will be disabled -> gate #1 cannot pass)"
