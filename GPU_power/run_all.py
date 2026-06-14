#!/usr/bin/env python3

import os
import json
import subprocess
import sys
from datetime import datetime

PROFILE_SCRIPT = "energy_profile_vllm.py"
BACKEND = "vllm"

MODELS = [
    # "google/gemma-7b",
    # "mistralai/Mistral-7B-v0.1",
    # "deepseek-ai/deepseek-llm-7b-base",
    # "meta-llama/Meta-Llama-3-8B",
    "Qwen/Qwen2-7B-Instruct",
    # "mistralai/Mixtral-8x7B-Instruct-v0.1"
    # "microsoft/Phi-3-mini-4k-instruct"
]

TASKS = [
    "alpaca",
    # "sharegpt",
    # "math",
   # "multi_turn",
]

DTYPES = ["float16"]

LIMIT = 50
BATCH_SIZE = 1
MAX_NEW_TOKENS = 64

GPU_ID = 0
POWER_INTERVAL_MS = 20
NVTX_START = 0
NVTX_COUNT = 10

TENSOR_PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.90
MAX_MODEL_LEN = 2048
NCU_LAUNCH_COUNT = 20000
NCU_REPLAY_MODE = "application"
SAVE_PREDICTIONS = False

LOG_ROOT = "logs"


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run(cmd, log_file):
    print(f"[{ts()}] RUN:", " ".join(cmd), flush=True)

    with open(log_file, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

    if proc.returncode != 0:
        print(f"[{ts()}] WARNING: failed -> {log_file}", flush=True)


def ncu_complete(status_path):
    if not os.path.exists(status_path):
        return False

    try:
        with open(status_path) as f:
            status = json.load(f)
    except Exception:
        return False

    ncu_csv = status.get("ncu_csv")
    return (
        status.get("status") == "ok"
        and isinstance(ncu_csv, str)
        and os.path.exists(ncu_csv)
    )


def intensity_complete(intensity_path):
    required_keys = [
        "energy_window_j",
        "avg_power_window_w",
        "scalar_flops",
        "tensor_insts",
        "bytes_window",
        "nj_per_scalar_flop",
        "nj_per_byte",
    ]

    if not os.path.exists(intensity_path):
        return False

    try:
        with open(intensity_path) as f:
            intensity = json.load(f)
    except Exception:
        return False

    return all(key in intensity for key in required_keys)


for model in MODELS:
    model_name = model.split("/")[-1]

    for task in TASKS:
        for dtype in DTYPES:

            run_dir = os.path.join(LOG_ROOT, f"{BACKEND}_{model_name}_{task}_{dtype}")
            os.makedirs(run_dir, exist_ok=True)

            results_json = os.path.join(run_dir, "results.json")
            intensity_json = os.path.join(run_dir, "results_intensity.json")
            ncu_status_json = os.path.join(run_dir, "ncu_status.json")

            print("=" * 60)
            print(f"[{ts()}] MODEL={model_name} TASK={task} DTYPE={dtype}")
            print("=" * 60)

            # -------------------------
            # PASS A: ENERGY
            # -------------------------
            if os.path.exists(results_json):
                print("[skip] energy already done")
            else:
                cmd = [
                    sys.executable, "-u", PROFILE_SCRIPT,
                    "--mode", "energy",
                    "--model", model,
                    "--task", task,
                    "--dtype", dtype,
                    "--limit", str(LIMIT),
                    "--batch_size", str(BATCH_SIZE),
                    "--max_new_tokens", str(MAX_NEW_TOKENS),
                    "--run_dir", run_dir,
                    "--gpu_id", str(GPU_ID),
                    "--power_interval_ms", str(POWER_INTERVAL_MS),
                    "--nvtx_start", str(NVTX_START),
                    "--nvtx_count", str(NVTX_COUNT),
                    "--tensor_parallel_size", str(TENSOR_PARALLEL_SIZE),
                    "--gpu_memory_utilization", str(GPU_MEMORY_UTILIZATION),
                    "--max_model_len", str(MAX_MODEL_LEN),
                ]
                if SAVE_PREDICTIONS:
                    cmd.append("--save_predictions")

                run(cmd, os.path.join(run_dir, "energy.log"))

            # -------------------------
            # PASS B: NCU
            # -------------------------
            if ncu_complete(ncu_status_json):
                print("[skip] ncu already done")
            else:
                cmd = [
                    sys.executable, "-u", PROFILE_SCRIPT,
                    "--mode", "ncu",
                    "--model", model,
                    "--task", task,
                    "--dtype", dtype,
                    "--limit", str(LIMIT),
                    "--batch_size", str(BATCH_SIZE),
                    "--max_new_tokens", str(MAX_NEW_TOKENS),
                    "--run_dir", run_dir,
                    "--gpu_id", str(GPU_ID),
                    "--power_interval_ms", str(POWER_INTERVAL_MS),
                    "--nvtx_start", str(NVTX_START),
                    "--nvtx_count", str(NVTX_COUNT),
                    "--tensor_parallel_size", str(TENSOR_PARALLEL_SIZE),
                    "--gpu_memory_utilization", str(GPU_MEMORY_UTILIZATION),
                    "--max_model_len", str(MAX_MODEL_LEN),
                    "--ncu_launch_count", str(NCU_LAUNCH_COUNT),
                    "--ncu_replay_mode", NCU_REPLAY_MODE,
                ]
                if SAVE_PREDICTIONS:
                    cmd.append("--save_predictions")

                run(cmd, os.path.join(run_dir, "ncu.log"))

            # -------------------------
            # PASS C: INTENSITY
            # -------------------------
            if intensity_complete(intensity_json):
                print("[skip] intensity already done")
            else:
                cmd = [
                    sys.executable, "-u", PROFILE_SCRIPT,
                    "--mode", "intensity",
                    "--run_dir", run_dir,
                ]

                run(cmd, os.path.join(run_dir, "intensity.log"))

print("=" * 60)
print(f"[{ts()}] ALL RUNS COMPLETE")
print("=" * 60)
