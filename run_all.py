#!/usr/bin/env python3

import os
import subprocess
from datetime import datetime

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

LOG_ROOT = "logs"


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run(cmd, log_file):
    print(f"[{ts()}] RUN:", " ".join(cmd), flush=True)

    with open(log_file, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

    if proc.returncode != 0:
        print(f"[{ts()}] WARNING: failed -> {log_file}", flush=True)


for model in MODELS:
    model_name = model.split("/")[-1]

    for task in TASKS:
        for dtype in DTYPES:

            run_dir = os.path.join(LOG_ROOT, f"{model_name}_{task}_{dtype}")
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
                    "python", "-u", "energy_profile.py",
                    "--mode", "energy",
                    "--model", model,
                    "--task", task,
                    "--dtype", dtype,
                    "--limit", str(LIMIT),
                    "--batch_size", str(BATCH_SIZE),
                    "--max_new_tokens", str(MAX_NEW_TOKENS),
                    "--run_dir", run_dir,
                ]

                run(cmd, os.path.join(run_dir, "energy.log"))

            # -------------------------
            # PASS B: NCU
            # -------------------------
            if os.path.exists(ncu_status_json):
                print("[skip] ncu already done")
            else:
                cmd = [
                    "python", "energy_profile.py",
                    "--mode", "ncu",
                    "--model", model,
                    "--task", task,
                    "--dtype", dtype,
                    "--limit", str(LIMIT),
                    "--batch_size", str(BATCH_SIZE),
                    "--max_new_tokens", str(MAX_NEW_TOKENS),
                    "--run_dir", run_dir,
                ]

                run(cmd, os.path.join(run_dir, "ncu.log"))

            # -------------------------
            # PASS C: INTENSITY
            # -------------------------
            if os.path.exists(intensity_json):
                print("[skip] intensity already done")
            else:
                cmd = [
                    "python", "energy_profile.py",
                    "--mode", "intensity",
                    "--run_dir", run_dir,
                ]

                run(cmd, os.path.join(run_dir, "intensity.log"))

print("=" * 60)
print(f"[{ts()}] ALL RUNS COMPLETE")
print("=" * 60)