#!/usr/bin/env python3

import os
import json
import pandas as pd

LOG_DIR = "logs"
OUT_CSV = "aggregate_results.csv"

records = []

def safe_load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}

for root, dirs, files in os.walk(LOG_DIR):

    # Only process folders that contain results.json
    if "results.json" not in files:
        continue

    results_path = os.path.join(root, "results.json")
    intensity_path = os.path.join(root, "results_intensity.json")

    data = safe_load_json(results_path)
    intensity = safe_load_json(intensity_path)

    # Skip empty / corrupted runs
    if not data:
        continue

    model = data.get("model")

    record = {
        # -------- identifiers --------
        "run_dir": root,
        "model": model,
        "model_short": model.split("/")[-1] if isinstance(model, str) else None,
        "task": data.get("task"),
        "dtype": data.get("dtype"),

        # -------- workload --------
        "batch_size": data.get("batch_size"),
        "max_new_tokens": data.get("max_new_tokens"),
        "examples": data.get("examples"),

        # -------- performance --------
        "seconds": data.get("seconds"),
        "avg_ttft_s": data.get("avg_ttft_s"),
        "avg_tokens_per_sec": data.get("avg_tokens_per_sec"),
        "aggregate_tokens_per_sec": data.get("aggregate_tokens_per_sec"),

        # -------- tokens --------
        "prompt_tokens": data.get("prompt_tokens"),
        "generated_tokens": data.get("generated_tokens"),

        # -------- energy (full run) --------
        "energy_total_j": data.get("energy_total_j"),
        "avg_power_w": data.get("avg_power_w"),
        "energy_per_token_j": data.get("energy_per_generated_token_j"),

        # -------- accuracy --------
        "accuracy": data.get("accuracy"),
        "num_accuracy_examples": data.get("num_accuracy_examples"),

        # -------- intensity (windowed) --------
        "energy_window_j": intensity.get("energy_window_j"),
        "avg_power_window_w": intensity.get("avg_power_window_w"),
        "scalar_flops": intensity.get("scalar_flops"),
        "tensor_insts": intensity.get("tensor_insts"),
        "bytes_window": intensity.get("bytes_window"),
        "nj_per_scalar_flop": intensity.get("nj_per_scalar_flop"),
        "nj_per_byte": intensity.get("nj_per_byte"),
    }

    records.append(record)

# -------------------------
# build dataframe
# -------------------------

if not records:
    print("[warn] no results found")
    exit(0)

df = pd.DataFrame(records)

# -------------------------
# cleaning / derived metrics
# -------------------------

# Avoid division errors
df["tokens_per_joule"] = df["generated_tokens"] / df["energy_total_j"]
df["tokens_per_joule"] = df["tokens_per_joule"].replace([float("inf")], None)

df["throughput_per_watt"] = df["aggregate_tokens_per_sec"] / df["avg_power_w"]

# -------------------------
# sort for readability
# -------------------------

df = df.sort_values(by=["model_short", "task", "dtype"])

# -------------------------
# save
# -------------------------

df.to_csv(OUT_CSV, index=False)

print(f"[saved] {OUT_CSV}")
print(df.head(10))