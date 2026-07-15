#!/usr/bin/env python3
"""
run_load_sweep.py

Resume-safe orchestrator for the offered-load energy sweep.

Per model:
  1. idle baseline           -> anchors P_static
  2. prefill-ceiling baseline-> near-TDP dynamic ceiling
  3. concurrency-1 gate run  -> gate_dram.py MUST pass or the model is skipped
     (no point sweeping a broken DRAM counter — handoff gate #1)
  4. grid: WORKLOADS x CONCURRENCY, climbing concurrency until throughput
     saturates, then energy_model + validate_waveform on each run.

Each run writes power_trace.csv, iter_log.csv, dram_trace.csv, run_meta.json,
results.json, calibration.json, waveform_validation.{json,png}.

Run inside the GPU container (RunAI pod or SLURM+Apptainer allocation), NOT on
the login node. See README_HANDOFF.md.
"""

import json
import os
import subprocess
import sys
from datetime import datetime

PROFILE = "energy_profile_load.py"

# Only models present in ~/models.py (byte accounting needs a config).
# Qwen is fully open (proven); the others are gated — they run if the HF token
# has license access, else that model is skipped (per-model try/except).
# Override with SWEEP_MODELS="a,b" (e.g. a single-model Llama-3 retry).
MODELS = [
    "Qwen/Qwen2-7B-Instruct",
    "meta-llama/Meta-Llama-3-8B",
    "mistralai/Mistral-7B-v0.1",
]
if os.environ.get("SWEEP_MODELS"):
    MODELS = [m for m in os.environ["SWEEP_MODELS"].split(",") if m.strip()]

# (input_len, output_len); include a realistic ~2048-ctx point.
WORKLOADS = [
    (512, 128),
    (2048, 128),
]

CONCURRENCY = [1, 2, 4, 8, 16, 32, 64]

DTYPE = "float16"
DURATION_S = 45
IDLE_S = 15
CEILING_S = 15
GPU_ID = 0
MAX_MODEL_LEN = 4096
GPU_MEM_UTIL = 0.90

# Stop climbing concurrency once tok/s gains stall (avoids wasting the tail).
SATURATION_GAIN = 1.03      # <3% gain counts as a stall
SATURATION_STALLS = 2

# Override with SWEEP_LOG_ROOT (e.g. a separate dir for A100 so resume-skip logic
# doesn't treat H200 runs as already-done).
LOG_ROOT = os.environ.get("SWEEP_LOG_ROOT", "logs/load_sweep")


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run(cmd, log_file):
    print(f"[{ts()}] RUN: {' '.join(cmd)}", flush=True)
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    with open(log_file, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"[{ts()}] WARNING rc={proc.returncode} -> {log_file}", flush=True)
    return proc.returncode


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def load_complete(run_dir):
    r = load_json(os.path.join(run_dir, "results.json"))
    return r.get("aggregate_tokens_per_sec") is not None


def profile_cmd(mode, model, run_dir, **kw):
    cmd = [sys.executable, "-u", PROFILE, "--mode", mode, "--model", model,
           "--dtype", DTYPE, "--run_dir", run_dir, "--gpu_id", str(GPU_ID),
           "--max_model_len", str(MAX_MODEL_LEN),
           "--gpu_memory_utilization", str(GPU_MEM_UTIL)]
    for k, v in kw.items():
        cmd += [f"--{k}", str(v)]
    return cmd


def do_baselines(model, model_dir):
    idle_dir = os.path.join(model_dir, "idle")
    if not load_json(os.path.join(model_dir, "baselines.json")).get("idle_power_w"):
        run(profile_cmd("idle", model, idle_dir, duration_s=IDLE_S),
            os.path.join(idle_dir, "run.log"))
    ceil_dir = os.path.join(model_dir, "prefill_ceiling")
    if not load_json(os.path.join(model_dir, "baselines.json")).get(
            "prefill_ceiling_power_w"):
        run(profile_cmd("prefill_ceiling", model, ceil_dir, duration_s=CEILING_S,
                        input_len=2048, concurrency=8),
            os.path.join(ceil_dir, "run.log"))


def gate_concurrency_1(model, model_dir):
    """Run c=1 on the first workload and gate the DRAM counter. Returns True to
    proceed with the model's sweep."""
    in_len, out_len = WORKLOADS[0]
    gate_dir = os.path.join(model_dir, f"gate_c1_in{in_len}_out{out_len}")
    if not load_complete(gate_dir):
        run(profile_cmd("load", model, gate_dir, concurrency=1,
                        input_len=in_len, output_len=out_len,
                        duration_s=DURATION_S),
            os.path.join(gate_dir, "run.log"))
    if not load_complete(gate_dir):
        print(f"[{ts()}] [skip model] c=1 run did not complete: {model}",
              flush=True)
        return False

    run([sys.executable, "-u", "gate_dram.py", "--run_dir", gate_dir],
        os.path.join(gate_dir, "gate.log"))
    gate = load_json(os.path.join(gate_dir, "gate_result.json"))
    # c=1 run is a valid data point regardless of the gate; analyze it.
    analyze(gate_dir)

    if gate.get("flagged_missing"):
        print(f"[{ts()}] [skip model] not in models.py: {model}", flush=True)
        return False
    passed = gate.get("passed")
    if passed is True:
        print(f"[{ts()}] DRAM gate PASSED for {model} "
              f"(ratio={gate.get('ratio'):.2f})", flush=True)
    elif passed is False:
        # Counter present but ratio out of range -> genuinely broken; abort.
        print(f"[{ts()}] [skip model] DRAM gate FAILED ({model}): "
              f"ratio={gate.get('ratio')}. Fix counter before sweeping.",
              flush=True)
        return False
    else:
        # Counter unavailable (no dcgmi) -> non-fatal. Energy calibration uses
        # NVML power + analytic bytes and does not need the DRAM counter.
        print(f"[{ts()}] [warn] DRAM gate unavailable ({model}): "
              f"{gate.get('error')}. Proceeding (calibration doesn't need it).",
              flush=True)
    return True


def analyze(run_dir):
    run([sys.executable, "-u", "energy_model.py", "--run_dir", run_dir],
        os.path.join(run_dir, "energy_model.log"))
    run([sys.executable, "-u", "validate_waveform.py", "--run_dir", run_dir],
        os.path.join(run_dir, "validate.log"))


def sweep_model(model):
    short = model.split("/")[-1]
    model_dir = os.path.join(LOG_ROOT, short)
    os.makedirs(model_dir, exist_ok=True)
    print("=" * 70)
    print(f"[{ts()}] MODEL {short}")
    print("=" * 70)

    do_baselines(model, model_dir)
    if not gate_concurrency_1(model, model_dir):
        return

    for (in_len, out_len) in WORKLOADS:
        best_tps = 0.0
        stalls = 0
        for c in CONCURRENCY:
            if c == 1 and (in_len, out_len) == WORKLOADS[0]:
                # already ran as the gate; read its throughput for saturation.
                gate_dir = os.path.join(model_dir,
                                        f"gate_c1_in{in_len}_out{out_len}")
                tps = load_json(os.path.join(gate_dir, "results.json")).get(
                    "aggregate_tokens_per_sec") or 0.0
                best_tps = max(best_tps, tps)
                continue

            run_dir = os.path.join(model_dir, f"c{c}_in{in_len}_out{out_len}")
            if not load_complete(run_dir):
                run(profile_cmd("load", model, run_dir, concurrency=c,
                                input_len=in_len, output_len=out_len,
                                duration_s=DURATION_S),
                    os.path.join(run_dir, "run.log"))
            if not load_complete(run_dir):
                print(f"[{ts()}] [skip] run failed: {run_dir}", flush=True)
                continue
            analyze(run_dir)

            tps = load_json(os.path.join(run_dir, "results.json")).get(
                "aggregate_tokens_per_sec") or 0.0
            if tps < best_tps * SATURATION_GAIN:
                stalls += 1
                if stalls >= SATURATION_STALLS:
                    print(f"[{ts()}] throughput saturated at c={c} "
                          f"({in_len},{out_len}); tps={tps:.1f} "
                          f"best={best_tps:.1f}. Stopping climb "
                          f"(remaining concurrency levels skipped).", flush=True)
                    break
            else:
                stalls = 0
            best_tps = max(best_tps, tps)


def aggregate_summary():
    """Roll every completed run dir into one CSV: one row per operating point."""
    import csv
    rows = []
    for root, _, files in os.walk(LOG_ROOT):
        if "results.json" not in files:
            continue
        res = load_json(os.path.join(root, "results.json"))
        if res.get("aggregate_tokens_per_sec") is None:
            continue
        meta = load_json(os.path.join(root, "run_meta.json"))
        cal = load_json(os.path.join(root, "calibration.json"))
        wav = load_json(os.path.join(root, "waveform_validation.json")).get("metrics", {})
        rows.append({
            "run_dir": root,
            "model": meta.get("model"),
            "gpu": meta.get("gpu_name"),
            "concurrency": meta.get("concurrency_or_rate"),
            "input_len": meta.get("input_len"),
            "output_len": meta.get("output_len"),
            "tokens_per_s": res.get("aggregate_tokens_per_sec"),
            "avg_power_w": res.get("avg_power_window_w"),
            "idle_power_w": meta.get("idle_power_w"),
            "e_bit_j_per_byte": cal.get("e_bit_j_per_byte"),
            "e_bit_aggregate": cal.get("e_bit_aggregate"),
            "p_static_w": cal.get("p_static_w"),
            "r2_energy": cal.get("r2_energy"),
            "waveform_mape_pct": wav.get("mean_abs_pct_err"),
            "waveform_rmse_w": wav.get("rmse_w"),
            "n_bins": cal.get("n_bins"),
        })
    if not rows:
        print(f"[{ts()}] no completed runs to summarize", flush=True)
        return
    rows.sort(key=lambda r: (str(r["model"]), r["input_len"] or 0,
                             r["output_len"] or 0, r["concurrency"] or 0))
    out = os.path.join(LOG_ROOT, "sweep_summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[{ts()}] wrote {out} ({len(rows)} operating points)", flush=True)


def main():
    os.makedirs(LOG_ROOT, exist_ok=True)
    for model in MODELS:
        try:
            sweep_model(model)
        except Exception as e:
            print(f"[{ts()}] [error] model {model}: {e}", flush=True)
        aggregate_summary()   # refresh after each model so partial results are usable
    print("=" * 70)
    print(f"[{ts()}] SWEEP COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
