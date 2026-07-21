#!/usr/bin/env python3
"""
run_ragged_sweep.py

Realistic ragged-workload sweep for the two-term energy fit. Drives real
variable-length prompts (alpaca + sharegpt) at a range of concurrency (+ a couple
Poisson arrival rates) across all 4 models, so the pooled per-bin data spans low
arithmetic intensity (c=1 pure decode) to high (sharegpt prefill-heavy / high
concurrency) — the spread needed to separate e_bit from e_flop.

Per run: energy_profile_load --task ... (variable-length prompts, natural EOS
capped) -> power_trace, iter_log, run_meta. Then energy_model --two_term exports
binned_table.csv per run. Per model: fit_two_term pools all its bins -> the
identifiable two-coefficient fit.

Resume-safe. Run in-container on H200 (or A100) via ragged_sweep.sbatch.
"""

import glob
import json
import os
import subprocess
import sys
from datetime import datetime

import run_load_sweep as base   # reuse run(), load_json(), load_complete()

MODELS = [
    "Qwen/Qwen2-7B-Instruct",
    "meta-llama/Meta-Llama-3-8B",
    "mistralai/Mistral-7B-v0.1",
    "google/gemma-7b",
]
TASKS = ["alpaca", "sharegpt"]
CONCURRENCY = [1, 4, 16, 64]     # spans memory-bound -> batched
POISSON_RATES = [8]              # one open-loop point per task
OUTPUT_CAP = 256                 # natural EOS, capped
DURATION_S = 45
IDLE_S = 15
GPU_ID = 0
MAX_MODEL_LEN = 8192             # room for long sharegpt prompts
GPU_MEM_UTIL = 0.90
LOG_ROOT = os.environ.get("RAGGED_LOG_ROOT") or "logs/ragged"
if os.environ.get("RAGGED_MODELS"):
    MODELS = [m for m in os.environ["RAGGED_MODELS"].split(",") if m.strip()]


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def profile_cmd(model, run_dir, **kw):
    cmd = [sys.executable, "-u", "energy_profile_load.py", "--mode", "load",
           "--model", model, "--dtype", "float16", "--run_dir", run_dir,
           "--gpu_id", str(GPU_ID), "--max_model_len", str(MAX_MODEL_LEN),
           "--gpu_memory_utilization", str(GPU_MEM_UTIL),
           "--output_len", str(OUTPUT_CAP)]
    for k, v in kw.items():
        cmd += [f"--{k}", str(v)]
    return cmd


def analyze(run_dir):
    # two-term export (binned_table.csv + per-run two-term fit) + one-term + waveform
    base.run([sys.executable, "-u", "energy_model.py", "--two_term",
              "--run_dir", run_dir], os.path.join(run_dir, "two_term.log"))
    base.run([sys.executable, "-u", "energy_model.py", "--run_dir", run_dir,
              "--phase", "decode"], os.path.join(run_dir, "energy_model.log"))
    base.run([sys.executable, "-u", "validate_waveform.py", "--run_dir", run_dir],
             os.path.join(run_dir, "validate.log"))


def do_idle(model, model_dir):
    idle_dir = os.path.join(model_dir, "idle")
    if not base.load_json(os.path.join(model_dir, "baselines.json")).get("idle_power_w"):
        base.run([sys.executable, "-u", "energy_profile_load.py", "--mode", "idle",
                  "--model", model, "--dtype", "float16", "--run_dir", idle_dir,
                  "--gpu_id", str(GPU_ID), "--max_model_len", str(MAX_MODEL_LEN),
                  "--gpu_memory_utilization", str(GPU_MEM_UTIL),
                  "--duration_s", str(IDLE_S)],
                 os.path.join(idle_dir, "run.log"))


def sweep_model(model):
    short = model.split("/")[-1]
    model_dir = os.path.join(LOG_ROOT, short)
    os.makedirs(model_dir, exist_ok=True)
    print("=" * 70); print(f"[{ts()}] RAGGED MODEL {short}"); print("=" * 70)

    do_idle(model, model_dir)
    if not base.load_json(os.path.join(model_dir, "baselines.json")).get("idle_power_w"):
        print(f"[{ts()}] [skip model] no idle baseline: {model}", flush=True)
        return

    for task in TASKS:
        for c in CONCURRENCY:
            rd = os.path.join(model_dir, f"{task}_c{c}")
            if not base.load_complete(rd):
                base.run(profile_cmd(model, rd, task=task, concurrency=c,
                                     duration_s=DURATION_S),
                         os.path.join(rd, "run.log"))
            if base.load_complete(rd):
                analyze(rd)
            else:
                print(f"[{ts()}] [skip] failed: {rd}", flush=True)
        for rate in POISSON_RATES:
            rd = os.path.join(model_dir, f"{task}_poisson{rate}")
            if not base.load_complete(rd):
                base.run(profile_cmd(model, rd, task=task, arrival_rate=rate,
                                     duration_s=DURATION_S),
                         os.path.join(rd, "run.log"))
            if base.load_complete(rd):
                analyze(rd)

    # pooled two-term fit across all this model's operating points
    fit_out = os.path.join(model_dir, "two_term_fit.json")
    base.run([sys.executable, "-u", "fit_two_term.py",
              "--glob", os.path.join(model_dir, "*"), "--out", fit_out],
             os.path.join(model_dir, "fit_two_term.log"))
    fit = base.load_json(fit_out)
    print(f"[{ts()}] {short} two-term: e_bit={fit.get('e_bit_j_per_byte')} "
          f"e_flop_pJ={fit.get('e_flop_pJ_per_flop')} R2={fit.get('r2_dynamic_energy')} "
          f"identifiable={fit.get('identifiable')} AI_span={fit.get('ai_span_ratio')}",
          flush=True)


def summarize():
    rows = []
    for fp in glob.glob(os.path.join(LOG_ROOT, "*", "two_term_fit.json")):
        d = base.load_json(fp)
        rows.append((os.path.basename(os.path.dirname(fp)), d))
    if not rows:
        return
    with open(os.path.join(LOG_ROOT, "two_term_summary.csv"), "w") as f:
        f.write("model,e_bit_j_per_byte,e_flop_pJ_per_flop,p_none,r2_dyn,"
                "identifiable,bytes_flops_r,ai_span_ratio,n_bins\n")
        for m, d in sorted(rows):
            f.write(f"{m},{d.get('e_bit_j_per_byte')},{d.get('e_flop_pJ_per_flop')},,"
                    f"{d.get('r2_dynamic_energy')},{d.get('identifiable')},"
                    f"{d.get('bytes_flops_pearson')},{d.get('ai_span_ratio')},"
                    f"{d.get('n_bins')}\n")
    print(f"[{ts()}] wrote {LOG_ROOT}/two_term_summary.csv", flush=True)


def main():
    os.makedirs(LOG_ROOT, exist_ok=True)
    for model in MODELS:
        try:
            sweep_model(model)
        except Exception as e:
            print(f"[{ts()}] [error] {model}: {e}", flush=True)
        summarize()
    print("=" * 70); print(f"[{ts()}] RAGGED SWEEP COMPLETE"); print("=" * 70)


if __name__ == "__main__":
    main()
