#!/usr/bin/env python3
"""
run_provisioning_sweep.py — B200 runs for the provisioning / "LIMINAL extended"
questions (HANDOFF_B200_PROVISIONING.md). Resume-safe; one arm per directory:

  relog     eager (as every run before 2026-10-16), Qwen2-7B, c=1..64, both tasks.
            Same grid as logs/B200 but logged with the energy counter and
            POWER_INSTANT and per-step prefix-cache hits -> a smear-free per-bin
            refit on serving data (checks e_gemm 0.94 pJ, e_kv ~ e_wbyte).
  graphs    CUDA graphs ON (--no_enforce_eager), Qwen2-7B c=1..64 both tasks and
            Qwen2.5-32B alpaca c=1,16,64. Every earlier serving run was eager, so
            how much of the "fixed per-step overhead" is eager-mode launch cost?
  highconc  CUDA graphs, max_num_seqs=512, c=128,256,512 (7B both tasks, 32B
            alpaca). Tests the model's extrapolated claim that B200 only beats
            H200 per token at very large batch; 32B at c=512 approaches the KV
            limit (capacity / preemption).
  prefill   CUDA graphs, unique synthetic prompts of 2048 / 8192 tokens, 16
            output tokens, c=4,16, 8192-token step budget: prefill-dominated
            serving, where B200 GEMMs run at the 1000 W cap.
  tp2       (2-GPU job) TP=2, CUDA graphs, both GPUs metered: Qwen2.5-32B and
            Qwen2-72B alpaca c=1,16,64.
  tp1ref    (2-GPU job) TP=1 72B with CUDA graphs, c=1,16 (comparator for tp2;
            72B TP=1 holds only ~16 x 4k contexts of KV).

  PROV_ARMS=relog,graphs,highconc,prefill python3 run_provisioning_sweep.py
Env: PROV_LOG_ROOT (default logs/B200_prov), PROV_DURATION_S (45), PROV_IDLE_S (15),
     PROV_SKIP_MODELS (comma list of HF ids to skip, e.g. an uncached 72B).
"""
import json
import os
import sys
from datetime import datetime

import run_load_sweep as base   # run(), load_json(), load_complete()

LOG_ROOT = os.environ.get("PROV_LOG_ROOT", "logs/B200_prov")
DURATION_S = float(os.environ.get("PROV_DURATION_S", 45))
IDLE_S = float(os.environ.get("PROV_IDLE_S", 15))
OUTPUT_CAP = 256
Q05 = "Qwen/Qwen2.5-0.5B-Instruct"
Q14 = "Qwen/Qwen2.5-14B-Instruct"
Q7, Q32, Q72 = "Qwen/Qwen2-7B-Instruct", "Qwen/Qwen2.5-32B-Instruct", "Qwen/Qwen2-72B-Instruct"
GRAPHS = {"no_enforce_eager": None}

# arm -> list of (model, engine flags, [(tag, load flags), ...])
def _ragged(tasks, concs):
    return [(f"{t}_c{c}", {"task": t, "concurrency": c}) for t in tasks for c in concs]


ARMS = {
    "relog": [(Q7, {}, _ragged(["alpaca", "sharegpt"], [1, 4, 16, 64]))],
    "graphs": [(Q7, GRAPHS, _ragged(["alpaca", "sharegpt"], [1, 4, 16, 64])),
               (Q32, GRAPHS, _ragged(["alpaca"], [1, 16, 64]))],
    "highconc": [(Q7, dict(GRAPHS, max_num_seqs=512), _ragged(["alpaca", "sharegpt"], [128, 256, 512])),
                 (Q32, dict(GRAPHS, max_num_seqs=512), _ragged(["alpaca"], [128, 256, 512]))],
    "prefill": [(Q7, dict(GRAPHS, max_num_batched_tokens=8192, max_model_len=10240),
                 [(f"in{n}_c{c}", {"input_len": n, "output_len": 16, "concurrency": c,
                                   "unique_prompts": None})
                  for n in (2048, 8192) for c in (4, 16)])],
    "tp2": [(Q32, dict(GRAPHS, tensor_parallel_size=2), _ragged(["alpaca"], [1, 16, 64])),
            (Q72, dict(GRAPHS, tensor_parallel_size=2, max_model_len=4096), _ragged(["alpaca"], [1, 16, 64]))],
    "tp1ref": [(Q72, dict(GRAPHS, max_model_len=4096), _ragged(["alpaca"], [1, 16]))],
    # Long-context sweep (HANDOFF_B200_LONGCTX.md): context varies at FIXED concurrency,
    # so KV bytes per token grow ~30x while weight bytes and GEMM FLOPs per token stay
    # fixed. That breaks the KV-vs-FLOP collinearity of every concurrency sweep.
    # Unique synthetic prompts (no prefix-cache hits), 512 decode tokens per request.
    "longctx": [(Q7, dict(GRAPHS, max_model_len=32768),
                 [(f"ctx{n}_c{c}", {"input_len": n, "output_len": 512, "concurrency": c,
                                    "unique_prompts": None})
                  for c in (4, 16) for n in (1024, 4096, 8192, 16384, 30720)])],
    "longctx32": [(Q32, dict(GRAPHS, max_model_len=32768),
                   [(f"ctx{n}_c4", {"input_len": n, "output_len": 512, "concurrency": 4,
                                    "unique_prompts": None})
                    for n in (1024, 8192, 30720)])],
    # Logger-overhead A/B: same as relog alpaca c=1/c=64 but with NVML_LOGGER_SLIM=1
    # (power, POWER_INSTANT and energy counter only). Compare tok/s with relog.
    "loggerab": [(Q7, {}, _ragged(["alpaca"], [1, 64]))],
    # A100 80GB PCIe via RunAI/KAI (runai/a100_job.sh). Same grids as the B200
    # graphs/relog arms so the GPUs compare directly; 14B instead of 32B (32B fp16
    # leaves ~no KV room on 80 GB). The A100-PCIe sits at its 300 W TDP under load,
    # so these runs test the power roofline as much as the coefficients.
    "a100_graphs": [(Q7, GRAPHS, _ragged(["alpaca", "sharegpt"], [1, 4, 16, 64])),
                    (Q14, GRAPHS, _ragged(["alpaca"], [1, 16, 64]))],
    "a100_eager": [(Q7, {}, _ragged(["alpaca"], [1, 16, 64]))],
    "a100_highconc": [(Q7, dict(GRAPHS, max_num_seqs=256), _ragged(["alpaca", "sharegpt"], [128, 256]))],
    # A100 follow-ups (2026-10-21). ladder_eager = the H200 size ladder grid
    # (logs/ragged_ladder, eager) so A100 vs H200 is measured-vs-measured.
    "a100_ladder_eager": [(m, {}, _ragged(["alpaca", "sharegpt"], [1, 4, 16, 64]))
                          for m in (Q05, "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct",
                                    "Qwen/Qwen2.5-7B-Instruct", Q14)],
    "a100_longctx": [(Q7, dict(GRAPHS, max_model_len=32768),
                      [(f"ctx{n}_c{c}", {"input_len": n, "output_len": 512, "concurrency": c,
                                         "unique_prompts": None})
                       for c in (4, 16) for n in (1024, 4096, 8192, 16384, 30720)])],
    "a100_prefill": [(Q7, dict(GRAPHS, max_num_batched_tokens=8192, max_model_len=10240),
                      [(f"in{n}_c{c}", {"input_len": n, "output_len": 16, "concurrency": c,
                                        "unique_prompts": None})
                       for n in (2048, 8192) for c in (4, 16)])],
    # A100 TP=2 (2 GPUs, both metered): 32B needs TP=2 on 80 GB A100s; 14B TP=2 vs the
    # TP=1 a100_graphs runs prices the second GPU.
    "a100_tp2": [(Q14, dict(GRAPHS, tensor_parallel_size=2), _ragged(["alpaca"], [1, 16, 64])),
                 (Q32, dict(GRAPHS, tensor_parallel_size=2, max_model_len=8192), _ragged(["alpaca"], [1, 16, 64]))],
    # MoE on A100, same grid and eager mode as the H200 logs/ragged_moe runs (measured
    # cross-GPU MoE comparison). ~61 GB of weights leaves ~11 GB KV on 80 GB.
    "a100_moe": [("Qwen/Qwen3-30B-A3B", {}, _ragged(["alpaca", "sharegpt"], [1, 4, 16, 64]))],
    "a100_moe_graphs": [("Qwen/Qwen3-30B-A3B", GRAPHS, _ragged(["alpaca", "sharegpt"], [1, 4, 16, 64]))],
    # Generic arms for another GPU type (A6000 etc.; set PROV_LOG_ROOT per GPU).
    "gpu_graphs": [(Q7, GRAPHS, _ragged(["alpaca", "sharegpt"], [1, 4, 16, 64])),
                   (Q14, GRAPHS, _ragged(["alpaca"], [1, 16, 64]))],
    "gpu_eager": [(Q7, {}, _ragged(["alpaca", "sharegpt"], [1, 4, 16, 64]))],
    "gpu_highconc": [(Q7, dict(GRAPHS, max_num_seqs=256), _ragged(["alpaca", "sharegpt"], [128, 256]))],
    "gpu_longctx": [(Q7, dict(GRAPHS, max_model_len=32768),
                     [(f"ctx{n}_c4", {"input_len": n, "output_len": 512, "concurrency": 4,
                                      "unique_prompts": None})
                      for n in (1024, 4096, 8192, 16384, 30720)])],
    # pipeline smoke tests (A5000, 0.5B): every code path in ~5 min
    "smoke": [(Q05, dict(GRAPHS, max_num_seqs=512), _ragged(["alpaca"], [4])
               + [("in1024_c4", {"input_len": 1024, "output_len": 16, "concurrency": 4,
                                 "unique_prompts": None})])],
    "smoke_eager": [(Q05, {}, _ragged(["alpaca"], [4]))],
    "smoke_tp2": [(Q05, dict(GRAPHS, tensor_parallel_size=2), _ragged(["alpaca"], [4]))],
    "smoke_longctx": [(Q05, dict(GRAPHS, max_model_len=8192),
                       [(f"ctx{n}_c2", {"input_len": n, "output_len": 64, "concurrency": 2,
                                        "unique_prompts": None}) for n in (512, 4096)])],
}


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def cmd(mode, model, run_dir, flags):
    c = [sys.executable, "-u", "energy_profile_load.py", "--mode", mode, "--model", model,
         "--dtype", "float16", "--run_dir", run_dir, "--gpu_id", "0",
         "--gpu_memory_utilization", "0.90", "--output_len", str(OUTPUT_CAP),
         "--max_model_len", "8192"]
    for k, v in flags.items():
        c += [f"--{k}"] if v is None else [f"--{k}", str(v)]
    return c          # later duplicates (e.g. --output_len, --max_model_len) win in argparse


def do_model(arm, model, eng, runs):
    # CUDA-graph arms need torch.compile (see engine_compat.py); eager arms keep
    # the historical environment exactly.
    if "no_enforce_eager" in eng:
        os.environ["VLLM_ALLOW_TORCH_COMPILE"] = "1"
    else:
        os.environ.pop("VLLM_ALLOW_TORCH_COMPILE", None)
    if arm == "loggerab":
        os.environ["NVML_LOGGER_SLIM"] = "1"
    else:
        os.environ.pop("NVML_LOGGER_SLIM", None)
    mdir = os.path.join(LOG_ROOT, arm, model.split("/")[-1])
    os.makedirs(mdir, exist_ok=True)
    print(f"[{ts()}] === {arm} / {model} engine={eng}", flush=True)
    if not base.load_json(os.path.join(mdir, "baselines.json")).get("idle_power_w"):
        idle = os.path.join(mdir, "idle")
        base.run(cmd("idle", model, idle, dict(eng, duration_s=IDLE_S)), os.path.join(idle, "run.log"))
    if not base.load_json(os.path.join(mdir, "baselines.json")).get("idle_power_w"):
        print(f"[{ts()}] [skip] no idle baseline for {mdir}", flush=True)
        return
    for tag, lf in runs:
        rd = os.path.join(mdir, tag)
        if not base.load_complete(rd):
            base.run(cmd("load", model, rd, dict(eng, duration_s=DURATION_S, **lf)),
                     os.path.join(rd, "run.log"))
        if base.load_complete(rd):
            base.run([sys.executable, "-u", "energy_model.py", "--two_term", "--run_dir", rd],
                     os.path.join(rd, "two_term.log"))
            r = base.load_json(os.path.join(rd, "results.json"))
            print(f"[{ts()}]   {tag}: tok/s={r.get('aggregate_tokens_per_sec'):.1f} "
                  f"reqs={r.get('completed_requests')} P_avg={r.get('avg_power_window_w')} "
                  f"E_counter={r.get('energy_window_j')}", flush=True)
        else:
            print(f"[{ts()}]   [failed] {rd} (see run.log)", flush=True)


def main():
    arms = [a for a in os.environ.get("PROV_ARMS", "relog,graphs,highconc,prefill").split(",") if a]
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"unknown arm {a}; choose from {list(ARMS)}")
    skip = [m for m in os.environ.get("PROV_SKIP_MODELS", "").split(",") if m]
    for a in arms:
        for model, eng, runs in ARMS[a]:
            if model in skip:
                print(f"[{ts()}] [skip] {a}/{model} (PROV_SKIP_MODELS: not in the model cache)", flush=True)
                continue
            try:
                do_model(a, model, eng, runs)
            except Exception as e:
                print(f"[{ts()}] [error] {a}/{model}: {e!r}", flush=True)
    print(f"[{ts()}] PROVISIONING SWEEP DONE ({','.join(arms)})", flush=True)


if __name__ == "__main__":
    main()
