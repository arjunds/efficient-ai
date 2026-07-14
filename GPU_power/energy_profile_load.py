#!/usr/bin/env python3
"""
energy_profile_load.py

Offered-load energy profiler for the LIMINAL energy extension. Drives vLLM with
N concurrent clients (NOT batch_size=N) so the iteration-level scheduler forms its
own ragged running batch, while capturing time-resolved NVML power and
per-iteration scheduler state on one shared wall clock.

Modes:
  load             N concurrent clients for a fixed steady-state window.
  idle             model loaded, zero traffic -> idle_power_w (anchors P_static).
  prefill_ceiling  sustained long-prompt prefills -> near-TDP dynamic ceiling.
  self_test        NO GPU/vLLM: synthesize iter_log + power_trace from known
                   (e_bit, P_static) so energy_model.py + validate_waveform.py can
                   be verified end-to-end anywhere.

Artifacts per run_dir (PROFILING_HANDOFF.md schema):
  power_trace.csv, iter_log.csv, dram_trace.csv, run_meta.json, results.json
  (+ iter_ctx.jsonl sidecar with per-request context lengths)

Example:
  python energy_profile_load.py --mode load --model meta-llama/Meta-Llama-3-8B \
      --dtype float16 --concurrency 8 --input_len 512 --output_len 128 \
      --duration_s 60 --run_dir logs/load_llama3_c8

  python energy_profile_load.py --mode self_test --run_dir logs/selftest
"""

import argparse
import asyncio
import csv
import json
import math
import os
import time
from typing import List, Optional

from nvml_logger import NvmlPowerLogger
from iter_logger import IterationStatsLogger
from dram_counter import DramCounter, peak_hbm_bw_for


# ------------------------------------------------------------------ helpers
def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_baselines(run_dir) -> dict:
    """idle_power_w / prefill_ceiling_power_w written by the baseline modes,
    looked up from the parent (sweep) dir or the run dir itself."""
    for cand in (os.path.join(run_dir, "baselines.json"),
                 os.path.join(os.path.dirname(run_dir), "baselines.json")):
        if os.path.exists(cand):
            try:
                with open(cand) as f:
                    return json.load(f)
            except Exception:
                pass
    return {}


def reconcile_dram_into_iter_log(iter_csv, dram: DramCounter):
    """Fill iter_log.dram_bytes by integrating the DRAM trace over each iter
    window [t_start, t_end]."""
    if not os.path.exists(iter_csv) or dram is None or dram.peak_bw is None:
        return 0
    with open(iter_csv) as f:
        rows = list(csv.reader(f))
    if not rows:
        return 0
    header = rows[0]
    try:
        i_t0 = header.index("t_start")
        i_t1 = header.index("t_end")
        i_db = header.index("dram_bytes")
    except ValueError:
        return 0
    filled = 0
    for r in rows[1:]:
        try:
            t0, t1 = float(r[i_t0]), float(r[i_t1])
        except (ValueError, IndexError):
            continue
        b = dram.bytes_in_window(t0, t1)
        if b is not None:
            r[i_db] = f"{b:.6e}"
            filled += 1
    with open(iter_csv, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    return filled


def avg_power_in_window(power_csv, t0, t1) -> Optional[float]:
    if not os.path.exists(power_csv):
        return None
    ps = []
    with open(power_csv) as f:
        for row in csv.DictReader(f):
            try:
                t = float(row["t_wall"])
                p = float(row["power_w"])
            except (ValueError, KeyError, TypeError):
                continue
            if t0 <= t <= t1:
                ps.append(p)
    return sum(ps) / len(ps) if ps else None


# ------------------------------------------------------------------ load mode
async def _run_load_async(args, engine, tokenizer, meta):
    from engine_compat import make_sampling_params, build_prompt, drain_generate

    sp = make_sampling_params(args.output_len, force_exact=True)
    completions: List[dict] = []
    stop_at = None
    counter = {"n": 0}

    def next_prompt():
        counter["n"] += 1
        return build_prompt(tokenizer, args.input_len,
                            text=f"Write a short story. #{counter['n']}")

    async def worker(wid: int):
        i = 0
        while stop_at is None or time.time() < stop_at:
            rid = f"w{wid}-{i}"
            i += 1
            try:
                res = await drain_generate(engine, next_prompt(), sp, rid)
                completions.append(res)
            except Exception:
                pass

    # Warmup: fill the pipeline so we measure steady state, not ramp-up.
    warm = [asyncio.create_task(
        drain_generate(engine, next_prompt(), sp, f"warm-{i}"))
        for i in range(args.concurrency)]
    await asyncio.gather(*warm, return_exceptions=True)

    win_t0 = time.time()
    stop_at = win_t0 + args.duration_s
    workers = [asyncio.create_task(worker(w)) for w in range(args.concurrency)]
    await asyncio.gather(*workers, return_exceptions=True)
    win_t1 = time.time()
    return win_t0, win_t1, completions


def run_load(args):
    from transformers import AutoTokenizer
    from engine_compat import build_async_engine

    token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=True, token=token, trust_remote_code=True)

    # Attach iteration logger (patches the scheduler class) BEFORE building the
    # engine so its scheduler instance uses the patched method.
    iter_log = IterationStatsLogger(
        out_csv=os.path.join(args.run_dir, "iter_log.csv"),
        ctx_sidecar=os.path.join(args.run_dir, "iter_ctx.jsonl"))
    iter_log.attach()

    engine, emeta = build_async_engine(args)

    nvml = NvmlPowerLogger(args.gpu_id,
                           os.path.join(args.run_dir, "power_trace.csv"),
                           interval_ms=args.power_interval_ms).start()
    dram = DramCounter(args.gpu_id, os.path.join(args.run_dir, "dram_trace.csv"),
                       interval_ms=args.dram_interval_ms,
                       gpu_name=nvml.gpu_name).start()
    if dram.error:
        print(f"[warn] DRAM counter: {dram.error}", flush=True)

    try:
        win_t0, win_t1, completions = asyncio.run(
            _run_load_async(args, engine, tokenizer, emeta))
    finally:
        dram.stop()
        nvml.stop()
        iter_log.detach()

    filled = reconcile_dram_into_iter_log(
        os.path.join(args.run_dir, "iter_log.csv"), dram)

    baselines = load_baselines(args.run_dir)
    from gate_dram import import_models, hf_to_model_key
    try:
        models = import_models()
        model_key = hf_to_model_key(args.model, models)
    except Exception:
        model_key = None

    meta = {
        "gpu_name": nvml.gpu_name,
        "power_limit_w": nvml.power_limit_w,
        "dtype": args.dtype,
        "model": args.model,
        "model_key": model_key,
        "tp_degree": args.tensor_parallel_size,
        "vllm_version": emeta.get("vllm_version"),
        "engine_api": emeta.get("engine_api"),
        "max_num_seqs": emeta.get("max_num_seqs"),
        "block_size": emeta.get("block_size"),
        "gpu_mem_util": args.gpu_memory_utilization,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "concurrency_or_rate": args.concurrency,
        "idle_power_w": baselines.get("idle_power_w"),
        "prefill_ceiling_power_w": baselines.get("prefill_ceiling_power_w"),
        "kv_cache_dtype": args.kv_cache_dtype,
        "window_wall_t0": win_t0,
        "window_wall_t1": win_t1,
    }
    write_json(os.path.join(args.run_dir, "run_meta.json"), meta)

    gen = sum(c["gen_tokens"] for c in completions)
    dur = win_t1 - win_t0
    results = {
        "mode": "load",
        "completed_requests": len(completions),
        "generated_tokens_window": gen,
        "window_duration_s": dur,
        "aggregate_tokens_per_sec": gen / dur if dur > 0 else None,
        "avg_power_window_w": avg_power_in_window(
            os.path.join(args.run_dir, "power_trace.csv"), win_t0, win_t1),
        "dram_rows_filled": filled,
        "nvml_summary": nvml.summary(),
        "dram_backend": dram.backend,
        "dram_total_bytes": dram.total_bytes(),
        "peak_hbm_bw_bytes_per_s": dram.peak_bw,
    }
    write_json(os.path.join(args.run_dir, "results.json"), results)
    print(f"[saved] {args.run_dir} "
          f"(reqs={len(completions)}, tok/s={results['aggregate_tokens_per_sec']}, "
          f"dram_filled={filled})", flush=True)
    return results


# ------------------------------------------------------------------ baselines
def run_idle(args):
    """Model loaded, zero traffic -> idle/static power."""
    from engine_compat import build_async_engine
    engine, emeta = build_async_engine(args)
    nvml = NvmlPowerLogger(args.gpu_id,
                           os.path.join(args.run_dir, "power_trace.csv"),
                           interval_ms=args.power_interval_ms).start()
    time.sleep(args.duration_s)
    nvml.stop()
    s = nvml.summary()
    idle_power = s.get("median_power_w")
    _update_baselines(args.run_dir, {"idle_power_w": idle_power,
                                     "gpu_name": s.get("gpu_name")})
    write_json(os.path.join(args.run_dir, "results.json"),
               {"mode": "idle", "idle_power_w": idle_power, "nvml_summary": s})
    print(f"[idle] idle_power_w={idle_power}", flush=True)
    return idle_power


def run_prefill_ceiling(args):
    """Sustained back-to-back long prefills -> near-TDP dynamic ceiling."""
    from transformers import AutoTokenizer
    from engine_compat import build_async_engine, make_sampling_params, \
        build_prompt, drain_generate

    token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=True, token=token, trust_remote_code=True)
    engine, emeta = build_async_engine(args)

    nvml = NvmlPowerLogger(args.gpu_id,
                           os.path.join(args.run_dir, "power_trace.csv"),
                           interval_ms=args.power_interval_ms).start()

    async def flood():
        sp = make_sampling_params(1, force_exact=True)  # 1 token: prefill-dominated
        stop = time.time() + args.duration_s
        n = 0
        while time.time() < stop:
            batch = [asyncio.create_task(drain_generate(
                engine, build_prompt(tokenizer, args.input_len or 2048, None),
                sp, f"pf-{n}-{k}")) for k in range(max(4, args.concurrency))]
            n += 1
            await asyncio.gather(*batch, return_exceptions=True)

    try:
        asyncio.run(flood())
    finally:
        nvml.stop()
    s = nvml.summary()
    ceiling = s.get("median_power_w")
    _update_baselines(args.run_dir, {"prefill_ceiling_power_w": ceiling})
    write_json(os.path.join(args.run_dir, "results.json"),
               {"mode": "prefill_ceiling", "prefill_ceiling_power_w": ceiling,
                "nvml_summary": s})
    print(f"[prefill_ceiling] ceiling_power_w={ceiling}", flush=True)
    return ceiling


def _update_baselines(run_dir, updates):
    """Merge baseline values into baselines.json in the run dir AND its parent
    (so a whole sweep under one parent shares them)."""
    for target_dir in {run_dir, os.path.dirname(run_dir) or "."}:
        path = os.path.join(target_dir, "baselines.json")
        cur = {}
        if os.path.exists(path):
            try:
                with open(path) as f:
                    cur = json.load(f)
            except Exception:
                cur = {}
        cur.update({k: v for k, v in updates.items() if v is not None})
        write_json(path, cur)


# ------------------------------------------------------------------ self_test
def run_self_test(args):
    """Synthesize a physically-consistent run with known coefficients, so the
    downstream model+validation can be checked with no GPU. Ground truth is
    written to selftest_truth.json for the fitters to compare against."""
    import random
    rng = random.Random(0)

    E_BIT = 1.3e-10       # J per byte (effective)
    P_STATIC = 70.0       # W
    WEIGHT_BYTES = 8.03e9 * 2   # ~8B params, fp16
    KV_BYTES_PER_TOKEN = 2 * 32 * 8 * 128 * 2  # Llama3-8B GQA, fp16

    os.makedirs(args.run_dir, exist_ok=True)
    power_csv = os.path.join(args.run_dir, "power_trace.csv")
    iter_csv = os.path.join(args.run_dir, "iter_log.csv")

    t = time.time()
    iters = []
    n_running = 1
    ctx = 512
    for k in range(400):
        # ragged: running set and contexts drift over time
        n_running = max(1, min(16, n_running + rng.choice([-1, 0, 0, 1, 1])))
        ctxs = [ctx + rng.randint(0, 300) + j * 40 for j in range(n_running)]
        kv_tokens_resident = sum(ctxs)
        decode_tokens = n_running
        prefill_tokens = 0
        t_iter = 0.010 + 0.0008 * n_running + rng.uniform(-0.001, 0.001)
        bytes_iter = WEIGHT_BYTES + kv_tokens_resident * KV_BYTES_PER_TOKEN
        e_iter = E_BIT * bytes_iter + P_STATIC * t_iter
        t0 = t
        t1 = t + t_iter
        iters.append((t0, t1, "decode", n_running, 0, kv_tokens_resident,
                      decode_tokens, prefill_tokens, e_iter, bytes_iter))
        t = t1

    with open(iter_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_start", "t_end", "phase", "n_running", "n_waiting",
                    "kv_tokens_resident", "kv_cache_pct", "prefill_tokens",
                    "decode_tokens", "dram_bytes"])
        for (t0, t1, ph, nr, nw, kv, dt, pt, e_iter, b_iter) in iters:
            w.writerow([f"{t0:.6f}", f"{t1:.6f}", ph, nr, nw, kv, "",
                        pt, dt, f"{b_iter:.6e}"])

    # Power trace: sample at 100Hz, power = E_iter / t_iter for the active iter,
    # plus measurement noise.
    with open(power_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_wall", "power_w", "sm_mhz", "mem_mhz", "gpu_util",
                    "mem_util", "temp_c"])
        for (t0, t1, ph, nr, nw, kv, dt, pt, e_iter, b_iter) in iters:
            dur = t1 - t0
            p = e_iter / dur if dur > 0 else P_STATIC
            ts = t0
            while ts < t1:
                w.writerow([f"{ts:.6f}", f"{p + rng.uniform(-3, 3):.3f}",
                            1400, 1200, 95, 60, 55])
                ts += 0.01

    write_json(os.path.join(args.run_dir, "run_meta.json"), {
        "mode": "self_test", "model": "synthetic-Llama3-8B",
        "model_key": "Llama3-8B", "dtype": "float16",
        "weight_bytes": WEIGHT_BYTES, "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
        "idle_power_w": P_STATIC, "concurrency_or_rate": 8,
        "window_wall_t0": iters[0][0], "window_wall_t1": iters[-1][1],
    })
    write_json(os.path.join(args.run_dir, "selftest_truth.json"), {
        "e_bit_j_per_byte": E_BIT, "p_static_w": P_STATIC,
        "weight_bytes": WEIGHT_BYTES, "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
    })
    print(f"[self_test] wrote synthetic run to {args.run_dir}\n"
          f"  truth: e_bit={E_BIT:.3e} J/byte, P_static={P_STATIC} W\n"
          f"  now run: python energy_model.py --run_dir {args.run_dir} "
          f"&& python validate_waveform.py --run_dir {args.run_dir}", flush=True)


# ------------------------------------------------------------------ cli
def build_arg_parser():
    ap = argparse.ArgumentParser(description="Offered-load vLLM energy profiler")
    ap.add_argument("--mode", default="load",
                    choices=["load", "idle", "prefill_ceiling", "self_test"])
    ap.add_argument("--model")
    ap.add_argument("--dtype", default="float16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--kv_cache_dtype", default="auto")
    ap.add_argument("--run_dir", default="logs/load_run")

    ap.add_argument("--concurrency", type=int, default=8,
                    help="number of concurrent clients (NOT batch_size)")
    ap.add_argument("--arrival_rate", type=float, default=0.0,
                    help=">0 => open-loop Poisson arrivals at this req/s (unused "
                         "unless you switch the worker model)")
    ap.add_argument("--input_len", type=int, default=512)
    ap.add_argument("--output_len", type=int, default=128)
    ap.add_argument("--duration_s", type=float, default=60.0)

    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--power_interval_ms", type=int, default=10)
    ap.add_argument("--dram_interval_ms", type=int, default=100)

    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--enforce_eager", action="store_true", default=True)
    ap.add_argument("--no_enforce_eager", dest="enforce_eager",
                    action="store_false")
    return ap


def main():
    args = build_arg_parser().parse_args()
    os.makedirs(args.run_dir, exist_ok=True)
    if args.mode == "self_test":
        run_self_test(args)
        return
    if not args.model:
        raise SystemExit(f"--model required for mode={args.mode}")
    if args.mode == "load":
        run_load(args)
    elif args.mode == "idle":
        run_idle(args)
    elif args.mode == "prefill_ceiling":
        run_prefill_ceiling(args)


if __name__ == "__main__":
    main()
