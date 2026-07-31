#!/usr/bin/env python3
"""
gate_dram.py

Fail-fast DRAM-counter gate (PROFILING_HANDOFF.md #1).

At concurrency-1 decode, each generated token reloads all active weights from
HBM, so measured `dram_bytes/token` must ≈ the model's weight+KV bytes. If it
doesn't, the byte counter is broken and there is no point burning a full sweep on
it. This module implements exactly the handoff's expectation using ~/models.py
(whose active_params() already handles MoE top-k and MLA), and provides a CLI to
gate a completed concurrency-1 run directory.

Usage:
  python gate_dram.py --run_dir logs/<run>          # gate one run dir
  python gate_dram.py --model meta-llama/Meta-Llama-3-8B --ctx 2048 \
      --dtype float16 --measured_bytes 1.6e13 --tokens 1000
"""

import argparse
import csv
import json
import os
import sys
from typing import Optional

DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "fp8": 1, "float32": 4, "auto": 2}

# HF model id -> models.py MODELS key. Extend as models are added.
HF_TO_KEY = {
    "mistralai/Mistral-7B-v0.1": "Mistral-7B-v0.1",
    "Qwen/Qwen2-7B-Instruct": "Qwen2-7B-Instruct",
    "google/gemma-7b": "gemma-7b",
    "meta-llama/Meta-Llama-3-8B": "Llama3-8B",
    "meta-llama/Meta-Llama-3-70B": "Llama3-70B",
    "meta-llama/Llama-3.1-8B": "Llama3-8B",
    "Qwen/Qwen3-30B-A3B": "Qwen3-30B",
    "Qwen/Qwen3-4B": "Qwen3-4B",
}


def import_models():
    """Import the MODELS db from models.py, searching likely locations."""
    candidates = [
        os.path.dirname(os.path.abspath(__file__)),
        os.path.expanduser("~"),
        "/workspace",
        os.getcwd(),
    ]
    for d in candidates:
        if d and os.path.exists(os.path.join(d, "models.py")) and d not in sys.path:
            sys.path.insert(0, d)
    try:
        from models import MODELS  # noqa
        return MODELS
    except Exception as e:
        raise ImportError(
            "Could not import MODELS from models.py. Place models.py in "
            f"GPU_power/ or ~/. Searched: {candidates}. ({e})"
        )


def hf_to_model_key(hf_id: str, models=None) -> Optional[str]:
    if hf_id in HF_TO_KEY:
        return HF_TO_KEY[hf_id]
    if models is None:
        return None
    short = hf_id.split("/")[-1]
    if short in models:
        return short
    # normalized fallback (case/dash-insensitive)
    norm = short.lower().replace("_", "-")
    for k in models:
        if k.lower().replace("_", "-") == norm:
            return k
    return None


def expected_dram_bytes_per_token(model_key, ctx, dtype, models=None,
                                  kv_dtype=None) -> float:
    if models is None:
        models = import_models()
    m = models[model_key]
    b = DTYPE_BYTES[dtype]
    kb = DTYPE_BYTES[kv_dtype] if kv_dtype else b
    return m.active_params() * b + m.kv_cache_bytes(ctx, kb)


def check_counter(model_key, ctx, dtype, measured_bytes, tokens, models=None,
                  kv_dtype=None) -> dict:
    exp_per_tok = expected_dram_bytes_per_token(model_key, ctx, dtype, models, kv_dtype)
    measured_per_tok = measured_bytes / tokens if tokens else float("nan")
    ratio = measured_per_tok / exp_per_tok if exp_per_tok else float("nan")
    passed = 0.8 <= ratio <= 1.5
    return {
        "model_key": model_key,
        "ctx": ctx,
        "dtype": dtype,
        "expected_bytes_per_token": exp_per_tok,
        "measured_bytes_per_token": measured_per_tok,
        "ratio": ratio,
        "passed": passed,
        "verdict": "ok" if passed else "DRAM counter off — fix before sweeping",
    }


# ---------------- run-dir gate ----------------
def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def gate_run_dir(run_dir: str) -> dict:
    models = import_models()
    meta_path = os.path.join(run_dir, "run_meta.json")
    iter_path = os.path.join(run_dir, "iter_log.csv")
    if not os.path.exists(meta_path):
        return {"error": f"missing {meta_path}"}
    if not os.path.exists(iter_path):
        return {"error": f"missing {iter_path}"}

    with open(meta_path) as f:
        meta = json.load(f)

    model_key = meta.get("model_key") or hf_to_model_key(meta.get("model", ""), models)
    if model_key is None or model_key not in models:
        return {
            "error": f"model not in models.py: {meta.get('model')} -> {model_key}. "
                     "Add its config on the analysis side before reconciling.",
            "flagged_missing": True,
            "model": meta.get("model"),
        }

    conc = meta.get("concurrency_or_rate")
    if conc not in (1, "1", None):
        return {"error": f"gate requires a concurrency-1 run; got {conc}"}

    dtype = meta.get("dtype", "float16")
    kv_dtype = meta.get("kv_cache_dtype")
    if kv_dtype in (None, "auto"):
        kv_dtype = None

    # Sum dram_bytes and decode tokens over decode-only iterations.
    total_bytes = 0.0
    total_decode_tokens = 0
    ctx_weighted = 0.0
    ctx_weight = 0.0
    have_bytes = False
    with open(iter_path) as f:
        for row in csv.DictReader(f):
            phase = (row.get("phase") or "").strip()
            if phase != "decode":
                continue
            db = _f(row.get("dram_bytes"))
            dt = _f(row.get("decode_tokens"))
            kv = _f(row.get("kv_tokens_resident"))
            nr = _f(row.get("n_running"))
            if dt:
                total_decode_tokens += int(dt)
                if kv and nr:
                    ctx_weighted += (kv / nr) * dt
                    ctx_weight += dt
            if db is not None:
                total_bytes += db
                have_bytes = True

    if not have_bytes:
        return {"error": "no dram_bytes in decode rows — DRAM counter produced "
                         "nothing (dcgmi missing or field unavailable?)",
                "model_key": model_key}
    if total_decode_tokens == 0:
        return {"error": "no decode tokens found in iter_log", "model_key": model_key}

    ctx = int(ctx_weighted / ctx_weight) if ctx_weight else int(
        (meta.get("input_len", 0) or 0) + (meta.get("output_len", 0) or 0) / 2)

    result = check_counter(model_key, ctx, dtype, total_bytes,
                           total_decode_tokens, models, kv_dtype)
    result["run_dir"] = run_dir
    result["total_measured_bytes"] = total_bytes
    result["total_decode_tokens"] = total_decode_tokens
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir")
    ap.add_argument("--model")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--measured_bytes", type=float)
    ap.add_argument("--tokens", type=float)
    args = ap.parse_args()

    if args.run_dir:
        result = gate_run_dir(args.run_dir)
        out_path = os.path.join(args.run_dir, "gate_result.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
    elif args.model and args.measured_bytes and args.tokens:
        models = import_models()
        key = hf_to_model_key(args.model, models) or args.model
        result = check_counter(key, args.ctx, args.dtype,
                               args.measured_bytes, args.tokens, models)
    else:
        ap.error("provide --run_dir OR (--model --measured_bytes --tokens)")

    print(json.dumps(result, indent=2))
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    sys.exit(main())
