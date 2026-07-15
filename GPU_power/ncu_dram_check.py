#!/usr/bin/env python3
"""
ncu_dram_check.py

Offline NCU-based reconciliation of the DRAM byte accounting (handoff gate #1,
without dcgmi). Two modes:

  --mode generate : run a short batch=1 vLLM decode under Nsight Compute (this
                    process; enforce_eager so real kernels are profiled). Writes
                    n_tokens.txt (generated token count).
  --mode parse    : read the exported ncu CSV, sum dram__bytes_read.sum +
                    dram__bytes_write.sum (unit-scaled), divide by generated
                    tokens, compare to analytic weight+KV bytes from models.py.

Per-decode-token HBM traffic should ≈ weight_bytes + KV(ctx) (each token reloads
all weights at batch=1). A short prompt + many decode tokens amortizes the single
prefill to a few percent.
"""

import argparse
import csv
import json
import os
import sys

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # keep kernels in-proc
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")


def do_generate(args):
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, dtype=args.dtype, enforce_eager=True,
              max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization,
              trust_remote_code=True)
    prompt_ids = list(range(10, 10 + args.prompt_len))  # tiny fixed prompt
    sp = SamplingParams(max_tokens=args.gen_tokens, min_tokens=args.gen_tokens,
                        ignore_eos=True, temperature=0.0)
    out = llm.generate({"prompt_token_ids": prompt_ids}, sp)
    n = len(out[0].outputs[0].token_ids)
    with open(os.path.join(args.run_dir, "n_tokens.txt"), "w") as f:
        f.write(str(n))
    print(f"[generate] prompt_len={args.prompt_len} generated={n} tokens")


UNIT_SCALE = {"byte": 1, "bytes": 1, "b": 1, "kbyte": 1e3, "mbyte": 1e6,
              "gbyte": 1e9, "tbyte": 1e12, "kib": 1024, "mib": 1024**2,
              "gib": 1024**3}


def _num(s):
    s = (s or "").strip().replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def parse_ncu_csv(path):
    """Sum dram read+write bytes from an ncu raw-page CSV (long format: one row
    per kernel×metric with Metric Name/Value/Unit columns)."""
    total = 0.0
    n_rows = 0
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    # find header
    hdr_i = next((i for i, r in enumerate(rows)
                  if any("Metric Name" in c for c in r)), None)
    if hdr_i is None:
        return None, 0
    hdr = rows[hdr_i]
    ni = next((j for j, c in enumerate(hdr) if "Metric Name" in c), None)
    vi = next((j for j, c in enumerate(hdr) if "Metric Value" in c), None)
    ui = next((j for j, c in enumerate(hdr) if "Metric Unit" in c), None)
    if ni is None or vi is None:
        return None, 0
    for r in rows[hdr_i + 1:]:
        if len(r) <= max(ni, vi):
            continue
        name = r[ni].strip()
        if name not in ("dram__bytes_read.sum", "dram__bytes_write.sum"):
            continue
        val = _num(r[vi])
        if val is None:
            continue
        unit = r[ui].strip().lower().replace("/s", "") if ui is not None and len(r) > ui else ""
        total += val * UNIT_SCALE.get(unit, 1)
        n_rows += 1
    return total, n_rows


def do_parse(args):
    from gate_dram import import_models, hf_to_model_key, expected_dram_bytes_per_token
    total_bytes, n_rows = parse_ncu_csv(args.csv)
    with open(os.path.join(args.run_dir, "n_tokens.txt")) as f:
        gen_tokens = int(f.read().strip())
    models = import_models()
    key = hf_to_model_key(args.model, models) or args.model
    ctx = args.prompt_len + args.gen_tokens // 2
    exp = expected_dram_bytes_per_token(key, ctx, args.dtype, models)
    measured_per_tok = total_bytes / gen_tokens if gen_tokens else None
    ratio = measured_per_tok / exp if (measured_per_tok and exp) else None
    result = {
        "model_key": key, "ctx": ctx, "gen_tokens": gen_tokens,
        "ncu_metric_rows": n_rows,
        "total_dram_bytes": total_bytes,
        "measured_bytes_per_token": measured_per_tok,
        "expected_bytes_per_token": exp,
        "ratio": ratio,
        "passed": (ratio is not None and 0.8 <= ratio <= 1.5),
        "note": "per-token includes one amortized prefill (~few %)",
    }
    with open(os.path.join(args.run_dir, "ncu_dram_result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["generate", "parse"])
    ap.add_argument("--model", default="Qwen/Qwen2-7B-Instruct")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--run_dir", default="logs/ncu_dram")
    ap.add_argument("--csv")
    ap.add_argument("--prompt_len", type=int, default=8)
    ap.add_argument("--gen_tokens", type=int, default=24)
    ap.add_argument("--max_model_len", type=int, default=2048)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.6)
    args = ap.parse_args()
    os.makedirs(args.run_dir, exist_ok=True)
    if args.mode == "generate":
        do_generate(args)
    else:
        do_parse(args)


if __name__ == "__main__":
    main()
