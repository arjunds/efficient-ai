#!/usr/bin/env python3
"""Measure a real reasoning model's output-length distribution, for the
"reasoning" replay trace (traces/build_reasoning_trace.py).

Qwen3-30B-A3B in thinking mode (its default chat template) answers every MATH-500
problem once with Qwen3's recommended thinking sampling (T 0.6, top-p 0.95, top-k 20),
capped at 16384 new tokens. Saves numbers only (no text): per problem the prompt
tokens, generated tokens, thinking tokens (up to </think>), finish reason, level.
Board power is logged alongside (an offline-batch reasoning energy point for free).

  VLLM_ALLOW_TORCH_COMPILE=1 python3 traces/gen_reasoning_lengths.py \
      --out traces/reasoning_lengths_qwen3-30b-a3b.json --power_dir logs/A100_prov/reasoning_gen
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engine_compat   # noqa: F401  (env defaults; tokenizer compat below)
from nvml_logger import NvmlPowerLogger

MATH = "/shared_data0/adsampat/traces/raw/math500_test.jsonl"
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
ap.add_argument("--out", required=True)
ap.add_argument("--power_dir", default=None)
ap.add_argument("--max_tokens", type=int, default=16384)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--max_num_seqs", type=int, default=256)
args = ap.parse_args()

engine_compat._install_transformers_tokenizer_compat()
from vllm import LLM, SamplingParams

probs = [json.loads(l) for l in open(MATH)]
if args.limit:
    probs = probs[:args.limit]
llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_tokens + 1024,
          gpu_memory_utilization=0.92, enforce_eager=False, max_num_seqs=args.max_num_seqs,
          trust_remote_code=True)
tok = llm.get_tokenizer()
end_think = tok.convert_tokens_to_ids("</think>")
prompts = [tok.apply_chat_template([{"role": "user", "content": p["problem"] + SUFFIX}],
                                   tokenize=False, add_generation_prompt=True) for p in probs]
sp = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=args.max_tokens, seed=0)

nvml = None
if args.power_dir:
    os.makedirs(args.power_dir, exist_ok=True)
    nvml = NvmlPowerLogger(0, os.path.join(args.power_dir, "power_trace.csv"), interval_ms=10).start()
t0 = time.time()
outs = llm.generate(prompts, sp)
t1 = time.time()
if nvml:
    nvml.stop()

rows = []
for p, o in zip(probs, outs):
    ids = list(o.outputs[0].token_ids)
    think = ids.index(end_think) + 1 if end_think in ids else None
    rows.append({"unique_id": p.get("unique_id"), "level": p.get("level"), "subject": p.get("subject"),
                 "prompt_tokens": len(o.prompt_token_ids), "gen_tokens": len(ids),
                 "think_tokens": think, "finish": o.outputs[0].finish_reason})
gen = sum(r["gen_tokens"] for r in rows)
meta = {"model": args.model, "dataset": "HuggingFaceH4/MATH-500 test", "max_tokens": args.max_tokens,
        "sampling": "T0.6 top_p0.95 top_k20 seed0, thinking on", "wall_s": t1 - t0,
        "gen_tokens": gen, "tok_per_s": gen / (t1 - t0),
        "truncated": sum(r["finish"] == "length" for r in rows),
        "window_wall_t0": t0, "window_wall_t1": t1,
        "power_summary": nvml.summary() if nvml else None}
json.dump({"meta": meta, "rows": rows}, open(args.out, "w"), indent=0)
L = sorted(r["gen_tokens"] for r in rows)
print(json.dumps(meta, default=str))
print("gen tokens p10/50/90/99:", [L[int(q * (len(L) - 1))] for q in (.1, .5, .9, .99)], flush=True)
