#!/usr/bin/env python3
"""Build the reasoning replay trace: real chat ARRIVALS (Azure 2023 conversation trace,
the same timestamps the azure23_conv replay uses) carrying a real reasoning model's
LENGTHS (traces/gen_reasoning_lengths.py: Qwen3-30B-A3B thinking mode on MATH-500;
prompt and output token counts drawn jointly, with replacement, seeded). No public trace
has reasoning-serving arrivals with lengths; this pairs the two measured halves.

  python3 traces/build_reasoning_trace.py traces/reasoning_lengths_qwen3-30b-a3b.json
"""
import gzip, json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "reasoning_lengths_qwen3-30b-a3b.json")
dst = os.path.join(HERE, "reasoning_math.jsonl.gz")
rows = json.load(open(src))["rows"]
pairs = [(r["prompt_tokens"], r["gen_tokens"]) for r in rows]
rng = random.Random(0)
n = 0
with gzip.open(os.path.join(HERE, "azure23_conv.jsonl.gz"), "rt") as f, gzip.open(dst, "wt") as g:
    for line in f:
        t = json.loads(line)["t"]
        i, o = rng.choice(pairs)
        g.write(json.dumps({"t": t, "in": i, "out": o}, separators=(",", ":")) + "\n")
        n += 1
outs = sorted(o for _, o in pairs)
print(f"{dst}: {n} requests; output tokens p10/50/90/99 = "
      f"{[outs[int(q * (len(outs) - 1))] for q in (.1, .5, .9, .99)]}, mean {sum(outs) / len(outs):.0f}")
