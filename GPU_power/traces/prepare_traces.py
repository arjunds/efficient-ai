#!/usr/bin/env python3
"""Normalize public LLM-serving traces into one replay format (stdlib only; runs on
the login node's python 3.6).

Each output line: {"t": arrival s from trace start, "in": prompt tokens,
"out": generated tokens[, "h": Mooncake 512-token prefix-block hash ids]}.

Sources (downloaded to RAW by the curl lines in traces/README.md):
  azure23_conv / azure23_code  Azure LLM inference 2023 (Splitwise), 1 h each
  azure24_conv                 Azure LLM inference 2024, first ~37 h of the week (ranged download)
  burstgpt                     BurstGPT v2 without failures, part 1 (61 days, ChatGPT + GPT-4)
  mooncake_conv / mooncake_tool  Mooncake FAST'25 conversation / tool-agent traces, 1 h each

  python3 traces/prepare_traces.py            # all; writes traces/*.jsonl.gz + stats
  cut(): one-hour windows of the big traces for the repo (B200 side can't see /shared_data0):
    azure24_conv_h32 = hour 32 (Monday 08:00 UTC, the 57 req/s peak; Sunday trough is 21);
    burstgpt_h521 = hour 521 (4.6 req/s, one of the five busiest hours, realistic lengths).
Large outputs (azure24_conv, burstgpt) go to OUT_BIG; the repo gets a per-minute
summary of them (traces/<name>_per_min.csv) for the fleet-sizing analysis.
"""
import csv, gzip, json, os, sys
from datetime import datetime

RAW = "/shared_data0/adsampat/traces/raw"
OUT_BIG = "/shared_data0/adsampat/traces"
HERE = os.path.dirname(os.path.abspath(__file__))


def ts_azure(s):
    # '2023-11-16 18:15:46.6805900' or '2024-05-12 00:00:00.001163+00:00'
    s = s.split("+")[0]
    if "." in s:
        a, frac = s.split(".")
    else:
        a, frac = s, "0"
    return datetime.strptime(a, "%Y-%m-%d %H:%M:%S").timestamp() + float("0." + frac)


def azure(fn):
    rows = []
    with open(os.path.join(RAW, fn)) as f:
        r = csv.reader(f); next(r)
        for x in r:
            if len(x) < 3:
                continue          # ranged download: last line may be cut
            try:
                rows.append((ts_azure(x[0]), int(x[1]), int(x[2]), None))
            except ValueError:
                continue
    return rows


def burstgpt(fn):
    rows = []
    with open(os.path.join(RAW, fn)) as f:
        r = csv.reader(f); next(r)
        for x in r:
            rows.append((float(x[0]), int(x[2]), int(x[3]), None))
    return rows


def mooncake(fn):
    rows = []
    with open(os.path.join(RAW, fn)) as f:
        for line in f:
            d = json.loads(line)
            rows.append((d["timestamp"] / 1000.0, d["input_length"], d["output_length"], d["hash_ids"]))
    return rows


SOURCES = {
    "azure23_conv": (azure, "azure23_conv.csv", HERE),
    "azure23_code": (azure, "azure23_code.csv", HERE),
    "azure24_conv": (azure, "azure24_conv_head.csv", OUT_BIG),
    "burstgpt": (burstgpt, "burstgpt_1.csv", OUT_BIG),
    "mooncake_conv": (mooncake, "mooncake_conversation.jsonl", HERE),
    "mooncake_tool": (mooncake, "mooncake_toolagent.jsonl", HERE),
}


def pct(v, q):
    return v[min(len(v) - 1, int(q * len(v)))]


def stats(name, rows):
    n = len(rows)
    T = rows[-1][0] - rows[0][0]
    ins = sorted(r[1] for r in rows); outs = sorted(r[2] for r in rows)
    return {"trace": name, "requests": n, "duration_s": round(T, 1), "req_per_s": round(n / T, 3),
            "in_p50": pct(ins, .5), "in_mean": round(sum(ins) / n), "in_p99": pct(ins, .99),
            "out_p50": pct(outs, .5), "out_mean": round(sum(outs) / n), "out_p99": pct(outs, .99),
            "decode_tok_per_s": round(sum(outs) / T, 1),
            "decode_share_of_tokens": round(sum(outs) / float(sum(outs) + sum(ins)), 3)}


def main():
    names = sys.argv[1:] or list(SOURCES)
    allstats = []
    for name in names:
        fn_load, src, outdir = SOURCES[name]
        rows = sorted(fn_load(src), key=lambda r: r[0])
        t0 = rows[0][0]
        rows = [(t - t0, i, max(1, o), h) for t, i, o, h in rows]
        path = os.path.join(outdir, name + ".jsonl.gz")
        with gzip.open(path, "wt") as f:
            for t, i, o, h in rows:
                d = {"t": round(t, 4), "in": i, "out": o}
                if h is not None:
                    d["h"] = h
                f.write(json.dumps(d, separators=(",", ":")) + "\n")
        if outdir == OUT_BIG:       # per-minute summary for the repo
            per = {}
            for t, i, o, _ in rows:
                m = int(t // 60)
                a = per.setdefault(m, [0, 0, 0])
                a[0] += 1; a[1] += i; a[2] += o
            with open(os.path.join(HERE, name + "_per_min.csv"), "w") as f:
                w = csv.writer(f)
                w.writerow(["minute", "requests", "prompt_tokens", "gen_tokens"])
                for m in range(max(per) + 1):
                    w.writerow([m] + per.get(m, [0, 0, 0]))
        s = stats(name, rows); s["file"] = path
        allstats.append(s)
        print(json.dumps(s), flush=True)
    with open(os.path.join(HERE, "trace_stats.json"), "w") as f:
        json.dump(allstats, f, indent=1)


if __name__ == "__main__":
    main()


def cut(src, start_s, dur_s, dst):
    """Copy [start_s, start_s + dur_s) of a normalized trace to dst, re-based to 0."""
    n = 0
    with gzip.open(src, "rt") as f, gzip.open(dst, "wt") as g:
        for line in f:
            d = json.loads(line)
            if d["t"] < start_s:
                continue
            if d["t"] >= start_s + dur_s:
                break
            d["t"] = round(d["t"] - start_s, 4)
            g.write(json.dumps(d, separators=(",", ":")) + "\n")
            n += 1
    print("cut", src, start_s, dur_s, "->", dst, n, "requests")
