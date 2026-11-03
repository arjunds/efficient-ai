#!/usr/bin/env python3
"""Fleet sizing over a real day (TRACES.md): the Azure 2024 per-minute request curve
(traces/azure24_conv_per_min.csv, ~37 h Sun->Mon) served by a fleet of decode GPUs, each
GPU's power-vs-load curve MEASURED by the azure24_peak replay levels (trace_analysis.py
-> trace_summary.json). Same request mix every minute (the peak hour's lengths; the
per-minute mean prompt/output lengths vary ~+-10% over the day).

Per GPU type, at a latency SLO (capacity c = highest measured req/s meeting it):
  peak     N = ceil(max demand / c) GPUs always on, load spread evenly
  consol   n(t) = ceil(D(t)/c) GPUs carry the load; the rest idle with the model loaded
  gated    as consol, the rest at the no-process idle (or off: --off)
P(r) is interpolated linearly between measured levels, P(0) = the vLLM-loaded idle.
  python3 fleet_analysis.py --summary logs/A100_prov/trace_summary.json --summary ... [--slo relaxed]
"""
import argparse
import csv
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "liminal_energy"))

DEEP_IDLE = {"A100-80-PCIe": 47.5, "A6000": 32.5, "B200": 235.3, "H200": None}   # W, no process


def interp(pts, x):
    """Piecewise-linear y(x) through sorted (x, y); flat beyond the last point."""
    if x <= pts[0][0]:
        return pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return pts[-1][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="append", required=True)
    ap.add_argument("--trace", default="azure24_conv_h32")
    ap.add_argument("--per_min", default=os.path.join(HERE, "traces/azure24_conv_per_min.csv"))
    ap.add_argument("--slo", default="relaxed", choices=["tight", "relaxed"])
    ap.add_argument("--off", action="store_true", help="gated GPUs draw 0 W (powered off)")
    ap.add_argument("--out", default=os.path.join(HERE, "fleet_summary.json"))
    a = ap.parse_args()

    D = [int(r["requests"]) / 60.0 for r in csv.DictReader(open(a.per_min))]
    D = D[:-1] if D and D[-1] < 0.5 * D[-2] else D          # partial last minute (ranged download)
    levels, caps = [], {}
    for p in a.summary:
        s = json.load(open(p))
        levels += [l for l in s["levels"] if l["trace"] == a.trace]
        caps.update({(c["gpu"], c["model"]): c for c in s["capacity"] if c["trace"] == a.trace and c["slo"] == a.slo})
    out = []
    for (gpu, model), c in sorted(caps.items()):
        L = sorted((l for l in levels if l["gpu"] == gpu and l["model"] == model), key=lambda l: l["req_per_s"])
        idle = next((l["idle_power_w"] for l in L if l.get("idle_power_w")), None)
        pts = ([(0.0, idle)] if idle else []) + [(l["req_per_s"], l["avg_power_w"]) for l in L]
        cap = c["req_per_s"]
        deep = 0.0 if a.off else (DEEP_IDLE.get(gpu) or idle)
        N = math.ceil(max(D) / cap)
        E = dict(peak=0.0, consol=0.0, gated=0.0)
        gpu_min = 0
        for d in D:
            n = max(1, math.ceil(d / cap))
            gpu_min += n
            E["peak"] += 60 * N * interp(pts, d / N)
            E["consol"] += 60 * (n * interp(pts, d / n) + (N - n) * idle)
            E["gated"] += 60 * (n * interp(pts, d / n) + (N - n) * deep)
        reqs = 60 * sum(D)
        row = dict(gpu=gpu, model=model, slo=a.slo, capacity_req_s=cap, fleet_peak_gpus=N,
                   mean_active_gpus=gpu_min / len(D), hours=len(D) / 60.0,
                   energy_kwh={k: v / 3.6e6 for k, v in E.items()},
                   j_per_request={k: v / reqs for k, v in E.items()},
                   saving_vs_peak={k: 1 - E[k] / E["peak"] for k in ("consol", "gated")})
        out.append(row)
        print("%-13s %-20s cap %.2f req/s  fleet %d (mean active %.1f)  J/req peak %.1f consol %.1f gated %.1f  "
              "(saves %.0f%% / %.0f%%)" % (gpu, model, cap, N, row["mean_active_gpus"], row["j_per_request"]["peak"],
                                           row["j_per_request"]["consol"], row["j_per_request"]["gated"],
                                           100 * row["saving_vs_peak"]["consol"], 100 * row["saving_vs_peak"]["gated"]))
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out, "| demand %.1f-%.1f req/s over %.1f h" % (min(D), max(D), len(D) / 60))


if __name__ == "__main__":
    main()
