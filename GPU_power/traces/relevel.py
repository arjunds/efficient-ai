#!/usr/bin/env python3
"""Recompute results.json["levels"] of trace-replay runs from their saved power_trace.csv
and requests.csv (e.g. after the 2026-10-28 energy_model POWER_INSTANT-parse fix left
energy None).   python3 traces/relevel.py <run_dir> [...]"""
import csv, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from energy_profile_load import level_results


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


for rd in sys.argv[1:]:
    res = json.load(open(os.path.join(rd, "results.json")))
    comps = []
    for r in csv.DictReader(open(os.path.join(rd, "requests.csv"))):
        comps.append(dict(t0=f(r["t_submit"]), t1=f(r["t_finish"]), t_first=f(r["t_first_token"]),
                          prompt_tokens=int(r["prompt_tokens"] or 0), gen_tokens=int(r["gen_tokens"] or 0),
                          cached_tokens=f(r.get("cached_tokens")), trace_speedup=f(r.get("trace_speedup"))))
    keys = ("trace_speedup", "warm_s", "trace", "t0", "t1", "t_last_arrival", "aborted", "torch_profile")
    levels = [{k: lv[k] for k in keys if k in lv} for lv in res["levels"]]
    res["levels"] = level_results(os.path.join(rd, "power_trace.csv"), levels, comps)
    res["levels_recomputed"] = "traces/relevel.py (energy_model POWER_INSTANT-parse fix, 2026-10-28)"
    json.dump(res, open(os.path.join(rd, "results.json"), "w"), indent=2)
    print(rd, [(lv["trace_speedup"], lv["j_per_token"] and round(lv["j_per_token"], 3)) for lv in res["levels"]])
