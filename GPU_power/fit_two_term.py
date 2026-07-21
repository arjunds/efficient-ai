#!/usr/bin/env python3
"""
fit_two_term.py

Pool the per-bin (bytes, flops, energy, dt) data across MANY operating points for
a model and fit the two-term energy model:

    E_bin = e_bit·bytes_bin + e_flop·flops_bin + P_static·dt

The two coefficients are only identifiable if the pooled data spans a range of
arithmetic intensity (flops/bytes): low-AI memory-bound bins (concurrency=1, pure
decode) AND high-AI compute-bound bins (heavy prefill / long prompts / high
concurrency). A single steady run is all low-AI and can't separate the terms —
hence pooling across the whole sweep.

P_static is per-model (idle baseline, re-measured for these runs). We subtract
each bin's per-run P_static·dt, then fit (e_bit, e_flop) on the pooled dynamic
energy.

  python3 fit_two_term.py --glob "logs/ragged/<Model>/*" --out logs/ragged/<Model>/two_term_fit.json
"""

import argparse
import csv
import glob
import json
import os

from energy_model import _ls_two_feature, _r2, _pearson


def load_bins(run_dir):
    """Return list of (bytes, flops, energy, dt, idle_power) for a run's bins."""
    tbl = os.path.join(run_dir, "binned_table.csv")
    meta_p = os.path.join(run_dir, "run_meta.json")
    if not os.path.exists(tbl) or not os.path.exists(meta_p):
        return []
    idle = json.load(open(meta_p)).get("idle_power_w")
    if idle is None:
        return []
    out = []
    for r in csv.DictReader(open(tbl)):
        try:
            out.append((float(r["bytes_bin"]), float(r["flops_bin"]),
                        float(r["energy_bin_j"]), float(r["dt_s"]), float(idle)))
        except (ValueError, KeyError):
            continue
    return out


def fit(run_dirs):
    bytes_, flops, dyn, dts, Es = [], [], [], [], []
    n_runs = 0
    for d in run_dirs:
        bins = load_bins(d)
        if bins:
            n_runs += 1
        for by, fl, E, dt, idle in bins:
            bytes_.append(by); flops.append(fl); dts.append(dt); Es.append(E)
            dyn.append(E - idle * dt)   # per-run idle-anchored dynamic energy

    result = {"n_runs": n_runs, "n_bins": len(bytes_)}
    if len(bytes_) < 8:
        result["error"] = f"too few pooled bins ({len(bytes_)})"
        return result

    e_bit, e_flop = _ls_two_feature(bytes_, flops, dyn)
    r_bf = _pearson(bytes_, flops)
    ai = [f / b for f, b in zip(flops, bytes_) if b]
    yhat = [(e_bit or 0) * b + (e_flop or 0) * f for b, f in zip(bytes_, flops)]

    result.update({
        "e_bit_j_per_byte": e_bit,
        "e_flop_j_per_flop": e_flop,
        "e_flop_pJ_per_flop": (e_flop * 1e12) if e_flop is not None else None,
        "r2_dynamic_energy": _r2(dyn, yhat),
        "bytes_flops_pearson": r_bf,
        "identifiable": (r_bf is not None and abs(r_bf) < 0.97),
        "arithmetic_intensity_min": min(ai) if ai else None,
        "arithmetic_intensity_max": max(ai) if ai else None,
        "ai_span_ratio": (max(ai) / min(ai)) if ai and min(ai) > 0 else None,
    })
    if not result["identifiable"]:
        result["warning"] = (f"bytes~flops collinear (r={r_bf:.3f}) — coefficients "
                             "not separable; sweep needs more prefill-heavy / "
                             "long-prompt (high-AI) points.")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True,
                    help='run-dir glob, e.g. "logs/ragged/Qwen2-7B-Instruct/*"')
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run_dirs = [d for d in glob.glob(args.glob) if os.path.isdir(d)]
    res = fit(run_dirs)
    res["glob"] = args.glob
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
