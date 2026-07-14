#!/usr/bin/env python3
"""
validate_waveform.py

The primary correctness gate for the energy extension. Replays a run's
per-iteration scheduler log through the calibrated model to synthesize a
predicted GPU power waveform, then compares it against the measured NVML trace on
the shared wall clock — RMSE / R² / MAE plus an overlay plot. This is a far
stronger check than matching a scalar average, because it tests that the model
tracks the ragged-batch dynamics iteration by iteration.

Predicted per-iteration power:  P_pred(t in [t0,t1]) = E_iter / (t1 - t0)
    E_iter = e_bit·(weight_bytes + kv_tokens_resident·kv_bytes_per_token)
             + P_static·(t1 - t0)
Timestamps between/without iterations fall back to P_static.

Usage:
  python validate_waveform.py --run_dir logs/<run>
  python validate_waveform.py --run_dir logs/<run> --calibration other/calibration.json
"""

import argparse
import bisect
import json
import math
import os
from typing import List, Optional

from energy_model import (
    load_power_samples, model_byte_constants, iter_rows, iter_bytes,
)


def build_predicted_series(run_dir, cal, meta, restrict_window=True):
    e_bit = cal["e_bit_j_per_byte"]
    p_static = cal["p_static_w"]
    weight_bytes = cal["weight_bytes"]
    kv_bpt = cal["kv_bytes_per_token"]

    rows = iter_rows(os.path.join(run_dir, "iter_log.csv"))
    windows = []  # (t0, t1, p_pred)
    for r in rows:
        try:
            t0 = float(r["t_start"]); t1 = float(r["t_end"])
        except (ValueError, KeyError):
            continue
        dt = t1 - t0
        if dt <= 0:
            continue
        b = iter_bytes(r, weight_bytes, kv_bpt)
        e_iter = e_bit * b + p_static * dt
        windows.append((t0, t1, e_iter / dt))
    windows.sort(key=lambda w: w[0])

    ts, ps = load_power_samples(os.path.join(run_dir, "power_trace.csv"))

    w0 = meta.get("window_wall_t0")
    w1 = meta.get("window_wall_t1")
    if restrict_window and w0 and w1:
        keep = [(t, p) for t, p in zip(ts, ps) if w0 <= t <= w1]
        ts = [t for t, _ in keep]; ps = [p for _, p in keep]

    starts = [w[0] for w in windows]
    pred = []
    for t in ts:
        i = bisect.bisect_right(starts, t) - 1
        if 0 <= i < len(windows) and windows[i][0] <= t <= windows[i][1]:
            pred.append(windows[i][2])
        else:
            pred.append(p_static)
    return ts, ps, pred, windows


def metrics(measured: List[float], predicted: List[float]) -> dict:
    n = len(measured)
    if n == 0:
        return {"error": "no overlapping samples"}
    resid = [m - p for m, p in zip(measured, predicted)]
    mse = sum(r * r for r in resid) / n
    rmse = math.sqrt(mse)
    mae = sum(abs(r) for r in resid) / n
    mbar = sum(measured) / n
    ss_tot = sum((m - mbar) ** 2 for m in measured)
    ss_res = sum(r * r for r in resid)
    r2 = 1 - ss_res / ss_tot if ss_tot else None
    return {
        "n_samples": n, "rmse_w": rmse, "mae_w": mae, "r2": r2,
        "mean_measured_w": mbar, "mean_predicted_w": sum(predicted) / n,
        "mean_abs_pct_err": 100.0 * mae / mbar if mbar else None,
    }


def plot_overlay(ts, measured, predicted, out_png, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib unavailable, skipping plot: {e}")
        return False
    t0 = ts[0]
    rel = [t - t0 for t in ts]
    plt.figure(figsize=(11, 5))
    plt.plot(rel, measured, linewidth=1.0, label="measured NVML", alpha=0.8)
    plt.plot(rel, predicted, linewidth=1.0, label="predicted model", alpha=0.8)
    plt.xlabel("time (s)"); plt.ylabel("power (W)")
    plt.title(title); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--calibration",
                    help="calibration.json (default: <run_dir>/calibration.json)")
    ap.add_argument("--no_restrict_window", action="store_true")
    args = ap.parse_args()

    cal_path = args.calibration or os.path.join(args.run_dir, "calibration.json")
    cal = json.load(open(cal_path))
    if "e_bit_j_per_byte" not in cal or cal.get("e_bit_j_per_byte") is None:
        raise SystemExit(f"calibration has no e_bit; run energy_model.py first "
                         f"({cal.get('error')})")
    meta = json.load(open(os.path.join(args.run_dir, "run_meta.json")))

    ts, measured, predicted, windows = build_predicted_series(
        args.run_dir, cal, meta, restrict_window=not args.no_restrict_window)
    m = metrics(measured, predicted)

    out_png = os.path.join(args.run_dir, "waveform_validation.png")
    title = (f"{meta.get('model_key') or meta.get('model')} "
             f"c={meta.get('concurrency_or_rate')} "
             f"R2={m.get('r2'):.3f} RMSE={m.get('rmse_w'):.1f}W"
             if m.get("r2") is not None else "waveform validation")
    plotted = plot_overlay(ts, measured, predicted, out_png, title)

    report = {"run_dir": args.run_dir, "calibration": cal_path,
              "n_iter_windows": len(windows), "metrics": m,
              "plot": out_png if plotted else None}
    out = os.path.join(args.run_dir, "waveform_validation.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
