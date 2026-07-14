#!/usr/bin/env python3
"""
validate_waveform.py

Primary correctness gate: replay a run's per-iteration scheduler log through the
calibrated model to predict the GPU power waveform, and compare to the measured
NVML trace on the shared wall clock.

Prediction is per TIME BIN (default 200 ms; see energy_model for why per-iteration
is too noisy):

    P_pred(bin) = e_bit · bytes_bin / Δ + P_static
    P_meas(bin) = ∫power / Δ   (mean measured power in the bin)

Reports RMSE, MAE, MAPE and R². NOTE: for a steady fixed-load run the measured
power is nearly flat, so R² (variance-explained) can look poor even when the level
is predicted to within a couple percent — MAPE/RMSE are the meaningful metrics
there. R² becomes informative for varying-load runs where power actually moves.

Usage: python validate_waveform.py --run_dir logs/<run> [--bin_s 0.2]
"""

import argparse
import json
import math
import os

from energy_model import bin_run, _r2, DEFAULT_BIN_S


def metrics(meas, pred) -> dict:
    n = len(meas)
    if n == 0:
        return {"error": "no overlapping bins"}
    resid = [m - p for m, p in zip(meas, pred)]
    rmse = math.sqrt(sum(r * r for r in resid) / n)
    mae = sum(abs(r) for r in resid) / n
    mbar = sum(meas) / n
    return {"n_bins": n, "rmse_w": rmse, "mae_w": mae, "r2": _r2(meas, pred),
            "mean_measured_w": mbar, "mean_predicted_w": sum(pred) / n,
            "mean_abs_pct_err": 100.0 * mae / mbar if mbar else None}


def plot_overlay(tmid, meas, pred, out_png, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib unavailable, skipping plot: {e}")
        return False
    t0 = tmid[0]
    rel = [t - t0 for t in tmid]
    plt.figure(figsize=(11, 5))
    plt.plot(rel, meas, linewidth=1.4, label="measured NVML", alpha=0.85)
    plt.plot(rel, pred, linewidth=1.4, label="predicted model", alpha=0.85)
    plt.xlabel("time (s)"); plt.ylabel("power (W)")
    plt.title(title); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--calibration")
    ap.add_argument("--bin_s", type=float, default=DEFAULT_BIN_S)
    args = ap.parse_args()

    cal_path = args.calibration or os.path.join(args.run_dir, "calibration.json")
    cal = json.load(open(cal_path))
    e_bit = cal.get("e_bit_j_per_byte")
    p_static = cal.get("p_static_w")
    if e_bit is None or p_static is None:
        raise SystemExit(f"calibration missing coefficients ({cal.get('error')}); "
                         "run energy_model.py first")

    # include_empty: idle bins (no decode activity) predict P_static — essential
    # for scoring active<->idle transitions in a scheduled/bursty run.
    bins, info = bin_run(args.run_dir, args.bin_s, cal.get("phase", "decode"),
                         include_empty=True)
    meas_w, pred_w, tmid = [], [], []
    for b in bins:
        meas_w.append(b["E"] / b["dt"])
        pred_w.append(e_bit * b["bytes"] / b["dt"] + p_static)
        tmid.append(0.5 * (b["b0"] + b["b1"]))

    m = metrics(meas_w, pred_w)
    meta = info["meta"]
    title = (f"{meta.get('model_key') or meta.get('model')} "
             f"c={meta.get('concurrency_or_rate')} "
             f"MAPE={m.get('mean_abs_pct_err', float('nan')):.1f}% "
             f"RMSE={m.get('rmse_w', float('nan')):.1f}W")
    out_png = os.path.join(args.run_dir, "waveform_validation.png")
    plotted = plot_overlay(tmid, meas_w, pred_w, out_png, title) if meas_w else False

    report = {"run_dir": args.run_dir, "calibration": cal_path,
              "bin_s": args.bin_s, "metrics": m,
              "plot": out_png if plotted else None}
    with open(os.path.join(args.run_dir, "waveform_validation.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
