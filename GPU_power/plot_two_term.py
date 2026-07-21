#!/usr/bin/env python3
"""
plot_two_term.py

Figures for the two-term energy model result. Reads logs/ragged/*/ (binned_table.csv,
calibration.json one-term, two_term_fit.json) and writes to plots_two_term/.

  1. coefficients_by_model.png  - e_bit & e_flop per model (near-constant = HW consts)
  2. ebit_drift_vs_two_term.png - single-term e_bit drifts with arithmetic intensity;
                                  two-term e_bit is a flat band (the money comparison)
  3. two_term_fit_quality.png   - pooled measured vs predicted dynamic energy/bin
"""

import csv
import glob
import json
import os
from collections import defaultdict

OUT = "plots_two_term"
ROOT = "logs/ragged"


def _load_json(p):
    try:
        return json.load(open(p))
    except Exception:
        return {}


def main():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[fatal] matplotlib unavailable: {e}")
        return
    os.makedirs(OUT, exist_ok=True)
    made = []

    # ---- gather ----
    models = sorted(os.path.basename(os.path.dirname(f))
                    for f in glob.glob(f"{ROOT}/*/two_term_fit.json"))
    fits = {m: _load_json(f"{ROOT}/{m}/two_term_fit.json") for m in models}

    # ---- fig 1: coefficients by model ----
    if models:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
        ebit = [fits[m].get("e_bit_j_per_byte", 0) * 1e10 for m in models]
        eflop = [fits[m].get("e_flop_pJ_per_flop", 0) for m in models]
        ax1.bar(models, ebit); ax1.set_ylabel("e_bit (1e-10 J/byte)")
        ax1.set_title("Memory coefficient e_bit\n(~constant across models = HW property)")
        ax2.bar(models, eflop, color="tab:orange"); ax2.set_ylabel("e_flop (pJ/flop)")
        ax2.set_title("Compute coefficient e_flop\n(~constant across models = HW property)")
        for ax in (ax1, ax2):
            ax.tick_params(axis="x", rotation=30)
            for lbl in ax.get_xticklabels():
                lbl.set_ha("right"); lbl.set_fontsize(8)
        fig.tight_layout(); p = f"{OUT}/coefficients_by_model.png"
        fig.savefig(p, dpi=150); plt.close(fig); made.append(p)

    # ---- fig 2: single-term e_bit drift vs arithmetic intensity ----
    # per operating point: one-term e_bit (calibration.json) vs mean AI (binned_table)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    any_pts = False
    for m in models:
        xs, ys = [], []
        for rd in glob.glob(f"{ROOT}/{m}/*"):
            cal = _load_json(os.path.join(rd, "calibration.json"))
            e1 = cal.get("e_bit_j_per_byte")
            tbl = os.path.join(rd, "binned_table.csv")
            if e1 is None or not os.path.exists(tbl):
                continue
            ais = []
            for r in csv.DictReader(open(tbl)):
                try:
                    by = float(r["bytes_bin"]); fl = float(r["flops_bin"])
                    if by:
                        ais.append(fl / by)
                except (ValueError, KeyError):
                    pass
            if ais:
                xs.append(sum(ais) / len(ais)); ys.append(e1 * 1e10); any_pts = True
        if xs:
            ax.scatter(xs, ys, s=45, alpha=0.75, label=m)
    for m in models:  # two-term e_bit as horizontal lines
        eb = fits[m].get("e_bit_j_per_byte")
        if eb:
            ax.axhline(eb * 1e10, ls="--", lw=1, alpha=0.5)
    if any_pts:
        ax.set_xscale("log"); ax.set_xlabel("mean arithmetic intensity (flops/byte)")
        ax.set_ylabel("single-term e_bit (1e-10 J/byte)")
        ax.set_title("Single-term e_bit DRIFTS with arithmetic intensity;\n"
                     "two-term e_bit (dashed) is a constant HW coefficient")
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
        fig.tight_layout(); p = f"{OUT}/ebit_drift_vs_two_term.png"
        fig.savefig(p, dpi=150); plt.close(fig); made.append(p)
    else:
        plt.close(fig)

    # ---- fig 3: two-term fit quality (pooled measured vs predicted dyn energy) ----
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for m in models:
        fit = fits[m]
        eb = fit.get("e_bit_j_per_byte"); ef = fit.get("e_flop_j_per_flop")
        if eb is None or ef is None:
            continue
        meas, pred = [], []
        for rd in glob.glob(f"{ROOT}/{m}/*"):
            tbl = os.path.join(rd, "binned_table.csv")
            meta = _load_json(os.path.join(rd, "run_meta.json"))
            idle = meta.get("idle_power_w")
            if idle is None or not os.path.exists(tbl):
                continue
            for r in csv.DictReader(open(tbl)):
                try:
                    by = float(r["bytes_bin"]); fl = float(r["flops_bin"])
                    E = float(r["energy_bin_j"]); dt = float(r["dt_s"])
                    meas.append(E - idle * dt); pred.append(eb * by + ef * fl)
                except (ValueError, KeyError):
                    pass
        if meas:
            ax.scatter(pred, meas, s=6, alpha=0.25, label=f"{m} (R2={fit.get('r2_dynamic_energy',0):.2f})")
    lim = ax.get_xlim()
    ax.plot(lim, lim, "k--", lw=1, alpha=0.6)
    ax.set_xlabel("predicted dynamic energy/bin (J)")
    ax.set_ylabel("measured dynamic energy/bin (J)")
    ax.set_title("Two-term model: predicted vs measured (pooled)")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); p = f"{OUT}/two_term_fit_quality.png"
    fig.savefig(p, dpi=150); plt.close(fig); made.append(p)

    print(f"[done] wrote {len(made)} figures to {OUT}/:")
    for p in made:
        print("  ", p)


if __name__ == "__main__":
    main()
