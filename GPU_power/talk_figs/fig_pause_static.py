#!/usr/bin/env python3
"""Zero-activity power at load temperature, from 1 s pauses inside a load (thermal_leakage.py
pulse phases; THERMAL_IDLE.md §11).

Per pause: 'end' = last 0.2 s of the pause (clock back at its idle level), 'trough' = lowest
50 ms running median (clock/voltage not yet back up). Grey: the same card's cool-down idle
(>10 s after load, 1 °C medians). Dashed: the constant P_s our model uses.
  PYTHONPATH=/shared_data0/adsampat/pydeps python3 talk_figs/fig_pause_static.py
"""
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MB = os.path.join(os.path.dirname(HERE), "microbench")
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
GROUPS = [("uncapped", "load below the cap (clock unchanged)", "#1baf7a"),
          ("capped stream", "capped stream (loaded at ~1290 MHz)", "#eda100"),
          ("capped GEMM", "capped GEMM (loaded at ~1035–1335 MHz)", "#eb6834")]
RUNS = [("A100_pulse1", "A100 80GB PCIe (81:00.0)", 102.5),
        ("A6000_pulse1", "RTX A6000 (A1:00.0)", 106.7)]


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def group(x):
    if not x["capped"]:
        return "uncapped"
    return "capped GEMM" if x["phase"] == "pulse:gemm" else "capped stream"


def cooldown_bins(tag):
    d = os.path.join(MB, f"thermal_{tag}")
    ph = json.load(open(os.path.join(d, "phases.json")))["phases"]
    R = list(csv.DictReader(open(os.path.join(d, "samples.csv"))))
    t, P, T, C = (np.array([fnum(r[k]) for r in R]) for k in ("t", "p_avg_w", "temp_c", "sm_mhz"))
    m = np.zeros(len(t), bool)
    for p in ph:
        if p["phase"].startswith(("cool:", "ramp_cool")):
            m |= (t >= p["t0"] + 10) & (t <= p["t1"])
    clk = np.nanmedian(C[m])
    m &= np.abs(C - clk) < 1
    b = [(c, np.median(P[m & (T == c)])) for c in np.unique(T[m]) if (m & (T == c)).sum() >= 20]
    return np.array(b), clk


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8), facecolor=SURF)
    stats = {}
    for a, (tag, lab, p_s) in zip(ax, RUNS):
        s = json.load(open(os.path.join(MB, f"thermal_{tag}", "summary.json")))
        X = [x for x in s["pauses"] if x.get("p_min_w") is not None]
        cb, clk = cooldown_bins(tag)
        if clk >= 1000:      # the A6000 cools down at 210 MHz (~30 W): a different state, not drawn
            a.plot(cb[:, 0], cb[:, 1], "-", color="#8a8984", lw=2.2, alpha=0.6, zorder=1,
                   label=f"cool-down idle, same card ({clk:.0f} MHz)")
        st = {}
        for g, glab, col in GROUPS:
            G = [x for x in X if group(x) == g]
            if not G:
                continue
            a.scatter([x["t_late_c"] for x in G], [x["p_late_w"] for x in G], s=22, color=col, zorder=3,
                      edgecolor="white", lw=0.5, label=f"{glab}: end of pause")
            a.scatter([x["temp_min_c"] for x in G], [x["p_min_w"] for x in G], s=24, marker="v", facecolor="none",
                      edgecolor=col, lw=1.1, zorder=2)
            d = [x["p_late_w"] - x["p_min_w"] for x in G]
            st[g] = dict(n=len(G), end_minus_trough_w=float(np.median(d)),
                         clk_load=float(np.median([x["clk_load"] for x in G])))
        a.scatter([], [], s=24, marker="v", facecolor="none", edgecolor=INK2, label="same pause: trough (clock not yet back up)")
        a.axhline(p_s, ls="--", color=INK, lw=1)
        a.text(0.99, p_s + 0.4, f"P_s our model uses: {p_s:.1f} W", fontsize=8, color=INK, ha="right",
               va="bottom", transform=a.get_yaxis_transform())
        a.set_title(lab, loc="left", fontsize=10, color=INK)
        a.set_xlabel("die temperature (°C)")
        a.set_ylabel("power with zero activity (W)")
        a.grid(color=GRID, lw=0.6); a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.set_facecolor(SURF)
        a.legend(fontsize=7, frameon=False, loc="upper left" if clk >= 1000 else "lower right")
        stats[tag] = st
    ax[1].text(0.02, 0.84, "No cool-down line: after ~10 s idle the\nA6000 drops to 210 MHz (~30 W). In a 1 s\n"
               "pause it stays at ~1900 MHz, so these\npoints are its clocks-up idle.",
               transform=ax[1].transAxes, ha="left", va="top", fontsize=7.5, color=INK2)
    fig.suptitle("Zero-activity power at load temperature, read in 1 s pauses inside a load. Below the cap it equals "
                 "the idle curve at that temperature;\nunder the cap the trough sits lower (A100: 4 W for stream, 17 W "
                 "for GEMM), because the card was running at lower clock and voltage.",
                 x=0.01, ha="left", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(HERE, "fig_pause_static.png"), dpi=180, facecolor=SURF)
    json.dump(stats, open(os.path.join(HERE, "fig_pause_static.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
