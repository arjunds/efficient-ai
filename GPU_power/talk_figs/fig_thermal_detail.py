#!/usr/bin/env python3
"""Follow-ups to fig_idle_vs_temp (THERMAL_IDLE.md §9): how repeatable is "an A100 at T idles
at X W", does it depend on thermal history, and what happens over time.
  fig_idle_cards.png   : A100 idle power vs die temperature for every card measured, with the
                         spread of 1 s means at each degree; one card measured twice
  fig_idle_history.png : per card, idle power vs die temperature by thermal history: warming up
                         at idle, cooling after heavy load, cooling after light load
  fig_thermal_time.png : power, die / memory temperature and SM clock over a whole run
                         (A100 vs A6000: the A6000's clock governor acts at idle, the A100's doesn't)
Cool-down samples start 10 s after the load stops (past NVML's ~1.3 s averaging tail and the
first-seconds excess); one clock state. Inputs: microbench/thermal_*/ (thermal_leakage.py).
  PYTHONPATH=/shared_data0/adsampat/pydeps python3 talk_figs/fig_thermal_detail.py
"""
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MB = os.path.join(os.path.dirname(HERE), "microbench")
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]   # fixed order
SKIP_S = 10
# (thermal dir, card label, colour slot, line style); node from `kai pods` (bus ids repeat across nodes)
A100_RUNS = [("A100", "node3 C1:00.0", 0, "-"), ("A100_run2", "node8 61:00.0", 1, "-"),
             ("A100_hist1", "node8 01:00.0", 2, "-"), ("A100_hist2", "node8 01:00.0, run 2", 2, "--"),
             ("A100_hist3", "node8 E1:00.0", 3, "-"), ("A100_hist4", "node3 E1:00.0", 4, "-"),
             ("A100_hist5", "node3 81:00.0", 5, "-")]
HIST_RUNS = [r for r in A100_RUNS if r[0].startswith("A100_hist")]


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def load(tag):
    d = os.path.join(MB, f"thermal_{tag}")
    ph = json.load(open(os.path.join(d, "phases.json")))
    R = list(csv.DictReader(open(os.path.join(d, "samples.csv"))))
    S = {k: np.array([fnum(r[k]) for r in R]) for k in ("t", "p_avg_w", "temp_c", "sm_mhz")}
    mp = os.path.join(d, "mem_temp.csv")
    if os.path.exists(mp):
        M = list(csv.DictReader(open(mp)))
        mt = np.array([fnum(r["t"]) for r in M]); mv = np.array([fnum(r["mem_temp_c"]) for r in M])
        S["mem_c"] = np.interp(S["t"], mt, mv) if np.isfinite(mv).any() else np.full(len(S["t"]), np.nan)
    else:
        S["mem_c"] = np.full(len(S["t"]), np.nan)
    return ph, S


def second_bins(ph, S, names):
    """1 s means (die T, mem T, P) over the named idle phases, past SKIP_S, at the modal clock."""
    out = []
    for p in ph["phases"]:
        if p["phase"] not in names:
            continue
        sk = 0 if p["phase"] == "baseline" else SKIP_S
        m = (S["t"] >= p["t0"] + sk) & (S["t"] <= p["t1"])
        if m.sum() < 50:
            continue
        m &= np.abs(S["sm_mhz"] - np.nanmedian(S["sm_mhz"][m])) < 1
        tb = np.floor(S["t"][m] - p["t0"])
        for b in np.unique(tb):
            w = tb == b
            out.append((S["temp_c"][m][w].mean(), np.nanmean(S["mem_c"][m][w]) if np.isfinite(S["mem_c"][m][w]).any()
                        else np.nan, S["p_avg_w"][m][w].mean()))
    return np.array(out).reshape(-1, 3)


def per_degree(B, min_n=3):
    rows = []
    for c in np.unique(np.round(B[:, 0])):
        w = np.round(B[:, 0]) == c
        if w.sum() >= min_n:
            rows.append((c, np.median(B[w, 2]), np.percentile(B[w, 2], 10), np.percentile(B[w, 2], 90), w.sum()))
    return np.array(rows)


def style(a):
    a.set_facecolor(SURF); a.grid(color=GRID, lw=0.8); a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)


COOLS = ("cool:stream", "cool:gemm", "cool:light", "ramp_cool")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2})
    stats = {}

    # ---- fig A: every A100 card, median and 10-90% of 1 s means per degree
    fig, a = plt.subplots(figsize=(8.6, 5.0), facecolor=SURF)
    ends = []
    for tag, lab, slot, ls in A100_RUNS:
        ph, S = load(tag)
        D = per_degree(second_bins(ph, S, COOLS))
        col = CAT[slot]
        a.fill_between(D[:, 0], D[:, 2], D[:, 3], color=col, alpha=0.18, lw=0)
        a.plot(D[:, 0], D[:, 1], ls, color=col, lw=1.8, label=lab)
        ends.append([D[-1, 0], D[-1, 1], lab])
        stats[tag] = dict(card=lab, by_degree={int(c): dict(p_med=m, p10=lo, p90=hi, n=int(n)) for c, m, lo, hi, n in D})
    h, l = a.get_legend_handles_labels()                 # legend ordered by idle level, top first
    order = sorted(range(len(l)), key=lambda i: -next(e[1] for e in ends if e[2] == l[i]))
    a.legend([h[i] for i in order], [l[i] for i in order], frameon=False, fontsize=8, loc="lower right",
             title="card (node + PCI bus)", title_fontsize=8)
    at = 50
    v = [s["by_degree"][at]["p_med"] for t, s in stats.items() if at in s["by_degree"] and t != "A100_hist2"]
    r1, r2 = stats["A100_hist1"]["by_degree"], stats["A100_hist2"]["by_degree"]
    rep = [abs(r1[c]["p_med"] - r2[c]["p_med"]) for c in r1 if c in r2]
    within = np.median([s["by_degree"][c]["p90"] - s["by_degree"][c]["p10"] for s in stats.values() for c in s["by_degree"]])
    a.axvline(at, color=INK2, lw=0.8, ls=":")
    txt = (f"at {at} °C, {len(v)} cards: {min(v):.0f}–{max(v):.0f} W\n"
           f"same card, two runs: within {max(rep):.1f} W (median {np.median(rep):.1f})\n"
           f"within a run, 10–90% of 1 s means: {within:.1f} W wide")
    a.text(0.02, 0.97, txt, transform=a.transAxes, va="top", fontsize=8.5, color=INK,
           bbox=dict(boxstyle="round,pad=0.4", fc=SURF, ec=GRID))
    a.set_xlabel("die temperature (°C)"); a.set_ylabel("idle power (W), clocks up, CUDA context open")
    a.set_title("A100 80GB PCIe: idle power vs die temperature, every card measured "
                "(line = median, band = 10–90% of 1 s means)", loc="left", fontsize=9.5, color=INK)
    style(a)
    fig.text(0.01, 0.01, f"Cool-downs only, from {SKIP_S} s after the load stops. node3 C1:00.0 cooled only to 48 °C "
             "inside its 10-minute window. Bus ids repeat across nodes; cards are node + bus.", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(HERE, "fig_idle_cards.png"), dpi=180, facecolor=SURF)

    # ---- fig B: thermal history per card
    hist = [("baseline", "idle at start of run (before any load)", 0), ("cool:stream", "cooling after heavy load (300 W)", 1),
            ("cool:light", "cooling after light load (~130 W)", 2)]
    fig, ax = plt.subplots(1, len(HIST_RUNS), figsize=(3.3 * len(HIST_RUNS), 3.9), facecolor=SURF, sharey=False)
    hstats = {}
    for a, (tag, lab, slot, ls) in zip(ax, HIST_RUNS):
        ph, S = load(tag)
        hs = {}
        for name, hl, s in hist:
            B = second_bins(ph, S, (name,))
            if len(B) == 0:
                continue
            a.plot(B[:, 0] + (np.random.default_rng(0).uniform(-0.25, 0.25, len(B))), B[:, 2], "o", ms=2.6,
                   color=CAT[s], alpha=0.55, mew=0, label=hl)
            D = per_degree(B)
            if len(D):
                a.plot(D[:, 0], D[:, 1], "-", color=CAT[s], lw=1.6)
            hs[name] = dict(by_degree={int(c): float(m) for c, m, *_ in D},
                            mem_minus_die=float(np.nanmedian(B[:, 1] - B[:, 0])))
        hstats[tag] = hs
        gap = np.nanmedian([h["mem_minus_die"] for h in hs.values()])
        a.set_title(f"{lab}\nmemory − die: {gap:+.0f} °C", loc="left", fontsize=8.5, color=INK)
        a.set_xlabel("die temperature (°C)"); style(a)
        a.set_xlim(None, min(a.get_xlim()[1], 52))
    ax[0].set_ylabel("idle power (W)")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, frameon=False, fontsize=8, loc="lower center", markerscale=2.5)
    fig.suptitle("Does idle power at a given die temperature depend on how the card got there? "
                 "(dots: 1 s means, jittered ±0.25 °C; lines: per-degree medians)", x=0.01, ha="left", fontsize=9.5, color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(os.path.join(HERE, "fig_idle_history.png"), dpi=180, facecolor=SURF)

    # ---- fig C: over time, A100 vs A6000
    runs = [("A100_hist1", "A100 (node8 01:00.0)"), ("A6000_run2", "RTX A6000 (81:00.0)")]
    fig, ax = plt.subplots(3, 2, figsize=(11, 6.4), facecolor=SURF, sharex="col",
                           gridspec_kw=dict(height_ratios=[1.3, 1, 0.7]))
    for j, (tag, lab) in enumerate(runs):
        ph, S = load(tag)
        t0 = S["t"][0]; tm = (S["t"] - t0) / 60
        k = max(1, len(tm) // 6000)
        for i in range(3):
            for p in ph["phases"]:
                if not p["phase"].startswith(("cool", "baseline", "ramp_cool")):
                    ax[i, j].axvspan((p["t0"] - t0) / 60, (p["t1"] - t0) / 60, color="#eda100", alpha=0.12, lw=0)
        ax[0, j].plot(tm[::k], S["p_avg_w"][::k], color=CAT[0], lw=0.8)
        ax[1, j].plot(tm[::k], S["temp_c"][::k], color=CAT[1], lw=1.2, label="die")
        if np.isfinite(S["mem_c"]).any():
            ax[1, j].plot(tm[::k], S["mem_c"][::k], color=CAT[2], lw=1.2, label="memory (HBM)")
        else:
            ax[1, j].text(0.99, 0.95, "no memory sensor (GDDR6)", transform=ax[1, j].transAxes, ha="right",
                          va="top", fontsize=7.5, color=INK2)
        ax[2, j].plot(tm[::k], S["sm_mhz"][::k], color=CAT[6], lw=1.0)
        ax[0, j].set_title(lab, loc="left", fontsize=10, color=INK)
        ax[0, j].set_ylabel("power (W)"); ax[1, j].set_ylabel("temperature (°C)"); ax[2, j].set_ylabel("SM clock (MHz)")
        ax[2, j].set_xlabel("minutes (shaded = load on)")
        for i in range(3):
            style(ax[i, j])
        if np.isfinite(S["mem_c"]).any():
            ax[1, j].legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.suptitle("Over time. A100: holds 1410 MHz at idle, so idle power follows temperature smoothly; memory runs "
                 "hotter than the die under load but tracks it once idle.\nA6000: the clock governor drops to 210 MHz "
                 "within seconds of going idle, a ~75 W step unrelated to temperature.",
                 x=0.01, ha="left", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(HERE, "fig_thermal_time.png"), dpi=180, facecolor=SURF)

    json.dump(dict(cards=stats, history=hstats, skip_s=SKIP_S), open(os.path.join(HERE, "fig_thermal_detail.json"), "w"),
              indent=1, default=float)
    print(txt)
    for tag, hs in hstats.items():
        print(tag, {n: {c: round(v, 2) for c, v in h["by_degree"].items()} for n, h in hs.items()},
              {n: round(h["mem_minus_die"], 1) for n, h in hs.items()})


if __name__ == "__main__":
    main()
