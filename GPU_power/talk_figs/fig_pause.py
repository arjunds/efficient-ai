#!/usr/bin/env python3
"""Load-pause experiment (thermal_leakage.py pulse phases; THERMAL_IDLE.md §11).

1 s pauses every 15 s inside long stream / GEMM loads and a stream ramp, sampled at 10 ms.
Per pause (summary.json "pauses"):
  late   = last 0.2 s of the pause: zero activity, clock back at its idle level, die still hot
  trough = lowest 50 ms running median in the pause: zero activity while clock/voltage are
           still near the loaded level (only differs from 'late' when the load ran capped)
Reference idle line: 1 C medians of the card's own cool-downs (>= 2 s after load, idle clock)
fitted with a quadratic; for the A6000, whose idle governor drops to 210 MHz, the pauses are
its only clocks-up idle data, so its reference is a linear fit of its own pause points.

  fig_pause.png : top, pause power vs die temperature per card with the idle line;
                  bottom, residual vs bandwidth, residual vs HBM-die gap, trough-late vs clock
  fig_pause.json: the numbers quoted in THERMAL_IDLE.md
  PYTHONPATH=/shared_data0/adsampat/pydeps python3 talk_figs/fig_pause.py
"""
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MB = os.path.join(os.path.dirname(HERE), "microbench")
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
# (thermal dir, title); a missing dir leaves a placeholder panel
RUNS = [("A100_pulse1", "A100 80GB PCIe (node8 81:00.0)"),
        ("A6000_pulse1", "RTX A6000 (A1:00.0)"),
        ("B200_pulse1", "B200")]
LOADS = [("pulse:stream", "stream, full occupancy", CAT[0]), ("pulse:gemm", "GEMM", CAT[1]),
         ("pulse_up", "stream ramp, up", CAT[2]), ("pulse_down", "stream ramp, down", CAT[5])]
CARD_MK = {"A100_pulse1": "o", "A6000_pulse1": "s", "B200_pulse1": "D"}


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def load_kind(phase):
    return next(k for k, _, _ in LOADS if phase.startswith(k))


def load(tag):
    d = os.path.join(MB, f"thermal_{tag}")
    if not os.path.exists(os.path.join(d, "summary.json")):
        return None
    s = json.load(open(os.path.join(d, "summary.json")))
    ph = json.load(open(os.path.join(d, "phases.json")))
    R = list(csv.DictReader(open(os.path.join(d, "samples.csv"))))
    S = {k: np.array([fnum(r[k]) for r in R]) for k in ("t", "p_avg_w", "temp_c", "sm_mhz")}
    S["mem_c"] = np.full(len(S["t"]), np.nan)
    mp = os.path.join(d, "mem_temp.csv")
    if os.path.exists(mp):
        M = list(csv.DictReader(open(mp)))
        mt = np.array([fnum(r["t"]) for r in M]); mv = np.array([fnum(r["mem_temp_c"]) for r in M])
        if np.isfinite(mv).any():
            S["mem_c"] = np.interp(S["t"], mt[np.isfinite(mv)], mv[np.isfinite(mv)])
    # idle line from the cool-downs, idle clock only
    m = np.zeros(len(S["t"]), bool)
    for p in ph["phases"]:
        if p["phase"].startswith(("cool", "ramp_cool")):
            m |= (S["t"] >= p["t0"] + 2) & (S["t"] <= p["t1"])
    fidle = float(np.nanmedian(S["sm_mhz"][m]))
    m &= np.abs(S["sm_mhz"] - fidle) < 1
    T = np.round(S["temp_c"])
    bins = np.array([(c, np.median(S["p_avg_w"][m & (T == c)])) for c in np.unique(T[m]) if (m & (T == c)).sum() >= 20])
    pz = [x for x in s["pauses"] if x.get("p_late_w") is not None]
    # drop pauses whose clock had not recovered by the end (A6000: first pause after its 210 MHz idle)
    c_mode = float(np.median([x["clk_late"] for x in pz]))
    n_all = len(pz)
    pz = [x for x in pz if x["clk_late"] >= 0.97 * c_mode]
    n_drop = n_all - len(pz)
    for x in pz:
        i = max(0, np.searchsorted(S["t"], x["t0"]) - 1)
        x["mem_gap_c"] = float(S["mem_c"][i] - S["temp_c"][i]) if np.isfinite(S["mem_c"][i]) else None
        x["kind"] = load_kind(x["phase"])
    loaded_clk = max(x["clk_late"] for x in pz)
    if fidle >= 0.9 * loaded_clk:      # idle holds the loaded clock: reference = the cool-down line
        q = np.polyfit(bins[:, 0], bins[:, 1], 2); ref_kind = "cool-down (quadratic)"
        lo, hi = bins[0, 0], bins[-1, 0]
    else:                              # A6000: idle governor drops the clock, use the pauses themselves
        X = np.array([x["t_late_c"] for x in pz]); Y = np.array([x["p_late_w"] for x in pz])
        q = np.polyfit(X, Y, 1); ref_kind = "pause self-fit (linear)"; lo, hi = X.min(), X.max()
    for x in pz:
        x["res_w"] = float(x["p_late_w"] - np.polyval(q, x["t_late_c"]))
        x["trough_gap_w"] = None if x.get("p_min_w") is None else float(x["p_min_w"] - x["p_late_w"])
    return dict(s=s, ph=ph, S=S, bins=bins, fidle=fidle, q=q, ref_kind=ref_kind, lo=lo, hi=hi, pz=pz, n_drop=n_drop,
                pcap=ph["info"]["power_limit_w"])


def style(a):
    a.set_facecolor(SURF)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        a.spines[sp].set_color(INK2)
    a.tick_params(colors=INK2, labelsize=8)
    a.grid(color=GRID, lw=0.6)
    a.set_axisbelow(True)
    a.xaxis.label.set_color(INK); a.yaxis.label.set_color(INK)


def stats(r):
    pz = r["pz"]
    res = np.array([x["res_w"] for x in pz]); bw = np.array([x["bw_gbs"] / 1e3 for x in pz])
    out = dict(n=len(pz), n_dropped_clock_not_recovered=r["n_drop"], ref=r["ref_kind"], idle_mhz=r["fidle"],
               res_median=float(np.median(res)), res_p10=float(np.percentile(res, 10)), res_p90=float(np.percentile(res, 90)),
               res_by_load={k: float(np.median([x["res_w"] for x in pz if x["kind"] == k])) for k, _, _ in LOADS
                            if any(x["kind"] == k for x in pz)},
               res_vs_bw_w_per_tbs=float(np.polyfit(bw, res, 1)[0]), corr_res_bw=float(np.corrcoef(res, bw)[0, 1]))
    g = [x for x in pz if x["mem_gap_c"] is not None]
    if len(g) > 4:
        mg = np.array([x["mem_gap_c"] for x in g]); rg = np.array([x["res_w"] for x in g])
        out.update(mem_gap_range=[float(mg.min()), float(mg.max())], res_vs_mem_gap_w_per_c=float(np.polyfit(mg, rg, 1)[0]),
                   corr_res_mem_gap=float(np.corrcoef(mg, rg)[0, 1]))
    for nm, sel in (("capped", True), ("uncapped", False)):
        L = [x for x in pz if bool(x["capped"]) == sel and x["trough_gap_w"] is not None]
        if L:
            out[f"trough_gap_{nm}"] = dict(median=float(np.median([x["trough_gap_w"] for x in L])),
                                           p10=float(np.percentile([x["trough_gap_w"] for x in L], 10)),
                                           clk_load=[float(min(x["clk_load"] for x in L)), float(max(x["clk_load"] for x in L))])
    for k in ("pulse:stream", "pulse:gemm"):
        L = [x for x in pz if x["kind"] == k]
        if L:
            out[f"{k}_trough_gap_median"] = float(np.median([x["trough_gap_w"] for x in L]))
            out[f"{k}_dT_load_to_late"] = float(np.median([x["t_load_c"] - x["t_late_c"] for x in L]))
    if r["ref_kind"].startswith("pause"):
        out["clocks_up_idle_fit"] = dict(w_per_c=float(r["q"][0]), p_at_40=float(np.polyval(r["q"], 40)),
                                         p_at_60=float(np.polyval(r["q"], 60)))
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    data = {tag: load(tag) for tag, _ in RUNS}
    js = {tag: stats(r) for tag, r in data.items() if r}
    fig, ax = plt.subplots(2, 3, figsize=(15.5, 8.6), facecolor=SURF, gridspec_kw=dict(height_ratios=[1.25, 1]))
    for j, (tag, title) in enumerate(RUNS):
        a = ax[0, j]; style(a); r = data[tag]
        if r is None:
            a.text(0.5, 0.5, f"{title}: run queued", transform=a.transAxes, ha="center", va="center", color=INK2, fontsize=10)
            a.set_xticks([]); a.set_yticks([]); a.set_title(title, loc="left", fontsize=10, color=INK)
            continue
        if r["ref_kind"].startswith("cool"):
            a.plot(r["bins"][:, 0], r["bins"][:, 1], "-", color=INK2, lw=2.2, alpha=0.55, zorder=1,
                   label=f"idle cool-down, {r['fidle']:.0f} MHz (1 °C medians)")
        else:
            xs = np.linspace(r["lo"], r["hi"], 10)
            a.plot(xs, np.polyval(r["q"], xs), "--", color=INK2, lw=1.2, zorder=1,
                   label=f"fit to the pauses: {r['q'][0]:.2f} W/°C")
            a.text(0.02, 0.03, f"its idle governor drops to {r['fidle']:.0f} MHz after ~10 s: "
                   f"{np.min(r['bins'][:, 1]):.0f}–{np.max(r['bins'][:, 1]):.0f} W, off this scale",
                   transform=a.transAxes, fontsize=7.5, color=INK2)
        for k, lab, col in LOADS:
            P = [x for x in r["pz"] if x["kind"] == k]
            if not P:
                continue
            a.plot([x["t_late_c"] for x in P], [x["p_late_w"] for x in P], "o", ms=4.5, color=col, mec="white", mew=0.4,
                   zorder=3, label=f"pause end: {lab}")
            C = [x for x in P if x["capped"] and x.get("p_min_w") is not None]
            if C:
                a.plot([x["temp_min_c"] for x in C], [x["p_min_w"] for x in C], "v", ms=5, mfc="none", color=col, mew=1.0,
                       zorder=2)
        a.plot([], [], "v", mfc="none", color=INK2, label="trough (still at loaded clock), capped loads only")
        a.set_title(title, loc="left", fontsize=10, color=INK)
        a.set_xlabel("die temperature (°C)"); a.set_ylabel("power with zero activity (W)")
        lo_y, hi_y = a.get_ylim(); a.set_ylim(lo_y, hi_y + 0.45 * (hi_y - lo_y))   # headroom for the legend
        a.legend(frameon=False, fontsize=7, loc="upper left")

    # bottom: pooled diagnostics
    a = ax[1, 0]; style(a)
    for tag, r in data.items():
        if not r:
            continue
        for k, _, col in LOADS:
            P = [x for x in r["pz"] if x["kind"] == k]
            a.plot([x["bw_gbs"] / 1e3 for x in P], [x["res_w"] for x in P], CARD_MK[tag], ms=4.5, color=col, mec="white", mew=0.4)
    a.axhline(0, color=INK2, lw=0.8)
    a.set_xlabel("DRAM bandwidth of the load just before the pause (TB/s)")
    a.set_ylabel("pause end minus idle line (W)")
    a.set_title("Does it matter what the card was doing? (same die temperature)", loc="left", fontsize=9, color=INK)
    a.legend(handles=[Line2D([], [], ls="", marker=CARD_MK[t], color=INK2, label=t.split("_")[0]) for t, r in data.items() if r],
             frameon=False, fontsize=7.5, loc="upper right")

    a = ax[1, 1]; style(a)
    for tag, r in data.items():
        if not r:
            continue
        for k, _, col in LOADS:
            P = [x for x in r["pz"] if x["kind"] == k and x["mem_gap_c"] is not None]
            a.plot([x["mem_gap_c"] for x in P], [x["res_w"] for x in P], CARD_MK[tag], ms=4.5, color=col, mec="white", mew=0.4)
    a.axhline(0, color=INK2, lw=0.8)
    a.set_xlabel("HBM minus die temperature when the pause starts (°C)")
    a.set_ylabel("pause end minus idle line (W)")
    a.set_title("Does hot HBM add leakage? (A6000: no memory sensor)", loc="left", fontsize=9, color=INK)

    a = ax[1, 2]; style(a)
    for tag, r in data.items():
        if not r:
            continue
        for k, _, col in LOADS:
            P = [x for x in r["pz"] if x["kind"] == k and x["trough_gap_w"] is not None]
            a.plot([x["clk_load"] / x["clk_late"] for x in P], [x["trough_gap_w"] for x in P], CARD_MK[tag], ms=4.5,
                   color=col, mec="white", mew=0.4)
    a.axhline(0, color=INK2, lw=0.8)
    a.set_xlabel("loaded SM clock / idle SM clock")
    a.set_ylabel("trough minus pause end (W)")
    a.set_title("Lower loaded clock (capped) → lower zero-activity power", loc="left", fontsize=9, color=INK)

    A = js.get("A100_pulse1", {})
    fig.suptitle("Load pauses: 1 s of zero activity every 15 s, at load temperature. At the same die temperature, power with the "
                 f"clock recovered sits on the idle line (A100 median {A.get('res_median', 0):+.1f} W),\nwhatever ran before and however "
                 "hot the HBM is. Only the clock matters: while still at a capped load's lower clock, zero-activity power is "
                 f"{-A.get('pulse:stream_trough_gap_median', 0):.0f} W (stream) to {-A.get('pulse:gemm_trough_gap_median', 0):.0f} W "
                 "(GEMM) lower on the A100.", x=0.01, ha="left", fontsize=9, color=INK)
    fig.text(0.01, 0.005, "Pause end = last 0.2 s of each 1 s pause. Trough = lowest 50 ms median; NVML averages power over ~25–100 ms, "
             "so troughs are smeared upward (a lower bound on the clock effect).\n"
             "Idle line: own cool-downs ≥2 s after load. Capped: A100 stream and GEMM, A6000 GEMM. Pauses whose clock had not recovered are dropped (A6000: 1).",
             fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.035, 1, 0.93))
    fig.savefig(os.path.join(HERE, "fig_pause.png"), dpi=170, facecolor=SURF)
    json.dump(js, open(os.path.join(HERE, "fig_pause.json"), "w"), indent=1)
    for t, v in js.items():
        print(t, json.dumps(v, default=float)[:900])


if __name__ == "__main__":
    main()
