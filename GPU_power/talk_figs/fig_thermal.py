#!/usr/bin/env python3
"""Idle power vs die temperature, and what it does to the trace breakdown (THERMAL_IDLE.md).
  fig_idle_vs_temp.png        : per GPU, idle power vs temperature for every card measured
                                (thermal cool-downs + the serving card's own idle gaps), the
                                constant P_s our figures use, and the serving temperature range
  fig_trace_breakdown_temp.png: fig_trace_breakdown with the static slice at each level's
                                measured die temperature: P_s(T) = P_s + k (T - T_ref)
Correction per level (A100, B200 only; see notes in the figure):
  T       = median die temperature over the level's steady window (power_trace.csv)
  T_ref   = temperature at which the serving card's own idle samples equal P_s
  k       = idle cool-down slope (thermal_leakage.py; card-to-card range in THERMAL_IDLE.md)
  static += k (T - T_ref) * window;  dynamic slices *= (1 + d_eb), d_eb = fixed-k e_b
  correction from the ramps (A100 -6.9%, B200 -2.8%); residual = measured - sum (unchanged total).
  PYTHONPATH=/shared_data0/adsampat/pydeps python3 talk_figs/fig_thermal.py
"""
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MB = os.path.join(ROOT, "microbench")
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
CARD_COLS = ["#2a78d6", "#eb6834", "#1baf7a"]          # categorical slots 1-3, fixed order
MODEL = "Qwen2.5-7B-Instruct"

# P_s our figures use (liminal_e.profile_params canonical), k from the idle cool-downs, and the
# fixed-k e_b correction from the ramps (THERMAL_IDLE.md §5/§8)
GPU = {
    "A100": dict(key="A100-80-PCIe", p_s=102.515, k=0.77, k_lo=0.71, k_hi=0.78, d_eb=-0.069,
                 runs=[("A100", "card C1:00.0"), ("A100_run2", "card 61:00.0")], serve="card 41:00.0 (serving)"),
    "B200": dict(key="B200", p_s=237.822, k=1.15, k_lo=0.99, k_hi=1.26, d_eb=-0.028,
                 runs=[("B200", "card dgx016"), ("B200_pairs", "card dgx007")], serve="serving card(s), idle gaps"),
}


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def read_power(path):
    R = list(csv.DictReader(open(path)))
    cols = {k: np.array([fnum(r.get(k)) for r in R]) for k in ("t_wall", "power_w", "temp_c", "gpu_util")}
    return cols


def cooldown_bins(tag):
    """1 C medians of idle power during the cool-downs (>2 s after load stops), one clock state."""
    ph = json.load(open(os.path.join(MB, f"thermal_{tag}", "phases.json")))
    ph = ph["phases"] if isinstance(ph, dict) else ph
    S = list(csv.DictReader(open(os.path.join(MB, f"thermal_{tag}", "samples.csv"))))
    t = np.array([fnum(r["t"]) for r in S]); P = np.array([fnum(r["p_avg_w"]) for r in S])
    T = np.array([fnum(r["temp_c"]) for r in S]); f = np.array([fnum(r["sm_mhz"]) for r in S])
    m = np.zeros(len(t), bool)
    for p in ph:
        if str(p.get("phase", "")).startswith(("cool:", "ramp_cool")):
            m |= (t >= p["t0"] + 2) & (t <= p["t1"])
    fmode = np.nanmedian(f[m])
    m &= np.abs(f - fmode) < 1
    out = []
    for c in np.unique(np.round(T[m])):
        w = m & (np.round(T) == c)
        if w.sum() >= 20:
            out.append((c, float(np.median(P[w]))))
    return np.array(out)


def serving_idle_and_levels(gk):
    """Serving card: idle samples (util 0 for >10 s) binned by temperature, and per-level median T."""
    s = json.load(open(os.path.join(ROOT, "logs", f"{gk}_prov", "trace_summary.json")))
    Ti, Pi, levT = [], [], {}
    for rd in sorted(set(r["run_dir"] for r in s["levels"])):
        pt = os.path.join(ROOT, rd, "power_trace.csv")
        rj = os.path.join(ROOT, rd, "results.json")
        if not (os.path.exists(pt) and os.path.exists(rj)):
            continue
        c = read_power(pt)
        u = np.nan_to_num(c["gpu_util"])
        busy_t = c["t_wall"][u > 0]
        last = np.full(len(u), -1e18)
        if len(busy_t):
            idx = np.searchsorted(busy_t, c["t_wall"], side="right") - 1
            last = np.where(idx >= 0, busy_t[np.clip(idx, 0, None)], -1e18)
        idle = (u == 0) & (c["t_wall"] - last > 10)
        Ti += list(c["temp_c"][idle]); Pi += list(c["power_w"][idle])
        ramp = {r["trace_speedup"]: r["ramp_s"] for r in s["levels"] if r["run_dir"] == rd}
        for lv in json.load(open(rj))["levels"]:
            x = lv["trace_speedup"]
            if x not in ramp:
                continue
            ws, we = lv["t0"] + ramp[x], lv["t_last_arrival"]
            if we - ws < 5:
                ws, we = lv["t0"], lv["t1"]
            w = (c["t_wall"] >= ws) & (c["t_wall"] <= we)
            levT[(rd, x)] = float(np.nanmedian(c["temp_c"][w])) if w.any() else np.nan
    Ti, Pi = np.array(Ti), np.array(Pi)
    ok = ~np.isnan(Ti) & ~np.isnan(Pi)
    Ti, Pi = Ti[ok], Pi[ok]
    bins = [(c, float(np.median(Pi[np.round(Ti) == c]))) for c in np.unique(np.round(Ti))
            if (np.round(Ti) == c).sum() >= 200]
    return np.array(bins), levT, s


def t_ref(bins, p_s):
    """Temperature where the serving card's binned idle crosses P_s (linear fit of the bins)."""
    k, a = np.polyfit(bins[:, 0], bins[:, 1], 1)
    return (p_s - a) / k


def corrected_shares(r, T, cfg, Tr, k):
    parts = dict(r["energy_parts_j"]); E = r["energy_j"]
    extra = k * (T - Tr) * r["window_s"]
    new = dict(parts)
    new["static"] = parts["static"] + extra
    for d in ("activity", "weights", "kv", "gemm"):
        new[d] = parts.get(d, 0.0) * (1 + cfg["d_eb"])
    new["residual"] = E - sum(v for kk, v in new.items() if kk != "residual")
    return {kk: v / E for kk, v in new.items()}


def style(a):
    a.set_facecolor(SURF); a.grid(color=GRID, lw=0.8); a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2})
    info = {}
    for g, cfg in GPU.items():
        bins, levT, s = serving_idle_and_levels(g)
        Tr = t_ref(bins, cfg["p_s"])
        Ts = [v for v in levT.values() if not np.isnan(v)]
        info[g] = dict(bins=bins, levT=levT, s=s, t_ref=Tr, t_serve=(min(Ts), float(np.median(Ts)), max(Ts)))
        print(f"{g}: P_s {cfg['p_s']:.1f} W reached by the serving card's idle at {Tr:.1f} C; "
              f"levels run at {min(Ts):.0f}-{max(Ts):.0f} C (median {np.median(Ts):.0f}); "
              f"P_s(T) at median = {cfg['p_s'] + cfg['k'] * (np.median(Ts) - Tr):.1f} W")

    # ---- figure 1: idle power vs temperature, per GPU
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURF)
    for a, (g, cfg) in zip(ax, GPU.items()):
        I = info[g]
        lo, med, hi = I["t_serve"]
        a.axvspan(lo, hi, color="#eda100", alpha=0.13, lw=0)
        a.text((lo + hi) / 2, 0.02, f"serving levels {lo:.0f}–{hi:.0f} °C", transform=a.get_xaxis_transform(),
               ha="center", va="bottom", fontsize=8, color=INK2)
        series = [(cooldown_bins(tag), lab, "thermal cool-down") for tag, lab in cfg["runs"]]
        series.append((I["bins"], cfg["serve"], "idle gaps in trace replays"))
        ends = []
        for (B, lab, src), col in zip(series, CARD_COLS):
            if len(B) == 0:
                continue
            a.plot(B[:, 0], B[:, 1], "-", color=col, lw=2)
            a.plot(B[:, 0], B[:, 1], "o", color=col, ms=3.5, mec=SURF, mew=0.6)
            ends.append([B[-1, 0], B[-1, 1], lab])
        ys = np.concatenate([s[0][:, 1] for s in series if len(s[0])])
        gap = 0.05 * (ys.max() - ys.min())
        ends.sort(key=lambda e: e[1])
        for i in range(1, len(ends)):                    # direct labels: nudge apart vertically
            ends[i].append(max(ends[i][1], ends[i - 1][-1] + gap) if len(ends[i - 1]) > 3 else max(ends[i][1], ends[i - 1][1] + gap))
        ends[0].append(ends[0][1])
        for xe, ye, lab, yl in ends:
            a.annotate(lab, (xe, ye), xytext=(xe + 0.6, yl), fontsize=8, color=INK, va="center")
        a.axhline(cfg["p_s"], color=INK, lw=1, ls="--")
        a.text(0.99, cfg["p_s"], f"P_s our figures use: {cfg['p_s']:.0f} W ", transform=a.get_yaxis_transform(),
               ha="right", va="bottom", fontsize=8, color=INK)
        Tm = med
        Pm = cfg["p_s"] + cfg["k"] * (Tm - I["t_ref"])
        a.annotate("", xy=(Tm, Pm), xytext=(Tm, cfg["p_s"]),
                   arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
        a.text(Tm + 0.5, (Pm + cfg["p_s"]) / 2, f"+{Pm - cfg['p_s']:.0f} W at the\nmedian serving\ntemperature ({Tm:.0f} °C)",
               fontsize=8, color=INK, va="center")
        a.set_xlabel("die temperature (°C)"); a.set_ylabel("idle power (W)")
        a.set_title(f"{g}: idle power rises {cfg['k_lo']:.2f}–{cfg['k_hi']:.2f} W/°C", loc="left", fontsize=10, color=INK)
        style(a)
        xs = np.concatenate([s[0][:, 0] for s in series if len(s[0])] + [[lo, hi]])
        a.set_xlim(xs.min() - 2, xs.max() + 12)
    fig.text(0.01, 0.01, "Lines: 1 °C medians, one clock state, >2 s after load (cool-downs) or >10 s idle (trace gaps). "
             "A6000 omitted: its idle drops to a 210 MHz state, so idle vs temperature is a clock question first.",
             fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(HERE, "fig_idle_vs_temp.png"), dpi=180, facecolor=SURF)

    # ---- figure 2: trace breakdown, as published vs at measured die temperature
    import fig_trace_roofline as ftr
    rows, caps = ftr.load()
    parts = [("static", "static (idle) power", "#b9b6ad"), ("activity", "memory activity", "#d8d4c8"),
             ("weights", "weight streaming", "#2a78d6"), ("kv", "KV cache", "#eb6834"),
             ("gemm", "GEMM compute", "#1baf7a"), ("residual", "unexplained", "#52514e")]
    bars, deltas = [], []
    for gk, gl in ftr.GPUS:
        for tk, tl, _ in ftr.TRACES:
            c = next((c for c in caps if c["gpu"] == gk and c["trace"] == tk and c["slo"] == "tight"), None)
            if not c:
                continue
            r = next((r for r in rows if r["gpu"] == gk and r["trace"] == tk and r["trace_speedup"] == c["trace_speedup"]), None)
            if not (r and r.get("energy_shares")):
                continue
            lab = f"{ftr.SHORT.get(tk, tl)}{'*' if gk == 'B200' and tk.startswith('mooncake') else ''}\n{gl}"
            if gl in GPU:
                cfg, I = GPU[gl], info[gl]
                T = I["levT"].get((r["run_dir"], r["trace_speedup"]), np.nan)
                sh = corrected_shares(r, T, cfg, I["t_ref"], cfg["k"])
                lo = corrected_shares(r, T, cfg, I["t_ref"], cfg["k_lo"])["static"]
                hi = corrected_shares(r, T, cfg, I["t_ref"], cfg["k_hi"])["static"]
                bars.append((lab, r["energy_shares"], sh, T, True))
                deltas.append(dict(gpu=gl, trace=tk, x=r["trace_speedup"], T=T,
                                   static_before=r["energy_shares"]["static"], static_after=sh["static"],
                                   static_lo=lo, static_hi=hi,
                                   resid_before=r["energy_shares"]["residual"], resid_after=sh["residual"]))
            else:
                bars.append((lab, r["energy_shares"], r["energy_shares"], np.nan, False))
    for d in deltas:
        print(f"  {d['gpu']:5s} {d['trace']:17s} x{d['x']:<5} T {d['T']:4.0f} C  static {100 * d['static_before']:5.1f} -> "
              f"{100 * d['static_after']:5.1f}% ({100 * (d['static_lo']):.1f}-{100 * d['static_hi']:.1f})  "
              f"unexplained {100 * d['resid_before']:+5.1f} -> {100 * d['resid_after']:+5.1f}%")
    json.dump(dict(gpu={g: dict(p_s=c["p_s"], k=c["k"], k_range=[c["k_lo"], c["k_hi"]], d_eb=c["d_eb"],
                                t_ref=info[g]["t_ref"], t_serve=info[g]["t_serve"]) for g, c in GPU.items()},
                   levels=deltas), open(os.path.join(HERE, "fig_trace_breakdown_temp.json"), "w"), indent=1)

    fig, axes = plt.subplots(2, 1, figsize=(max(8, 0.75 * len(bars)), 8.2), facecolor=SURF, sharex=True)
    x = np.arange(len(bars))
    for a, which, title in ((axes[0], 1, "As in fig 9: constant P_s (cold idle)"),
                            (axes[1], 2, "At each level's measured die temperature: P_s(T) = P_s + k·(T − T_ref), "
                                         "dynamic terms refit with temperature")):
        bot_p, bot_n = np.zeros(len(bars)), np.zeros(len(bars))
        for k, lab, col in parts:
            v = np.array([100 * b[which].get(k, 0.0) for b in bars])
            pos = np.clip(v, 0, None); neg = np.clip(v, None, 0)
            alpha = [1.0 if b[4] or which == 1 else 0.35 for b in bars]
            for i in range(len(bars)):
                a.bar(x[i], pos[i], bottom=bot_p[i], color=col, width=0.7, alpha=alpha[i], ec=SURF, lw=0.6,
                      label=lab if i == 0 else None)
                a.bar(x[i], neg[i], bottom=bot_n[i], color=col, width=0.7, alpha=alpha[i], ec=SURF, lw=0.6)
            bot_p += pos; bot_n += neg
        if which == 2:
            for i, b in enumerate(bars):
                if b[4]:
                    s0 = 100 * b[1]["static"]
                    a.plot([i - 0.35, i + 0.35], [s0, s0], color=INK, lw=1.2)
                    a.text(i, max(bot_p[i], 100) + 2, f"{b[3]:.0f}°C\n+{100 * (b[2]['static'] - b[1]['static']):.1f}",
                           ha="center", fontsize=7, color=INK2)
                else:
                    a.text(i, max(bot_p[i], 100) + 2, "not\ncorrected", ha="center", fontsize=6.5, color=INK2)
            a.plot([], [], color=INK, lw=1.2, label="top of static slice before")
        a.axhline(100, color=INK2, lw=0.8, ls="--")
        a.set_ylim(-8, 118); a.set_ylabel("share of measured energy (%)")
        a.set_title(title, loc="left", fontsize=10, color=INK); style(a)
    axes[1].set_xticks(x); axes[1].set_xticklabels([b[0] for b in bars], fontsize=7.5)
    axes[1].legend(ncol=7, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    A, B = GPU["A100"], GPU["B200"]
    fig.text(0.01, 0.01,
             f"Number above each bar: median die temperature and change in static share (points). k = idle cool-down slope "
             f"(A100 {A['k']}, B200 {B['k']} W/°C); T_ref = where the serving card's idle equals P_s "
             f"(A100 {info['A100']['t_ref']:.0f} °C, B200 {info['B200']['t_ref']:.0f} °C); dynamic terms x{1 + A['d_eb']:.3f} (A100), "
             f"x{1 + B['d_eb']:.3f} (B200).\nA6000 faded: its thermal data are at a 210 MHz idle state, not the clocks-up state "
             "it serves in. Totals (J/token) unchanged.", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(HERE, "fig_trace_breakdown_temp.png"), dpi=180, facecolor=SURF)
    print("wrote fig_idle_vs_temp.png, fig_trace_breakdown_temp.png")


if __name__ == "__main__":
    main()
