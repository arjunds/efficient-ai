#!/usr/bin/env python3
"""Figures for PI_UPDATE_2026-10.md (response to the PI's feedback). Run from GPU_power/:
  PYTHONPATH=/shared_data0/adsampat/pydeps python3 pi_update_figs/make_figs.py
Inputs (all committed): microbench/results_B200_steady.json (+ _square_multi_trace.csv),
microbench/steady_calib_B200.json, gpu_coefficients.json, balance_runs.csv,
balance_analysis.json."""
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "pi_update_figs"
R = json.load(open("microbench/results_B200_steady.json"))
SC = json.load(open("microbench/steady_calib_B200.json"))
COEF = json.load(open("gpu_coefficients.json"))
BA = json.load(open("balance_analysis.json"))
CAPPED = 0.05
PEAK_BW, PEAK_F, LIMIT = 8e12, 2.25e15, 1000.0
IDLE = SC["idle_deep_w"]; BUDGET = LIMIT - IDLE
A0 = SC["grid"]["intercept_w"]; EB = SC["e_byte_full_clock"]; EF = SC["llmgemm_fit"]["e_flop"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def med(p, k):
    v = [r[k] for r in p["repeats"] if r.get(k) is not None]
    return float(np.median(v)) if v else None


def save(fig, name):
    fig.tight_layout(); p = os.path.join(OUT, name); fig.savefig(p, dpi=150); plt.close(fig); print("wrote", p)


# ---------------------------------------------------------------- U1 smoothing
def fig_smoothing():
    rows = [r for r in csv.DictReader(open("microbench/results_B200_steady_square_multi_trace.csv"))]
    t = np.array([float(r["t"]) for r in rows]); pi = np.array([float(r["p_inst_w"]) for r in rows])
    pa = np.array([float(r["p_avg_w"]) for r in rows]); on = np.array([float(r["on"] or 0) for r in rows])
    per = np.array([float(r["period_s"]) if r["period_s"] else np.nan for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw=dict(width_ratios=[1.6, 1]))
    a = ax[0]; sel = per == 0.5
    t0 = t[sel][0]; w = sel & (t >= t0 + 0.5) & (t <= t0 + 3.0)
    tt = t[w] - t[w][0]
    swing = np.percentile(pi[per == 2.0], 97) - IDLE
    a.fill_between(tt, IDLE, IDLE + on[w] * swing, step="post", color="#dddddd", label="load on (true)")
    a.plot(tt, pi[w], color="#1f77b4", lw=1.2, label="POWER_INSTANT (fit: 0.10 s window)")
    a.plot(tt, pa[w], color="#d62728", lw=1.6, label="GetPowerUsage (fit: 0.85 s window)")
    for b in np.arange(0, tt[-1] + 0.01, 0.2):
        a.axvline(b, color="k", lw=0.3, alpha=0.3)
    a.set_xlabel("time (s); thin lines = our 200 ms bins"); a.set_ylabel("board power (W)")
    a.set_title("(a) B200, 0.5 s on/off load: what each NVML reading sees", loc="left")
    a.legend(fontsize=7.5, loc="upper right", ncol=1, framealpha=0.9)
    b = ax[1]; ps = [0.1, 0.25, 0.5, 1.0, 2.0]; fi, fa = [], []
    for p in ps:
        s = per == p
        fi.append((np.percentile(pi[s], 97) - np.percentile(pi[s], 3)) / swing)
        fa.append((np.percentile(pa[s], 97) - np.percentile(pa[s], 3)) / swing)
    b.plot(ps, fi, "o-", color="#1f77b4", label="POWER_INSTANT"); b.plot(ps, fa, "s-", color="#d62728", label="GetPowerUsage")
    b.axvline(0.2, color="k", ls=":", lw=1); b.text(0.21, 0.08, "200 ms bin", fontsize=7.5)
    b.set_xscale("log"); b.set_ylim(0, 1.1); b.set_xticks(ps); b.set_xticklabels([str(p) for p in ps])
    b.set_xlabel("on/off period (s)"); b.set_ylabel("fraction of true power swing seen")
    b.set_title("(b) GetPowerUsage misses sub-second swings", loc="left"); b.legend(fontsize=7.5)
    save(fig, "fig_u1_nvml_smoothing.png")


# ---------------------------------------------------------------- U2 HBM blast
def fig_hbm():
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    pts = []
    for p in R["grid"]["points"]:
        pts.append(("occupancy grid", med(p, "bytes_per_s"), med(p, "p_dyn_w"), med(p, "frac_sw_power_cap"), med(p, "sm_mhz")))
    for p in R["stream"]:
        if (p.get("ws_bytes") or 0) >= 8 * R["gpu"]["l2_bytes"]:
            pts.append(("1-4 GB stream", med(p, "bytes_per_s"), med(p, "p_dyn_w"), med(p, "frac_sw_power_cap"), med(p, "sm_mhz")))
    for p in R["vendor"]:
        pts.append(("torch sum/copy", med(p, "bytes_per_s"), med(p, "p_dyn_w"), med(p, "frac_sw_power_cap"), med(p, "sm_mhz")))
    mk = {"occupancy grid": "o", "1-4 GB stream": "s", "torch sum/copy": "^"}
    for lab in mk:
        for capped in (False, True):
            s = [q for q in pts if q[0] == lab and (q[3] >= CAPPED) == capped]
            if s:
                ax.scatter([q[1] / PEAK_BW for q in s], [q[2] for q in s], marker=mk[lab], s=40,
                           facecolor="white" if capped else "#1f77b4", edgecolor="#d62728" if capped else "#1f77b4",
                           label=f"{lab}{' (at power cap)' if capped else ''}", zorder=3)
    x = np.linspace(0, 1.0, 50)
    ax.plot(x, A0 + EB * x * PEAK_BW, color="#1f77b4", lw=1,
            label=f"full-clock fit: {A0:.0f} W + {EB*1e12:.0f} pJ/B x BW")
    ax.axhline(BUDGET, color="#d62728", ls="--", lw=1); ax.text(0.02, BUDGET + 12, f"power limit ({BUDGET:.0f} W above idle)", color="#d62728", fontsize=7.5)
    ax.axvline(EB and (BUDGET - A0) / EB / PEAK_BW, color="grey", ls=":", lw=1)
    ax.text((BUDGET - A0) / EB / PEAK_BW - 0.02, 120, "max BW at\nfull clock\n(73%)", ha="right", fontsize=7.5, color="grey")
    best = max(pts, key=lambda q: q[1])
    ax.annotate(f"{best[1]/PEAK_BW:.0%} of peak,\nSM throttled to {best[4]:.0f} MHz", (best[1] / PEAK_BW, best[2]),
                xytext=(0.55, 880), fontsize=7.5, arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1100)
    ax.set_xlabel("achieved DRAM bandwidth (fraction of 8 TB/s peak)"); ax.set_ylabel("dynamic power (W above idle)")
    ax.set_title("B200 'blast HBM' proxy kernels (10 s steady loops)", loc="left")
    ax.legend(fontsize=7, loc="lower right")
    save(fig, "fig_u2_hbm_blast.png")


# ---------------------------------------------------------------- U3 power roofline
def fig_roofline():
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    ai = np.logspace(-0.3, 4, 200)
    ax.plot(ai, np.minimum(PEAK_F, PEAK_BW * ai) / 1e12, color="k", lw=1.2, label="datasheet roofline")
    pw = (BUDGET - A0) / (EB / ai + EF)
    ax.plot(ai, np.minimum(pw, np.minimum(PEAK_F, PEAK_BW * ai)) / 1e12, color="#d62728", lw=1.5, ls="--",
            label=f"1000 W limit at full clock ({EF*1e12:.2f} pJ/FLOP)")
    jf = SC["llmgemm_capped"]["total_j_per_flop_M2048plus"]; efc = float(np.mean(jf))
    pwc = (BUDGET - A0) / (EB / ai + efc)
    ax.plot(ai, np.minimum(pwc, np.minimum(PEAK_F, PEAK_BW * ai)) / 1e12, color="#d62728", lw=1, ls=":",
            label=f"1000 W limit, throttled clocks (~{efc*1e12:.1f} pJ/FLOP)")
    col = {"qkv": "#1f77b4", "o": "#2ca02c", "mlp_up": "#9467bd", "mlp_down": "#ff7f0e"}
    for shp, c in col.items():
        for capped in (False, True):
            s = [p for p in R["llmgemm"] if p["shape"] == shp and ((med(p, "frac_sw_power_cap") or 0) >= CAPPED) == capped]
            if s:
                ax.scatter([p["arith_intensity"] for p in s], [med(p, "flops_per_s") / 1e12 for p in s], s=30,
                           facecolor="white" if capped else c, edgecolor=c, lw=1.2, zorder=3,
                           label=f"{shp} GEMM" if not capped else None)
    rows = [r for r in csv.DictReader(open("balance_runs.csv")) if r["gpu"] == "B200" and r["valid"] == "True"]
    ax.scatter([float(r["ai_flop_per_byte"]) for r in rows], [float(r["realized_tflops"]) for r in rows],
               marker="x", color="grey", s=22, label="serving runs (whole-run average)", zorder=2)
    ax.scatter([], [], facecolor="white", edgecolor="k", label="open = ran at power cap")
    ax.text(7e2, 25, "capped GEMMs run above the\nfull-clock line by dropping SM\nclock to 1.1-1.7 GHz:\n50-58% of dense peak", fontsize=7)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1, 4000); ax.set_xlim(0.5, 1e4)
    ax.set_xlabel("arithmetic intensity (FLOP / DRAM byte); decode ~ tokens per step")
    ax.set_ylabel("achieved TFLOP/s")
    ax.set_title("B200: the power limit, not the datasheet roofline, binds prefill", loc="left")
    ax.legend(fontsize=7, loc="upper left")
    save(fig, "fig_u3_power_roofline.png")


# ---------------------------------------------------------------- U4 coefficients
def fig_coef():
    b = COEF["B200"]; h = COEF["H200"]; rl = BA["runlevel"]; pin = SC["serving_refit_fixed_egemm"]
    sets = {
        "B200": [("per-bin serving fit\n(in Sept brief)", b["e_wbyte"], b["e_kvbyte"], b["e_gemm"]),
                 ("whole-run serving fit\n(no smear)", rl["B200/fit_root"]["e_wbyte"], rl["B200/fit_root"]["e_kvbyte"], rl["B200/fit_root"]["e_gemm"]),
                 ("steady microbench\n(+ serving refit)", pin["e_wbyte"], pin["e_kvbyte"], EF)],
        "H200": [("per-bin serving fit\n(in Sept brief)", h["e_wbyte"], h["e_kvbyte"], h["e_gemm"]),
                 ("whole-run serving fit\n(no smear)", rl["H200/fit_root"]["e_wbyte"], rl["H200/fit_root"]["e_kvbyte"], rl["H200/fit_root"]["e_gemm"])]}
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw=dict(width_ratios=[3, 2]))
    cols = ["#888888", "#1f77b4", "#2ca02c"]
    for a, (g, S) in zip(ax, sets.items()):
        x = np.arange(3); wd = 0.8 / len(S)
        for i, (lab, w, k, f) in enumerate(S):
            v = [w * 1e10, k * 1e10, f * 1e12]
            bars = a.bar(x + (i - (len(S) - 1) / 2) * wd, v, wd, color=cols[i], label=lab)
            for bb, vv in zip(bars, v):
                a.text(bb.get_x() + bb.get_width() / 2, vv + 0.05, f"{vv:.2f}", ha="center", fontsize=7)
        a.set_xticks(x); a.set_xticklabels(["weight byte\n(1e-10 J/B)", "KV byte\n(1e-10 J/B)", "GEMM FLOP\n(pJ)"])
        a.set_title(f"{g}", loc="left"); a.set_ylim(0, 3.9); a.legend(fontsize=7, loc="upper right")
    save(fig, "fig_u4_coefficients.png")


# ---------------------------------------------------------------- U5 balance (measured)
def fig_balance():
    rows = [r for r in csv.DictReader(open("balance_runs.csv")) if r["valid"] == "True"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    a = ax[0]; parts = ["share_static", "share_weights", "share_kv", "share_gemm"]
    cols = ["#888888", "#1f77b4", "#9467bd", "#d62728"]; labs = ["static (idle)", "weight bytes", "KV bytes", "GEMM FLOPs"]
    k = 0; xs, tk = [], []
    for g in ("H200", "B200"):
        for c in ("1", "4", "16", "64"):
            s = [r for r in rows if r["gpu"] == g and r["conc"] == c and r["model"] in ("Qwen2-7B-Instruct", "Mistral-7B-v0.1")]
            bot = 0
            for p, cc, lb in zip(parts, cols, labs):
                v = max(0, float(np.mean([float(r[p]) for r in s])))
                a.bar(k, v, bottom=bot, color=cc, label=lb if k == 0 else None); bot += v
            xs.append(k); tk.append(f"{g}\nc={c}"); k += 1
        k += 0.6
    a.set_xticks(xs); a.set_xticklabels(tk, fontsize=7.5); a.set_ylabel("share of energy"); a.set_ylim(0, 1.18)
    a.legend(fontsize=7, ncol=4, loc="upper center"); a.set_title("(a) where the energy goes (7B, measured runs)", loc="left")
    b = ax[1]; M = BA["matched"]
    for mdl, mk in (("Mistral-7B-v0.1", "o"), ("Qwen2-7B-Instruct", "s"), ("Qwen2.5-32B-Instruct", "^")):
        s = [r for r in M if r["model"] == mdl]
        b.scatter([r["speedup"] for r in s], [r["jtok_ratio"] for r in s], marker=mk, s=30, label=mdl.replace("-Instruct", ""))
    b.axvline(1.67, color="grey", ls=":"); b.text(1.62, 1.45, "B200 BW\nratio 1.67", ha="right", fontsize=7.5, color="grey")
    b.axhline(1.0, color="k", lw=0.6)
    b.set_xlim(0.9, 1.75); b.set_ylim(0.9, 1.6)
    b.set_xlabel("B200 throughput / H200 (same model, load)"); b.set_ylabel("B200 J/token / H200")
    b.set_title("(b) B200 converts little of its extra BW; idle 2x not repaid", loc="left"); b.legend(fontsize=7)
    save(fig, "fig_u5_balance.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_smoothing(); fig_hbm(); fig_roofline(); fig_coef(); fig_balance()
