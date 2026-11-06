#!/usr/bin/env python3
"""Three figures for the thesis "LIMINAL's TDP energy is wrong off the cap; its peak-bandwidth time
model misses at high concurrency and long context; mixed-step FA2 attention is a missing term".
  fig_thesis_tdp.png  : J/token error of LIMINAL as published (LIMINAL time x TDP) vs LIMINAL-E,
                        against measured power / TDP; MAPE by GPU at the cap vs below it
  fig_thesis_time.png : B200 throughput predicted/measured vs concurrency and prompt length:
                        LIMINAL (peak BW + 114 us/layer), LIMINAL-E, LIMINAL-E + high-c terms
  fig_thesis_fa2.png  : FA2 mixed-step penalty, measured vs parameter-free prediction (A100/A6000);
                        profiler kernel split; high-c throughput error with and without the terms
Inputs: liminal_energy/validation_v2_runs.csv (validate_v2.py), timemodel_highc.json
(timemodel_highc.py). H200 (Sept, eager, old logger, no microbench) is left out.
  PYTHONPATH=/shared_data0/adsampat/pydeps python3 talk_figs/fig_thesis.py
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: F401
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
GPUS = [("A6000", "A6000 (300 W)", "#eda100"), ("A100-80-PCIe", "A100 (300 W)", "#1baf7a"),
        ("B200", "B200 (1000 W)", "#eb6834")]
TDP = {"A6000": 300.0, "A100-80-PCIe": 300.0, "B200": 1000.0}
C_LIM, C_BASE, C_FULL = "#8a8780", "#2a78d6", "#0b4fa0"


def style(a):
    a.set_facecolor(SURF); a.grid(color=GRID, lw=0.8); a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_runs():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "liminal_energy", "validation_v2_runs.csv"))))
    out = []
    for r in rows:
        if r["gpu"] not in TDP or r["moe"] == "True" or r["tp"] != "1":
            continue
        x = {k: fnum(v) for k, v in r.items()}
        x.update(gpu=r["gpu"], arm=r["arm"], key=r["key"], model=r["model"], task=r["task"], eager=r["eager"] == "True",
                 capped=r["meas_capped"] == "True")
        out.append(x)
    return out


def ape(p, m):
    return 100.0 * abs(p / m - 1.0)


# ---------------------------------------------------------------------------------------------
def fig_tdp(runs):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.4, 4.6), facecolor=SURF, gridspec_kw=dict(width_ratios=[1.15, 1]))
    for gk, gl, col in GPUS:
        R = [x for x in runs if x["gpu"] == gk and x["j_meas"] and x["p_meas"]]
        pf = [100 * x["p_meas"] / TDP[gk] for x in R]
        a.scatter(pf, [x["j_liminal_tdp"] / x["j_meas"] for x in R], marker="x", s=22, color=col, alpha=0.55, lw=1.1)
        Z = [(p, x["j_zs"] / x["j_meas"]) for p, x in zip(pf, R) if x.get("j_zs")]
        a.scatter([z[0] for z in Z], [z[1] for z in Z], s=20, color=col, edgecolor="white", lw=0.5, label=gl, zorder=3)
    a.axhline(1, color=INK2, lw=1)
    a.set_yscale("log"); a.set_ylim(0.5, 4); a.set_yticks([0.5, 0.75, 1, 1.5, 2, 3, 4])
    a.set_yticklabels(["0.5", "0.75", "1", "1.5", "2", "3", "4"]); a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.set_xlim(15, 108); a.axvline(93, color=INK2, lw=0.8, ls=":")
    a.text(92, 0.53, "at cap ", fontsize=8, color=INK2, ha="right")
    a.set_xlabel("measured average power, % of TDP"); a.set_ylabel("predicted / measured J per token")
    a.set_title("(a) Charging TDP over-predicts energy wherever the GPU is below its cap\n"
                "x = LIMINAL as published (its time x TDP)   dot = LIMINAL-E zero-shot",
                loc="left", fontsize=9.5, color=INK)
    a.legend(frameon=False, fontsize=8.5, loc="upper right", bbox_to_anchor=(0.9, 1.0)); style(a)

    # (b) MAPE by GPU x cap state
    groups, vals = [], {"LIMINAL (TDP)": [], "LIMINAL-E zero-shot": [], "LIMINAL-E calibrated": []}
    for gk, gl, _ in GPUS:
        for cap, cl in ((True, "at cap"), (False, "below cap")):
            R = [x for x in runs if x["gpu"] == gk and x["capped"] == cap and x["j_meas"]]
            if len(R) < 3:
                continue
            groups.append("%s\n%s (n=%d)" % (gl.split(" ")[0], cl, len(R)))
            vals["LIMINAL (TDP)"].append(np.mean([ape(x["j_liminal_tdp"], x["j_meas"]) for x in R]))
            vals["LIMINAL-E zero-shot"].append(np.mean([ape(x["j_zs"], x["j_meas"]) for x in R if x.get("j_zs")]))
            vals["LIMINAL-E calibrated"].append(np.mean([ape(x["j_cal"], x["j_meas"]) for x in R if x.get("j_cal")]))
    x = np.arange(len(groups)); wd = 0.26
    for i, (k, col) in enumerate(zip(vals, (C_LIM, C_BASE, C_FULL))):
        bars = b.bar(x + (i - 1) * wd, vals[k], wd, color=col, label=k)
        for xx, v in zip(x + (i - 1) * wd, vals[k]):
            b.text(xx, v + 1.5, "%.0f" % v, ha="center", fontsize=7.5, color=INK2)
    b.set_xticks(x); b.set_xticklabels(groups, fontsize=8)
    b.set_ylabel("J/token mean absolute error (%)")
    b.set_title("(b) Error by GPU and cap state (dense, TP1; H200 omitted)", loc="left", fontsize=9.5, color=INK)
    b.legend(frameon=False, fontsize=8.5, loc="upper left"); style(b)
    fig.text(0.01, 0.01, "Runs: Qwen2/2.5 0.5B-72B, alpaca/sharegpt/long-context, c=1-512, eager and CUDA graphs "
             "(liminal_energy/validation_v2_runs.csv). At cap = measured power >= 93% of TDP.\n"
             "A100 below cap: LIMINAL's 13% is error cancellation (its time over-predicts speed on <=3B models, offsetting the "
             "TDP over-charge); LIMINAL-E zero-shot's 27% is the same host overhead, removed by calibration.", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(HERE, "fig_thesis_tdp.png"), dpi=180, facecolor=SURF)


# ---------------------------------------------------------------------------------------------
def fig_time(runs, hc):
    H = {(r["gpu"], r["key"]): r for r in hc["runs"]}
    R = [x for x in runs if x["gpu"] == "B200" and not x["eager"] and x["arm"] in
         ("graphs", "highconc", "longctx", "longctx32", "tp1ref") and x["tok_s_meas"]]
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 4.5), facecolor=SURF, sharey=True)
    series = [("LIMINAL (peak BW + 114 us/layer)", lambda x, h: x["tok_s_liminal"], C_LIM, "s"),
              ("LIMINAL-E", lambda x, h: h and h.get("tok|canonical|base"), C_BASE, "o"),
              ("LIMINAL-E + high-c terms", lambda x, h: h and h.get("tok|canonical|full (B200 terms + FA2)"), C_FULL, "D")]
    for a, longctx in zip(axs, (False, True)):
        S = [x for x in R if x["arm"].startswith("longctx") == longctx]
        for i, (lab, f, col, mk) in enumerate(series):
            pts = []
            for x in S:
                p = f(x, H.get(("B200", x["key"])))
                if p:
                    xv = x["P"] if longctx else x["conc"]
                    pts.append((xv * (1 + 0.06 * (i - 1)), p / x["tok_s_meas"], "32B" in x["model"] or "72B" in x["model"]))
            a.scatter([p[0] for p in pts if not p[2]], [p[1] for p in pts if not p[2]], marker=mk, s=26, color=col,
                      label=lab, zorder=3, edgecolor="white", lw=0.4)
            a.scatter([p[0] for p in pts if p[2]], [p[1] for p in pts if p[2]], marker=mk, s=26, facecolor=SURF,
                      edgecolor=col, lw=1.1, zorder=3)
        a.axhline(1, color=INK2, lw=1); a.set_xscale("log"); style(a)
    axs[0].axvspan(110, 600, color="#f3efe6", zorder=0)
    axs[0].text(125, 1.62, "7B c=128-512: fit set\nfor the high-c terms", fontsize=7.5, color=INK2)
    axs[0].set_xlabel("concurrent requests (short prompts: alpaca, sharegpt)")
    axs[1].set_xlabel("prompt length, tokens (long-context arm, c=4-16)")
    axs[0].set_ylabel("predicted / measured tokens per second")
    axs[0].set_ylim(0.8, 1.75)
    axs[0].set_title("(a) Short prompts: LIMINAL misses only the 7B at c>=64, a per-sequence\n"
                     "host cost (4.8 us/seq), not bandwidth: the 32B (open) is predicted fine", loc="left", fontsize=9.5, color=INK)
    axs[1].set_title("(b) Long context (KV-byte dominated): assuming peak bandwidth\n"
                     "over-predicts throughput by 16-64%", loc="left", fontsize=9.5, color=INK)
    axs[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.text(0.01, 0.01, "B200, dense TP1, CUDA graphs (Oct runs). Filled = Qwen2-7B; open = 32B/72B (the 32B c=128-512 "
             "points are held out from the high-c fit).\nLIMINAL-E time terms were fitted on B200/H200 serving data, so "
             "this panel tests the functional form, not zero-shot transfer.", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(HERE, "fig_thesis_time.png"), dpi=180, facecolor=SURF)


# ---------------------------------------------------------------------------------------------
def fig_fa2(hc):
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(14.2, 4.4), facecolor=SURF,
                                  gridspec_kw=dict(width_ratios=[1.1, 0.8, 1]))
    for gk, gl, col in GPUS[:2]:
        E = [e for e in hc["evidence"] if e["gpu"] == gk and e.get("mix_penalty_ms") is not None
             and e.get("fa2_pred_ms") is not None]
        a.scatter([e["fa2_pred_ms"] for e in E], [e["mix_penalty_ms"] for e in E], s=30, color=col,
                  edgecolor="white", lw=0.5, label="%s (%d runs)" % (gl, len(E)), zorder=3)
    lim = 60
    a.plot([-5, lim], [-5, lim], color=INK2, lw=1, ls="--", label="y = x")
    a.set_xlim(-2, lim); a.set_ylim(-6, lim)
    a.set_xlabel("predicted extra time per mixed step (ms)\n(g-1) x decode KV bytes at the streaming rate; no fitted constant")
    a.set_ylabel("measured extra time per mixed step (ms)")
    a.set_title("(a) Mixed prefill/decode steps: measured penalty\nvs the parameter-free FA2 term", loc="left",
                fontsize=9.5, color=INK)
    a.legend(frameon=False, fontsize=8.5, loc="upper left"); style(a)

    # (b) profiler: GPU time per step (A5000, Qwen2.5-7B, azure23_code; logs/diag_admission, TRACES.md)
    kinds = ["pure decode\n(294 steps)", "mixed\n(6 steps)"]
    attn, gemm, other = np.array([7.0, 79.0]), np.array([82.0, 89.0]), np.array([2.0, 3.0])
    b.bar(kinds, gemm, color="#b9b6ad", label="GEMM")
    b.bar(kinds, attn, bottom=gemm, color="#d03b3b", label="attention")
    b.bar(kinds, other, bottom=gemm + attn, color="#e6e5e0", label="other")
    for i, t in enumerate(["7 ms attn\ngrid [1,32,4]\n32 KV splits,\nGQA packed", "79 ms attn\ngrid [1,b,28]\n1 split,\nno GQA packing"]):
        b.text(i, gemm[i] + attn[i] + other[i] + 4, t, ha="center", fontsize=7.5, color=INK2)
    b.set_ylim(0, 225); b.set_ylabel("GPU time per step (ms)")
    b.set_title("(b) Profiler: why (A5000, FA2)", loc="left", fontsize=9.5, color=INK)
    b.legend(frameon=False, fontsize=8, loc="upper left"); style(b)

    # (c) throughput error with and without the terms (zero-shot profile)
    V = hc["validation"]["zeroshot"]
    sets = [("A100 high-c c128-256", "A100\nc=128-256"), ("A6000 high-c c128-256", "A6000\nc=128-256"),
            ("A100 graphs c<=64 (7B, 14B)", "A100\nc<=64"), ("A6000 c<=64 (graphs+eager)", "A6000\nc<=64")]
    x = np.arange(len(sets)); wd = 0.36
    for i, (k, lab, col) in enumerate((("base", "LIMINAL-E", C_BASE), ("full (B200 terms + FA2)", "+ FA2 / host terms", C_FULL))):
        v = [V[s][k]["tok"]["mape"] for s, _ in sets]
        c.bar(x + (i - 0.5) * wd, v, wd, color=col, label=lab)
        for xx, vv in zip(x + (i - 0.5) * wd, v):
            c.text(xx, vv + 0.8, "%.0f" % vv, ha="center", fontsize=7.5, color=INK2)
    c.set_xticks(x); c.set_xticklabels(["%s\n(n=%d)" % (l, V[s]["base"]["tok"]["n"]) for s, l in sets], fontsize=8)
    c.set_ylabel("tokens/s mean absolute error (%)")
    c.set_title("(c) Zero-shot throughput error: the term fixes high c\nwithout hurting low c", loc="left",
                fontsize=9.5, color=INK)
    c.legend(frameon=False, fontsize=8.5, loc="upper right"); style(c)
    fig.text(0.01, 0.01, "(a, c) A100/A6000 vLLM runs, Qwen2-7B and Qwen2.5-14B (timemodel_highc.json); high-c sets are "
             "small (3-4 runs). (b) torch profiler, one azure23_code level (logs/diag_admission).", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(HERE, "fig_thesis_fa2.png"), dpi=180, facecolor=SURF)


def main():
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2})
    runs = load_runs()
    hc = json.load(open(os.path.join(ROOT, "timemodel_highc.json")))
    fig_tdp(runs); fig_time(runs, hc); fig_fa2(hc)
    print("wrote fig_thesis_{tdp,time,fa2}.png from", len(runs), "runs")


if __name__ == "__main__":
    main()
