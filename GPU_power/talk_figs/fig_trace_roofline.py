#!/usr/bin/env python3
"""Real decode workloads on the energy roofline (TRACES.md) - paper/talk figures.
  fig_trace_roofline.png : per GPU, achieved FLOP/s vs arithmetic intensity of every trace x
                           load level, over the bandwidth, full-clock power and compute roofs
  fig_trace_jtok.png     : J/token vs request rate per trace, per GPU; open marker = misses
                           the tight SLO (TPOT p99 50 ms / TTFT p99 1 s)
  fig_trace_breakdown.png: energy shares at each trace's tight-SLO capacity, per GPU
Inputs: logs/<GPU>_prov/trace_summary.json (trace_analysis.py).
  PYTHONPATH=/shared_data0/adsampat/pydeps python3 talk_figs/fig_trace_roofline.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "liminal_energy"))
import liminal_e as lim  # noqa: E402

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
SHORT = {"azure23_conv": "Az chat", "azure24_conv_h32": "Az '24", "burstgpt_h521": "Burst",
         "azure23_code": "Az code", "mooncake_conv": "MC chat", "mooncake_tool": "MC agent", "reasoning_math": "Reason"}
TRACES = [  # key, label, color
    ("azure23_conv", "Azure chat '23", "#2a78d6"), ("azure24_conv_h32", "Azure chat '24 peak", "#6aa6e8"),
    ("burstgpt_h521", "BurstGPT", "#1baf7a"), ("azure23_code", "Azure code '23", "#9a6fd1"),
    ("mooncake_conv", "Mooncake chat (long ctx)", "#eb6834"), ("mooncake_tool", "Mooncake agent", "#c2410c"),
    ("reasoning_math", "Reasoning", "#eda100")]
GPUS = [("B200", "B200"), ("A100-80-PCIe", "A100"), ("A6000", "A6000")]


def load():
    rows, caps = [], []
    for g in ("B200", "A100", "A6000", "H200"):
        p = os.path.join(ROOT, "logs", f"{g}_prov", "trace_summary.json")
        if os.path.exists(p):
            s = json.load(open(p))
            rows += s["levels"]; caps += s["capacity"]
    return rows, caps


def style(a):
    a.set_facecolor(SURF); a.grid(color=GRID, lw=0.8); a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)


def ok(r):
    return (r.get("tpot_p99_ms") or 1e9) <= 50 and (r.get("ttft_p99_ms") or 1e9) <= 1000


def main():
    rows, caps = load()
    gpus = [(k, l) for k, l in GPUS if any(r["gpu"] == k for r in rows)]
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2})
    # 1. roofline placement
    fig, ax = plt.subplots(1, len(gpus), figsize=(5.2 * len(gpus), 4.6), facecolor=SURF, squeeze=False)
    Is = np.logspace(0, 3, 200)
    for a, (gk, gl) in zip(ax[0], gpus):
        g = lim.gpu(gk)
        rf = [lim.roofline(g, I) for I in Is]
        a.plot(Is, [r["bw_roof"] / 1e12 for r in rf], color=INK2, ls="--", lw=1, label="bandwidth roof")
        a.plot(Is, [r["power_roof"] / 1e12 for r in rf], color="#d03b3b", lw=1.6, label="power roof (full clock)")
        a.axhline(g["peak_flops"] / 1e12, color=INK2, ls=":", lw=1, label="compute roof")
        for tk, tl, col in TRACES:
            R = sorted([r for r in rows if r["gpu"] == gk and r["trace"] == tk and r.get("intensity")],
                       key=lambda r: r["trace_speedup"])
            if not R:
                continue
            a.plot([r["intensity"] for r in R], [r["flops_per_s"] / 1e12 for r in R], "-", color=col, lw=1, alpha=0.6)
            for r in R:
                a.plot(r["intensity"], r["flops_per_s"] / 1e12, "o", ms=6, color=col if ok(r) else SURF,
                       mec=col, mew=1.4, label=tl if r is R[0] else None)
        a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(1, 1000); a.set_ylim(1, g["peak_flops"] / 1e12 * 2)
        a.set_xlabel("arithmetic intensity (FLOP / DRAM byte)"); a.set_ylabel("achieved TFLOP/s")
        a.set_title(f"{gl}: real decode traffic on the roofline", loc="left", fontsize=10, color=INK); style(a)
    h, l = [], []
    for a in ax[0]:
        for hh, ll in zip(*a.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.02))
    fig.text(0.01, 0.01, "Qwen2.5-7B, decode only (prompts prefix-cache-warmed). Each line = one trace at increasing "
             "arrival rate; open marker = misses TPOT p99 50 ms / TTFT p99 1 s.", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.9)); fig.savefig(os.path.join(HERE, "fig_trace_roofline.png"), dpi=180, facecolor=SURF)

    # 2. J/token vs request rate
    fig, ax = plt.subplots(1, len(gpus), figsize=(5.2 * len(gpus), 4.3), facecolor=SURF, squeeze=False, sharey=True)
    for a, (gk, gl) in zip(ax[0], gpus):
        for tk, tl, col in TRACES:
            R = sorted([r for r in rows if r["gpu"] == gk and r["trace"] == tk and r.get("j_per_token")],
                       key=lambda r: r["req_per_s"])
            if not R:
                continue
            a.plot([r["req_per_s"] for r in R], [r["j_per_token"] for r in R], "-", color=col, lw=1.4)
            for r in R:
                a.plot(r["req_per_s"], r["j_per_token"], "o", ms=5.5, color=col if ok(r) else SURF, mec=col, mew=1.3,
                       label=tl if r is R[0] else None)
        a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("request rate served by one GPU (req/s)")
        a.set_title(gl, loc="left", fontsize=10, color=INK); style(a)
    ax[0][0].set_ylabel("energy per generated token (J)")
    h, l = [], []
    for a in ax[0]:
        for hh, ll in zip(*a.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, loc="upper center", ncol=7, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(os.path.join(HERE, "fig_trace_jtok.png"), dpi=180, facecolor=SURF)

    # 3. breakdown at tight-SLO capacity
    parts = [("static", "static (idle) power", "#b9b6ad"), ("activity", "memory activity", "#d8d4c8"),
             ("weights", "weight streaming", "#2a78d6"), ("kv", "KV cache", "#eb6834"),
             ("gemm", "GEMM compute", "#1baf7a"), ("residual", "unexplained", "#52514e")]
    bars = []
    for gk, gl in gpus:
        for tk, tl, _ in TRACES:
            c = next((c for c in caps if c["gpu"] == gk and c["trace"] == tk and c["slo"] == "tight"), None)
            if not c:
                continue
            r = next((r for r in rows if r["gpu"] == gk and r["trace"] == tk and r["trace_speedup"] == c["trace_speedup"]), None)
            if r and r.get("energy_shares"):
                bars.append((f"{SHORT.get(tk, tl)}\n{gl}", r["energy_shares"], r["j_per_token"]))
    if bars:
        fig, a = plt.subplots(figsize=(max(8, 0.75 * len(bars)), 4.6), facecolor=SURF)
        x = np.arange(len(bars))
        bot_p, bot_n = np.zeros(len(bars)), np.zeros(len(bars))
        for k, lab, col in parts:
            v = np.array([100 * b[1].get(k, 0.0) for b in bars])
            pos = np.clip(v, 0, None); neg = np.clip(v, None, 0)
            a.bar(x, pos, bottom=bot_p, color=col, label=lab, width=0.7); bot_p += pos
            a.bar(x, neg, bottom=bot_n, color=col, width=0.7); bot_n += neg
        for i, b in enumerate(bars):
            a.text(i, max(bot_p[i], 100) + 2, f"{b[2]:.2f}", ha="center", fontsize=7.5, color=INK2)
        a.set_xticks(x); a.set_xticklabels([b[0] for b in bars], fontsize=7.5)
        a.set_ylabel("share of measured energy (%)"); a.axhline(100, color=INK2, lw=0.8, ls="--")
        a.set_ylim(min(-5, bot_n.min() - 2), 112)
        a.set_title("Where the energy goes at each workload's tight-SLO capacity\n"
                    "(LIMINAL-E terms on the measured timeline; number = J/token)", loc="left", fontsize=10, color=INK)
        style(a); a.legend(ncol=6, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22))
        fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_trace_breakdown.png"), dpi=180, facecolor=SURF)
    print("wrote fig_trace_roofline.png, fig_trace_jtok.png, fig_trace_breakdown.png;", len(rows), "levels,", len(bars), "bars")


if __name__ == "__main__":
    main()
