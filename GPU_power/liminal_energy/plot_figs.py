#!/usr/bin/env python3
"""plot_figs.py (v2, 4 measured GPUs) -- draw LIMINAL-E figures from figs_data.json and
validation_v2_runs.csv.  Separate process from the model code (matplotlib from pydeps):
  PYTHONPATH=/shared_data0/adsampat/pydeps python3 liminal_energy/plot_figs.py
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs")
os.makedirs(FIG, exist_ok=True)
D = json.load(open(os.path.join(HERE, "figs_data.json")))
RUNS = list(csv.DictReader(open(os.path.join(HERE, "validation_v2_runs.csv"))))

# reference palette (dataviz skill references/palette.md), light mode, fixed slot order
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
GCOL = {"H200": BLUE, "B200": ORANGE, "A100-80-PCIe": AQUA, "A6000": YELLOW, "H100-SXM": MAGENTA, "L40S": VIOLET}
SHORT = {"H200": "H200", "B200": "B200", "A100-80-PCIe": "A100", "A6000": "A6000", "H100-SXM": "H100*",
         "L40S": "L40S*"}
ORDER = ["A6000", "A100-80-PCIe", "H200", "B200"]
TITLE = {"A6000": "RTX A6000 (GDDR6, 300 W)", "A100-80-PCIe": "A100 80GB PCIe (HBM2e, 300 W)",
         "H200": "H200 (HBM3e, 700 W)", "B200": "B200 (HBM3e, 1000 W)"}
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.facecolor": SURF,
    "axes.facecolor": SURF, "savefig.facecolor": SURF, "legend.frameon": False,
    "axes.titlesize": 10, "axes.titlecolor": INK, "lines.linewidth": 2})


def f(r, k):
    try:
        return float(r[k])
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------- (a) rooflines
def fig_a():
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.2), sharey=True, sharex=True)
    for ax, g in zip(axes.flat, ORDER):
        a = D["a"][g]
        I = a["I"]
        roof = [min(b, a["compute"]) for b in a["bw_roof"]]
        ax.plot(I, [x / 1e12 for x in roof], color=INK, lw=1.6, label="bandwidth / compute roof (datasheet)")
        ax.plot(I, [x / 1e12 for x in a["power_full"]], color=ORANGE, lw=1.6, ls="--",
                label="power roof, full-clock coefficients")
        if a["power_cap"]:
            ax.plot(I, [x / 1e12 for x in a["power_cap"]], color=ORANGE, lw=2.4,
                    label="at-cap roofline (fitted on capped LLM-shaped kernels)")
        mc = a["micro_capped"]
        ax.scatter([p["I"] for p in mc], [p["flops"] / 1e12 for p in mc], s=20, color=AQUA, edgecolors=SURF,
                   lw=0.6, zorder=3, label="microbench kernel, at the power cap")
        mu = a["micro_uncapped"]
        if mu:
            ax.scatter([p["I"] for p in mu], [p["flops"] / 1e12 for p in mu], s=20, facecolors="none",
                       edgecolors=AQUA, lw=1.2, zorder=3, label="microbench kernel, uncapped")
        sv = [p for p in D["a"]["serving"] if p["gpu"] == g and not p["prefill"]]
        ax.scatter([p["I"] for p in sv], [p["flops"] / 1e12 for p in sv], s=11, color=BLUE, edgecolors=SURF,
                   lw=0.4, zorder=4, label="measured vLLM serving run (per GPU)")
        pf = [p for p in D["a"]["serving"] if p["gpu"] == g and p["prefill"]]
        if pf:
            ax.scatter([p["I"] for p in pf], [p["flops"] / 1e12 for p in pf], s=24, marker="D", color=BLUE,
                       edgecolors=SURF, lw=0.5, zorder=4, label="measured prefill-heavy run")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.5, 4e4)
        ax.set_ylim(0.3, 3500)
        rf, rc = a["rates_full"], a["rates_cap"]
        note = "BW %.2f TB/s, peak %.0f TF\nfull-clock power roof: %.0f%% BW, %.0f%% peak" % (
            a["bw"] / 1e12, a["compute"] / 1e12, 100 * rf["bw_frac"], 100 * rf["flops_frac"])
        if rc:
            note += "\nat-cap roofline: %.0f%% BW, %.0f%% peak (measured)" % (100 * rc["bw_frac"],
                                                                           100 * rc["flops_frac"])
        else:
            note += "\n(no microbench: power roof PREDICTED)"
        ax.text(300, 0.4, note, color=INK2, fontsize=7.2, va="bottom")
        ax.axvline(rf["perf_ridge"], color=INK, lw=0.7, ls=":")
        ax.set_title(TITLE[g], loc="left")
    for ax in axes[1]:
        ax.set_xlabel("arithmetic intensity (FLOP per DRAM byte; decode ~ tokens per step)")
    for ax in axes[:, 0]:
        ax.set_ylabel("attainable TFLOP/s")
    hs, ls = [], []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in ls:
                hs.append(h)
                ls.append(l)
    fig.legend(hs, ls, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle("LIMINAL-E rooflines with power roofs, 4 GPUs (dotted: performance ridge)", x=0.01, ha="left",
                 fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(os.path.join(FIG, "fig_a_power_roofline.png"), dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- (b) energy roofline
COMP = [("static", "idle + activity (x time)", INK2), ("weights", "weight bytes", BLUE),
        ("kv", "KV bytes", ORANGE), ("compute", "GEMM FLOPs (incl. prefill)", AQUA)]
MEAS_KEY = {"A6000": ("gpu_graphs", "alpaca"), "A100-80-PCIe": ("a100_graphs", "alpaca"),
            "H200": ("sept_H200", "alpaca"), "B200": ("graphs", "alpaca")}


def fig_b():
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), sharex=True)
    for j, g in enumerate(ORDER):
        for i, c in enumerate(("short", "long")):
            ax = axes[i, j]
            d = D["b"]["%s_%s" % (g, c)]
            rows = d["rows"]
            B = [x["batch"] for x in rows]
            labs = []
            for k, lab, col in COMP:
                ax.plot(B, [max(x[k], 1e-6) for x in rows], color=col, lw=1.6)
                labs.append([max(rows[-1][k], 1e-6), lab, col, "normal"])
            ax.plot(B, [x["total"] for x in rows], color=INK, lw=2.4)
            labs.append([rows[-1]["total"], "total", INK, "bold"])
            labs.sort(key=lambda z: z[0])
            for q in range(1, len(labs)):
                labs[q][0] = max(labs[q][0], labs[q - 1][0] * 1.5)
            for y, lab, col, wt in labs:
                ax.text(B[-1] * 1.15, y, lab, color=col, fontsize=7, va="center", weight=wt)
            cp = [x for x in rows if x["capped"]]
            if cp:
                ax.axvspan(cp[0]["batch"], B[-1], color=ORANGE, alpha=0.07, lw=0)
                ax.text(cp[0]["batch"] * 1.05, 3.0, "at power cap", color=ORANGE, fontsize=7)
            ax.axvline(d["energy_ridge"], color=AQUA, lw=0.9, ls=":")
            if c == "short":
                arm, task = MEAS_KEY[g]
                ms = [r for r in RUNS if r["gpu"] == g and r["arm"] == arm and r["task"] == task
                      and r["model"] == "Qwen2-7B-Instruct"]
                ax.scatter([f(r, "conc") for r in ms], [f(r, "j_meas") for r in ms], s=30, color=SURF,
                           edgecolors=INK, lw=1.3, zorder=5, label="measured, Qwen2-7B alpaca")
                if j == 0:
                    ax.legend(loc="lower left", fontsize=7.5)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(1e-3 if c == "short" else 5e-3, 12)
            ax.set_xlim(0.8, 4000)
            ttl = "P=25, G=147" if c == "short" else "P=8192, G=512"
            ax.set_title("%s\n%s (B*=%.0f)" % (TITLE[g], ttl, d["energy_ridge"]), loc="left", fontsize=9)
            if j == 0:
                ax.set_ylabel("J per generated token")
            if i == 1:
                ax.set_xlabel("concurrent sequences")
    fig.suptitle("Energy per token vs batch, Qwen2-7B, by component (canonical coefficients; dotted: energy "
                 "ridge B*; shaded: predicted at the power cap)", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 0.97, 0.95))
    fig.savefig(os.path.join(FIG, "fig_b_energy_roofline.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- (c) provisioning map
def fig_c():
    C = D["c"]
    panels = [("short (P=256, G=256)|measured", "short context, measured GPUs"),
              ("long (P=8192, G=1024)|measured", "long context, measured GPUs"),
              ("short (P=256, G=256)|with_priors", "short context, + H100*/L40S* priors"),
              ("long (P=8192, G=1024)|with_priors", "long context, + H100*/L40S* priors")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.6), sharey=True)
    for ax, (key, ttl) in zip(axes.flat, panels):
        grid = C["cells"][key]
        nm, nc = len(C["models"]), len(C["concs"])
        for i, row in enumerate(grid):
            for j, x in enumerate(row):
                if not x:
                    continue
                prior = x["prior"]
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=GCOL[x["gpu"]],
                                       alpha=0.45 if prior else 0.85, edgecolor=SURF, lw=2,
                                       hatch="//" if prior else None))
                jm = x["j"] * 1e3
                txt = "%s\n%s mJ" % (SHORT[x["gpu"]], ("%.0f" % jm) if jm >= 10 else ("%.1f" % jm))
                if x["n_gpus"] > 1:
                    txt += "\n%dx TP%d" % (x["n_gpus"], x["tp"])
                ax.text(j, i, txt, ha="center", va="center", fontsize=6.8,
                        color=INK if (prior or x["gpu"] == "A6000") else "white",
                        weight="bold" if prior else "normal")
        ax.set_xlim(-0.5, nc - 0.5)
        ax.set_ylim(nm - 0.5, -0.5)
        ax.set_xticks(range(nc))
        ax.set_xticklabels([str(c) for c in C["concs"]])
        ax.set_yticks(range(nm))
        ax.set_yticklabels(C["models"])
        ax.set_xlabel("concurrent requests")
        ax.grid(False)
        ax.set_title(ttl, loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
    names = ["H200", "B200", "A100-80-PCIe", "A6000", "H100-SXM", "L40S"]
    h = [Rectangle((0, 0), 1, 1, facecolor=GCOL[n], hatch="///" if n in ("H100-SXM", "L40S") else None,
                   edgecolor=SURF) for n in names]
    fig.legend(h, ["H200", "B200", "A100 80GB PCIe", "RTX A6000", "H100 SXM (prior)", "L40S (prior)"],
               loc="lower center", ncol=6, fontsize=8.5)
    fig.suptitle("Provisioning map (PREDICTED): lowest J/token at TPOT <= 50 ms; cell = winner, J per generated "
                 "token, total GPUs x TP degree (eager time model, common host)", x=0.01, ha="left", fontsize=10,
                 color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(os.path.join(FIG, "fig_c_provisioning_map.png"), dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- (d) validation parity
def fig_d():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharex=True, sharey=True)
    panels = [("j_zs", "ZERO-SHOT: microbench + datasheet + idle only", ("A6000", "A100-80-PCIe", "B200")),
              ("j_cal", "canonical / calibrated", ("A6000", "A100-80-PCIe", "H200", "B200"))]
    for ax, (k, ttl, gs) in zip(axes, panels):
        for g in gs:
            rs = [r for r in RUNS if r["gpu"] == g and r["moe"] != "True" and r.get(k)]
            ax.scatter([f(r, "j_meas") for r in rs], [f(r, k) for r in rs], s=14, color=GCOL[g],
                       edgecolors=SURF, lw=0.4, label="%s (n=%d)" % (SHORT[g], len(rs)), zorder=3)
        ax.plot([0.01, 40], [0.01, 40], color=MUTED, lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("measured J/token")
        ax.set_title(ttl, loc="left", fontsize=9.5)
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("LIMINAL-E predicted J/token")
    fig.suptitle("J/token predicted vs measured (dense models; MoE excluded)", x=0.01, ha="left", fontsize=10,
                 color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(FIG, "fig_d_validation_parity.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    fig_a()
    fig_b()
    fig_c()
    fig_d()
    print("wrote", sorted(os.listdir(FIG)))
