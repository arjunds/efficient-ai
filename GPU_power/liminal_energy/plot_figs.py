#!/usr/bin/env python3
"""plot_figs.py -- draw LIMINAL-E figures from figs_data.json + validation_runs.csv.
Separate process from the model code (matplotlib from /shared_data0/adsampat/pydeps):
  PYTHONPATH=/shared_data0/adsampat/pydeps python3 liminal_energy/plot_figs.py
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs")
os.makedirs(FIG, exist_ok=True)
D = json.load(open(os.path.join(HERE, "figs_data.json")))
RUNS = list(csv.DictReader(open(os.path.join(HERE, "validation_runs.csv"))))

# reference palette (dataviz skill references/palette.md), light mode
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.facecolor": SURF,
    "axes.facecolor": SURF, "savefig.facecolor": SURF, "legend.frameon": False,
    "axes.titlesize": 10, "axes.titlecolor": INK, "lines.linewidth": 2})


def f(r, k):
    return float(r[k])


# ---------------------------------------------------------------- (a) rooflines
def fig_a():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=True)
    for ax, g in zip(axes, ("H200", "B200")):
        a = D["a"][g]
        I = a["I"]
        roof = [min(b, a["compute"]) for b in a["bw_roof"]]
        ax.plot(I, [x / 1e12 for x in roof], color=INK, lw=1.6)
        ax.plot(I, [x / 1e12 for x in a["power_full"]], color=ORANGE, lw=2, label="power roof, full clock")
        if g == "B200":
            ax.plot(I, [x / 1e12 for x in a["power_cap"]], color=ORANGE, lw=1.4, ls="--",
                    label="power roof, clock-throttled (DVFS capped coefficients)")
            pts = D["a"]["B200_microbench_gemm"]
            un = [p for p in pts if not p["capped"]]
            ca = [p for p in pts if p["capped"]]
            ax.scatter([p["I"] for p in un], [p["flops"] / 1e12 for p in un], s=22, facecolors="none",
                       edgecolors=AQUA, lw=1.3, zorder=3, label="steady GEMM microbench, uncapped")
            ax.scatter([p["I"] for p in ca], [p["flops"] / 1e12 for p in ca], s=22, color=AQUA,
                       edgecolors=SURF, lw=0.8, zorder=3, label="steady GEMM microbench, at 1000 W cap")
        sv = [r for r in RUNS if r["gpu"] == g and r["arm"] != "prefill"]
        ax.scatter([f(r, "ai_meas") for r in sv], [f(r, "flops_meas") / f(r, "tp") / 1e12 for r in sv], s=14,
                   color=BLUE, edgecolors=SURF, lw=0.6, zorder=4, label="measured vLLM serving runs (per GPU)")
        if g == "B200":
            pf = [r for r in RUNS if r["arm"] == "prefill"]
            ax.scatter([f(r, "ai_meas") for r in pf], [f(r, "flops_meas") / 1e12 for r in pf], s=26,
                       marker="D", color=BLUE, edgecolors=SURF, lw=0.6, zorder=4,
                       label="measured prefill-heavy runs (100% capped)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.5, 4e4)
        ax.set_ylim(0.5, 3500)
        ax.set_xlabel("arithmetic intensity (FLOP per DRAM byte; decode ~ tokens per step)")
        rf, rc = a["rates_full"], a["rates_cap"]
        ax.text(3e3, a["compute"] / 1e12 * 1.15, "compute roof %.0f TF" % (a["compute"] / 1e12),
                color=INK, fontsize=8)
        note = "bandwidth roof %.1f TB/s\nfull-clock power roof: %.1f TB/s (%.0f%% of BW),\n   %.0f TF (%.0f%% of peak)" % (
            a["bw"] / 1e12, rf["bw_p"] / 1e12, 100 * rf["bw_frac"], rf["phi_p"] / 1e12, 100 * rf["flops_frac"])
        if g == "B200":
            note += "\nthrottled (DVFS) roof: %.1f TB/s (%.0f%%), %.0f TF (%.0f%%)" % (
                rc["bw_p"] / 1e12, 100 * rc["bw_frac"], rc["phi_p"] / 1e12, 100 * rc["flops_frac"])
        else:
            note += "\n-> power never binds pure streaming"
        ax.text(420, 0.58, note, color=INK2, fontsize=7.2, va="bottom")
        if g == "B200":
            ax.set_title("B200 (power roof MEASURED: steady microbench)", loc="left")
        else:
            ax.set_title("H200 (power roof PREDICTED from serving coefficients)", loc="left")
        ax.axvline(rf["roof_knee"], color=ORANGE, lw=0.8, ls=":")
        ax.text(rf["roof_knee"] * 0.93, 0.62, "energy ridge\n%.0f FLOP/B" % rf["roof_knee"], color=ORANGE,
                fontsize=7.5, ha="right")
        ax.axvline(rf["perf_ridge"], color=INK, lw=0.8, ls=":")
        ax.text(rf["perf_ridge"] * 1.08, 8.0, "perf. ridge\n%.0f FLOP/B" % rf["perf_ridge"], color=INK, fontsize=7.5)
    axes[0].set_ylabel("attainable TFLOP/s")
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle("LIMINAL-E roofline: bandwidth, compute and power roofs (P_s + P_0 + e_b*bytes/s + e_f*FLOP/s <= P_cap)",
                 x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
    fig.savefig(os.path.join(FIG, "fig_a_power_roofline.png"), dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------- (b) energy roofline
COMP = [("static", "static (idle x time)", INK2), ("weights", "weight bytes", BLUE),
        ("kv", "KV bytes", ORANGE), ("compute", "GEMM FLOPs (incl. prefill)", AQUA)]


def fig_b():
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for j, g in enumerate(("H200", "B200")):
        for i, c in enumerate(("short", "long")):
            ax = axes[i, j]
            d = D["b"]["%s_%s" % (g, c)]
            rows = d["rows"]["canonical"]
            B = [x["batch"] for x in rows]
            if "kv_low" in d["rows"]:
                lo = d["rows"]["kv_low"]
                hi = d["rows"]["kv_high"]
                ax.fill_between(B, [x["kv"] for x in lo], [x["kv"] for x in hi], color=ORANGE, alpha=0.15,
                                lw=0)
                ax.fill_between(B, [x["total"] for x in lo], [x["total"] for x in hi], color=INK, alpha=0.08,
                                lw=0)
            labs = []
            for k, lab, col in COMP:
                ax.plot(B, [x[k] for x in rows], color=col, lw=1.8)
                labs.append([rows[-1][k], lab, col, "normal"])
            ax.plot(B, [x["total"] for x in rows], color=INK, lw=2.6)
            labs.append([rows[-1]["total"], "total", INK, "bold"])
            labs.sort(key=lambda z: z[0])
            for q in range(1, len(labs)):          # dodge: >= 1.45x apart on the log axis
                labs[q][0] = max(labs[q][0], labs[q - 1][0] * 1.45)
            for y, lab, col, wt in labs:
                ax.text(B[-1] * 1.12, y, lab, color=col, fontsize=7.5, va="center", weight=wt)
            cp = [x for x in rows if x["capped"]]
            if cp:
                ax.axvspan(cp[0]["batch"], B[-1], color=ORANGE, alpha=0.07, lw=0)
                ax.text(cp[0]["batch"] * 1.05, 0.25 if c == "short" else 2.0, "power-\ncapped", color=ORANGE, fontsize=7.5)
            ax.axvline(d["energy_ridge"], color=AQUA, lw=0.9, ls=":")
            ax.text(d["energy_ridge"] * 0.93, 3.5, "energy ridge\nB*=%.0f" % d["energy_ridge"], ha="right", va="top",
                    color=AQUA, fontsize=7.5)
            if c == "short":
                ms = [r for r in RUNS if r["gpu"] == g and r["model"] == "Qwen2-7B-Instruct" and
                      r["task"] == "alpaca" and r["set"] == "Sept"]
                ax.scatter([f(r, "conc") for r in ms], [f(r, "j_meas") for r in ms], s=34, color=SURF,
                           edgecolors=INK, lw=1.4, zorder=5, label="measured (Sept, eager)")
                if g == "B200":
                    ms2 = [r for r in RUNS if r["model"] == "Qwen2-7B-Instruct" and r["task"] == "alpaca"
                           and r["arm"] in ("highconc",)]
                    ax.scatter([f(r, "conc") for r in ms2], [f(r, "j_meas") for r in ms2], s=34, marker="s",
                               color=SURF, edgecolors=INK, lw=1.4, zorder=5,
                               label="measured (Oct, CUDA graphs, c>=128)")
                ax.legend(loc="lower left", fontsize=7.5, framealpha=0.9, frameon=True, edgecolor=SURF,
                          bbox_to_anchor=(0.0, 0.08))
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(1e-3 if c == "short" else 5e-3, 5)
            ax.set_xlim(0.8, 3000)
            ttl = "alpaca-like: P=25, G=147" if c == "short" else "long context: P=8192, G=512"
            ax.set_title("%s, Qwen2-7B, %s" % (g, ttl), loc="left")
            if j == 0:
                ax.set_ylabel("J per generated token")
            if i == 1:
                ax.set_xlabel("concurrent sequences (batch)")
    fig.suptitle("Energy roofline: J/token = P_s t/B + e_w W/B + e_kv ctx kv_B + e_g 2N (1+P/G)    "
                 "(shaded: KV-vs-compute split undetermined, B200 kv_low..kv_high)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 0.93, 0.96))
    fig.savefig(os.path.join(FIG, "fig_b_energy_roofline.png"), dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------- (c) provisioning map
def fig_c():
    C = D["c"]
    panels = [("short (P=256, G=256)|measured", "short context (P=256, G=256)"),
              ("long (P=8192, G=1024)|measured", "long context (P=8192, G=1024)"),
              ("long (P=8192, G=1024)|measured_kv_as_weight", "long context, if a KV byte = a weight byte")]
    cols = {"H200": BLUE, "B200": ORANGE}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    for ax, (key, ttl) in zip(axes, panels):
        grid = C["cells"][key]
        nm, nc = len(C["models"]), len(C["concs"])
        idx = [[(0 if (x and x["gpu"] == "H200") else 1) if x else 2 for x in row] for row in grid]
        ax.imshow(idx, cmap=ListedColormap([BLUE, ORANGE, GRID]), vmin=0, vmax=2, aspect="auto", alpha=0.85)
        for i, row in enumerate(grid):
            for j, x in enumerate(row):
                if not x:
                    continue
                j_mj = x["j"] * 1e3
                txt = ("%.0f" % j_mj if j_mj >= 10 else "%.1f" % j_mj) + " mJ"
                if x["n_gpus"] > 1:
                    txt += "\n%dx (TP%d)" % (x["n_gpus"], x["tp"])
                ax.text(j, i, txt, ha="center", va="center", fontsize=7, color="white")
        ax.set_xticks(range(nc))
        ax.set_xticklabels([str(c) for c in C["concs"]])
        ax.set_yticks(range(nm))
        ax.set_yticklabels(C["models"])
        ax.set_xlabel("concurrent requests")
        ax.grid(False)
        ax.set_title(ttl, loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
    h = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cols.values()]
    fig.legend(h, ["H200 lowest J/token", "B200 lowest J/token"], loc="lower center", ncol=2, fontsize=8.5)
    fig.suptitle("Provisioning map (PREDICTED): GPU, TP and replicas minimizing J/token at TPOT <= 50 ms "
                 "(eager vLLM time model; cell = J per generated token; Nx (TPk) = N GPUs in total at TP degree k)", x=0.01, ha="left", fontsize=10,
                 color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(os.path.join(FIG, "fig_c_provisioning_map.png"), dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------- (d) validation parity
def fig_d():
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    groups = [("Sept H200 (eager)", lambda r: r["arm"] == "sept_H200", BLUE, "o"),
              ("Sept B200 (eager)", lambda r: r["arm"] == "sept_B200", ORANGE, "o"),
              ("Oct B200 graphs/highconc/TP", lambda r: r["arm"] in ("graphs", "highconc", "tp2", "tp1ref"),
               AQUA, "s")]
    for lab, sel, col, mk in groups:
        rs = [r for r in RUNS if sel(r)]
        ax.scatter([f(r, "j_meas") for r in rs], [f(r, "j_mode") for r in rs], s=18, color=col, marker=mk,
                   edgecolors=SURF, lw=0.5, label=lab, zorder=3)
    ax.plot([0.01, 40], [0.01, 40], color=MUTED, lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("measured J/token")
    ax.set_ylabel("LIMINAL-E predicted J/token")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("J/token predicted vs measured (Sept: in-sample fit data; Oct: new runs)", loc="left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_d_validation_parity.png"), dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    fig_a()
    fig_b()
    fig_c()
    fig_d()
    print("wrote", sorted(os.listdir(FIG)))
