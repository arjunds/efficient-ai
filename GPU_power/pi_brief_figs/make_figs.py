#!/usr/bin/env python3
"""New figures for PI_BRIEF.md. Run in the container from GPU_power/ with
PYTHONPATH=/shared_data0/adsampat/pydeps (matplotlib). Numbers are taken from
gpu_coefficients.json, microbench/a5000_summary.json and FINDINGS_*.md; the phase
figure is recomputed from logs/ragged binned tables with the canonical coefficients."""
import csv, glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

OUT = "pi_brief_figs"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#ffffff"
plt.rcParams.update({
    "figure.dpi": 200, "font.size": 10.5, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "text.color": INK})

coef = json.load(open("gpu_coefficients.json"))


# ---------------------------------------------------------------- fig: cross-GPU J/byte
def fig_crossgpu():
    h200 = coef["H200"]["direct_c1_jbyte"]["Qwen2-7B-Instruct"] * 1e10       # 1.076
    b200 = coef["B200"]["direct_c1_jbyte"]["Qwen2-7B-Instruct"] * 1e10       # 1.252
    a5 = json.load(open("microbench/a5000_summary.json"))
    a5_7b = [r["jbyte_direct"] * 1e10 for r in a5["direct_c1"] if "Qwen2-7B" in r["run_dir"]]
    a5_lo, a5_hi = min(a5_7b), max(a5_7b)                                        # 2.69-2.78
    law_b200 = h200 * 4.8 / 8.0                                                  # 0.646
    law_a5 = a5["refs"]["law_pred_A5000"] * 1e10                                 # 6.7
    stream_lo, stream_hi = 2.6, 3.1                                              # FINDINGS_A5000 table

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    xs = [0, 1, 2.3, 3.3]
    w = 0.56
    # measured
    ax.bar(xs[0], h200, w, color=BLUE, zorder=3)
    ax.bar(xs[1], b200, w, color=BLUE, zorder=3)
    ax.bar(xs[2], (a5_lo + a5_hi) / 2, w, color=ORANGE, zorder=3, hatch="///",
           edgecolor=SURF, linewidth=0)
    ax.bar(xs[3], stream_hi - stream_lo, w, bottom=stream_lo, color=ORANGE, alpha=0.55,
           zorder=3)
    ax.errorbar(xs[2], (a5_lo + a5_hi) / 2, yerr=[[(a5_hi - a5_lo) / 2]] * 2, fmt="none",
                ecolor=INK, capsize=4, lw=1.2, zorder=4)
    # lower-bound arrow on capped serving point
    ax.annotate("", xy=(xs[2], 3.55), xytext=(xs[2], a5_hi + 0.05),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2), zorder=5)
    # law predictions
    for x0, x1, v, lab in [(xs[1] - w / 2 - 0.06, xs[1] + w / 2 + 0.06, law_b200, "law: 0.65"),
                           (xs[2] - w / 2 - 0.06, xs[3] + w / 2 + 0.06, law_a5, "law: 6.7")]:
        ax.plot([x0, x1], [v, v], color=INK, lw=1.6, ls=(0, (4, 2)), zorder=5)
        ax.text(x1 + 0.05, v, lab, va="center", ha="left", fontsize=8.5, color=INK2)
    # value labels
    ax.text(xs[0], h200 + 0.12, f"{h200:.2f}\n(anchor)", ha="center", fontsize=9)
    ax.text(xs[1], b200 + 0.12, f"{b200:.2f}\n1.16× H200", ha="center", fontsize=9)
    ax.text(xs[2] - 0.34, 3.2, "≈2.7–2.8\nlower bound\n(capped at 100 W)",
            ha="right", va="center", fontsize=8.5)
    ax.text(xs[3], stream_hi + 0.12, "2.6–3.1\nfull clocks", ha="center", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(["H200\nserving", "B200\nserving", "A5000\nserving (c=1)",
                        "A5000\nDRAM-stream\nmicrobench"], fontsize=9.5)
    ax.set_ylabel("energy per byte  (×10⁻¹⁰ J/B)")
    ax.set_ylim(0, 7.4); ax.set_xlim(-0.55, 4.45)
    ax.grid(axis="x", visible=False)
    ax.axvline(1.65, color=GRID, lw=1)
    ax.text(0.5, 7.15, "HBM3e", ha="center", color=BLUE, fontweight="bold")
    ax.text(2.8, 7.15, "GDDR6", ha="center", color=ORANGE, fontweight="bold")
    handles = [Patch(color=BLUE, label="measured, HBM3e"),
               Patch(color=ORANGE, label="measured, GDDR6"),
               plt.Line2D([], [], color=INK, ls=(0, (4, 2)), lw=1.6,
                          label="1/bandwidth-law prediction (from H200)")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.93), fontsize=8.5,
              frameon=False)
    ax.set_title("Serving: Qwen2-7B at concurrency 1, dynamic energy ÷ analytic bytes (no regression).\nMicrobench: A5000 DRAM stream at full clocks.",
                 fontsize=9.5, color=INK2, loc="left")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_crossgpu_jbyte.png"); plt.close(fig)
    print("crossgpu", h200, b200, a5_lo, a5_hi, law_b200, law_a5)


# ---------------------------------------------------------------- fig: pipeline
def box(ax, x, y, w, h, text, fc, ec, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=INK)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.2, shrinkA=0, shrinkB=0))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(10, 4.3))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.3); ax.axis("off")
    LB, LO, LA, LG, LW = "#e3eefb", "#fde6dc", "#dcf3ea", "#f3f2ef", "#fbfaf8"
    # row 1: measurement
    box(ax, 0.1, 3.0, 1.75, 0.95, "Real traffic\nalpaca + sharegpt\nc = 1…64, Poisson", LG, MUTED)
    box(ax, 2.2, 3.0, 1.5, 0.95, "vLLM 0.10.2\ncontinuous\nbatching, fp16", LG, MUTED)
    box(ax, 4.1, 3.55, 2.3, 0.6, "NVML power (~10 ms)", LO, ORANGE)
    box(ax, 4.1, 2.8, 2.3, 0.6, "per-iteration scheduler log", LB, BLUE)
    box(ax, 6.9, 3.0, 1.5, 0.95, "~200 ms bins\n(one clock)", LG, MUTED)
    arrow(ax, 1.85, 3.47, 2.2, 3.47); arrow(ax, 3.7, 3.65, 4.1, 3.85); arrow(ax, 3.7, 3.3, 4.1, 3.1)
    arrow(ax, 6.4, 3.85, 6.9, 3.65); arrow(ax, 6.4, 3.1, 6.9, 3.3)
    # row 2: fit
    box(ax, 7.8, 1.95, 2.1, 0.55, "Y: measured J − idle·Δt", LO, ORANGE, fs=8.5)
    box(ax, 7.8, 1.3, 2.1, 0.55, "X: analytic bytes, FLOPs\n(models.py, LIMINAL-style)", LB, BLUE, fs=7.8)
    arrow(ax, 7.65, 3.0, 8.3, 2.5)
    box(ax, 3.3, 1.3, 4.1, 1.2,
        "Energy model: OLS per GPU → e_wbyte, e_kvbyte, e_gemm\n"
        "E − P_static·Δt = e_wbyte·W + e_kvbyte·KV + e_gemm·F\n"
        "cross-check: direct J/byte at c = 1 (no regression)", LW, INK2, fs=8.5)
    arrow(ax, 7.8, 2.2, 7.4, 2.05); arrow(ax, 7.8, 1.55, 7.4, 1.7)
    box(ax, 0.1, 1.3, 2.8, 1.2, "Time model\nhost overhead + roofline\n(realized utilization,\nfit to iteration timing)",
        LW, INK2, fs=8.5)
    arrow(ax, 4.6, 2.8, 2.5, 2.5)
    # row 3: uses (both use both models)
    ax.plot([1.5, 1.5, 5.35, 5.35], [1.3, 1.0, 1.0, 1.3], color=INK2, lw=1.2)
    arrow(ax, 2.6, 1.0, 2.6, 0.75); arrow(ax, 6.6, 1.0, 6.6, 0.75)
    ax.plot([5.35, 6.6], [1.0, 1.0], color=INK2, lw=1.2)
    box(ax, 0.9, 0.05, 3.4, 0.7, "GPU recommender\n(J/token per phase, with P(best))", LA, AQUA, fs=8.5)
    box(ax, 4.9, 0.05, 3.4, 0.7, "Fleet power control\n(economic MPC, simulation)", LA, AQUA, fs=8.5)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_pipeline.png"); plt.close(fig)


# ---------------------------------------------------------------- fig: max vs sum
def fig_max_sum():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    res = ["weight stream", "KV stream", "tensor compute"]
    cols = [BLUE, AQUA, ORANGE]
    t = [4.0, 1.2, 0.8]            # schematic lengths only
    e = [1.5, 0.4, 0.4]
    ax = axes[0]
    ax.barh([2, 1, 0], [0, 0, 0])  # placeholder axis
    y = 1.0
    ax.barh(y + 0.55, 2.0, 0.4, left=0, color="#c9c8c3")
    ax.text(1.0, y + 0.55, "host CPU\noverhead", ha="center", va="center", fontsize=8)
    for i, (r, c, v) in enumerate(zip(res, cols, t)):
        ax.barh(y - 0.05 - 0.42 * i, v, 0.34, left=2.0, color=c, alpha=0.9)
        ax.text(2.0 + v + 0.1, y - 0.05 - 0.42 * i, r, va="center", fontsize=8.5)
    ax.axvline(6.0, color=INK, ls=(0, (4, 2)), lw=1.2)
    ax.text(6.05, 1.75, "t_iter ≈ overhead + max(…)\n(bottleneck sets time)", fontsize=8.5,
            va="top")
    ax.set_title("Performance (LIMINAL roofline): a MAX", loc="left", fontsize=10.5)
    ax.set_xlim(0, 9.5); ax.set_ylim(-0.4, 1.9); ax.set_yticks([]); ax.set_xticks([])
    ax.set_xlabel("time within one iteration (schematic)"); ax.grid(False)
    ax = axes[1]
    left = 0
    for r, c, v in zip(res, cols, e):
        ax.barh(0.5, v, 0.45, left=left, color=c, edgecolor=SURF, linewidth=2)
        left += v
    for r, c, v, lx in zip(res, cols, e, [0.75, 1.7, 2.1]):
        pass
    ax.text(0.75, 0.5, "e_wbyte·W", ha="center", va="center", fontsize=8.5, color="white",
            fontweight="bold")
    ax.text(1.7, 0.95, "e_kvbyte·KV", ha="center", fontsize=8.5)
    ax.text(2.1, 0.05, "e_gemm·F", ha="center", fontsize=8.5, va="top")
    ax.text(left + 0.1, 0.5, "= E_iter", va="center", fontsize=9.5, fontweight="bold")
    ax.text(0, 1.55, "power = E_iter / t_iter      perf/W = tokens·s⁻¹ / power\n"
            "(all resources draw power at once, so terms add)", fontsize=8.5, va="top")
    ax.set_title("Energy (this work): a SUM", loc="left", fontsize=10.5)
    ax.set_xlim(0, 3.3); ax.set_ylim(-0.4, 1.9); ax.set_yticks([]); ax.set_xticks([])
    ax.set_xlabel("dynamic energy of one iteration (schematic)"); ax.grid(False)
    for a in axes:
        a.spines["left"].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_max_vs_sum.png"); plt.close(fig)


# ---------------------------------------------------------------- fig: phase split (canonical)
def fig_phase():
    c = coef["H200"]; ew, ek, eg = c["e_wbyte"], c["e_kvbyte"], c["e_gemm"]
    groups = {0: [], 1: [], 2: []}
    for rd in glob.glob("logs/ragged/*/*"):
        t = os.path.join(rd, "binned_table.csv")
        if not os.path.exists(t):
            continue
        for r in csv.DictReader(open(t)):
            try:
                W = float(r["weight_bytes_bin"]); K = float(r["kv_bytes_bin"])
                pc = r.get("prefill_tokens_computed")
                if pc in (None, ""):
                    G = float(r["gemm_flops_bin"]); p = float(r["prefill_tokens"])
                else:
                    G = float(r["gemm_flops_bin_computed"]); p = float(pc)
                d = float(r["decode_tokens"])
            except (KeyError, ValueError):
                continue
            if p + d <= 0:
                continue
            pf = p / (p + d)
            g = 0 if pf < 0.05 else 2 if pf > 0.5 else 1
            tot = ew * W + ek * K + eg * G
            groups[g].append((ew * W / tot, ek * K / tot, eg * G / tot))
    labels = ["decode-heavy\n(prefill < 5% of tokens)", "mixed\n(5–50%)", "prefill-heavy\n(> 50%)"]
    sh = np.array([np.mean(groups[g], axis=0) * 100 for g in range(3)])
    ns = [len(groups[g]) for g in range(3)]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    x = np.arange(3)
    bottoms = np.zeros(3)
    names = ["weight bytes (e_wbyte)", "KV bytes (e_kvbyte)", "GEMM FLOPs (e_gemm)"]
    for j, (col, nm) in enumerate(zip([BLUE, AQUA, ORANGE], names)):
        ax.bar(x, sh[:, j], 0.58, bottom=bottoms, color=col, label=nm, edgecolor=SURF,
               linewidth=2)
        for i in range(3):
            if sh[i, j] >= 4:
                ax.text(x[i], bottoms[i] + sh[i, j] / 2, f"{sh[i, j]:.0f}%", ha="center",
                        va="center", fontsize=9, color="white" if j != 1 else INK,
                        fontweight="bold")
        bottoms += sh[:, j]
    ax.set_xticks(x); ax.set_xticklabels([f"{l}\nn={n:,} bins" for l, n in zip(labels, ns)],
                                          fontsize=8.5)
    ax.set_ylabel("share of dynamic energy (%)"); ax.set_ylim(0, 100)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, fontsize=8, frameon=False)
    ax.set_title("H200, 4 dense 7–8B models, canonical 3-term coefficients", fontsize=9.5,
                 color=INK2, loc="left")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_phase_split.png"); plt.close(fig)
    print("phase shares (w, kv, gemm) %:", np.round(sh, 1).tolist(), "n:", ns)


if __name__ == "__main__":
    which = sys.argv[1:] or ["crossgpu", "pipeline", "maxsum", "phase"]
    for w in which:
        {"crossgpu": fig_crossgpu, "pipeline": fig_pipeline, "maxsum": fig_max_sum,
         "phase": fig_phase}[w]()
