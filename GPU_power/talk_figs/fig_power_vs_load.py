#!/usr/bin/env python3
"""Power draw vs load on four GPUs (Qwen2-7B, alpaca prompts) - talk figure.
(a) average board power as % of the GPU's power limit, (b) tokens per joule,
(c) share of energy that is idle power. H200 = September eager runs (only H200 data);
B200 / A100 / A6000 = October CUDA-graph runs.
  PYTHONPATH=/shared_data0/adsampat/pydeps python3 talk_figs/fig_power_vs_load.py"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
GPUS = [  # label, cap W, marker, color, [(conc, run_dir)]
    ("H200 (700 W)", 700, "o", "#2a78d6", [(c, f"logs/ragged/Qwen2-7B-Instruct/alpaca_c{c}") for c in (1, 4, 16, 64)]),
    ("B200 (1000 W)", 1000, "s", "#eb6834", [(c, f"logs/B200_prov/{'highconc' if c > 64 else 'graphs'}/Qwen2-7B-Instruct/alpaca_c{c}")
                                             for c in (1, 4, 16, 64, 128, 256, 512)]),
    ("A100 PCIe (300 W)", 300, "^", "#1baf7a", [(c, f"logs/A100_prov/{'a100_highconc' if c > 64 else 'a100_graphs'}/Qwen2-7B-Instruct/alpaca_c{c}")
                                                for c in (1, 4, 16, 64, 128, 256)]),
    ("A6000 (300 W)", 300, "D", "#eda100", [(c, f"logs/A6000_prov/{'gpu_highconc' if c > 64 else 'gpu_graphs'}/Qwen2-7B-Instruct/alpaca_c{c}")
                                            for c in (1, 4, 16, 64, 128, 256)]),
]


def load(rd):
    try:
        r = json.load(open(os.path.join(rd, "results.json"))); m = json.load(open(os.path.join(rd, "run_meta.json")))
    except Exception:
        return None
    win = r["window_duration_s"]; E = r.get("energy_window_j") or r["avg_power_window_w"] * win
    return dict(P=E / win, tok_j=r["generated_tokens_window"] / E, idle=m["idle_power_w"])


plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2})
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURF)
out = {}
for label, cap, mk, col, runs in GPUS:
    pts = [(c, load(rd)) for c, rd in runs]
    pts = [(c, d) for c, d in pts if d]
    out[label] = [(c, round(d["P"], 1), round(d["tok_j"], 2), round(d["idle"] / d["P"], 3)) for c, d in pts]
    xs = [c for c, _ in pts]
    for a, ys in ((ax[0], [100 * d["P"] / cap for _, d in pts]), (ax[1], [d["tok_j"] for _, d in pts]),
                  (ax[2], [100 * d["idle"] / d["P"] for _, d in pts])):
        a.plot(xs, ys, "-", color=col, lw=2, marker=mk, ms=7, mec=SURF, mew=1.5, zorder=3,
               label=label if a is ax[0] else None)
    idle_pct = 100 * pts[0][1]["idle"] / cap
    ax[0].plot([0.8, 0.95], [idle_pct, idle_pct], color=col, lw=3, solid_capstyle="butt")
for a in ax:
    a.set_xscale("log", base=2); a.set_xticks([1, 4, 16, 64, 256]); a.set_xticklabels(["1", "4", "16", "64", "256"])
    a.set_xlim(0.75, 700); a.set_facecolor(SURF); a.grid(axis="y", color=GRID, lw=0.8); a.set_axisbelow(True)
    a.set_xlabel("concurrent requests")
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
ax[0].axhline(100, color=INK2, ls="--", lw=1); ax[0].text(1.0, 102, "power limit", color=INK2, fontsize=8.5)
ax[0].set_ylim(0, 112); ax[0].set_ylabel("average power, % of the GPU's power limit")
ax[0].set_title("(a) Power drawn vs the limit\n(bars at left: idle with the model loaded)", loc="left", fontsize=10, color=INK)
ax[1].set_yscale("log"); ax[1].set_ylabel("generated tokens per joule")
ax[1].set_title("(b) Energy efficiency (higher is better)", loc="left", fontsize=10, color=INK)
ax[2].set_ylim(0, 100); ax[2].set_ylabel("idle power as % of total energy")
ax[2].set_title("(c) How much of the energy is just idle power", loc="left", fontsize=10, color=INK)
fig.text(0.01, 0.01, "Qwen2-7B, Alpaca prompts. H200: September runs (eager mode, the only H200 data); B200/A100/A6000: October runs "
         "(CUDA graphs). Power from the NVML energy counter where validated, else POWER_INSTANT / GetPowerUsage.",
         fontsize=7.5, color=INK2)
fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False,
           fontsize=9.5, labelcolor=INK, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0.04, 1, 0.93))
fig.savefig("talk_figs/fig_power_vs_load.png", dpi=160, facecolor=SURF)
json.dump(out, open("talk_figs/power_vs_load.json", "w"), indent=1)
print(json.dumps(out, indent=1))
