#!/usr/bin/env python3
"""B200 / H200 energy per token vs concurrency (meeting figure, 2026-10-16).
Solid: B200 measured (CUDA graphs, energy counter; logs/B200_prov) / H200 predicted
(recommender time model). Hollow: both GPUs measured (eager, Sept; balance_analysis.json).
  PYTHONPATH=/shared_data0/adsampat/pydeps python3 pi_update_figs/fig_b200_vs_h200.py"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = json.load(open("logs/B200_prov/provisioning_summary.json"))
BA = json.load(open("balance_analysis.json"))
RUNS = S["runs"]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
SERIES = [  # (label, model key, task, color, label y-offset)
    ("7B, short prompts (Alpaca)", "Qwen2-7B-Instruct", "alpaca", "#2a78d6", 0.0),
    ("7B, long contexts (ShareGPT)", "Qwen2-7B-Instruct", "sharegpt", "#eb6834", -0.025),
    ("32B, short prompts (Alpaca)", "Qwen2.5-32B-Instruct", "alpaca", "#1baf7a", 0.035),
]

plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2})
fig, ax = plt.subplots(figsize=(8.6, 4.9), facecolor=SURF)
ax.set_facecolor(SURF)
ax.axhspan(0.75, 1.0, color="#eef4fc", zorder=0)
ax.axhline(1.0, color=INK2, lw=1, zorder=1)
ax.text(1.05, 0.79, "B200 cheaper per token", color=INK2, fontsize=9)
ax.text(1.05, 1.53, "H200 cheaper per token", color=INK2, fontsize=9)

for label, model, task, col, dy in SERIES:
    pts = []
    for r in S["highconc"]:
        arm, m, tag = r["key"].split("/")
        if m == model and tag.startswith(task + "_c"):
            cap = (RUNS.get(r["key"]) or {}).get("cap_frac") or 0
            pts.append((r["conc"], r["b200_meas"] / r["h200_pred"], cap))
    pts.sort()
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    ax.plot(xs, ys, color=col, lw=2, zorder=3)
    for x, y, cap in pts:
        ax.plot(x, y, "o", ms=8, color=col, mec=SURF, mew=2, zorder=4)
        if cap >= 0.5:
            ax.plot(x, y, "o", ms=15, mfc="none", mec=INK2, mew=1.2, zorder=4)
    ax.text(xs[-1] * 1.35, ys[-1] + dy, label, color=INK, fontsize=9, va="center")
    meas = [m_ for m_ in BA["matched"] if m_["model"] == model and m_["task"] == task]
    for m_ in meas:
        ax.plot(m_["conc"], m_["jtok_ratio"], "o", ms=8, mfc=SURF, mec=col, mew=2, zorder=5)

# legend for encodings (identity is carried by the direct labels)
h1, = ax.plot([], [], "o-", color=INK2, lw=2, ms=8, mec=SURF, mew=2,
              label="B200 measured (CUDA graphs) ÷ H200 predicted")
h2, = ax.plot([], [], "o", ms=8, mfc=SURF, mec=INK2, mew=2, label="both GPUs measured (Sept, eager mode)")
h3, = ax.plot([], [], "o", ms=15, mfc="none", mec=INK2, mew=1.2, label="B200 at its 1000 W power cap (≥50% of samples)")
ax.legend(handles=[h1, h2, h3], loc="upper right", fontsize=8.5, frameon=False, labelcolor=INK)

ax.set_xscale("log", base=2)
ticks = [1, 4, 16, 64, 128, 256, 512]
ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks])
ax.set_xlim(0.8, 3800); ax.set_ylim(0.75, 1.6)
ax.set_xlabel("concurrent requests")
ax.set_ylabel("energy per token, B200 ÷ H200")
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("B200 costs 1.2–1.5× H200 per token at normal load;\nit breaks even only near 256 concurrent "
             "requests, and only for long contexts or larger models", loc="left", fontsize=10.5, color=INK)
fig.text(0.01, 0.012, "H200 side of the solid lines is the recommender's prediction (H200 access revoked). Its time model was fitted on\n"
         "eager-mode runs, so the low-concurrency ratios are likely understated. Hollow points: same model and task, both GPUs measured.",
         fontsize=7.5, color=INK2)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig("pi_update_figs/fig_u6_b200_vs_h200.png", dpi=160, facecolor=SURF)
print("wrote pi_update_figs/fig_u6_b200_vs_h200.png")
