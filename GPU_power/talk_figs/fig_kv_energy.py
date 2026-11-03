#!/usr/bin/env python3
"""KV-byte energy on B200 from the long-context sweep (talk figure, single panel):
decode-only energy per generated token vs KV bytes per token for 7B c=4, 7B c=16 and
32B c=4 (each request's prefill window excluded), with fitted slopes (= e_kv) against a
weight-byte reference slope (1.25e-10 J/B).
  PYTHONPATH=/shared_data0/adsampat/pydeps python3 talk_figs/fig_kv_energy.py"""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, ".")
from longctx_analysis import run_bins

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
E_W = 1.25e-10


def per_run(rd, clean=True):
    b = [x for x in run_bins(rd) if x["prefill"] == 0 and x["decode"] > 0
         and (not clean or not x["in_prefill_window"])]
    if len(b) < 5:
        return None
    tok = sum(x["decode"] for x in b)
    return sum(x["K"] for x in b) / tok, sum(x["Y"] for x in b) / tok


def series(pattern, clean=True):
    pts = []
    for rd in glob.glob(pattern):
        p = per_run(rd, clean)
        if p:
            pts.append(p)
    pts.sort()
    return np.array(pts)


plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2})
fig, ax0 = plt.subplots(1, 1, figsize=(7.6, 5.0), facecolor=SURF)
ax = [ax0]
out = {}

# (b) clean slopes
b = ax[0]
cfg = [("7B, 4 concurrent", "logs/B200_prov/longctx/Qwen2-7B-Instruct/ctx*_c4", "#2a78d6", "o"),
       ("7B, 16 concurrent", "logs/B200_prov/longctx/Qwen2-7B-Instruct/ctx*_c16", "#eb6834", "s"),
       ("32B, 4 concurrent", "logs/B200_prov/longctx32/Qwen2.5-32B-Instruct/ctx*_c4", "#1baf7a", "^")]
for lab, pat, col, mk in cfg:
    s = series(pat)
    x, y = s[:, 0], s[:, 1]
    A = np.c_[np.ones_like(x), x]
    (c0, sl), *_ = np.linalg.lstsq(A, y, rcond=None)
    r2 = 1 - ((y - A @ np.array([c0, sl])) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    # plot the increase over the 1k-context point, so the three configs share an origin
    hollow = "4 concurrent" in lab and "7B" in lab        # drawn on top: overlaps the c=16 line
    b.plot(x / 1e9, y - (c0 + sl * x[0]), mk, ms=11 if hollow else 7, mfc="none" if hollow else col,
           mec=col, mew=2 if hollow else 1.5, zorder=5 if hollow else 3)
    xx = np.linspace(x.min(), x.max(), 20)
    b.plot(xx / 1e9, sl * (xx - x[0]), "--" if hollow else "-", color=col, lw=2, zorder=4 if hollow else 2,
           label=f"{lab}: slope {sl*1e10:.2f}e-10 J/B (×{sl/E_W:.2f} weight byte), R² {r2:.3f}")
    out[lab] = dict(slope=sl, intercept=c0, r2=r2, pts=s.tolist())
xx = np.linspace(0, 8.3, 20)
b.plot(xx, E_W * xx * 1e9, ":", color=INK2, lw=1.5, label="reference: a weight byte (1.25e-10 J/B)")
b.set_xlabel("KV bytes read per generated token (GB)")
b.set_ylabel("extra energy per token vs the 1k-context run (J)")
b.set_title("A KV byte costs 1.28–1.29× a weight byte (B200)\nslope of decode energy vs KV bytes; context 1k → 30k at fixed concurrency",
            loc="left", fontsize=10, color=INK)
b.legend(fontsize=8, frameon=False, loc="upper left")
for x_ in ax:
    x_.set_facecolor(SURF); x_.grid(color=GRID, lw=0.8); x_.set_axisbelow(True)
    for s_ in ("top", "right"):
        x_.spines[s_].set_visible(False)
fig.text(0.01, 0.012, "B200, CUDA graphs, energy counter; decode steps only (prefill windows excluded). Each point is one 60 s run\n"
         "(context 1k, 4k, 8k, 16k, 30k; 512 generated tokens per request). 7B at 4 and 16 concurrent overlap (hollow = 4).\n"
         "Independent check: long-output runs give 1.58e-10 J/B (x1.26). NCU: DRAM/analytic 1.012 at 8k and 30k context.",
         fontsize=7, color=INK2)
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.savefig("talk_figs/fig_kv_energy.png", dpi=300, facecolor=SURF)
fig.savefig("talk_figs/fig_kv_energy.svg", facecolor=SURF)
json.dump(out, open("talk_figs/kv_energy.json", "w"), indent=1, default=float)
print(json.dumps({k: v.get("slope") for k, v in out.items() if isinstance(v, dict) and "slope" in v}))
