#!/usr/bin/env python3
"""In-load leakage slope vs DRAM bandwidth, from up/down ramp pairs (thermal_leakage.py).

Each pair holds the same stream occupancy on the up and down legs; the down leg runs hotter,
so dP/dT at matched bandwidth is the in-load W/C. Only pairs with dT >= 4 C and no capping
are fitted (below that a +-1 W error moves the ratio by >= 0.25 W/C). The idle cool-down
slope of the same card is drawn at BW = 0.
  python3 microbench/thermal_pairs_fig.py     # writes microbench/fig_inload_k.{png,json}
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = [("A100_run2", "A100 80GB PCIe (61:00.0)", "#d03b3b"),
        ("A6000_run2", "RTX A6000 (81:00.0)", "#2ca02c"),
        ("B200_pairs", "B200 (dgx007 DF:00.0)", "#1f77b4")]
MIN_DT = 4.0


def load(tag):
    s = json.load(open(os.path.join(HERE, f"thermal_{tag}", "summary.json")))
    up = {x["nprog"]: x for x in s["ramp_levels"] if x["direction"] == "up"}
    pairs = []
    for h in s["hysteresis"]:
        u = up[h["nprog"]]
        pairs.append(dict(bw_tbs=u["bw_gbs"] / 1e3, dT=h["dT"], dP=h["dP"], k=h["w_per_c"],
                          used=h["dT"] >= MIN_DT and u["capped_frac"] == 0))
    cd = [c["fits"]["skip2s"]["w_per_c"] for n, c in s["cooldowns"].items()
          if n != "ramp_cool" and c.get("fits", {}).get("skip2s")]
    return pairs, cd


def main():
    out = {}
    for tag, _, _ in RUNS:
        pairs, cd = load(tag)
        u = [p for p in pairs if p["used"]]
        x = np.array([p["bw_tbs"] for p in u]); y = np.array([p["k"] for p in u])
        b, a = np.polyfit(x, y, 1)
        out[tag] = dict(pairs=pairs, cooldown_k=cd, fit_k0=a, fit_dk_per_tbs=b,
                        r2=float(np.corrcoef(x, y)[0, 1] ** 2), n=len(u))
        print(f"{tag:11s} k(BW) = {a:.2f} + {b:.2f}/TBps  R2 {out[tag]['r2']:.2f} n={len(u)}  "
              f"idle cool-down {min(cd):.2f}-{max(cd):.2f}")
    json.dump(out, open(os.path.join(HERE, "fig_inload_k.json"), "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for tag, label, col in RUNS:
        r = out[tag]
        for used, mk in ((True, "o"), (False, "x")):
            P = [p for p in r["pairs"] if p["used"] == used]
            ax.plot([p["bw_tbs"] for p in P], [p["k"] for p in P], mk, color=col, alpha=1 if used else 0.35,
                    label=f"{label}: k = {r['fit_k0']:.2f} + {r['fit_dk_per_tbs']:.2f}·BW" if used else None)
        xs = np.linspace(0, max(p["bw_tbs"] for p in r["pairs"] if p["used"]), 20)
        ax.plot(xs, r["fit_k0"] + r["fit_dk_per_tbs"] * xs, "-", color=col, lw=1)
        ax.errorbar([0], [np.mean(r["cooldown_k"])], yerr=[[np.mean(r["cooldown_k"]) - min(r["cooldown_k"])],
                    [max(r["cooldown_k"]) - np.mean(r["cooldown_k"])]], fmt="*", ms=11, color=col, mec="k")
    ax.plot([], [], "k*", ms=9, label="idle cool-down slope (BW = 0)")
    ax.plot([], [], "kx", alpha=0.35, label=f"pair with ΔT < {MIN_DT:.0f} °C (not fitted)")
    ax.set_xscale("symlog", linthresh=0.1); ax.set_ylim(0, 3.2)
    ax.set_xlabel("DRAM bandwidth (TB/s)"); ax.set_ylabel("dP/dT at matched bandwidth (W/°C)")
    ax.set_title("Leakage slope under load, from up/down ramp pairs (20 s holds)", loc="left", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_inload_k.png"), dpi=150)


if __name__ == "__main__":
    main()
