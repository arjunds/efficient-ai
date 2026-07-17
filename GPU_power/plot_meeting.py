#!/usr/bin/env python3
"""
plot_meeting.py

Generate the PPT-ready figures from the collected data. Reads the result CSVs /
run dirs (auto-includes whatever models/points are present) and writes PNGs to
plots_meeting/. Pure stdlib + matplotlib.

If matplotlib won't import in the cluster container, run this on the analysis
side (the input CSVs are small and self-contained); or copy plots_meeting/ back.

  python3 plot_meeting.py [--sweep_csv logs/load_sweep/sweep_summary.csv]
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

OUT = "plots_meeting"


def load_sweep(path):
    rows = []
    if not os.path.exists(path):
        return rows
    for r in csv.DictReader(open(path)):
        try:
            r["concurrency"] = int(float(r["concurrency"]))
            r["input_len"] = int(float(r["input_len"]))
            for k in ("tokens_per_s", "avg_power_w", "idle_power_w",
                      "e_bit_j_per_byte", "waveform_mape_pct"):
                r[k] = float(r[k]) if r.get(k) not in (None, "", "None") else None
        except (ValueError, KeyError):
            continue
        r["model_short"] = (r.get("model") or "?").split("/")[-1]
        rows.append(r)
    return rows


def _fig(plt):
    return plt.subplots(figsize=(8, 5))


def plot_vs_concurrency(rows, plt, workload=512):
    series = defaultdict(list)
    for r in rows:
        if r["input_len"] != workload or r["e_bit_j_per_byte"] is None:
            continue
        series[r["model_short"]].append(r)
    for m in series:
        series[m].sort(key=lambda r: r["concurrency"])

    specs = [
        ("tokens_per_joule", lambda r: r["tokens_per_s"] / r["avg_power_w"],
         "Tokens per Joule", "tokens_per_joule_vs_concurrency.png",
         "Energy efficiency scales with concurrency (batching amortizes weight loads)"),
        ("avg_power_w", lambda r: r["avg_power_w"], "Avg power (W)",
         "power_vs_concurrency.png",
         "Power rises modestly while throughput scales ~linearly"),
        ("e_bit", lambda r: r["e_bit_j_per_byte"] * 1e10, "e_bit (1e-10 J/byte)",
         "ebit_vs_concurrency.png",
         "e_bit grows with load: memory-bound -> compute-bound"),
    ]
    made = []
    for key, fn, ylabel, fname, title in specs:
        fig, ax = _fig(plt)
        for m, rs in sorted(series.items()):
            xs = [r["concurrency"] for r in rs]
            ys = [fn(r) for r in rs]
            ax.plot(xs, ys, marker="o", label=m)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("offered concurrency (clients)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\n(input={workload}, output=128)")
        ax.grid(alpha=0.3, which="both"); ax.legend()
        fig.tight_layout(); p = os.path.join(OUT, fname); fig.savefig(p, dpi=150)
        plt.close(fig); made.append(p)
    return made


def plot_pareto(rows, plt, workload=512):
    fig, ax = _fig(plt)
    series = defaultdict(list)
    for r in rows:
        if r["input_len"] != workload or not r["tokens_per_s"] or not r["avg_power_w"]:
            continue
        j_per_tok = r["avg_power_w"] / r["tokens_per_s"]
        series[r["model_short"]].append((r["tokens_per_s"], j_per_tok, r["concurrency"]))
    for m, pts in sorted(series.items()):
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=m)
        for tps, jpt, c in pts:
            ax.annotate(f"c{c}", (tps, jpt), fontsize=7, xytext=(3, 3),
                        textcoords="offset points")
    ax.set_xlabel("throughput (tokens/s)"); ax.set_ylabel("energy per token (J/token)")
    ax.set_title("Throughput vs energy/token (down-right = better)\n(input=512, output=128)")
    ax.set_xscale("log"); ax.grid(alpha=0.3, which="both"); ax.legend()
    fig.tight_layout(); p = os.path.join(OUT, "throughput_vs_energy_per_token.png")
    fig.savefig(p, dpi=150); plt.close(fig); return [p]


def plot_longctx(plt):
    pts = []
    for d in glob.glob("logs/longctx/*/ctx*_c*"):
        try:
            meta = json.load(open(os.path.join(d, "run_meta.json")))
            cal = json.load(open(os.path.join(d, "calibration.json")))
            res = json.load(open(os.path.join(d, "results.json")))
        except Exception:
            continue
        pts.append((meta.get("input_len"), meta.get("concurrency_or_rate"),
                    cal.get("e_bit_j_per_byte"), res.get("avg_power_window_w")))
    if not pts:
        return []
    by_c = defaultdict(list)
    for ctx, c, eb, pw in pts:
        if ctx and eb:
            by_c[c].append((ctx, eb, pw))
    made = []
    for ylabel, sel, fname, title in [
        ("e_bit (1e-10 J/byte)", lambda t: t[1] * 1e10, "ebit_vs_context.png",
         "e_bit rises with context as KV bytes rival weights"),
        ("avg power (W)", lambda t: t[2], "power_vs_context.png",
         "Long context raises power (KV traffic + attention compute)")]:
        fig, ax = _fig(plt)
        for c, arr in sorted(by_c.items(), key=lambda kv: kv[0]):
            arr.sort()
            ax.plot([a[0] for a in arr], [sel(a) for a in arr], marker="o",
                    label=f"c={c}")
        ax.set_xlabel("context length (tokens)"); ax.set_ylabel(ylabel)
        ax.set_title(title + "\n(Qwen2-7B)"); ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout(); p = os.path.join(OUT, fname); fig.savefig(p, dpi=150)
        plt.close(fig); made.append(p)
    return made


def plot_waveform(plt, series_csv="logs/wave/waveform_series.csv"):
    if not os.path.exists(series_csv):
        return []
    t, meas, pred = [], [], []
    for r in csv.DictReader(open(series_csv)):
        t.append(float(r["t_rel_s"])); meas.append(float(r["measured_w"]))
        pred.append(float(r["predicted_w"]))
    fig, ax = _fig(plt)
    ax.plot(t, meas, lw=1.4, label="measured (NVML)", alpha=0.85)
    ax.plot(t, pred, lw=1.4, label="predicted (model)", alpha=0.85)
    ax.set_xlabel("time (s)"); ax.set_ylabel("power (W)")
    ax.set_title("Predicted vs measured GPU power waveform\n"
                 "bursty active<->idle load (the validation)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); p = os.path.join(OUT, "waveform_overlay.png")
    fig.savefig(p, dpi=150); plt.close(fig); return [p]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_csv", default="logs/load_sweep/sweep_summary.csv")
    args = ap.parse_args()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[fatal] matplotlib unavailable: {e}\n"
              "Run this on the analysis side; inputs are small CSVs/JSONs.")
        return
    os.makedirs(OUT, exist_ok=True)
    rows = load_sweep(args.sweep_csv) or load_sweep("sweep_summary_H200.csv")
    made = []
    if rows:
        made += plot_vs_concurrency(rows, plt)
        made += plot_pareto(rows, plt)
    made += plot_longctx(plt)
    made += plot_waveform(plt)
    print(f"[done] wrote {len(made)} figures to {OUT}/:")
    for p in made:
        print("  ", p)


if __name__ == "__main__":
    main()
