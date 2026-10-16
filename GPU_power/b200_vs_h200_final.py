#!/usr/bin/env python3
"""
b200_vs_h200_final.py — the B200 vs H200 energy-per-token comparison, on equal footing.

Problem: the only MEASURED H200 runs (Sept) used eager mode; the B200 now has eager and
CUDA-graph runs. CUDA graphs only remove launch time from each step; they do not change
energy per byte or per FLOP (B200: e_wbyte 1.260e-10 eager vs 1.257e-10 graphs). So

    J/tok(graphs) = J/tok(eager) - P_idle * dt_step / tokens_per_step

where dt_step = mean step time saved. Step 1 VALIDATES that on B200 (eager -> graphs,
both measured, same logger). Step 2 applies it to the measured Sept H200 and B200 runs
with the B200-measured dt_step (launch time is host-side, so the same kernel sequence
saves about the same time on either GPU - an assumption, stated). Step 3 tabulates the
three comparisons: (a) both measured eager, (b) both graphs-adjusted from measured
eager, (c) B200 measured graphs vs H200 recommender prediction (needed for c >= 128).

Qwen2-7B only: it is the one model with matched eager/graphs B200 runs.
  python3 b200_vs_h200_final.py  -> b200_vs_h200_final.json + printed tables
"""
import csv
import json
import os

ROOT = "logs/B200_prov"
MODEL = "Qwen2-7B-Instruct"


def jl(p):
    try:
        return json.load(open(p))
    except Exception:
        return {}


def run_stats(rd):
    """J/token (energy counter if logged, else avg power), mean step time and tokens/step
    over the measurement window."""
    res, meta = jl(os.path.join(rd, "results.json")), jl(os.path.join(rd, "run_meta.json"))
    if not res.get("aggregate_tokens_per_sec"):
        return None
    # validity (as balance_analysis.py): a stalled load generator leaves few requests
    # and average power at/below idle (2 Sept B200 sharegpt runs)
    conc = meta.get("concurrency_or_rate") if isinstance(meta.get("concurrency_or_rate"), int) else 1
    if (res.get("completed_requests") or 0) < max(5, 2 * conc) or \
            (res.get("avg_power_window_w") or 0) <= (meta.get("idle_power_w") or 0):
        return None
    t0, t1 = meta["window_wall_t0"], meta["window_wall_t1"]
    win = t1 - t0
    gen = res["generated_tokens_window"]
    E = res.get("energy_window_j") or (res["avg_power_window_w"] * win)
    steps = 0
    for r in csv.DictReader(open(os.path.join(rd, "iter_log.csv"))):
        try:
            a, b = float(r["t_start"]), float(r["t_end"])
        except (ValueError, KeyError):
            continue
        if t0 <= a and b <= t1 and b > a:
            steps += 1
    return dict(j_tok=E / gen, tok_s=gen / win, step_s=win / steps if steps else None,
                tok_per_step=gen / steps if steps else None, idle=meta["idle_power_w"])


def main():
    out = {"validation_b200": [], "comparison": []}
    tags = [f"{t}_c{c}" for t in ("alpaca", "sharegpt") for c in (1, 4, 16, 64)]
    dt = {}
    print("== 1. validate on B200: predict graphs J/token from eager J/token ==")
    print(f"{'run':<14}{'eager mJ':>10}{'step e/g ms':>14}{'saved ms':>10}{'pred graphs':>13}{'meas graphs':>13}{'err':>7}")
    for tag in tags:
        e = run_stats(os.path.join(ROOT, "relog", MODEL, tag))
        g = run_stats(os.path.join(ROOT, "graphs", MODEL, tag))
        if not (e and g):
            continue
        d = e["step_s"] - g["step_s"]
        dt[tag] = d
        pred = e["j_tok"] - e["idle"] * d / e["tok_per_step"]
        err = pred / g["j_tok"] - 1
        out["validation_b200"].append(dict(tag=tag, eager=e["j_tok"], graphs_meas=g["j_tok"], graphs_pred=pred,
                                           step_saved_s=d, err=err))
        print(f"{tag:<14}{e['j_tok']*1e3:>10.0f}{e['step_s']*1e3:>7.2f}/{g['step_s']*1e3:<6.2f}{d*1e3:>10.2f}"
              f"{pred*1e3:>13.0f}{g['j_tok']*1e3:>13.0f}{err*100:>6.1f}%")

    print("\n== 2./3. B200 vs H200, Qwen2-7B ==")
    print("   (a) both measured, eager (Sept)   (b) both adjusted to CUDA graphs from (a)"
          "   (c) B200 measured graphs / H200 predicted")
    print(f"{'run':<14}{'(a) B/H':>9}{'(b) B/H':>9}{'(b) if H200 saves 0.5-1.5x':>28}{'(c) B/H':>9}{'  B200 graphs mJ/tok':>21}{'  H200 graphs est':>18}")
    hc = {r["key"]: r for r in jl(os.path.join(ROOT, "provisioning_summary.json")).get("highconc", [])}
    for tag in tags:
        h = run_stats(os.path.join("logs/ragged", MODEL, tag))
        b = run_stats(os.path.join("logs/B200", MODEL, tag))
        bg = run_stats(os.path.join(ROOT, "graphs", MODEL, tag))
        if not (h and b and tag in dt):
            continue
        d = dt[tag]
        h_g = h["j_tok"] - h["idle"] * d / h["tok_per_step"]
        b_g = b["j_tok"] - b["idle"] * d / b["tok_per_step"]
        c = hc.get(f"graphs/{MODEL}/{tag}")
        # sensitivity: H200 saves 0.5x or 1.5x the B200-measured launch time per step
        h_lo = h["j_tok"] - h["idle"] * 1.5 * d / h["tok_per_step"]
        h_hi = h["j_tok"] - h["idle"] * 0.5 * d / h["tok_per_step"]
        row = dict(tag=tag, a=b["j_tok"] / h["j_tok"], b=b_g / h_g,
                   b_range=[b_g / h_hi, b_g / h_lo],
                   c=(c["b200_meas"] / c["h200_pred"]) if c else None,
                   b200_graphs_meas=bg and bg["j_tok"], h200_graphs_est=h_g, b200_graphs_est=b_g,
                   h200_eager=h["j_tok"], b200_eager=b["j_tok"])
        out["comparison"].append(row)
        print(f"{tag:<14}{row['a']:>9.2f}{row['b']:>9.2f}{row['b_range'][0]:>19.2f}-{row['b_range'][1]:<8.2f}{(row['c'] or float('nan')):>9.2f}"
              f"{(bg['j_tok']*1e3 if bg else float('nan')):>21.0f}{h_g*1e3:>18.0f}")
    for k in ("a", "b", "c"):
        v = [r[k] for r in out["comparison"] if r[k]]
        if v:
            print(f"   ({k}) range {min(v):.2f}-{max(v):.2f}, median {sorted(v)[len(v)//2]:.2f}")
    print("\n   c >= 128 (no H200 measurement exists): B200 measured / H200 predicted")
    for k, r in sorted(hc.items(), key=lambda kv: (kv[0].split("/")[1], kv[1]["conc"])):
        if r["conc"] >= 128:
            print(f"   {k:<44} {r['b200_meas']/r['h200_pred']:.2f}")
            out.setdefault("high_conc_predicted", []).append(dict(key=k, conc=r["conc"],
                                                                  ratio=r["b200_meas"] / r["h200_pred"]))
    json.dump(out, open("b200_vs_h200_final.json", "w"), indent=1)
    print("\nwrote b200_vs_h200_final.json")


if __name__ == "__main__":
    main()
