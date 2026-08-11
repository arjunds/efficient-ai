#!/usr/bin/env python3
"""
fit_channels.py

(#19) Two analyses on the ragged per-bin data, pooled across a model set:

A. CHANNEL SPLIT — does decomposing the two lumped terms into physical channels
   (weight-bytes, KV-bytes, GEMM-flops, attn-flops) improve the fit, and are the
   extra coefficients identifiable (positive, CI excludes 0, held-out improves)?
   Compares nested models: 2-term (bytes,flops) vs 3/4-term. Keeps a split only
   if it earns its degrees of freedom.

B. PHASE DECOMPOSITION — using the fitted 2-term coefficients, what fraction of a
   bin's dynamic energy is compute (e_flop*flops) vs memory (e_bit*bytes), split
   by whether the bin is prefill-heavy or decode-heavy? Demonstrates the
   recommender's premise: prefill = compute-bound, decode = memory-bound.

  python3 fit_channels.py logs/ragged
"""
import csv, glob, json, os, sys, random
import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else "logs/ragged"
COLS = ["weight_bytes_bin", "kv_bytes_bin", "gemm_flops_bin", "attn_flops_bin",
        "bytes_bin", "flops_bin", "energy_bin_j", "dt_s",
        "prefill_tokens", "decode_tokens"]


def load(root):
    rows = []
    for rd in glob.glob(os.path.join(root, "*", "*")):
        tbl = os.path.join(rd, "binned_table.csv"); mp = os.path.join(rd, "run_meta.json")
        if not (os.path.exists(tbl) and os.path.exists(mp)):
            continue
        idle = json.load(open(mp)).get("idle_power_w")
        if idle is None:
            continue
        for r in csv.DictReader(open(tbl)):
            try:
                d = {c: float(r[c]) for c in COLS}
            except (KeyError, ValueError):
                continue
            d["dyn"] = d["energy_bin_j"] - idle * d["dt_s"]
            d["model"] = os.path.basename(os.path.dirname(rd))
            rows.append(d)
    return rows


def fit(X, y):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef
    ss = ((y - yhat) ** 2).sum(); tot = ((y - y.mean()) ** 2).sum()
    return coef, 1 - ss / tot


def boot_ci(X, y, n=200, seed=0):
    rng = np.random.default_rng(seed)
    cs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        c, _ = fit(X[idx], y[idx]); cs.append(c)
    cs = np.array(cs)
    return np.percentile(cs, 2.5, axis=0), np.percentile(cs, 97.5, axis=0)


def held_out(rows, cols):
    models = sorted(set(r["model"] for r in rows))
    if len(models) < 2:
        return None
    errs = []
    for h in models:
        tr = [r for r in rows if r["model"] != h]; te = [r for r in rows if r["model"] == h]
        Xtr = np.array([[r[c] for c in cols] for r in tr]); ytr = np.array([r["dyn"] for r in tr])
        Xte = np.array([[r[c] for c in cols] for r in te]); yte = np.array([r["dyn"] for r in te])
        coef, _ = fit(Xtr, ytr); pred = Xte @ coef
        mape = np.mean(np.abs((yte - pred) / np.where(yte == 0, np.nan, yte))) * 100
        errs.append(np.nanmean(mape) if np.isfinite(mape) else np.nan)
    return float(np.nanmean(errs))


def main():
    rows = load(ROOT)
    print(f"ROOT={ROOT}  n_bins={len(rows)}  models={sorted(set(r['model'] for r in rows))}")
    if len(rows) < 20:
        print("too few bins"); return
    y = np.array([r["dyn"] for r in rows])

    print("\n=== A. CHANNEL SPLIT (nested models) ===")
    variants = [
        ("2-term  [bytes, flops]", ["bytes_bin", "flops_bin"]),
        ("3-term  [wbytes, kvbytes, flops]", ["weight_bytes_bin", "kv_bytes_bin", "flops_bin"]),
        ("3-term  [bytes, gemm, attn]", ["bytes_bin", "gemm_flops_bin", "attn_flops_bin"]),
        ("4-term  [wbytes, kvbytes, gemm, attn]",
         ["weight_bytes_bin", "kv_bytes_bin", "gemm_flops_bin", "attn_flops_bin"]),
    ]
    for label, cols in variants:
        X = np.array([[r[c] for c in cols] for r in rows])
        coef, r2 = fit(X, y)
        lo, hi = boot_ci(X, y)
        ho = held_out(rows, cols)
        # scale coeffs to readable units: bytes->1e-10 J/byte, flops->pJ/flop
        parts = []
        for c, v, l, h in zip(cols, coef, lo, hi):
            unit = 1e10 if "bytes" in c else 1e12
            u = "e-10" if "bytes" in c else "pJ"
            sign = "" if (l * h > 0) else "  (CI spans 0!)"
            parts.append(f"{c.replace('_bin','')}={v*unit:.3f}{u}[{l*unit:.2f},{h*unit:.2f}]{sign}")
        print(f"\n {label}\n   R2={r2:.3f}  held-out MAPE={ho:.1f}%" if ho else f"\n {label}\n   R2={r2:.3f}")
        for p in parts:
            print("   ", p)

    print("\n=== B. PHASE DECOMPOSITION (2-term coeffs) ===")
    X2 = np.array([[r["bytes_bin"], r["flops_bin"]] for r in rows])
    (e_bit, e_flop), _ = fit(X2, y)
    print(f" e_bit={e_bit*1e10:.3f}e-10 J/byte, e_flop={e_flop*1e12:.3f} pJ/flop")
    groups = {"decode-heavy (prefill<5%)": [], "mixed (5-50%)": [], "prefill-heavy (>50%)": []}
    for r in rows:
        tot = r["prefill_tokens"] + r["decode_tokens"]
        if tot <= 0:
            continue
        pf = r["prefill_tokens"] / tot
        g = ("decode-heavy (prefill<5%)" if pf < 0.05 else
             "prefill-heavy (>50%)" if pf > 0.5 else "mixed (5-50%)")
        comp = e_flop * r["flops_bin"]; mem = e_bit * r["bytes_bin"]
        if comp + mem > 0:
            groups[g].append(comp / (comp + mem))
    print(f" {'phase group':<28}{'n_bins':>8}{'compute share':>15}{'memory share':>14}")
    for g, vals in groups.items():
        if vals:
            cs = float(np.mean(vals))
            print(f" {g:<28}{len(vals):>8}{cs*100:>14.1f}%{(1-cs)*100:>13.1f}%")


if __name__ == "__main__":
    main()
