#!/usr/bin/env python3
"""Size-transfer test on the ladder (data in hand). Does the 3-term model fit on
SOME sizes predict a HELD-OUT size's per-bin energy? This turns 'coefficients look
similar per size' into 'coefficients fit on one size range predict another'.

  A. Leave-one-size-out: fit on the other 5 sizes, predict the held size.
  B. Extrapolation: fit on small (<=3B), predict large (>=14B); and vice-versa.
"""
import glob, os, csv, json, sys
import numpy as np

ROOT = "logs/ragged_ladder"
SIZES = [("Qwen2.5-0.5B-Instruct", 0.49), ("Qwen2.5-1.5B-Instruct", 1.54),
         ("Qwen2.5-3B-Instruct", 3.09), ("Qwen2.5-7B-Instruct", 7.07),
         ("Qwen2.5-14B-Instruct", 13.99), ("Qwen2.5-32B-Instruct", 31.98)]
FEAT = ["weight_bytes_bin", "kv_bytes_bin", "gemm_flops_bin"]


def load(name):
    W = {c: [] for c in FEAT}; W["dyn"] = []
    for rd in glob.glob(os.path.join(ROOT, name, "*")):
        t = os.path.join(rd, "binned_table.csv"); m = os.path.join(rd, "run_meta.json")
        if not (os.path.exists(t) and os.path.exists(m)):
            continue
        idle = json.load(open(m)).get("idle_power_w")
        if idle is None:
            continue
        for r in csv.DictReader(open(t)):
            try:
                for c in FEAT:
                    W[c].append(float(r[c]))
                W["dyn"].append(float(r["energy_bin_j"]) - idle * float(r["dt_s"]))
            except (KeyError, ValueError):
                pass
    X = np.c_[[W[c] for c in FEAT]].T if W["dyn"] else np.empty((0, 3))
    return X, np.array(W["dyn"])


def fit(X, y):
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    return c


def score(coef, X, y):
    yh = X @ coef
    r2 = 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    mape = np.mean(np.abs((y - yh) / np.where(y == 0, np.nan, y))) * 100
    return r2, np.nanmean(mape)


data = {n: load(n) for n, _ in SIZES}

print("=== A. Leave-one-SIZE-out (fit other 5, predict held-out size) ===")
print(f"{'held-out size':<22}{'n_bins':>8}{'R2':>8}{'MAPE%':>8}"
      f"{'e_wbyte':>10}{'e_kv':>8}{'e_gemm':>8}")
for held, ap in SIZES:
    Xh, yh = data[held]
    if len(yh) < 20:
        continue
    Xtr = np.vstack([data[n][0] for n, _ in SIZES if n != held])
    ytr = np.concatenate([data[n][1] for n, _ in SIZES if n != held])
    coef = fit(Xtr, ytr)
    r2, mape = score(coef, Xh, yh)
    print(f"{held.replace('Qwen2.5-','').replace('-Instruct',''):<22}{len(yh):>8}"
          f"{r2:>8.3f}{mape:>8.1f}{coef[0]*1e10:>10.3f}{coef[1]*1e10:>8.2f}{coef[2]*1e12:>8.3f}")

print("\n=== B. Extrapolation across the size range ===")
def group(pred):
    return [n for n, ap in SIZES if pred(ap)]
def run(train_names, test_names, label):
    Xtr = np.vstack([data[n][0] for n in train_names])
    ytr = np.concatenate([data[n][1] for n in train_names])
    coef = fit(Xtr, ytr)
    Xte = np.vstack([data[n][0] for n in test_names])
    yte = np.concatenate([data[n][1] for n in test_names])
    r2, mape = score(coef, Xte, yte)
    tr = "+".join(n.split("-")[1] for n in train_names)
    te = "+".join(n.split("-")[1] for n in test_names)
    print(f" {label:<26} train[{tr}] -> test[{te}]:  R2={r2:.3f}  MAPE={mape:.1f}%")
run(group(lambda a: a <= 3.1), group(lambda a: a >= 13), "small->large")
run(group(lambda a: a >= 13), group(lambda a: a <= 3.1), "large->small")
run(["Qwen2.5-7B-Instruct"], group(lambda a: a != 7.07), "one 7B -> all others")
