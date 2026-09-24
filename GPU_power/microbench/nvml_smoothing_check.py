#!/usr/bin/env python3
"""
nvml_smoothing_check.py -- does NVML's 1 s power averaging bias the per-bin fits?

NVML docs (nvmlDeviceGetPowerUsage): "On Ampere (except GA100) or newer GPUs, the
API returns power averaged over 1 sec interval." H200 (Hopper) and B200
(Blackwell) are newer than Ampere, so the per-bin energy in logs/ragged and
logs/B200 is the integral of a 1 s TRAILING moving average of true power. For
200 ms bins that is Y_obs = k * Y_true with k ~= [0.1,0.2,0.2,0.2,0.2,0.1] over
the current and 5 previous bins (plus the ~0.5 s group delay).

Convolution is linear, so the unbiased fix is to smear the regressors with the
same kernel: Y_obs = (k*X) beta. This script fits the pooled 3-term model both
ways (raw X, and k*X) and reports the coefficient change. Run-level ratios
(direct J/byte) are integrals and unaffected either way.

  python3 microbench/nvml_smoothing_check.py logs/ragged logs/B200
"""
import csv, glob, json, os, sys
import numpy as np

K = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])          # weights for lags 0..5 bins
FEAT = ["weight_bytes_bin", "kv_bytes_bin", "gemm_flops_bin"]


def load(rd):
    meta = json.load(open(os.path.join(rd, "run_meta.json")))
    idle = meta.get("idle_power_w")
    rows = list(csv.DictReader(open(os.path.join(rd, "binned_table.csv"))))
    if idle is None or len(rows) < 20:
        return None
    t = np.array([float(r["t_mid"]) for r in rows])
    dt = np.array([float(r["dt_s"]) for r in rows])
    X = np.array([[float(r[c]) for c in FEAT] for r in rows])
    Y = np.array([float(r["energy_bin_j"]) for r in rows]) - idle * dt
    return t, dt, X, Y


def kernel(n):
    """Trailing boxcar of n bins convolved with the bin (half weights at ends)."""
    if n <= 1:
        return np.array([1.0])
    k = np.ones(n + 1); k[0] = k[-1] = 0.5
    return k / k.sum()


def smear(t, X, bin_s=0.2, k=None):
    """Causal smear by K on the time grid (missing bins count as zero work)."""
    idx = np.round((t - t[0]) / bin_s).astype(int)
    grid = np.zeros((idx.max() + 1, X.shape[1]))
    grid[idx] = X
    out = np.zeros_like(grid)
    for lag, w in enumerate(K if k is None else k):
        out[lag:] += w * grid[:len(grid) - lag]
    return out[idx]


def fit(X, Y):
    c, *_ = np.linalg.lstsq(X, Y, rcond=None)
    r2 = 1 - ((Y - X @ c) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()
    return c, r2


def main():
    res = {}
    for root in sys.argv[1:]:
        Xr, Xs, Ys = [], [], []
        for tb in sorted(glob.glob(os.path.join(root, "*", "*", "binned_table.csv"))):
            d = load(os.path.dirname(tb))
            if d is None:
                continue
            t, dt, X, Y = d
            Xs.append(smear(t, X)); Xr.append(X); Ys.append(Y)
        if not Ys:
            continue
        Xr, Xs, Y = np.vstack(Xr), np.vstack(Xs), np.concatenate(Ys)
        (cr, r2r), (cs, r2s) = fit(Xr, Y), fit(Xs, Y)
        res[root] = {"n_bins": int(len(Y)), "raw": {"coef": cr.tolist(), "r2": float(r2r)},
                     "smeared_X": {"coef": cs.tolist(), "r2": float(r2s)},
                     "rel_change": ((cs - cr) / cr).tolist()}
        print(f"{root}: n={len(Y)}")
        for nm, a, b in zip(["e_wbyte", "e_kvbyte", "e_gemm"], cr, cs):
            print(f"   {nm:<9} raw={a:.4e}  smeared-X={b:.4e}  change={100*(b-a)/a:+.1f}%")
        print(f"   R2 raw={r2r:.3f}  smeared-X={r2s:.3f}")
        # scan the averaging window: which trailing window best explains the data?
        scan = []
        Ts = [load(os.path.dirname(tb)) for tb in sorted(glob.glob(os.path.join(root, "*", "*", "binned_table.csv")))]
        Ts = [d for d in Ts if d is not None]
        for n in range(0, 11):
            Xn = np.vstack([smear(d[0], d[2], k=kernel(n)) for d in Ts])
            c, r2 = fit(Xn, Y)
            scan.append({"window_s": 0.2 * n, "r2": float(r2), "coef": c.tolist()})
        best = max(scan, key=lambda z: z["r2"])
        print("   window scan (s: R2): " + "  ".join(f"{z['window_s']:.1f}:{z['r2']:.4f}" for z in scan)
              + f"   -> best {best['window_s']:.1f} s")
        res[root]["window_scan"] = scan
    json.dump(res, open("microbench/nvml_smoothing_check.json", "w"), indent=1)


if __name__ == "__main__":
    main()
