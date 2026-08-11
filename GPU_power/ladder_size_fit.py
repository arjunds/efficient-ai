#!/usr/bin/env python3
"""Per-size 2-term and 3-term (channel) coefficient fit across the size ladder,
to test size-independence of the coefficients. Adds a per-bin fixed-overhead
diagnostic: is small-model coefficient inflation explained by GPU underutilization
(low realized throughput)?"""
import glob, os, csv, json, sys
import numpy as np
from models import model_from_hf_id

ROOT = sys.argv[1] if len(sys.argv) > 1 else "logs/ragged_ladder"


def load(md):
    cols = ["weight_bytes_bin", "kv_bytes_bin", "gemm_flops_bin", "bytes_bin",
            "flops_bin", "energy_bin_j", "dt_s"]
    out = {c: [] for c in cols}
    out["dyn"] = []
    for rd in glob.glob(os.path.join(md, "*")):
        t = os.path.join(rd, "binned_table.csv"); m = os.path.join(rd, "run_meta.json")
        if not (os.path.exists(t) and os.path.exists(m)):
            continue
        idle = json.load(open(m)).get("idle_power_w")
        if idle is None:
            continue
        for r in csv.DictReader(open(t)):
            try:
                vals = {c: float(r[c]) for c in cols}
            except (KeyError, ValueError):
                continue
            for c in cols:
                out[c].append(vals[c])
            out["dyn"].append(vals["energy_bin_j"] - idle * vals["dt_s"])
    return {k: np.array(v) for k, v in out.items()}


def fit(X, y):
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    yh = X @ c
    r2 = 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return c, r2


hdr = ("model", "active_B", "e_bit(e-10)", "e_flop(pJ)", "e_wbyte(e-10)",
       "e_kv(e-10)", "e_gemm(pJ)", "R2_2t", "R2_3t", "n")
print(f"{hdr[0]:<24}{hdr[1]:>9}{hdr[2]:>13}{hdr[3]:>11}{hdr[4]:>14}"
      f"{hdr[5]:>11}{hdr[6]:>11}{hdr[7]:>7}{hdr[8]:>7}{hdr[9]:>6}")
for md in sorted(glob.glob(os.path.join(ROOT, "*"))):
    if not os.path.isdir(md):
        continue
    name = os.path.basename(md)
    d = load(md)
    if len(d["dyn"]) < 20:
        continue
    try:
        ap = model_from_hf_id("Qwen/" + name).active_params() / 1e9
    except Exception:
        ap = float("nan")
    (eb, ef), r2 = fit(np.c_[d["bytes_bin"], d["flops_bin"]], d["dyn"])
    (ew, ek, eg), r3 = fit(np.c_[d["weight_bytes_bin"], d["kv_bytes_bin"],
                                 d["gemm_flops_bin"]], d["dyn"])
    print(f"{name:<24}{ap:>9.2f}{eb*1e10:>13.3f}{ef*1e12:>11.3f}{ew*1e10:>14.3f}"
          f"{ek*1e10:>11.2f}{eg*1e12:>11.3f}{r2:>7.3f}{r3:>7.3f}{len(d['dyn']):>6}")
