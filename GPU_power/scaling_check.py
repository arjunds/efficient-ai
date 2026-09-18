#!/usr/bin/env python3
"""
scaling_check.py — test the cross-GPU coefficient scaling law against the
PUBLISHED H200 coefficients, without needing the H200 raw logs.

WHY THIS EXISTS: pool_gpus.py is the right tool once you hold every GPU's
binned_table.csv, but it anchors on a *log directory*. The H200 logs live on the
old cluster and were not part of the repo, so on a fresh server there is nothing
to anchor on. The H200 numbers we need are, however, recorded in
HANDOFF_CROSSGPU.md / PROPOSAL.md, and that is all the scaling test requires.
Once the H200 logs are copied over, prefer:

    python3 pool_gpus.py H200:logs/ragged B200:logs/B200

which refits both from raw bins and is strictly better.

The law under test (anchored on H200):
    e_wbyte, e_kvbyte  ∝ 1 / HBM_bandwidth
    e_gemm             ∝ 1 / peak_dense_FLOPS

Usage:
    python3 scaling_check.py B200:logs/B200 [LABEL:dir ...]
"""
import json
import os
import sys

import numpy as np

from pool_gpus import GPU_SPECS, load_gpu, fit3, specs_for

# --- H200 reference, MEASURED (HANDOFF_CROSSGPU.md table; fit_channels 3-term,
#     pooled dense models, logs/ragged on the old cluster). ---
H200 = dict(
    label="H200 (published)",
    bw=4.8e12, peak_flops=990e12, p_cap=700.0, p_static=118.0,
    e_wbyte=1.077e-10, e_kvbyte=3.380e-10, e_gemm=0.673e-12,
    e_bit=1.12e-10, e_flop=0.87e-12,
)


def report(label, log_dir):
    W, capinfo = load_gpu(log_dir)
    n = len(W["dyn"])
    if n < 20:
        print(f"[skip] {label}: only {n} bins under {log_dir}")
        return None
    sp = specs_for(label, log_dir)
    if sp is None:
        print(f"[warn] {label}: no specs -- add to pool_gpus.GPU_SPECS or drop a "
              f"specs.json in {log_dir}")
        return None
    f = fit3(W)
    avg_p = float(np.mean(capinfo)) if capinfo else None
    capped = bool(avg_p and avg_p > 0.97 * sp["p_cap"])
    return dict(label=label, sp=sp, fit=f, avg_p=avg_p, capped=capped, n=n)


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    rows = [r for r in (report(*a.split(":", 1)) for a in args) if r]
    if not rows:
        print("nothing to report")
        return

    print(f"\n=== measured coefficients ===")
    print(f"{'GPU':<12}{'bw(TB/s)':>9}{'peak(TF)':>9}{'e_wbyte':>10}{'e_kvbyte':>10}"
          f"{'e_gemm(pJ)':>11}{'R2':>6}{'held%':>7}{'avgP':>7}{'cap':>6}{'n':>7}")
    print(f"{'H200*':<12}{H200['bw']/1e12:>9.2f}{H200['peak_flops']/1e12:>9.0f}"
          f"{H200['e_wbyte']*1e10:>10.3f}{H200['e_kvbyte']*1e10:>10.2f}"
          f"{H200['e_gemm']*1e12:>11.3f}{'-':>6}{'-':>7}{'-':>7}{'no':>6}{'-':>7}")
    for r in rows:
        f, sp = r["fit"], r["sp"]
        held = f"{f['held']:.1f}" if f["held"] is not None else "-"
        ap = f"{r['avg_p']:.0f}" if r["avg_p"] else "-"
        print(f"{r['label']:<12}{sp['bw']/1e12:>9.2f}{sp['peak_flops']/1e12:>9.0f}"
              f"{f['e_wbyte']*1e10:>10.3f}{f['e_kvbyte']*1e10:>10.2f}"
              f"{f['e_gemm']*1e12:>11.3f}{f['r2']:>6.2f}{held:>7}{ap:>7}"
              f"{'YES' if r['capped'] else 'no':>6}{r['n']:>7}")
    print("  * H200 row is the published measurement, not refit here.")

    print(f"\n=== scaling law vs H200 ===")
    print(f"  predicted = H200 coeff x (H200 spec / this GPU's spec)")
    print(f"{'GPU':<12}{'coeff':<10}{'measured':>11}{'predicted':>11}{'err%':>8}")
    ok = True
    for r in rows:
        if r["capped"]:
            print(f"  [{r['label']}] POWER-CAPPED (avg {r['avg_p']:.0f} W vs cap "
                  f"{r['sp']['p_cap']:.0f} W) -- coefficients are not meaningful; "
                  f"see HANDOFF gotcha #1")
            continue
        f, sp = r["fit"], r["sp"]
        bw_r = H200["bw"] / sp["bw"]
        fl_r = H200["peak_flops"] / sp["peak_flops"]
        for name, meas, pred, scale in (
                ("e_wbyte", f["e_wbyte"], H200["e_wbyte"] * bw_r, 1e10),
                ("e_kvbyte", f["e_kvbyte"], H200["e_kvbyte"] * bw_r, 1e10),
                ("e_gemm", f["e_gemm"], H200["e_gemm"] * fl_r, 1e12)):
            err = 100 * (meas - pred) / pred if pred else float("nan")
            flag = "" if abs(err) <= 20 else "   <-- outside +/-20%"
            if abs(err) > 20:
                ok = False
            print(f"{r['label']:<12}{name:<10}{meas*scale:>11.3f}{pred*scale:>11.3f}"
                  f"{err:>7.0f}%{flag}")
    print("\n  Success criterion (HANDOFF): measured within ~10-20% of predicted.")
    print(f"  => {'PASS' if ok else 'at least one coefficient outside 20%'}")


if __name__ == "__main__":
    main()
