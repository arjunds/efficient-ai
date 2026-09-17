#!/usr/bin/env python3
"""
pool_gpus.py — pool multiple GPUs' ragged logs, fit per-GPU energy coefficients,
and test the cross-GPU scaling law:  e_byte ∝ 1/HBM_bandwidth,  e_flop ∝ 1/peak.

Usage (one arg per GPU, LABEL:log_dir; log_dir is scanned for binned_table.csv at
any depth):

  python3 pool_gpus.py H200:logs/ragged L40S:logs/L40S A100-SXM:logs/A100

GPU specs (HBM bytes/s, peak fp16 FLOP/s, power cap W) come from the built-in
GPU_SPECS table below (extend it for new parts) or from a specs.json placed in the
log dir: {"bw":.., "peak_flops":.., "p_cap":..}. The scaling law is anchored on
the GPU whose label contains the ANCHOR (default H200), whose coefficients are
measured; other GPUs' measured coefficients are compared to the anchor scaled by
their spec ratios.

Writes plots_proposal/fig7_gpu_scaling.png and prints a measured-vs-predicted table.
Pure stdlib + numpy (+ matplotlib if available).
"""
import glob, os, csv, json, sys
import numpy as np

ANCHOR = "H200"
# datasheet specs: HBM bandwidth (bytes/s), peak fp16 dense (FLOP/s), power cap (W)
GPU_SPECS = {
    "H200":        dict(bw=4.8e12,   peak_flops=990e12, p_cap=700.0),
    "H100":        dict(bw=3.35e12,  peak_flops=990e12, p_cap=700.0),
    "A100-SXM":    dict(bw=2.039e12, peak_flops=312e12, p_cap=400.0),
    "A100-PCIe":   dict(bw=1.94e12,  peak_flops=312e12, p_cap=300.0),
    "A100":        dict(bw=2.039e12, peak_flops=312e12, p_cap=400.0),
    "L40S":        dict(bw=0.864e12, peak_flops=362e12, p_cap=350.0),
    "L40":         dict(bw=0.864e12, peak_flops=181e12, p_cap=300.0),
    "A6000":       dict(bw=0.768e12, peak_flops=155e12, p_cap=300.0),
    "V100":        dict(bw=0.90e12,  peak_flops=125e12, p_cap=300.0),
    "RTX4090":     dict(bw=1.008e12, peak_flops=330e12, p_cap=450.0),
    "MI300X":      dict(bw=5.3e12,   peak_flops=1300e12, p_cap=750.0),
}
FEAT3 = ["weight_bytes_bin", "kv_bytes_bin", "gemm_flops_bin"]


def specs_for(label, log_dir):
    sj = os.path.join(log_dir, "specs.json")
    if os.path.exists(sj):
        return json.load(open(sj))
    for key in sorted(GPU_SPECS, key=len, reverse=True):
        if key.lower().replace("-", "") in label.lower().replace("-", ""):
            return GPU_SPECS[key]
    return None


def find_run_dirs(log_dir):
    """binned_table.csv can sit at any depth under log_dir."""
    return sorted({os.path.dirname(p)
                   for p in glob.glob(os.path.join(log_dir, "**", "binned_table.csv"),
                                      recursive=True)})


def load_gpu(log_dir):
    """Pool bins across all runs. Returns per-bin arrays + per-run cap-check info."""
    W = {c: [] for c in FEAT3}; W["dyn"] = []; W["model"] = []
    capinfo = []
    for rd in find_run_dirs(log_dir):
        mp = os.path.join(rd, "run_meta.json")
        if not os.path.exists(mp):
            continue
        idle = json.load(open(mp)).get("idle_power_w")
        if idle is None:
            continue
        model = os.path.basename(os.path.dirname(rd))
        rj = os.path.join(rd, "results.json")
        if os.path.exists(rj):
            capinfo.append(json.load(open(rj)).get("avg_power_window_w", 0.0))
        for r in csv.DictReader(open(os.path.join(rd, "binned_table.csv"))):
            try:
                for c in FEAT3:
                    W[c].append(float(r[c]))
                W["dyn"].append(float(r["energy_bin_j"]) - idle*float(r["dt_s"]))
                W["model"].append(model)
            except (KeyError, ValueError):
                pass
    return W, capinfo


def fit3(W):
    X = np.c_[[np.array(W[c]) for c in FEAT3]].T
    y = np.array(W["dyn"])
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    yh = X @ c
    r2 = 1 - ((y-yh)**2).sum()/((y-y.mean())**2).sum()
    # held-out by model (if >1 model)
    models = sorted(set(W["model"]))
    ho = None
    if len(models) > 1:
        errs = []
        m = np.array(W["model"])
        for h in models:
            tr, te = m != h, m == h
            if te.sum() < 5 or tr.sum() < 5:
                continue
            ci, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
            pe = X[te] @ ci
            mape = np.nanmean(np.abs((y[te]-pe)/np.where(y[te] == 0, np.nan, y[te])))*100
            errs.append(mape)
        ho = float(np.nanmean(errs)) if errs else None
    return dict(e_wbyte=c[0], e_kvbyte=c[1], e_gemm=c[2], r2=r2, held=ho, n=len(y))


def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    gpus = {}
    for a in sys.argv[1:]:
        label, log_dir = a.split(":", 1)
        W, capinfo = load_gpu(log_dir)
        if len(W["dyn"]) < 20:
            print(f"[skip] {label}: only {len(W['dyn'])} bins in {log_dir}")
            continue
        sp = specs_for(label, log_dir)
        if sp is None:
            print(f"[warn] {label}: no specs (add to GPU_SPECS or specs.json); "
                  "coefficients fit but excluded from scaling")
        fit = fit3(W)
        capped = bool(sp and capinfo and np.mean(capinfo) > 0.97*sp["p_cap"])
        gpus[label] = dict(fit=fit, sp=sp, capped=capped,
                           avg_p=float(np.mean(capinfo)) if capinfo else None)

    # report
    print(f"\n{'GPU':<14}{'bw(TB/s)':>9}{'peak(TF)':>9}{'e_wbyte':>11}{'e_kvbyte':>10}"
          f"{'e_gemm(pJ)':>11}{'R2':>6}{'held%':>7}{'cap?':>6}{'n':>7}")
    for label, g in gpus.items():
        f, sp = g["fit"], g["sp"]
        bw = sp["bw"]/1e12 if sp else float("nan")
        pk = sp["peak_flops"]/1e12 if sp else float("nan")
        held = f"{f['held']:.1f}" if f["held"] is not None else "-"
        print(f"{label:<14}{bw:>9.2f}{pk:>9.0f}{f['e_wbyte']*1e10:>11.3f}"
              f"{f['e_kvbyte']*1e10:>10.2f}{f['e_gemm']*1e12:>11.3f}{f['r2']:>6.2f}"
              f"{held:>7}{'YES' if g['capped'] else 'no':>6}{f['n']:>7}")

    # scaling test, anchored
    anchor = next((l for l in gpus if ANCHOR.lower() in l.lower()), None)
    if not anchor or not gpus[anchor]["sp"]:
        print(f"\n[scaling] no anchor '{ANCHOR}' with specs — skipping scaling test")
        return
    A = gpus[anchor]; Af, Asp = A["fit"], A["sp"]
    print(f"\n=== scaling law vs {anchor} (predict e_* by datasheet ratio) ===")
    print("  (capped GPUs excluded — their linear fit is invalid)")
    print(f"{'GPU':<14}{'e_wbyte meas':>13}{'pred':>9}{'err%':>7}   "
          f"{'e_gemm meas':>12}{'pred':>9}{'err%':>7}")
    pts = []
    for label, g in gpus.items():
        if label == anchor or not g["sp"] or g["capped"]:
            continue
        f, sp = g["fit"], g["sp"]
        pred_wb = Af["e_wbyte"] * Asp["bw"]/sp["bw"]          # e_byte ∝ 1/bw
        pred_gm = Af["e_gemm"]  * Asp["peak_flops"]/sp["peak_flops"]  # e_flop ∝ 1/peak
        ew_err = 100*(f["e_wbyte"]-pred_wb)/pred_wb
        eg_err = 100*(f["e_gemm"]-pred_gm)/pred_gm
        print(f"{label:<14}{f['e_wbyte']*1e10:>13.3f}{pred_wb*1e10:>9.3f}{ew_err:>6.0f}%   "
              f"{f['e_gemm']*1e12:>12.3f}{pred_gm*1e12:>9.3f}{eg_err:>6.0f}%")
        pts.append((label, sp, f))

    _plot(anchor, A, pts, gpus)


def _plot(anchor, A, pts, gpus):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("plot skipped:", e); return
    os.makedirs("plots_proposal", exist_ok=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3), dpi=150)
    # all measured (uncapped) incl anchor
    good = [(anchor, A["sp"], A["fit"])] + pts
    # panel 1: e_wbyte vs 1/bw, line through origin from anchor
    inv_bw = np.array([1/s["bw"] for _, s, _ in good])
    ew = np.array([f["e_wbyte"] for _, _, f in good])
    xs = np.linspace(0, inv_bw.max()*1.1, 50)
    slope = A["fit"]["e_wbyte"] * A["sp"]["bw"]
    a1.plot(xs*1e12, slope*xs*1e10, "--", color="#a33862", label="∝ 1/BW (anchor)")
    a1.scatter(inv_bw*1e12, ew*1e10, s=60, color="#1f5fbf", zorder=3, edgecolor="w")
    for lbl, s, f in good:
        a1.annotate(lbl, (1/s["bw"]*1e12, f["e_wbyte"]*1e10), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    a1.set_xlabel("1 / HBM bandwidth  (1e-12 s/byte)")
    a1.set_ylabel("e_wbyte  (×10⁻¹⁰ J/byte)")
    a1.set_title("Memory coeff scales with 1/bandwidth"); a1.grid(alpha=.3); a1.legend(fontsize=8)
    # panel 2: e_gemm vs 1/peak
    inv_pk = np.array([1/s["peak_flops"] for _, s, _ in good])
    eg = np.array([f["e_gemm"] for _, _, f in good])
    xs2 = np.linspace(0, inv_pk.max()*1.1, 50)
    slope2 = A["fit"]["e_gemm"] * A["sp"]["peak_flops"]
    a2.plot(xs2*1e12, slope2*xs2*1e12, "--", color="#d1701a", label="∝ 1/peak (anchor)")
    a2.scatter(inv_pk*1e12, eg*1e12, s=60, color="#d1701a", zorder=3, edgecolor="w")
    for lbl, s, f in good:
        a2.annotate(lbl, (1/s["peak_flops"]*1e12, f["e_gemm"]*1e12), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    a2.set_xlabel("1 / peak FLOPS  (1e-12 s/flop)")
    a2.set_ylabel("e_gemm  (pJ/flop)")
    a2.set_title("Compute coeff scales with 1/peak-FLOPS"); a2.grid(alpha=.3); a2.legend(fontsize=8)
    fig.suptitle("Cross-GPU coefficient scaling — do measured e_* fall on the anchor line?",
                 fontsize=11)
    fig.tight_layout(); fig.savefig("plots_proposal/fig7_gpu_scaling.png")
    print("\nwrote plots_proposal/fig7_gpu_scaling.png")


if __name__ == "__main__":
    main()
