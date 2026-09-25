#!/usr/bin/env python3
"""
reconcile_gpus.py — one consistent cross-GPU comparison.

Why: the H200 corpus was binned with the canonical models.py (active_params
EXCLUDES the input-embedding gather, which reads only the batch's rows), while the
B200 runs were binned with a reconstructed models.py that counted both embedding
tables (~7.7% more weight bytes for Qwen2-7B). Re-bin everything with the canonical
file (bind ~/models.py), then compare with no convention mismatch.

Outputs
  * per-run table: realized bandwidth, MBU, direct J/byte (ratio of sums, no
    regression), for every GPU/model/task/concurrency  -> realized_utilization.csv
  * pooled 3-term fit per GPU with bootstrap 95% CIs
  * gpu_coefficients.json — the single source of truth other tools read

Usage (inside the container):
  python3 reconcile_gpus.py            # uses already-binned tables
  (re-bin first with: python3 energy_model.py --two_term --run_dir <dir>)
"""
import csv, glob, json, os, re, sys
from datetime import date
import numpy as np

# GPU groups: which log roots belong to which GPU, and which root is the
# well-conditioned calibration set (7B-class, per FINDINGS_B200 identifiability).
GROUPS = {
    "H200": dict(roots=["logs/ragged", "logs/ragged_ladder"], fit_root="logs/ragged",
                 gpu_name="NVIDIA H200", mem_tech="HBM3e",
                 bw=4.8e12, peak_flops=9.9e14, p_cap=700.0),
    "B200": dict(roots=["logs/B200", "logs/B200_32B", "logs/B200_large"], fit_root="logs/B200",
                 gpu_name="NVIDIA B200", mem_tech="HBM3e",
                 bw=8.0e12, peak_flops=2.25e15, p_cap=1000.0),
    "A5000": dict(roots=["logs/A5000"], fit_root="logs/A5000",
                  gpu_name="NVIDIA RTX A5000", mem_tech="GDDR6",
                  bw=7.68e11, peak_flops=None, p_cap=230.0),
}
FEAT = ["weight_bytes_bin", "kv_bytes_bin", "gemm_flops_bin"]


def parse_tag(tag):
    m = re.match(r"(alpaca|sharegpt)_(?:c(\d+)|poisson(\d+))", tag)
    if not m:
        return None, None, None
    return m.group(1), (int(m.group(2)) if m.group(2) else None), \
        (float(m.group(3)) if m.group(3) else None)


def run_record(rd):
    tbl = os.path.join(rd, "binned_table.csv"); mp = os.path.join(rd, "run_meta.json")
    if not (os.path.exists(tbl) and os.path.exists(mp)):
        return None
    meta = json.load(open(mp)); idle = meta.get("idle_power_w")
    if idle is None:
        return None
    res = {}
    rj = os.path.join(rd, "results.json")
    if os.path.exists(rj):
        res = json.load(open(rj))
    cols = {c: [] for c in FEAT}; dyn = []; dt = []; graw = []; n_corr = 0
    for r in csv.DictReader(open(tbl)):
        try:
            for c in FEAT:
                if c == "gemm_flops_bin" and r.get("gemm_flops_bin_computed") not in (None, ""):
                    cols[c].append(float(r["gemm_flops_bin_computed"])); n_corr += 1
                else:
                    cols[c].append(float(r[c]))
            graw.append(float(r["gemm_flops_bin"]))
            dyn.append(float(r["energy_bin_j"]) - idle*float(r["dt_s"])); dt.append(float(r["dt_s"]))
        except (KeyError, ValueError):
            pass
    if not dyn:
        return None
    task, conc, rate = parse_tag(os.path.basename(rd))
    W = np.array(cols["weight_bytes_bin"]); K = np.array(cols["kv_bytes_bin"])
    G = np.array(cols["gemm_flops_bin"]); Y = np.array(dyn); T = np.array(dt)
    return dict(run_dir=rd, model=os.path.basename(os.path.dirname(rd)), task=task,
                conc=conc, rate=rate, idle=idle, W=W, K=K, G=G, Y=Y, T=T,
                G_raw=np.array(graw), flops_corrected=(n_corr == len(dyn)),
                tok_s=res.get("aggregate_tokens_per_sec"),
                p_avg=res.get("avg_power_window_w"))


def fit3(recs, n_boot=200, seed=0):
    X = np.vstack([np.c_[r["W"], r["K"], r["G"]] for r in recs])
    y = np.concatenate([r["Y"] for r in recs])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yh = X @ coef; r2 = 1 - ((y-yh)**2).sum()/((y-y.mean())**2).sum()
    rng = np.random.default_rng(seed); boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y)); c, *_ = np.linalg.lstsq(X[i], y[i], rcond=None)
        boots.append(c)
    b = np.array(boots); lo, hi = np.percentile(b, 2.5, 0), np.percentile(b, 97.5, 0)
    # held-out by model
    models = sorted({r["model"] for r in recs}); ho = []
    for h in models:
        tr = [r for r in recs if r["model"] != h]; te = [r for r in recs if r["model"] == h]
        if not tr or not te:
            continue
        Xtr = np.vstack([np.c_[r["W"], r["K"], r["G"]] for r in tr]); ytr = np.concatenate([r["Y"] for r in tr])
        Xte = np.vstack([np.c_[r["W"], r["K"], r["G"]] for r in te]); yte = np.concatenate([r["Y"] for r in te])
        c, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None); pe = Xte @ c
        ho.append(np.nanmean(np.abs((yte-pe)/np.where(yte == 0, np.nan, yte)))*100)
    cv = float(y.std()/y.mean()) if y.mean() else float("nan")
    return dict(coef=coef, lo=lo, hi=hi, r2=float(r2), heldout=float(np.mean(ho)) if ho else None,
                n_bins=int(len(y)), cv_dyn=cv, models=models)


def main():
    out = {"_meta": {
        "convention": "canonical ~/models.py: active_params excludes the input-embedding "
                      "gather (reads only batch rows); lm_head counted once. All runs re-binned "
                      "with it — H200 and B200 are now apples-to-apples.",
        "generated": str(date.today()), "script": "reconcile_gpus.py",
        "direct_jbyte_def": "c=1 runs: sum(dynamic energy)/sum(weight+KV bytes) over "
                            "active bins; no regression. Includes the small (~1-5%) "
                            "compute-energy share, so it is a slight upper bound on "
                            "memory energy per byte.",
        "flops": "gemm_flops use prefix-cache-corrected COMPUTED prefill tokens "
                 "(controls/prefix_cache_refit.py) when present; *_uncorrected keys "
                 "keep the logged-token fit. Logged prefill_tokens ignore prefix-cache "
                 "hits (26-36% at c=64) and biased e_gemm ~17-19% low."}}
    util_rows = []
    for gpu, g in GROUPS.items():
        recs = [r for root in g["roots"] for rd in sorted(glob.glob(os.path.join(root, "*", "*")))
                if os.path.isdir(rd) and (r := run_record(rd))]
        if not recs:
            print(f"[{gpu}] no data yet"); continue
        idles = [r["idle"] for r in recs]
        entry = dict(gpu_name=g["gpu_name"], mem_tech=g["mem_tech"], bw=g["bw"],
                     peak_flops=g["peak_flops"], p_cap=g["p_cap"],
                     p_static=float(np.median(idles)), p_static_range=[min(idles), max(idles)])
        # per-run realized utilization + direct J/byte
        print(f"\n===== {gpu} ({len(recs)} runs) =====")
        print(f"{'model':<24}{'task':<9}{'c':>4}{'BW TB/s':>9}{'MBU':>6}{'J/byte (direct)':>17}{'tok/s':>8}")
        direct = {}
        for r in recs:
            B = r["W"].sum() + r["K"].sum(); T = r["T"].sum()
            bw_real = B/T if T else 0.0; mbu = bw_real/g["bw"]
            jb = r["Y"].sum()/B if B else float("nan")
            util_rows.append(dict(gpu=gpu, model=r["model"], task=r["task"], conc=r["conc"],
                                  rate=r["rate"], realized_bw=bw_real, mbu=mbu, jbyte_direct=jb,
                                  tok_s=r["tok_s"], p_avg=r["p_avg"], idle=r["idle"],
                                  weight_bytes_per_bin=float(r["W"].mean()), run_dir=r["run_dir"]))
            if r["conc"] == 1:
                direct.setdefault(r["model"], []).append(jb)
                print(f"{r['model']:<24}{r['task']:<9}{r['conc']:>4}{bw_real/1e12:>9.2f}{mbu:>6.2f}"
                      f"{jb:>17.3e}{(r['tok_s'] or 0):>8.1f}")
        entry["direct_c1_jbyte"] = {m: float(np.mean(v)) for m, v in direct.items()}
        # pooled fit on the well-conditioned calibration root
        frecs = [r for r in recs if r["run_dir"].startswith(g["fit_root"] + "/")]
        if frecs:
            f = fit3(frecs)
            c, lo, hi = f["coef"], f["lo"], f["hi"]
            entry.update(e_wbyte=float(c[0]), e_kvbyte=float(c[1]), e_gemm=float(c[2]),
                         e_wbyte_ci=[float(lo[0]), float(hi[0])], e_kvbyte_ci=[float(lo[1]), float(hi[1])],
                         e_gemm_ci=[float(lo[2]), float(hi[2])], fit_r2=f["r2"],
                         fit_heldout_mape=f["heldout"], fit_n_bins=f["n_bins"],
                         fit_cv_dyn=f["cv_dyn"], fit_dataset=g["fit_root"], fit_models=f["models"])
            n_corr = sum(r["flops_corrected"] for r in frecs)
            entry["flops_prefix_cache_corrected"] = f"{n_corr}/{len(frecs)} runs"
            fu = fit3([dict(r, G=r["G_raw"]) for r in frecs], n_boot=100)
            entry.update(e_gemm_uncorrected=float(fu["coef"][2]),
                         e_wbyte_uncorrected=float(fu["coef"][0]),
                         e_kvbyte_uncorrected=float(fu["coef"][1]))
            print(f"  [prefix-cache corrected FLOPs in {n_corr}/{len(frecs)} runs; "
                  f"uncorrected e_gemm={fu['coef'][2]*1e12:.3f} pJ]")
            print(f"  3-term fit [{g['fit_root']}]: e_wbyte={c[0]:.3e} [{lo[0]:.3e},{hi[0]:.3e}]  "
                  f"e_kvbyte={c[1]:.3e}  e_gemm={c[2]*1e12:.3f} pJ [{lo[2]*1e12:.3f},{hi[2]*1e12:.3f}]  "
                  f"R2={f['r2']:.3f} held-out={f['heldout']}  CV(dyn)={f['cv_dyn']:.3f}")
        # per-model fit quality on other roots (diagnostic only)
        for root in g["roots"]:
            if root == g["fit_root"]:
                continue
            rr = [r for r in recs if r["run_dir"].startswith(root + "/")]
            for m in sorted({r["model"] for r in rr}):
                sub = [r for r in rr if r["model"] == m]
                if sum(len(r["Y"]) for r in sub) > 50:
                    f = fit3(sub, n_boot=50)
                    print(f"  diag {root}/{m}: e_wbyte={f['coef'][0]:.3e} R2={f['r2']:.3f} CV={f['cv_dyn']:.3f}")
        # Validity gate: a GPU pinned at its enforced power limit is in the
        # constant-power regime (HANDOFF gotcha #1) — its coefficients are not
        # physics. Keep it OUT of the coefficient table so downstream tools
        # (recommend_gpu.py auto-loads entries) don't consume garbage.
        lim = []
        for r in recs:
            mp = os.path.join(r["run_dir"], "run_meta.json")
            try:
                m = json.load(open(mp))
                v = m.get("enforced_power_limit_w") or m.get("power_limit_w")
                if v:
                    lim.append(float(v))
            except Exception:
                pass
        enforced = min(lim) if lim else g["p_cap"]
        p_avgs = [r["p_avg"] for r in recs if r["p_avg"]]
        capped_frac = (sum(p >= 0.95*enforced for p in p_avgs)/len(p_avgs)) if p_avgs else 0.0
        if capped_frac > 0.5 or (entry.get("fit_r2") is not None and entry["fit_r2"] < 0):
            reason = (f"{capped_frac:.0%} of runs at >=95% of enforced power limit "
                      f"{enforced:.0f} W; fit R2={entry.get('fit_r2')}")
            print(f"  !! {gpu} EXCLUDED from coefficient table: {reason}")
            out.setdefault("_excluded", {})[gpu] = dict(reason=reason, enforced_limit_w=enforced,
                                                       direct_c1_jbyte_capped=entry.get("direct_c1_jbyte"))
            continue
        out[gpu] = entry
    json.dump(out, open("gpu_coefficients.json", "w"), indent=2)
    with open("realized_utilization.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(util_rows[0].keys())); w.writeheader(); w.writerows(util_rows)
    print("\nwrote gpu_coefficients.json, realized_utilization.csv")
    # headline: the anchor model on both GPUs
    a = out.get("H200", {}).get("direct_c1_jbyte", {}).get("Qwen2-7B-Instruct")
    b = out.get("B200", {}).get("direct_c1_jbyte", {}).get("Qwen2-7B-Instruct")
    if a and b:
        pred = a * 4.8e12/8.0e12
        print(f"\nHEADLINE Qwen2-7B c=1 direct J/byte: H200 {a:.3e}  B200 {b:.3e}  "
              f"ratio {b/a:.2f}  (1/BW law predicted B200 {pred:.3e}, i.e. ratio 0.60)")


if __name__ == "__main__":
    main()
