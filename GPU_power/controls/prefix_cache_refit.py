#!/usr/bin/env python3
"""
controls/prefix_cache_refit.py — prefix-cache correction for the ENERGY model.

vLLM's IterationStats.num_prompt_tokens (our iter_log `prefill_tokens`) counts
the full prompt at the first-token iteration, including prefix-cache hits, which
are never computed. binned_table.csv's gemm_flops_bin = 2*P_active*(prefill +
decode) therefore over-counts prefill GEMM FLOPs by the cached fraction
(26-36% of prompt tokens at c=64 on the real-prompt runs). This script

  (a) ADDS two columns to every real-prompt run's binned_table.csv
        prefill_tokens_computed, gemm_flops_bin_computed
      (existing columns and values are left byte-identical; rows re-written with
      the original strings). The computed prefill per iter row comes from the
      RNG-replay + tokenized-pool matching in controls/sysid.py
      (computed_prefill; match rate 100% on current-harness runs).
      A sanity check recomputes gemm_flops_bin from the logged prefill and
      requires it to match the existing column.
  (b) re-fits the 3-term model (e_wbyte, e_kvbyte, e_gemm; P_static from idle)
      exactly as reconcile_gpus.fit3 does (OLS, 200-sample bootstrap CI,
      held-out-by-model MAPE) on logs/ragged (H200) and logs/B200 (B200), with
      the original vs the computed FLOPs.
  (c) checks whether the B200 < H200 e_gemm ordering survives (bootstrap).

Writes controls/prefix_cache_refit.json. CPU-only. From GPU_power/ in-container:
    python3 controls/prefix_cache_refit.py            # add columns + refit
    python3 controls/prefix_cache_refit.py --no_write # refit only (no csv edits)
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import sysid as S  # noqa: E402

GROUPS = {"H200": dict(fit_root="logs/ragged", roots=["logs/ragged", "logs/ragged_ladder",
                                                    "logs/ragged_moe"]),
          "B200": dict(fit_root="logs/B200", roots=["logs/B200", "logs/B200_32B",
                                                   "logs/B200_large"])}
OUT_JSON = os.path.join(HERE, "prefix_cache_refit.json")
NEW_COLS = ["prefill_tokens_computed", "gemm_flops_bin_computed"]


def correct_run(rd, toklens, write=True):
    """Compute per-bin computed prefill / GEMM FLOPs; optionally add columns."""
    R = S.load_run(rd, toklens, with_power=False)
    A = R["A"]
    tm = 0.5 * (A["t_start"] + A["t_end"])
    order = np.argsort(tm, kind="stable")
    tm = tm[order]
    pf_log = A["prefill_tokens"][order]
    pf_cmp = R["prefill_comp"][order]
    dec = A["decode_tokens"][order]
    mm = R["consts"]["gemm_flops_per_token"]
    path = os.path.join(S.ROOT, rd, "binned_table.csv")
    with open(path) as f:
        rows = list(csv.reader(f))
    hdr, body = rows[0], rows[1:]
    ix = {c: hdr.index(c) for c in hdr}
    out_pf, out_g, chk = [], [], []
    for r in body:
        t_mid, dt = float(r[ix["t_mid"]]), float(r[ix["dt_s"]])
        lo = np.searchsorted(tm, t_mid - 0.5 * dt, side="left")
        hi = np.searchsorted(tm, t_mid + 0.5 * dt, side="left")
        sl = slice(lo, hi)
        g_log = mm * (pf_log[sl].sum() + dec[sl].sum())
        g_cmp = mm * (pf_cmp[sl].sum() + dec[sl].sum())
        g_old = float(r[ix["gemm_flops_bin"]])
        chk.append(abs(g_log - g_old) / max(g_old, 1.0))
        out_pf.append(pf_cmp[sl].sum())
        out_g.append(g_cmp)
    chk = np.array(chk)
    ok = float(np.mean(chk < 1e-3))
    if write and ok > 0.99:
        keep = [c for c in hdr if c not in NEW_COLS]
        new_hdr = keep + NEW_COLS
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(new_hdr)
            for r, a, b in zip(body, out_pf, out_g):
                w.writerow([r[ix[c]] for c in keep] + [f"{a:.1f}", f"{b:.6e}"])
    return dict(run=rd, gpu=R["gpu"], model=R["model"], match_rate=R["match_rate"],
                cached_frac=R["cached_frac"], bins=len(body), recompute_ok_frac=ok,
                gemm_orig=float(sum(float(r[ix["gemm_flops_bin"]]) for r in body)),
                gemm_comp=float(sum(out_g)), written=bool(write and ok > 0.99)), \
        np.array(out_g)


def load_recs(gpu, toklens, write):
    recs, runinfo = [], []
    for root in GROUPS[gpu]["roots"]:
        for rd in sorted(glob.glob(os.path.join(S.ROOT, root, "*", "*"))):
            rel = os.path.relpath(rd, S.ROOT)
            if os.path.basename(rd) == "idle" or not os.path.exists(os.path.join(rd, "binned_table.csv")):
                continue
            meta = S.load_json(os.path.join(rd, "run_meta.json"), {})
            if not meta.get("task") or meta.get("idle_power_w") is None:
                continue
            with open(os.path.join(rd, "binned_table.csv")) as fh:
                if "gemm_flops_bin" not in fh.readline():
                    continue          # old schema (e.g. ragged_moe): no channel columns
            info, g_cmp = correct_run(rel, toklens, write)
            runinfo.append(info)
            W, K, G, Y = [], [], [], []
            idle = meta["idle_power_w"]
            for r in csv.DictReader(open(rd + "/binned_table.csv")):
                W.append(float(r["weight_bytes_bin"])); K.append(float(r["kv_bytes_bin"]))
                G.append(float(r["gemm_flops_bin"]))
                Y.append(float(r["energy_bin_j"]) - idle * float(r["dt_s"]))
            recs.append(dict(run_dir=rel, root=root, model=os.path.basename(os.path.dirname(rd)),
                             W=np.array(W), K=np.array(K), G=np.array(G), Gc=g_cmp,
                             Y=np.array(Y), crashed=rel in S.BAD_RUNS))
    return recs, runinfo


def fit3(recs, gkey="G", n_boot=200, seed=0):
    """Same estimator as reconcile_gpus.fit3, plus the bootstrap samples."""
    X = np.vstack([np.c_[r["W"], r["K"], r[gkey]] for r in recs])
    y = np.concatenate([r["Y"] for r in recs])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yh = X @ coef
    r2 = 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        c, *_ = np.linalg.lstsq(X[i], y[i], rcond=None)
        boots.append(c)
    b = np.array(boots)
    models = sorted({r["model"] for r in recs})
    ho = {}
    for h in models:
        tr = [r for r in recs if r["model"] != h]
        te = [r for r in recs if r["model"] == h]
        Xtr = np.vstack([np.c_[r["W"], r["K"], r[gkey]] for r in tr])
        ytr = np.concatenate([r["Y"] for r in tr])
        Xte = np.vstack([np.c_[r["W"], r["K"], r[gkey]] for r in te])
        yte = np.concatenate([r["Y"] for r in te])
        c, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
        pe = Xte @ c
        ho[h] = float(np.nanmean(np.abs((yte - pe) / np.where(yte == 0, np.nan, yte))) * 100)
    # share of dynamic energy attributed to GEMM
    gshare = float((coef[2] * X[:, 2]).sum() / y.sum())
    return dict(coef=coef.tolist(), lo=np.percentile(b, 2.5, 0).tolist(),
                hi=np.percentile(b, 97.5, 0).tolist(), boots=b, r2=float(r2),
                heldout=float(np.mean(list(ho.values()))), heldout_by_model=ho,
                n_bins=int(len(y)), gemm_energy_share=gshare, models=models)


def summarize(f):
    return {k: v for k, v in f.items() if k != "boots"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_write", action="store_true")
    args = ap.parse_args()
    toklens = S.load_json(S.TOKLEN_JSON, {})
    coef_json = S.load_json(S.COEF_JSON, {})
    out = {"_meta": dict(
        what="3-term energy fit with prefix-cache-corrected GEMM FLOPs",
        estimator="reconcile_gpus.fit3 (OLS, P_static=idle, 200 bootstrap, held-out-by-model)",
        columns_added=NEW_COLS, script="controls/prefix_cache_refit.py")}
    fits = {}
    for gpu, g in GROUPS.items():
        recs, info = load_recs(gpu, toklens, write=not args.no_write)
        frecs = [r for r in recs if r["root"] == g["fit_root"]]
        fo, fc = fit3(frecs, "G"), fit3(frecs, "Gc")
        # excluding the two crashed B200 runs (sensitivity)
        fx = fit3([r for r in frecs if not r["crashed"]], "Gc") if any(r["crashed"] for r in frecs) else None
        # all roots of the GPU (incl. ladder / 32B / 72B / MoE) - diagnostic
        fa_o, fa_c = fit3(recs, "G", n_boot=50), fit3(recs, "Gc", n_boot=50)
        tot_o = sum(r["G"].sum() for r in frecs)
        tot_c = sum(r["Gc"].sum() for r in frecs)
        pf_share = None
        fits[gpu] = (fo, fc)
        ent = dict(
            fit_root=g["fit_root"], n_runs=len(frecs),
            gemm_flops_reduction_pct=float(100 * (1 - tot_c / tot_o)),
            original=summarize(fo), corrected=summarize(fc),
            delta=dict(
                e_gemm_pct=100 * (fc["coef"][2] / fo["coef"][2] - 1),
                e_wbyte_pct=100 * (fc["coef"][0] / fo["coef"][0] - 1),
                e_kvbyte_pct=100 * (fc["coef"][1] / fo["coef"][1] - 1),
                r2=fc["r2"] - fo["r2"], heldout_mape=fc["heldout"] - fo["heldout"]),
            reproduces_gpu_coefficients_json=dict(
                e_wbyte=coef_json.get(gpu, {}).get("e_wbyte"),
                e_kvbyte=coef_json.get(gpu, {}).get("e_kvbyte"),
                e_gemm=coef_json.get(gpu, {}).get("e_gemm")),
            corrected_excluding_crashed_runs=summarize(fx) if fx else None,
            all_roots_original=summarize(fa_o), all_roots_corrected=summarize(fa_c),
            runs=info)
        out[gpu] = ent
        print(f"\n===== {gpu}  fit root {g['fit_root']}  ({len(frecs)} runs, GEMM FLOPs "
              f"-{ent['gemm_flops_reduction_pct']:.1f}%)")
        for name, f in (("original ", fo), ("corrected", fc)):
            c, lo, hi = f["coef"], f["lo"], f["hi"]
            print(f"  {name}: e_wbyte={c[0]:.4e} [{lo[0]:.4e},{hi[0]:.4e}]  e_kvbyte={c[1]:.3e} "
                  f"[{lo[1]:.3e},{hi[1]:.3e}]  e_gemm={c[2]*1e12:.3f} pJ [{lo[2]*1e12:.3f},{hi[2]*1e12:.3f}]"
                  f"  R2={f['r2']:.4f}  held-out={f['heldout']:.2f}%  GEMM share={100*f['gemm_energy_share']:.1f}%")
        d = ent["delta"]
        print(f"  delta: e_gemm {d['e_gemm_pct']:+.1f}%  e_wbyte {d['e_wbyte_pct']:+.2f}%  "
              f"e_kvbyte {d['e_kvbyte_pct']:+.1f}%  R2 {d['r2']:+.4f}  held-out {d['heldout_mape']:+.2f} pp")
        if fx:
            print(f"  corrected, excl. crashed runs: e_gemm={fx['coef'][2]*1e12:.3f} pJ "
                  f"[{fx['lo'][2]*1e12:.3f},{fx['hi'][2]*1e12:.3f}] R2={fx['r2']:.4f}")
        print(f"  all roots: e_gemm orig {fa_o['coef'][2]*1e12:.3f} -> corr {fa_c['coef'][2]*1e12:.3f} pJ, "
              f"R2 {fa_o['r2']:.4f} -> {fa_c['r2']:.4f}")
        bad = [i["run"] for i in info if i["recompute_ok_frac"] < 0.99]
        if bad:
            print(f"  WARNING recompute mismatch (columns not written): {bad}")
    # (c) ordering
    order = {}
    for tag, j in (("original", 0), ("corrected", 1)):
        bh = fits["H200"][j]["boots"][:, 2]
        bb = fits["B200"][j]["boots"][:, 2]
        ratio = fits["B200"][j]["coef"][2] / fits["H200"][j]["coef"][2]
        p = float(np.mean(bb[:, None] < bh[None, :]))
        order[tag] = dict(e_gemm_H200_pJ=fits["H200"][j]["coef"][2] * 1e12,
                          e_gemm_B200_pJ=fits["B200"][j]["coef"][2] * 1e12,
                          ratio_B200_over_H200=ratio, P_B200_lt_H200=p)
        print(f"\n[{tag}] e_gemm B200/H200 = {ratio:.3f}  "
              f"({fits['B200'][j]['coef'][2]*1e12:.3f} vs {fits['H200'][j]['coef'][2]*1e12:.3f} pJ), "
              f"bootstrap P(B200 < H200) = {p:.3f}")
    out["ordering"] = order
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {os.path.relpath(OUT_JSON, S.ROOT)}")


if __name__ == "__main__":
    main()
