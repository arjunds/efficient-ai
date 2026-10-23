#!/usr/bin/env python3
"""
prefill_spread_compare.py -- verify and evaluate energy_model.py's prefill spreading
(TIMEMODEL_HIGHC.md, Task 2).  CPU only; run in the container (model configs).

  (a) bit-identity: runs WITHOUT requests.csv (Sept B200/H200/ladder/MoE/longctx) bin exactly as
      the pre-change energy_model.py (a frozen copy is passed with --orig); runs WITH requests.csv
      and spreading forced off also bin exactly as before.
  (b) run totals: prompt tokens, computed prompt tokens, GEMM FLOPs over the window, spread vs not.
  (c) per-bin fits, energy counter, before/after spreading, by arm group:
        free 3-term   Y = E - P_idle*dt = e_w*W + e_kv*K + e_g*G      (G = computed GEMM FLOPs)
        e_g only      with e_w, e_kv held at the settled B200 values (1.310e-10, 1.548e-10)
      and the mean |residual| of bins that contain prefill.
  Writes binned_table_spread.csv (budget mode) next to each compared run's binned_table.csv
  (never overwrites binned_table.csv) and prefill_spread_compare.json.

  git show 281de59:GPU_power/energy_model.py > /tmp/energy_model_orig.py    # pre-change copy
  python3 prefill_spread_compare.py --orig /tmp/energy_model_orig.py      # (a) is skipped without it
"""
import argparse
import csv
import glob
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)
import energy_model as em  # noqa: E402

E_W, E_KV, E_G_STEADY = 1.310e-10, 1.548e-10, 0.943e-12      # settled B200 (UPDATE 6 review), steady GEMM

OLD_RUNS = ["logs/B200/*/*", "logs/ragged/*/*", "logs/ragged_moe/*/*", "logs/longctx/*/*", "logs/B200_32B/*/*"]
GROUPS = {
    "longctx 7B (c4/c16, 1k-30k)": "logs/B200_prov/longctx/Qwen2-7B-Instruct/ctx*",
    "longctx32 32B": "logs/B200_prov/longctx32/Qwen2.5-32B-Instruct/ctx*",
    "prefill arm 7B (in2048/in8192, capped)": "logs/B200_prov/prefill/Qwen2-7B-Instruct/in*",
    "sharegpt 7B graphs c1-64": "logs/B200_prov/graphs/Qwen2-7B-Instruct/sharegpt_c*",
    "sharegpt 7B highconc c128-512": "logs/B200_prov/highconc/Qwen2-7B-Instruct/sharegpt_c*",
    "alpaca 7B graphs+highconc (control: short prompts)": "logs/B200_prov/*/Qwen2-7B-Instruct/alpaca_c*",
}


def load_orig(path):
    spec = importlib.util.spec_from_file_location("energy_model_orig", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["energy_model_orig"] = mod
    spec.loader.exec_module(mod)
    return mod


def same_bins(a, b):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if set(x) != set(y):
            return False
        for k in x:
            if x[k] != y[k]:
                return False
    return True


def export_text(mod, rd, bins, psrc, **kw):
    """Write via the module's exporter into a scratch dir (never the run dir) and read back."""
    d = os.path.join("/tmp", "btdir_%d" % os.getpid())
    os.makedirs(d, exist_ok=True)
    mod._export_binned_table(d, bins, psrc, **kw)
    return open(os.path.join(d, kw.get("name", "binned_table.csv"))).read()


def check_identity(orig):
    out = []
    rds = []
    for pat in OLD_RUNS:
        rds += [rd for rd in sorted(glob.glob(pat)) if os.path.exists(os.path.join(rd, "iter_log.csv"))][:3]
    rds += ["logs/B200_prov/longctx/Qwen2-7B-Instruct/ctx16384_c4", "logs/B200_prov/highconc/Qwen2-7B-Instruct/sharegpt_c128"]
    for rd in rds:
        has_req = os.path.exists(os.path.join(rd, "requests.csv"))
        try:
            b0, i0 = orig.bin_run(rd, phase=None)
        except Exception as e:                                     # unresolvable model etc.
            out.append(dict(run=rd, error=str(e)[:80]))
            continue
        kw = dict(spread_prefill=False) if has_req else {}
        b1, i1 = em.bin_run(rd, phase=None, **kw)
        t0 = export_text(orig, rd, b0, i0.get("power_source"))
        t1 = export_text(em, rd, b1, i1.get("power_source"))
        bd0, _ = orig.bin_run(rd, phase="decode")
        bd1, _ = em.bin_run(rd, phase="decode")
        rec = dict(run=rd, requests_csv=has_req, forced_off=has_req, n_bins=len(b0),
                   bins_identical=same_bins(b0, b1), table_identical=(t0 == t1),
                   decode_phase_identical=same_bins(bd0, bd1))
        committed = os.path.join(rd, "binned_table.csv")
        if os.path.exists(committed):
            rec["matches_committed_binned_table"] = (open(committed).read() == t1)
        out.append(rec)
        print("  %-62s req=%-5s bins_identical=%s table_identical=%s decode_identical=%s committed=%s" % (
            rd, has_req, rec["bins_identical"], rec["table_identical"], rec["decode_phase_identical"],
            rec.get("matches_committed_binned_table")))
    return out


def fit_bins(bins, idle):
    rows = []
    for b in bins:
        G = b["gemm_flops_computed"] if b.get("gemm_flops_computed") is not None else b["gemm_flops"]
        rows.append((b["weight_bytes"], b["kv_bytes"], G, b["E"] - idle * b["dt"], b["prefill"] > 0))
    A = np.array([r[:3] for r in rows])
    y = np.array([r[3] for r in rows])
    pf = np.array([r[4] for r in rows])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    r2 = 1 - ((y - A @ c) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    yr = y - E_W * A[:, 0] - E_KV * A[:, 1]
    eg_all = float(A[:, 2] @ yr / (A[:, 2] @ A[:, 2]))
    eg_pf = float(A[pf, 2] @ yr[pf] / (A[pf, 2] @ A[pf, 2])) if pf.sum() >= 5 else None
    res_fixed = y - E_W * A[:, 0] - E_KV * A[:, 1] - E_G_STEADY * A[:, 2]
    yhat = E_W * A[:, 0] + E_KV * A[:, 1] + eg_all * A[:, 2]
    r2_fixed = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return dict(n=len(y), n_prefill_bins=int(pf.sum()), e_w=float(c[0]), e_kv=float(c[1]), e_g_pJ=float(c[2] * 1e12),
                r2_free=float(r2), e_g_pJ_fixedWK=eg_all * 1e12, r2_fixedWK=float(r2_fixed),
                e_g_pJ_fixedWK_prefill_bins=(eg_pf * 1e12 if eg_pf is not None else None),
                mean_abs_resid_prefill_bins_at_steady_J=float(np.abs(res_fixed[pf]).mean()) if pf.any() else None,
                mean_abs_resid_all_bins_at_steady_J=float(np.abs(res_fixed).mean()))


def compare():
    out = {}
    for gname, pat in GROUPS.items():
        rds = [rd for rd in sorted(glob.glob(pat)) if os.path.exists(os.path.join(rd, "requests.csv"))]
        if not rds:
            continue
        acc = {"off": [], "budget": [], "duration": []}
        tot = []
        for rd in rds:
            idle = json.load(open(os.path.join(rd, "run_meta.json"))).get("idle_power_w")
            res = {}
            for mode in ("off", "budget", "duration"):
                kw = dict(spread_prefill=False) if mode == "off" else dict(spread_mode=mode)
                bins, info = em.bin_run(rd, phase=None, **kw)
                res[mode] = (bins, info)
                acc[mode].append((bins, idle))
            bo, io = res["off"]
            bb, ib = res["budget"]
            em._export_binned_table(rd, bb, ib.get("power_source"), spread="budget", name="binned_table_spread.csv")
            S = lambda bs, k: sum((b[k] or 0.0) for b in bs)
            sp = ib.get("prefill_spread", {})
            tot.append(dict(run=os.path.relpath(rd, "logs"), bins=len(bo),
                            prefill_off=S(bo, "prefill"), prefill_spread=S(bb, "prefill"),
                            computed_off=S(bo, "prefill_computed"), computed_spread=S(bb, "prefill_computed"),
                            gemm_off=S(bo, "gemm_flops_computed"), gemm_spread=S(bb, "gemm_flops_computed"),
                            prefill_bins_off=sum(1 for b in bo if b["prefill"] > 0),
                            prefill_bins_spread=sum(1 for b in bb if b["prefill"] > 0),
                            moved_tokens=sp.get("tokens_moved_off_report_row"), attributed=sp.get("requests_attributed"),
                            requests=sp.get("requests_total"), overflow=sp.get("computed_overflow")))
        fits = {}
        for mode, lst in acc.items():
            bins = [b for bs, _ in lst for b in bs]
            # idle differs per run: subtract per run before pooling
            pooled = []
            for bs, idle in lst:
                for b in bs:
                    pooled.append(dict(b, E=b["E"] - idle * b["dt"]))
            fits[mode] = fit_bins(pooled, 0.0)
        out[gname] = dict(runs=tot, fits=fits)
        print("\n== %s (%d runs) ==" % (gname, len(rds)))
        for t in tot:
            print("   %-48s prefill tok %10.0f -> %10.0f  computed %10.0f -> %10.0f  GEMM %.4e -> %.4e  "
                  "prefill bins %4d -> %4d  moved %9.0f tok (%s/%s req, overflow %.0f)" % (
                      t["run"], t["prefill_off"], t["prefill_spread"], t["computed_off"], t["computed_spread"],
                      t["gemm_off"], t["gemm_spread"], t["prefill_bins_off"], t["prefill_bins_spread"],
                      t["moved_tokens"] or 0, t["attributed"], t["requests"], t["overflow"] or 0))
        for mode, f in fits.items():
            print("   %-8s n=%5d (pf bins %4d)  free: e_w %.3e e_kv %.3e e_g %6.3f pJ R2 %.3f | W,K fixed: e_g %6.3f pJ "
                  "(R2 %.3f), prefill bins only %s pJ | mean|res| @steady: pf bins %s J, all %.3f J" % (
                      mode, f["n"], f["n_prefill_bins"], f["e_w"], f["e_kv"], f["e_g_pJ"], f["r2_free"],
                      f["e_g_pJ_fixedWK"], f["r2_fixedWK"],
                      "%6.3f" % f["e_g_pJ_fixedWK_prefill_bins"] if f["e_g_pJ_fixedWK_prefill_bins"] is not None else "  -  ",
                      "%.3f" % f["mean_abs_resid_prefill_bins_at_steady_J"] if f["mean_abs_resid_prefill_bins_at_steady_J"] is not None else "-",
                      f["mean_abs_resid_all_bins_at_steady_J"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", default="/tmp/energy_model_orig.py",
                    help="frozen copy of energy_model.py before the spreading change")
    a = ap.parse_args()
    res = {}
    if os.path.exists(a.orig):
        print("== (a) bit-identity vs the pre-change energy_model.py ==")
        res["identity"] = check_identity(load_orig(a.orig))
    res["compare"] = compare()
    json.dump(res, open(os.path.join(HERE, "prefill_spread_compare.json"), "w"), indent=1, default=float)
    print("\nwrote prefill_spread_compare.json")


if __name__ == "__main__":
    main()
