#!/usr/bin/env python3
"""
longctx_analysis.py — separate KV-byte energy from compute energy using the
long-context sweep (HANDOFF_B200_LONGCTX.md). CPU-only, numpy + stdlib.

Why this works where concurrency sweeps did not: at fixed concurrency, context length
changes KV bytes per decode step ~30x while weight bytes and GEMM FLOPs per step stay
fixed. (Decode attention FLOPs grow with KV, so e_kvbyte here INCLUDES the attention
compute that comes with reading a KV byte - which is what the energy model needs.)

  A. decode-only bins (no computed prefill): Y = e_w*W + e_kv*K + e_g*G   (free, then
     e_g pinned at the steady 0.943 pJ to show sensitivity) -> e_kv / e_w with CI
  B. prefill-containing bins, e_w and e_kv held at A: e_gemm from prefill
  C. all bins, free 3-term (for comparison with the concurrency-sweep fits)
  D. per run: tok/s, J per output token, KV share of decode bytes, % at power cap
  E. logger A/B: loggerab (slim NVML logger) vs relog tok/s at the same load

  python3 longctx_analysis.py [--root logs/B200_prov] [--arms longctx,longctx32]
"""
import argparse
import bisect
import csv
import glob
import json
import os

import numpy as np

E_G_STEADY = 0.943e-12


def jload(p):
    try:
        return json.load(open(p))
    except Exception:
        return {}


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def prefill_windows(rd):
    """[t_submit, t_first_token] per request. vLLM reports a request's prompt tokens
    only on the step its prefill FINISHES; with chunked prefill (2048-token step
    budget) a long prompt's earlier chunk steps report prefill_tokens = 0 and look
    like decode. On the B200 long-context runs those hidden steps took 34-73 ms (7B)
    and 155-240 ms (32B) vs ~5 ms decode steps, and grew with context - they made
    'decode-only' fits convex and degenerate. A bin overlapping any request's prefill
    window is therefore NOT decode-only."""
    iv = []
    p = os.path.join(rd, "requests.csv")
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            a, f = fnum(r.get("t_submit")), fnum(r.get("t_first_token"))
            if a is not None and f is not None:
                iv.append((a, f))
    iv.sort()
    return iv


def _overlaps(iv, b0, b1, max_len=600.0):
    lo = max(0, bisect.bisect_left(iv, (b0 - max_len, 0.0)) - 1)
    hi = bisect.bisect_right(iv, (b1, float("inf")))
    return any(s <= b1 and b0 <= f for s, f in iv[lo:hi])


def run_bins(rd):
    idle = jload(os.path.join(rd, "run_meta.json")).get("idle_power_w")
    out = []
    if idle is None:
        return out
    iv = prefill_windows(rd)
    for b in csv.DictReader(open(os.path.join(rd, "binned_table.csv"))):
        tm, dt = fnum(b["t_mid"]), fnum(b["dt_s"])
        pc = fnum(b.get("prefill_tokens_computed"))
        pc = fnum(b.get("prefill_tokens")) if pc is None else pc
        g = fnum(b.get("gemm_flops_bin_computed")) or fnum(b["gemm_flops_bin"])
        in_pf = _overlaps(iv, tm - dt / 2, tm + dt / 2) if iv else False
        out.append(dict(W=fnum(b["weight_bytes_bin"]), K=fnum(b["kv_bytes_bin"]), G=g, in_prefill_window=in_pf,
                        Y=fnum(b["energy_bin_j"]) - idle * fnum(b["dt_s"]),
                        prefill=pc or 0.0, decode=fnum(b["decode_tokens"]) or 0.0,
                        src=b.get("power_source")))
    return out


def ols(X, y, n_boot=500, seed=0):
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - ((y - X @ c) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    rng = np.random.default_rng(seed); B = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y)); b, *_ = np.linalg.lstsq(X[i], y[i], rcond=None); B.append(b)
    B = np.array(B)
    return c, np.percentile(B, 2.5, 0), np.percentile(B, 97.5, 0), float(r2), B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="logs/B200_prov")
    ap.add_argument("--arms", default="longctx,longctx32")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    R = {}
    for arm in a.arms.split(","):
        for mdir in sorted(glob.glob(os.path.join(a.root, arm, "*"))):
            if not os.path.isdir(mdir):
                continue
            model = os.path.basename(mdir)
            rds = [rd for rd in sorted(glob.glob(os.path.join(mdir, "*")))
                   if os.path.basename(rd) != "idle"
                   and os.path.exists(os.path.join(rd, "binned_table.csv"))]
            if not rds:
                continue
            print(f"\n===== {arm} / {model}: {len(rds)} runs =====")
            # D. per run
            print(f"{'run':<16}{'tok/s':>8}{'mJ/out-tok':>11}{'W':>6}{'KV share of decode bytes':>26}{'cap%':>6}{'src':>8}")
            allb = []
            for rd in rds:
                bins = run_bins(rd); allb += bins
                res = jload(os.path.join(rd, "results.json"))
                win = res.get("window_duration_s") or 1
                E = res.get("energy_window_j"); gen = res.get("generated_tokens_window") or 0
                dec = [b for b in bins if b["prefill"] == 0 and b["decode"] > 0 and not b["in_prefill_window"]]
                kvs = (sum(b["K"] for b in dec) / sum(b["K"] + b["W"] for b in dec)) if dec else None
                tr = list(csv.DictReader(open(os.path.join(rd, "power_trace.csv"))))
                cr = [int(r["clk_reasons"]) for r in tr if (r.get("clk_reasons") or "").strip()]
                cap = (sum(1 for x in cr if x & 0x84) / len(cr)) if cr else None
                print(f"{os.path.basename(rd):<16}{res.get('aggregate_tokens_per_sec') or 0:>8.0f}"
                      f"{(E / gen * 1e3) if (E and gen) else float('nan'):>11.0f}{(E / win) if E else float('nan'):>6.0f}"
                      f"{(kvs or 0) * 100:>25.0f}%{(cap or 0) * 100:>6.0f}{(bins[0]['src'] if bins else '-'):>8}")
            rec = {}
            # A0. slope method (headline): at fixed concurrency, truly-decode energy per
            # output token vs KV bytes per output token is a straight line; slope = e_kv,
            # intercept = weight + GEMM energy per token. Needs no e_gemm assumption.
            by_c = {}
            for rd in rds:
                tag = os.path.basename(rd)
                if "_c" not in tag:
                    continue
                d = [b for b in run_bins(rd) if b["prefill"] == 0 and b["decode"] > 0
                     and not b["in_prefill_window"]]
                if len(d) >= 5:
                    tok = sum(b["decode"] for b in d)
                    by_c.setdefault(tag.split("_c")[-1], []).append(
                        (tag, sum(b["Y"] for b in d) / tok, sum(b["K"] for b in d) / tok,
                         sum(b["W"] for b in d) / tok, sum(b["G"] for b in d) / tok))
            rec["slope"] = {}
            for c, rows in sorted(by_c.items(), key=lambda kv: int(kv[0])):
                if len(rows) < 3:
                    continue
                yv = np.array([r[1] for r in rows]); kv_ = np.array([r[2] for r in rows])
                A = np.c_[np.ones_like(kv_), kv_]
                (a0, s), *_ = np.linalg.lstsq(A, yv, rcond=None)
                r2s = 1 - ((yv - A @ np.array([a0, s])) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
                wbar, gbar = np.mean([r[3] for r in rows]), np.mean([r[4] for r in rows])
                print(f"A0. slope, c={c} ({len(rows)} contexts): e_kv = {s:.3e} J/B "
                      f"(x{s / 1.25e-10:.2f} a 1.25e-10 weight byte)  intercept {a0:.4f} J/tok  R2 {r2s:.4f}  "
                      f"-> intercept implies e_gemm {(a0 - 1.25e-10 * wbar) / gbar * 1e12:.2f} pJ if e_w = 1.25e-10")
                rec["slope"][c] = dict(e_kvbyte=s, intercept_j_per_tok=a0, r2=r2s, n=len(rows))
            # A. decode-only bins
            dec = [b for b in allb if b["prefill"] == 0 and b["decode"] > 0 and not b["in_prefill_window"]]
            if len(dec) >= 30:
                X = np.array([[b["W"], b["K"], b["G"]] for b in dec]); y = np.array([b["Y"] for b in dec])
                corr = np.corrcoef(X[:, 1], X[:, 2])[0, 1]
                c, lo, hi, r2, B = ols(X, y)
                ratio = B[:, 1] / B[:, 0]
                print(f"A. decode-only bins (n={len(dec)}, corr(KV,FLOP)={corr:.2f}): e_w {c[0]:.3e} [{lo[0]:.3e},{hi[0]:.3e}]  "
                      f"e_kv {c[1]:.3e} [{lo[1]:.3e},{hi[1]:.3e}]  e_g {c[2]*1e12:.3f} pJ [{lo[2]*1e12:.3f},{hi[2]*1e12:.3f}]  R2 {r2:.3f}")
                print(f"   e_kv / e_w = {c[1]/c[0]:.2f}  [{np.percentile(ratio, 2.5):.2f}, {np.percentile(ratio, 97.5):.2f}]")
                yp = y - E_G_STEADY * X[:, 2]
                cp, lop, hip, _, _ = ols(X[:, :2], yp)
                print(f"   e_g pinned 0.943 pJ: e_w {cp[0]:.3e}  e_kv {cp[1]:.3e} [{lop[1]:.3e},{hip[1]:.3e}]  e_kv/e_w {cp[1]/cp[0]:.2f}")
                rec["decode_only"] = dict(n=len(dec), corr_kv_flop=float(corr), e_wbyte=c[0], e_kvbyte=c[1], e_gemm=c[2],
                                          ci_lo=lo.tolist(), ci_hi=hi.tolist(), r2=r2,
                                          kv_over_w=float(c[1] / c[0]),
                                          kv_over_w_ci=[float(np.percentile(ratio, 2.5)), float(np.percentile(ratio, 97.5))],
                                          pinned_e_g=dict(e_wbyte=cp[0], e_kvbyte=cp[1]))
                # B. prefill bins with memory terms fixed
                pre = [b for b in allb if b["prefill"] > 0]
                if len(pre) >= 10:
                    yr = np.array([b["Y"] - c[0] * b["W"] - c[1] * b["K"] for b in pre])
                    G = np.array([b["G"] for b in pre])
                    eg = float(G @ yr / (G @ G))
                    print(f"B. prefill bins (n={len(pre)}), e_w/e_kv fixed from A: e_gemm {eg*1e12:.3f} pJ "
                          f"(steady uncapped 0.943; capped GEMM loops 0.57-0.67)")
                    rec["prefill_e_gemm"] = eg
            else:
                print(f"A. only {len(dec)} decode-only bins; not enough")
            # C. all bins
            if len(allb) >= 30:
                X = np.array([[b["W"], b["K"], b["G"]] for b in allb]); y = np.array([b["Y"] for b in allb])
                c, lo, hi, r2, _ = ols(X, y, n_boot=200)
                print(f"C. all bins (n={len(allb)}): e_w {c[0]:.3e}  e_kv {c[1]:.3e}  e_g {c[2]*1e12:.3f} pJ  R2 {r2:.3f}")
                rec["all_bins"] = dict(e_wbyte=c[0], e_kvbyte=c[1], e_gemm=c[2], r2=r2)
            R[f"{arm}/{model}"] = rec

    # E. logger A/B
    ab = {}
    for tag in ("alpaca_c1", "alpaca_c64"):
        r0 = jload(os.path.join(a.root, "relog", "Qwen2-7B-Instruct", tag, "results.json"))
        r1 = jload(os.path.join(a.root, "loggerab", "Qwen2-7B-Instruct", tag, "results.json"))
        if r0.get("aggregate_tokens_per_sec") and r1.get("aggregate_tokens_per_sec"):
            ab[tag] = dict(relog=r0["aggregate_tokens_per_sec"], slim=r1["aggregate_tokens_per_sec"])
    if ab:
        print("\nE. logger A/B (eager, Qwen2-7B): tok/s full logger -> slim logger "
              "(September runs, old logger: c1 182, c64 ~8900 tok/s)")
        for k, v in ab.items():
            print(f"   {k}: {v['relog']:.0f} -> {v['slim']:.0f}  (x{v['slim']/v['relog']:.2f})")
        R["logger_ab"] = ab
    path = a.out or os.path.join(a.root, "longctx_summary.json")
    json.dump(R, open(path, "w"), indent=1, default=float)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
