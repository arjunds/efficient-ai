#!/usr/bin/env python3
"""
steady_calib_analysis.py — turn a steady-state microbenchmark run into calibrated
energy coefficients, independent of per-bin NVML smoothing (EnergAIzer-style
protocol: every point is a >= steady_s back-to-back loop).

Answers the PI's notes directly:
  * "blast HBM with constant reads"  -> J/byte at (near-)saturated bandwidth,
    J/byte vs bandwidth fraction (grid sweep), max sustained BW fraction
  * DRAM vs on-chip energy           -> DRAM-resident vs L2-resident streaming
  * compute coefficient free of smear -> 2-feature fit on LLM-shaped GEMM loops
  * overhead power                   -> J per kernel launch
  * 200 ms / smoothing question      -> boxcar-window + lag fit on square waves
  * balance                          -> energy ridge e_byte/e_flop vs perf ridge
and then refits the SERVING model with e_gemm held at the steady-state value.

Usage (inside the container):
  python3 microbench/steady_calib_analysis.py --results microbench/results_B200.json \
      --gpu B200 --serving_root logs/B200 --out microbench/steady_calib_B200.json
Missing sections are skipped, so it also runs on older result files.
"""
import argparse, csv, glob, json, math, os
import numpy as np


def m(d, k):
    return d.get(k + "_mean") if isinstance(d, dict) else None


def peak_bw(gpu_info, coef):
    if coef.get("bw"):
        return float(coef["bw"])
    w, mhz = gpu_info.get("mem_bus_width_bits"), gpu_info.get("max_mem_mhz")
    return (w / 8) * mhz * 1e6 * 2 if w and mhz else None


def ols(X, y, origin=True):
    X = np.asarray(X, float); y = np.asarray(y, float)
    if not origin:
        X = np.c_[np.ones(len(y)), X]
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    yh = X @ c
    r2 = 1 - ((y - yh) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-30)
    return c, float(r2)


def boot(X, y, n=500, seed=0):
    rng = np.random.default_rng(seed); X = np.asarray(X, float); y = np.asarray(y, float)
    cs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        c, *_ = np.linalg.lstsq(X[i], y[i], rcond=None); cs.append(c)
    cs = np.array(cs)
    return np.percentile(cs, 2.5, 0), np.percentile(cs, 97.5, 0)


def smear_fit(trace_csv):
    """Fit p(t) ~ a + b * (boxcar_w * on)(t - lag) for each power column."""
    rows = list(csv.DictReader(open(trace_csv)))
    t = np.array([float(r["t"]) for r in rows]); on = np.array([float(r["on"]) for r in rows])
    dt = float(np.median(np.diff(t)))
    out = {}
    for col in ("p_inst_w", "p_avg_w"):
        try:
            p = np.array([float(r[col]) for r in rows])
        except (KeyError, ValueError):
            continue
        best = None
        for w in np.arange(0.0, 2.01, 0.05):
            k = max(1, int(round(w / dt)))
            sm = np.convolve(on, np.ones(k) / k, mode="full")[:len(on)]  # causal boxcar
            for lag in np.arange(-0.3, 1.01, 0.02):
                s = int(round(lag / dt))
                x = np.roll(sm, s)
                if s > 0: x[:s] = x[s]
                elif s < 0: x[s:] = x[s - 1]
                (a, b), r2 = ols(x[:, None], p, origin=False)
                if best is None or r2 > best[2]:
                    best = (float(w), float(lag), r2, float(b))
        out[col] = dict(window_s=best[0], lag_s=best[1], r2=best[2], amplitude_w=best[3])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--gpu", required=True, help="key in gpu_coefficients.json (e.g. B200)")
    ap.add_argument("--coef", default="gpu_coefficients.json")
    ap.add_argument("--serving_root", default=None, help="e.g. logs/B200 (binned tables)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    R = json.load(open(a.results))
    allc = json.load(open(a.coef)) if os.path.exists(a.coef) else {}
    C = allc.get(a.gpu) or allc.get("_excluded", {}).get(a.gpu, {}) or {}
    G = R.get("gpu", {})
    l2 = G.get("l2_bytes") or 0
    BWp = peak_bw(G, C)
    out = {"gpu": a.gpu, "gpu_info": G, "peak_bw": BWp, "source": a.results}
    idle = R.get("idle_pre", {}); p0 = R.get("idle_p0_pre", {})
    out["idle_deep_w"] = idle.get("p_w"); out["idle_clocks_up_w"] = p0.get("p_w")
    print(f"== {a.gpu}: idle(deep) {idle.get('p_w')} W, idle(clocks-up) {p0.get('p_w')} W, "
          f"peak BW {BWp/1e12 if BWp else float('nan'):.2f} TB/s, enforced limit "
          f"{G.get('power_limit_enforced_w')} W")

    # ---- 1. DRAM streaming at (near-)saturation + L2 split ---------------------
    dram, l2pts = [], []
    for s in R.get("stream", []):
        ws = s.get("ws_bytes", 0)
        row = dict(label=s["label"], ws=ws, bw=m(s, "bytes_per_s"), jb=m(s, "j_per_byte_dyn"),
                   jb_p0=m(s, "j_per_byte_dyn_vs_p0"), sm=m(s, "sm_mhz"),
                   cap=m(s, "frac_sw_power_cap"))
        if l2 and ws >= 8 * l2:
            dram.append(row)
        elif l2 and ws <= l2 // 2 and ".ca" not in s.get("cache", "") and "L1" not in s["label"]:
            l2pts.append(row)
    for v in R.get("vendor", []):
        dram.append(dict(label=v["label"], ws=None, bw=m(v, "bytes_per_s"), jb=m(v, "j_per_byte_dyn"),
                         jb_p0=m(v, "j_per_byte_dyn_vs_p0"), sm=m(v, "sm_mhz"),
                         cap=m(v, "frac_sw_power_cap")))
    if dram:
        jb = [d["jb"] for d in dram if d["jb"]]
        bwmax = max(d["bw"] for d in dram if d["bw"])
        out["dram_stream"] = dict(points=dram, j_per_byte_median=float(np.median(jb)),
                                  max_bw=bwmax, max_bw_frac=(bwmax / BWp if BWp else None))
        print(f"-- DRAM streaming ('blast HBM'): J/B median {np.median(jb):.3e}; max sustained BW "
              f"{bwmax/1e12:.2f} TB/s = {bwmax/BWp if BWp else float('nan'):.0%} of peak")
        for d in dram:
            print(f"   {d['label']:<24} BW {d['bw']/1e12:6.2f} TB/s ({d['bw']/BWp if BWp else 0:4.0%})  "
                  f"J/B {d['jb']:.3e}  sm {d['sm'] or 0:.0f} MHz  capfrac {d['cap'] or 0:.2f}")
    if l2pts and dram:
        jl2 = float(np.median([p["jb"] for p in l2pts if p["jb"]]))
        out["l2_stream"] = dict(points=l2pts, j_per_byte_median=jl2,
                                onchip_share=jl2 / out["dram_stream"]["j_per_byte_median"])
        print(f"-- L2-resident stream J/B {jl2:.3e} -> on-chip share of a DRAM-streamed byte "
              f"~{out['l2_stream']['onchip_share']:.0%}")

    # ---- 2. grid: J/B vs bandwidth fraction (is energy/byte flat up to saturation?) --
    gp = (R.get("grid") or {}).get("points") or []
    if len(gp) >= 3:
        bw = np.array([m(p, "bytes_per_s") for p in gp]); pd = np.array([m(p, "p_dyn_w") for p in gp])
        (a0, e), r2 = ols(bw[:, None], pd, origin=False)
        out["grid"] = dict(marginal_j_per_byte=float(e), intercept_w=float(a0), r2=r2,
                           points=[dict(nprog=p.get("nprog"), bw=m(p, "bytes_per_s"),
                                        bw_frac=(m(p, "bytes_per_s") / BWp if BWp else None),
                                        jb=m(p, "j_per_byte_dyn")) for p in gp])
        print(f"-- grid (occupancy sweep): P_dyn = {a0:.1f} W + {e:.3e} J/B * BW  (R2 {r2:.3f}); "
              f"BW frac {min(bw)/BWp if BWp else 0:.0%}..{max(bw)/BWp if BWp else 0:.0%}")

    # ---- 3. LLM-shaped GEMMs: steady-state 2-feature fit ------------------------
    lg = R.get("llmgemm") or []
    if len(lg) >= 4:
        X = np.array([[m(p, "bytes_per_s"), m(p, "flops_per_s")] for p in lg])
        y = np.array([m(p, "p_dyn_w") for p in lg])
        c, r2 = ols(X, y); lo, hi = boot(X, y)
        out["llmgemm_fit"] = dict(e_byte=float(c[0]), e_flop=float(c[1]),
                                  e_byte_ci=[float(lo[0]), float(hi[0])],
                                  e_flop_ci=[float(lo[1]), float(hi[1])], r2=r2, n=len(y))
        capfrac = np.mean([m(p, "frac_sw_power_cap") or 0 for p in lg])
        out["llmgemm_fit"]["frac_points_power_capped"] = float(capfrac)
        if capfrac > 0.5:
            print(f"   !! {capfrac:.0%} of points ran at the power cap -> constant-power regime; "
                  "this fit is NOT a valid calibration")
        print(f"-- LLM-shaped GEMM loops (steady, n={len(y)}): e_byte {c[0]:.3e} [{lo[0]:.3e},{hi[0]:.3e}] J/B, "
              f"e_flop {c[1]*1e12:.3f} [{lo[1]*1e12:.3f},{hi[1]*1e12:.3f}] pJ/flop, R2 {r2:.3f}")
        for p in lg:
            if p.get("M") in (1, 8192):
                print(f"   {p['label']:<22} AI {p.get('arith_intensity',0):7.1f}  BW {m(p,'bytes_per_s')/1e12:5.2f} TB/s  "
                      f"TF {m(p,'flops_per_s')/1e12:7.1f}  P_dyn {m(p,'p_dyn_w'):6.1f} W")

    # ---- 4. launch overhead power ----------------------------------------------
    la = R.get("launch") or []
    if la:
        out["launch"] = [dict(label=p["label"], launches_per_s=m(p, "bytes_per_s"),
                              p_dyn_w=m(p, "p_dyn_w"), j_per_launch=m(p, "j_per_byte_dyn")) for p in la]
        for p in out["launch"]:
            print(f"-- launch-only {p['label']}: {p['launches_per_s']/1e3:.0f} k launches/s, "
                  f"P_dyn {p['p_dyn_w']:.1f} W, {p['j_per_launch']*1e6:.2f} uJ/launch")

    # ---- 5. smoothing: window vs lag -------------------------------------------
    for key, suffix in (("square_multi", "_square_multi_trace.csv"), ("square", "_square_trace.csv")):
        tr = a.results.replace(".json", suffix)
        if os.path.exists(tr):
            out[key + "_fit"] = smear_fit(tr)
            for col, f in out[key + "_fit"].items():
                print(f"-- {key} {col}: boxcar window {f['window_s']:.2f} s, lag {f['lag_s']:+.2f} s, R2 {f['r2']:.3f}")

    # ---- 6. balance: energy ridge vs performance ridge --------------------------
    eb = (out.get("llmgemm_fit") or {}).get("e_byte") or (out.get("dram_stream") or {}).get("j_per_byte_median")
    ef = (out.get("llmgemm_fit") or {}).get("e_flop") or C.get("e_gemm")
    pk = C.get("peak_flops") or G.get("peak_flops")
    if eb and ef:
        out["balance"] = dict(energy_ridge_flop_per_byte=eb / ef,
                              perf_ridge_flop_per_byte=(pk / BWp if pk and BWp else None))
        print(f"-- balance: energy ridge e_byte/e_flop = {eb/ef:.0f} FLOP/B"
              + (f"; performance ridge peak/BW = {pk/BWp:.0f} FLOP/B" if pk and BWp else ""))

    # ---- 7. serving: compare + refit with e_gemm held at steady-state value -----
    if C:
        cmp = {}
        for k in ("e_wbyte", "e_kvbyte", "e_gemm"):
            if C.get(k) is not None:
                cmp[k + "_serving"] = C[k]
        if "dram_stream" in out and C.get("e_wbyte"):
            cmp["stream_over_serving_wbyte"] = out["dram_stream"]["j_per_byte_median"] / C["e_wbyte"]
        if "llmgemm_fit" in out and C.get("e_gemm"):
            cmp["steady_over_serving_egemm"] = out["llmgemm_fit"]["e_flop"] / C["e_gemm"]
        out["vs_serving"] = cmp
        print("-- vs serving fit:", json.dumps({k: (round(v, 4) if isinstance(v, float) and v > 1e-3 else v)
                                                 for k, v in cmp.items()}))
    if a.serving_root and "llmgemm_fit" in out:
        W, K, Gf, Y = [], [], [], []
        for rd in sorted(glob.glob(os.path.join(a.serving_root, "*", "*"))):
            tb, mp = os.path.join(rd, "binned_table.csv"), os.path.join(rd, "run_meta.json")
            if not (os.path.exists(tb) and os.path.exists(mp)):
                continue
            idl = json.load(open(mp)).get("idle_power_w")
            if idl is None:
                continue
            for r in csv.DictReader(open(tb)):
                try:
                    g = r.get("gemm_flops_bin_computed") or r["gemm_flops_bin"]
                    W.append(float(r["weight_bytes_bin"])); K.append(float(r["kv_bytes_bin"]))
                    Gf.append(float(g)); Y.append(float(r["energy_bin_j"]) - idl * float(r["dt_s"]))
                except (KeyError, ValueError):
                    pass
        if len(Y) > 50:
            W, K, Gf, Y = map(np.array, (W, K, Gf, Y))
            efix = out["llmgemm_fit"]["e_flop"]
            c2, r2f = ols(np.c_[W, K], Y - efix * Gf)
            c3, r23 = ols(np.c_[W, K, Gf], Y)
            out["serving_refit_fixed_egemm"] = dict(e_gemm_fixed=efix, e_wbyte=float(c2[0]),
                                                    e_kvbyte=float(c2[1]), r2=r2f, n_bins=int(len(Y)),
                                                    free_3term=dict(e_wbyte=float(c3[0]), e_kvbyte=float(c3[1]),
                                                                    e_gemm=float(c3[2]), r2=r23))
            print(f"-- serving refit (e_gemm fixed {efix*1e12:.3f} pJ): e_wbyte {c2[0]:.3e}, e_kvbyte {c2[1]:.3e} "
                  f"(free 3-term: {c3[0]:.3e} / {c3[1]:.3e} / {c3[2]*1e12:.3f} pJ, R2 {r23:.3f})")

    out_path = a.out or a.results.replace(".json", "_calib.json")
    json.dump(out, open(out_path, "w"), indent=1, default=float)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
