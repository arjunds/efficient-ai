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


CAPPED = 0.05   # frac of samples in SW power cap above which a point is "capped": the GPU
                # throttles SM clock/voltage to hold the limit, so J/B and J/FLOP drop and
                # power stops tracking work. Capped points are NEVER used in a calibration
                # fit (B200 2026-10-07: including them gave e_flop 0.51 vs 0.94 pJ uncapped).


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
        unc = [d["jb"] for d in dram if d["jb"] and (d["cap"] or 0) < CAPPED]
        cap = [d["jb"] for d in dram if d["jb"] and (d["cap"] or 0) >= CAPPED]
        bwmax = max(d["bw"] for d in dram if d["bw"])
        out["dram_stream"] = dict(points=dram,
                                  j_per_byte_median=float(np.median(unc)) if unc else None,
                                  j_per_byte_median_capped=float(np.median(cap)) if cap else None,
                                  n_uncapped=len(unc), n_capped=len(cap),
                                  max_bw=bwmax, max_bw_frac=(bwmax / BWp if BWp else None))
        print(f"-- DRAM streaming ('blast HBM'): J/B median uncapped "
              f"{np.median(unc) if unc else float('nan'):.3e} (n={len(unc)}), CAPPED/throttled "
              f"{np.median(cap) if cap else float('nan'):.3e} (n={len(cap)}); max sustained BW "
              f"{bwmax/1e12:.2f} TB/s = {bwmax/BWp if BWp else float('nan'):.0%} of peak")
        for d in dram:
            print(f"   {d['label']:<24} BW {d['bw']/1e12:6.2f} TB/s ({d['bw']/BWp if BWp else 0:4.0%})  "
                  f"J/B {d['jb']:.3e}  sm {d['sm'] or 0:.0f} MHz  capfrac {d['cap'] or 0:.2f}")
    # ---- 2. grid: J/B vs bandwidth fraction (is energy/byte flat up to saturation?) --
    gp = (R.get("grid") or {}).get("points") or []
    gu = [p for p in gp if (m(p, "frac_sw_power_cap") or 0) < CAPPED]
    if len(gu) >= 3:
        bw = np.array([m(p, "bytes_per_s") for p in gu]); pd = np.array([m(p, "p_dyn_w") for p in gu])
        (a0, e), r2 = ols(bw[:, None], pd, origin=False)
        out["grid"] = dict(marginal_j_per_byte=float(e), intercept_w=float(a0), r2=r2,
                           n_uncapped=len(gu), n_capped=len(gp) - len(gu),
                           points=[dict(nprog=p.get("nprog"), bw=m(p, "bytes_per_s"),
                                        bw_frac=(m(p, "bytes_per_s") / BWp if BWp else None),
                                        jb=m(p, "j_per_byte_dyn"), sm=m(p, "sm_mhz"),
                                        cap=m(p, "frac_sw_power_cap")) for p in gp])
        print(f"-- grid (occupancy sweep, {len(gu)} uncapped of {len(gp)}): P_dyn = {a0:.1f} W + {e:.3e} J/B * BW "
              f"(R2 {r2:.3f}); uncapped BW {min(bw)/BWp if BWp else 0:.0%}..{max(bw)/BWp if BWp else 0:.0%} of peak")
    # full-clock DRAM e_byte: uncapped streams if any, else the uncapped grid slope
    eb_full = ((out.get("dram_stream") or {}).get("j_per_byte_median")
               or (out.get("grid") or {}).get("marginal_j_per_byte"))
    out["e_byte_full_clock"] = eb_full
    if l2pts and eb_full:
        jl2 = float(np.median([p["jb"] for p in l2pts if p["jb"]]))
        out["l2_stream"] = dict(points=l2pts, j_per_byte_median=jl2, onchip_share=jl2 / eb_full)
        print(f"-- L2-resident stream J/B {jl2:.3e} -> on-chip share of a full-clock DRAM-streamed byte "
              f"~{out['l2_stream']['onchip_share']:.0%} (per L2 crossing; DRAM streams cross L2 ~1.5x)")

    # ---- 3. LLM-shaped GEMMs: steady-state 2-feature fit ------------------------
    lg_all = R.get("llmgemm") or []
    lg = [p for p in lg_all if (m(p, "frac_sw_power_cap") or 0) < CAPPED]
    lgc = [p for p in lg_all if (m(p, "frac_sw_power_cap") or 0) >= CAPPED]
    if lgc:
        jf = [m(p, "p_dyn_w") / m(p, "flops_per_s") for p in lgc if (p.get("M") or 0) >= 2048]
        out["llmgemm_capped"] = dict(n=len(lgc), labels=[p["label"] for p in lgc],
                                     tflops_range=[min(m(p, "flops_per_s") for p in lgc) / 1e12,
                                                   max(m(p, "flops_per_s") for p in lgc) / 1e12],
                                     sm_mhz_range=[min(m(p, "sm_mhz") for p in lgc), max(m(p, "sm_mhz") for p in lgc)],
                                     total_j_per_flop_M2048plus=[min(jf), max(jf)] if jf else None)
        print(f"   ({len(lgc)}/{len(lg_all)} GEMM points ran at the power cap, SM "
              f"{out['llmgemm_capped']['sm_mhz_range'][0]:.0f}-{out['llmgemm_capped']['sm_mhz_range'][1]:.0f} MHz: "
              "EXCLUDED from the fit"
              + (f"; at M>=2048 they deliver P_dyn/FLOP {min(jf)*1e12:.2f}-{max(jf)*1e12:.2f} pJ (throttled V/f)" if jf else "") + ")")
    if len(lg) >= 4:
        X = np.array([[m(p, "bytes_per_s"), m(p, "flops_per_s")] for p in lg])
        y = np.array([m(p, "p_dyn_w") for p in lg])
        c, r2 = ols(X, y); lo, hi = boot(X, y)
        out["llmgemm_fit"] = dict(e_byte=float(c[0]), e_flop=float(c[1]),
                                  e_byte_ci=[float(lo[0]), float(hi[0])],
                                  e_flop_ci=[float(lo[1]), float(hi[1])], r2=r2, n=len(y),
                                  n_capped_excluded=len(lgc),
                                  max_tflops=float(X[:, 1].max() / 1e12))
        if eb_full:   # e_byte pinned at the full-clock streaming value
            yf = y - eb_full * X[:, 0]
            out["llmgemm_fit"]["e_flop_ebyte_pinned"] = float((X[:, 1] @ yf) / (X[:, 1] @ X[:, 1]))
        print(f"-- LLM-shaped GEMM loops (steady, uncapped n={len(y)}): e_byte {c[0]:.3e} [{lo[0]:.3e},{hi[0]:.3e}] J/B, "
              f"e_flop {c[1]*1e12:.3f} [{lo[1]*1e12:.3f},{hi[1]*1e12:.3f}] pJ/flop, R2 {r2:.3f}"
              + (f"; e_byte pinned -> e_flop {out['llmgemm_fit']['e_flop_ebyte_pinned']*1e12:.3f} pJ" if eb_full else ""))
    elif lg_all:
        print(f"   !! only {len(lg)} uncapped GEMM points -> no valid steady e_flop")
        for p in lg_all:
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
    eb = eb_full or (out.get("llmgemm_fit") or {}).get("e_byte")
    ef = (out.get("llmgemm_fit") or {}).get("e_flop") or C.get("e_gemm")
    pk = C.get("peak_flops") or G.get("peak_flops")
    if eb and ef:
        out["balance"] = dict(energy_ridge_flop_per_byte=eb / ef, e_byte=eb, e_flop=ef,
                              perf_ridge_flop_per_byte=(pk / BWp if pk and BWp else None))
        print(f"-- balance (full-clock, uncapped coefficients): energy ridge e_byte/e_flop = {eb/ef:.0f} FLOP/B"
              + (f"; performance ridge peak/BW = {pk/BWp:.0f} FLOP/B" if pk and BWp else ""))
        # power roofline: what the power limit lets you sustain at full clock
        lim = G.get("power_limit_enforced_w"); idl = idle.get("p_w")
        if lim and idl and pk and BWp:
            a0 = (out.get("grid") or {}).get("intercept_w") or 0.0
            budget = lim - idl
            out["power_roofline"] = dict(budget_w=budget,
                                         p_dyn_at_peak_bw_w=a0 + eb * BWp, p_dyn_at_peak_flops_w=ef * pk,
                                         bw_frac_within_tdp_full_clock=min(1.0, (budget - a0) / (eb * BWp)),
                                         flops_frac_within_tdp_full_clock=min(1.0, budget / (ef * pk)))
            pr = out["power_roofline"]
            print(f"-- power roofline: budget {budget:.0f} W above idle; full-clock P_dyn at peak BW "
                  f"{pr['p_dyn_at_peak_bw_w']:.0f} W, at peak FLOPs {pr['p_dyn_at_peak_flops_w']:.0f} W -> at full clock "
                  f"the limit allows {pr['bw_frac_within_tdp_full_clock']:.0%} of peak BW, "
                  f"{pr['flops_frac_within_tdp_full_clock']:.0%} of peak FLOPs")

    # ---- 7. serving: compare + refit with e_gemm held at steady-state value -----
    if C:
        cmp = {}
        for k in ("e_wbyte", "e_kvbyte", "e_gemm"):
            if C.get(k) is not None:
                cmp[k + "_serving"] = C[k]
        if eb_full and C.get("e_wbyte"):
            cmp["fullclock_stream_over_serving_wbyte"] = eb_full / C["e_wbyte"]
        if (out.get("dram_stream") or {}).get("j_per_byte_median_capped") and C.get("e_wbyte"):
            cmp["capped_stream_over_serving_wbyte"] = out["dram_stream"]["j_per_byte_median_capped"] / C["e_wbyte"]
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
            c2, _ = ols(np.c_[W, K], Y - efix * Gf)
            yh = c2[0] * W + c2[1] * K + efix * Gf          # R2 on Y, comparable to the free fit
            r2f = float(1 - ((Y - yh) ** 2).sum() / ((Y - Y.mean()) ** 2).sum())
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
