#!/usr/bin/env python3
"""
kv_longctx.py -- does the H200 long-context sweep (logs/longctx, Qwen2-7B, July 2026,
synthetic fixed-length prompts 4k/16k/30k x c=1/8/16) discriminate the KV coefficient?

KV bytes per step vary ~7x at fixed concurrency while GEMM FLOPs stay fixed, which breaks
the KV/FLOP collinearity of the ragged 7B data (FINDINGS_B200.md "UPDATE 5 review").

Method (run level, 55 s windows -> NVML smoothing irrelevant):
  measured: window avg power, time-weighted running batch n, mean iteration time t
            (iter_log.csv), idle from run_meta.json
  predicted power = P_idle + (e_w*W + e_kv*K + e_g*Fg)/t,
     W = canonical weight bytes, K = n*(P+G/2)*kv_bytes_per_token (each sequence's
     attention reads its own context), Fg = 2*N_act*n (decode GEMMs only).
CAVEATS (stated in LIMINAL_E.md):
  * synthetic prompts are identical -> prefix cache: prefill is ~0 after the first request,
    and kv_tokens_resident shows ONE shared prompt.  Concurrent sequences read the same
    physical KV blocks, so some reads may hit L2 -> DRAM KV bytes <= K above, which would
    bias the fitted e_kv LOW.
  * older harness (GetPowerUsage window average, July 2026).
Writes liminal_energy/kv_longctx.json.
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(1, PARENT)
os.chdir(PARENT)
import liminal_e as le  # noqa: E402

ROOT = "logs/longctx/Qwen2-7B-Instruct"


def load(d):
    m = json.load(open(os.path.join(d, "run_meta.json")))
    r = json.load(open(os.path.join(d, "results.json")))
    t0, t1 = m.get("window_wall_t0"), m.get("window_wall_t1")
    busy = nr = nit = 0.0
    for x in csv.DictReader(open(os.path.join(d, "iter_log.csv"))):
        a, b = float(x["t_start"]), float(x["t_end"])
        if t0 and (b < t0 or a > t1):
            continue
        busy += b - a
        nit += 1
        nr += float(x["n_running"] or 0) * (b - a)
    return dict(run=os.path.basename(d), P=float(m["input_len"]), G=float(m["output_len"]),
                conc=int(m["concurrency_or_rate"]), idle=float(m["idle_power_w"]),
                p_meas=float(r["avg_power_window_w"]), tok_s=float(r["aggregate_tokens_per_sec"]),
                n=nr / busy, t=busy / nit)


def main():
    runs = []
    for name in sorted(os.listdir(ROOT)):
        d = os.path.join(ROOT, name)
        if os.path.exists(os.path.join(d, "run_meta.json")):
            runs.append(load(d))
    g = le.gpu("H200")
    m = le.model("Qwen/Qwen2-7B-Instruct")
    kvpt = m.kv_bytes_per_token(le.DTYPE_B)
    W = le.rg.weight_params(m, 1) * le.DTYPE_B
    for r in runs:
        r["K"] = r["n"] * (r["P"] + r["G"] / 2.0) * kvpt
        r["Fg"] = 2.0 * m.active_params() * r["n"]
        r["W"] = W
    profiles = {
        "canonical (e_kv 3.29e-10, e_g 0.79 pJ)": (g["e_wbyte"], g["e_kvbyte"], g["e_gemm"]),
        "runlevel (e_kv 2.42e-10, e_g 1.10 pJ)": tuple(le.PROFILES["H200"]["runlevel"][k]
                                                      for k in ("e_wbyte", "e_kvbyte", "e_gemm")),
        "kv_as_weight (e_kv 1.07e-10)": (g["e_wbyte"], g["e_wbyte"], g["e_gemm"]),
    }
    out = dict(runs=runs, profiles={})
    for name, (ew, ekv, eg) in profiles.items():
        errs = []
        for r in runs:
            p = r["idle"] + (ew * r["W"] + ekv * r["K"] + eg * r["Fg"]) / r["t"]
            r["p_" + name.split()[0]] = p
            errs.append(100 * (p / r["p_meas"] - 1))
        out["profiles"][name] = dict(mape=sum(abs(e) for e in errs) / len(errs),
                                     bias=sum(errs) / len(errs), errs=errs)
    # free e_kv (e_w, e_g held at each profile's values), LS through origin on residual power
    for name, (ew, _, eg) in list(profiles.items())[:2]:
        num = den = 0.0
        for r in runs:
            y = r["p_meas"] - r["idle"] - (ew * r["W"] + eg * r["Fg"]) / r["t"]
            x = r["K"] / r["t"]
            num += x * y
            den += x * x
        ekv = num / den
        errs = [100 * ((r["idle"] + (ew * r["W"] + ekv * r["K"] + eg * r["Fg"]) / r["t"]) / r["p_meas"] - 1)
                for r in runs]
        out["profiles"]["fit e_kv | " + name.split()[0]] = dict(
            e_kv=ekv, mape=sum(abs(e) for e in errs) / len(errs), bias=sum(errs) / len(errs), errs=errs)
    # c=16 slope: dP/d(KV rate) across contexts at fixed concurrency (model-free-ish)
    c16 = sorted([r for r in runs if r["conc"] == 16], key=lambda r: r["P"])
    if len(c16) >= 2:
        a, b = c16[0], c16[-1]
        dyn = lambda r: r["p_meas"] - r["idle"] - (g["e_wbyte"] * r["W"] + g["e_gemm"] * r["Fg"]) / r["t"]
        out["c16_two_point_e_kv"] = (dyn(b) - dyn(a)) / (b["K"] / b["t"] - a["K"] / a["t"])
        # reuse check: incremental KV bytes / incremental step time.  Above peak BW means
        # the counted KV reads cannot all come from DRAM (shared prefix blocks hit L2).
        out["c16_incremental_kv_bw"] = (b["K"] - a["K"]) / (b["t"] - a["t"])
    json.dump(out, open(os.path.join(HERE, "kv_longctx.json"), "w"), indent=1)
    print("%-14s %5s %5s %6s %6s %8s | measured W | predicted W per profile" % ("run", "n", "t ms", "K GB", "KV TB/s", "W TB/s"))
    for r in runs:
        print("%-14s %5.1f %5.2f %6.2f %6.2f %8.2f | %5.0f | %s" % (
            r["run"], r["n"], r["t"] * 1e3, r["K"] / 1e9, r["K"] / r["t"] / 1e12, r["W"] / r["t"] / 1e12,
            r["p_meas"], "  ".join("%5.0f" % r["p_" + k.split()[0]] for k in profiles)))
    for k, v in out["profiles"].items():
        print("  %-45s MAPE %5.1f%%  bias %+5.1f%%  %s" % (k, v["mape"], v["bias"],
                                                          ("e_kv %.3g" % v["e_kv"]) if "e_kv" in v else ""))
    print("  c=16 incremental KV bytes / incremental step time: %.2f TB/s (H200 peak 4.8)" % (
        out["c16_incremental_kv_bw"] / 1e12))
    print("  c=16 two-point e_kv (4k vs 30k, e_w/e_g canonical): %.3g J/B" % out.get("c16_two_point_e_kv", float("nan")))


if __name__ == "__main__":
    main()
