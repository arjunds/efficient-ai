#!/usr/bin/env python3
"""
a5000_analysis.py -- the numbers quoted in FINDINGS_A5000.md, in one place.

  python3 microbench/a5000_analysis.py      (in the container; writes microbench/a5000_summary.json)

1. Direct c=1 serving J/byte (reconcile_gpus.py definition: sum(E - idle*dt) /
   sum(weight+KV bytes) over active bins) for every A5000 c=1 run.
2. Like-for-like on ONE GPU (logs/A5000_samegpu/gpu_*): serving c=1 vs steady
   GEMV / DRAM-stream kernels in the SAME power-capped DVFS state (SM 210 MHz),
   as a function of the unknown floor F at that state:  J/B(F) = (P - F) / BW.
3. Full-clock (duty-cycle, short-gap) kernel J/byte and J/flop.
"""
import csv, glob, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from energy_model import model_byte_constants  # noqa: E402

H200, B200 = 1.076e-10, 1.252e-10               # reconciled direct c=1 Qwen2-7B (lead, 2026-09-24)
LAW = 1.076e-10 * 4.8e12 / 0.768e12             # 1/BW prediction for A5000


def run_direct(rd):
    meta = json.load(open(os.path.join(rd, "run_meta.json")))
    res = json.load(open(os.path.join(rd, "results.json")))
    idle = meta.get("idle_power_w")
    W = K = E = T = 0.0
    for r in csv.DictReader(open(os.path.join(rd, "binned_table.csv"))):
        W += float(r["weight_bytes_bin"]); K += float(r["kv_bytes_bin"])
        E += float(r["energy_bin_j"]); T += float(r["dt_s"])
    P = [r for r in csv.DictReader(open(os.path.join(rd, "power_trace.csv")))
         if meta["window_wall_t0"] <= float(r["t_wall"]) <= meta["window_wall_t1"]]
    sm = np.mean([float(r["sm_mhz"]) for r in P if r["sm_mhz"]])
    p_inst = np.mean([float(r["power_inst_w"]) for r in P if r.get("power_inst_w")]) if P and "power_inst_w" in P[0] else None
    B = W + K
    return {"run_dir": rd, "idle_w": idle, "tok_s": res.get("aggregate_tokens_per_sec"),
            "p_window_w": res.get("avg_power_window_w"), "p_inst_w": p_inst,
            "p_active_bins_w": E / T, "bw_active": B / T, "mbu": B / T / 768e9,
            "sm_mhz": float(sm), "jbyte_direct": (E - idle * T) / B, "jbyte_gross": E / B,
            "kv_share_of_bytes": K / B}


def main():
    out = {"refs": {"H200_direct": H200, "B200_direct": B200, "law_pred_A5000": LAW}}
    # 1. direct c=1
    rows = []
    for rd in sorted(glob.glob("logs/A5000/*/*_c1")) + sorted(glob.glob("logs/A5000_samegpu/*/*/alpaca_c1")):
        if os.path.exists(os.path.join(rd, "binned_table.csv")):
            rows.append(run_direct(rd))
    out["direct_c1"] = rows
    print(f"{'run':<58}{'idle':>6}{'P':>7}{'sm':>6}{'GB/s':>7}{'MBU':>6}{'J/B direct':>12}{'x H200':>8}{'J/B gross':>11}")
    for r in rows:
        print(f"{r['run_dir']:<58}{r['idle_w']:>6.1f}{r['p_active_bins_w']:>7.1f}{r['sm_mhz']:>6.0f}"
              f"{r['bw_active']/1e9:>7.1f}{r['mbu']:>6.2f}{r['jbyte_direct']:>12.3e}{r['jbyte_direct']/H200:>8.2f}"
              f"{r['jbyte_gross']:>11.3e}")

    # 2. same-GPU like-for-like vs floor
    for g in sorted(glob.glob("logs/A5000_samegpu/gpu_*")):
        mb = json.load(open(os.path.join(g, "microbench.json")))
        kern = {}
        for sec in ("stream", "vendor", "gemv"):
            for r in mb.get(sec, []):
                kern[r["label"]] = (r["p_w_mean"], r["bytes_per_s_mean"], r.get("sm_mhz_mean"))
        serv = {}
        for rd in sorted(glob.glob(os.path.join(g, "*", "alpaca_c1"))):
            d = run_direct(rd)
            serv[os.path.basename(os.path.dirname(rd))] = (d["p_active_bins_w"], d["bw_active"], d["sm_mhz"], d["idle_w"])
        floors = [21.0, 40.0, 50.0] + sorted({round(v[3], 1) for v in serv.values()})
        sel = {k: v for k, v in kern.items() if k in ("stream_cg_256.00MB", "stream_cg_1024.00MB",
                                                        "torch_sum_1GB", "torch_copy_512MB", "gemv_b1_512MB", "gemv_b16_512MB")}
        items = [(f"serve {k}", v[:3]) for k, v in serv.items()] + list(sel.items())
        tab = []
        print(f"\nSAME GPU {g}: J/byte = (P - F)/BW in the capped state, F = unknown floor at that state")
        print(f"{'workload':<32}{'P W':>7}{'GB/s':>7}{'sm':>6}" + "".join(f"{'F='+str(f):>12}" for f in floors))
        for name, (p, bw, sm) in items:
            js = [(p - f) / bw for f in floors]
            tab.append({"workload": name, "p_w": p, "bw": bw, "sm_mhz": sm,
                        "jbyte_vs_floor": dict(zip([str(f) for f in floors], js))})
            print(f"{name:<32}{p:>7.1f}{bw/1e9:>7.1f}{(sm or 0):>6.0f}" + "".join(f"{j:>12.3e}" for j in js))
        # ratio serving / gemv_b1 across floors
        g1 = sel.get("gemv_b1_512MB")
        ratios = {}
        for k, v in serv.items():
            if g1:
                ratios[k] = {str(f): ((v[0] - f) / v[1]) / ((g1[0] - f) / g1[1]) for f in floors}
        print("serving / gemv_b1 J/B ratio by floor:", json.dumps(ratios, indent=None))
        out.setdefault("same_gpu", {})[g] = {"table": tab, "floors": floors, "serving_over_gemv": ratios,
                                             "idle_deep": mb["idle_pre"], "idle_p0": mb["idle_p0_pre"],
                                             "idle_post": mb.get("idle_post"),
                                             "lowgrid": mb.get("lowgrid", {}).get("fit_active_only"),
                                             "aliasing": mb.get("aliasing")}

    # 3. full-clock kernels
    fc = {}
    if os.path.exists("microbench/results_A5000_duty10.json"):
        d10 = json.load(open("microbench/results_A5000_duty10.json"))["duty10"]
        for k, v in d10.items():
            fc[f"duty10:{k}"] = {"e_ols": v["fit_ols"]["e"], "se": v["fit_ols"]["se_e"],
                                 "e_origin": v["e_origin"], "e_point_mean": v["e_point_mean"],
                                 "e_point_std": v["e_point_std"], "unit": v["unit"],
                                 "burst_rate_mean": float(np.mean([p["burst_rate"] for p in v["points"]]))}
    if os.path.exists("microbench/results_A5000_burstscan.json"):
        bs = json.load(open("microbench/results_A5000_burstscan.json"))["burstscan"]
        for k, v in bs.items():
            short = [p for p in v["points"] if p["t_burst_s"] / v["duty"] <= 0.03]   # period <= 30 ms
            if short:
                es = [p["e_per_unit"] for p in short]
                fc[f"burstscan_short:{k}"] = {"e_mean": float(np.mean(es)), "e_std": float(np.std(es)),
                                              "n": len(es), "unit": v["unit"],
                                              "burst_rate_mean": float(np.mean([p["burst_rate"] for p in short]))}
    if os.path.exists("microbench/results_A5000.json"):
        du = json.load(open("microbench/results_A5000.json"))["duty"]
        for k, v in du.items():
            if v.get("fit_pooled"):
                fc[f"duty1.5ms:{k}"] = {"e": v["fit_pooled"]["e"], "se": v["fit_pooled"]["se_e"],
                                        "unit": v["unit"], "burst_rate_mean": v.get("burst_rate_mean"),
                                        "note": "1.5 ms bursts, periods 20-100 ms: low-duty floors may sag"}
    out["full_clock"] = fc
    print("\nFULL-CLOCK (short-gap duty) kernel energy:")
    for k, v in fc.items():
        e = v.get("e_ols") or v.get("e_mean") or v.get("e")
        print(f"  {k:<40} e={e:.3e} J/{v['unit']}  " + ", ".join(f"{a}={b:.3e}" for a, b in v.items()
                                                              if isinstance(b, float) and a not in ('e_ols', 'e_mean', 'e')))
    json.dump(out, open("microbench/a5000_summary.json", "w"), indent=1)
    print("\nwrote microbench/a5000_summary.json")


if __name__ == "__main__":
    main()
