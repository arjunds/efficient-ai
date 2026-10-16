#!/usr/bin/env python3
"""
figdata.py -- compute the data behind the LIMINAL-E figures (container, no matplotlib):
  (a) rooflines with power roofs (B200, H200) + measured serving and microbench points
  (b) energy-per-token "roofline" vs batch, by component (static, weights, KV, compute)
  (c) provisioning maps: best GPU x TP x replicas and J/token over model size x concurrency
Writes liminal_energy/figs_data.json; plot_figs.py draws it.
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(1, PARENT)
os.chdir(PARENT)

import liminal_e as le  # noqa: E402


def logspace(a, b, n):
    return [10 ** (a + (b - a) * i / (n - 1)) for i in range(n)]


def fig_a():
    out = {}
    Is = logspace(-0.5, 4.5, 200)
    for gname in ("H200", "B200"):
        g = le.gpu(gname)
        full = le.roofline_curves(g, Is, "full")
        cap = le.roofline_curves(g, Is, "cap")
        out[gname] = dict(I=Is, bw_roof=[r["bw_roof"] for r in full], compute=g["peak_flops"],
                          power_full=[r["power_roof"] for r in full],
                          power_cap=[r["power_roof"] for r in cap],
                          rates_full=le.power_limited_rates(g, "full"),
                          rates_cap=le.power_limited_rates(g, "cap"),
                          measured_roof=le.POWER_ROOF.get(gname, {}).get("src"),
                          bw=g["bw"], p_cap=g["p_cap"], p_static=g["p_static"])
    # B200 steady microbench LLM-shaped GEMMs (microbench/results_B200_steady.json)
    mb = json.load(open("microbench/results_B200_steady.json"))
    pts = []
    for d in mb["llmgemm"]:
        reps = d["repeats"]
        fl = sum(r["flops_per_s"] for r in reps) / len(reps)
        by = sum(r["bytes_per_s"] for r in reps) / len(reps)
        capf = sum(r["frac_sw_power_cap"] for r in reps) / len(reps)
        sm = sum(r["sm_mhz"] for r in reps) / len(reps)
        pts.append(dict(label=d["label"], I=fl / by, flops=fl, capped=capf > 0.05, sm_mhz=sm,
                        p_w=sum(r["p_w"] for r in reps) / len(reps)))
    out["B200_microbench_gemm"] = pts
    return out


def fig_b():
    """J/token vs batch, Qwen2-7B.  Short = alpaca-like (P=25, G=147, the measured mix);
    long = P=8192, G=512.  Eager time model (the mode of the overlaid Sept runs)."""
    m = le.model("Qwen/Qwen2-7B-Instruct")
    batches = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    ctxs = {"short": (25.0, 147.0), "long": (8192.0, 512.0)}
    out = {}
    for gname in ("H200", "B200"):
        g = le.gpu(gname)
        profs = ["canonical"] + [p for p in ("kv_low", "kv_high") if p in le.PROFILES.get(gname, {})]
        for cname, (P, G) in ctxs.items():
            cap_seqs = min(le.kv_capacity_seqs(g, m, 1, P + G), 1024)
            rows = {}
            for pf in profs:
                rr = []
                for b in batches:
                    if b > cap_seqs:
                        continue
                    r = le.serving(g, m, P, G, conc=b, profile=pf, time_model="eager")
                    n = r["n"]
                    rr.append(dict(batch=b, total=r["j_per_token"], static=r["e_static"] / n,
                                   weights=r["split"]["weights"] / n, kv=r["split"]["kv"] / n,
                                   compute=r["split"]["compute"] / n, capped=r["capped"],
                                   bound=r["bound"], power=r["power"], t=r["t"]))
                rows[pf] = rr
            e_w, e_kv, e_g, _, _ = le.coefficients(g)
            W = le.rg.weight_params(m, 1) * le.DTYPE_B
            # batch where weight J/token = compute J/token (compute incl. each token's prefill)
            b_star = e_w * W / (e_g * 2.0 * m.active_params() * (1.0 + P / G))
            # batch where weight J/token = KV J/token
            b_kv = W / ((P + G / 2.0) * m.kv_bytes_per_token(le.DTYPE_B)) * e_w / e_kv
            out["%s_%s" % (gname, cname)] = dict(
                rows=rows, P=P, G=G, max_batch=cap_seqs, energy_ridge=b_star, kv_ridge=b_kv,
                energy_ridge_bare=e_w / e_g,
                perf_ridge=g["peak_flops"] / g["bw"],
                first_capped=next((x["batch"] for x in rows["canonical"] if x["capped"]), None))
    return out


def fig_c():
    models = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct",
              "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-32B-Instruct",
              "Qwen/Qwen2-72B-Instruct"]
    concs = [1, 4, 16, 64, 256, 1024]
    ctxs = {"short (P=256, G=256)": (256.0, 256.0), "long (P=8192, G=1024)": (8192.0, 1024.0)}
    sets = {"measured": (["H200", "B200"], "canonical"),
            "measured_kv_as_weight": (["H200", "B200"], "kv_as_weight"),
            "all": (["H200", "B200", "H100-SXM", "A100-80-SXM", "L40S"], "canonical")}
    out = dict(models=[x.split("/")[-1].replace("-Instruct", "") for x in models], concs=concs,
               params=[le.model(x).active_params() for x in models], cells={})
    for cname, (P, G) in ctxs.items():
        for sname, (names, prof) in sets.items():
            grid = []
            for mid in models:
                row = []
                for c in concs:
                    cand = le.provision_conc(mid, c, P, G, slo_tpot=0.05, gpu_names=names,
                                             time_model="eager", profile=prof)
                    if not cand:
                        row.append(None)
                        continue
                    b = cand[0]
                    second = next((x for x in cand if x["gpu"] != b["gpu"]), None)
                    row.append(dict(gpu=b["gpu"], tp=b["tp"], replicas=b["replicas"], n_gpus=b["n_gpus"],
                                    j=b["j_per_token"], tpot=b["tpot"], capped=b["capped"],
                                    measured=b["measured"],
                                    runner_up=second and dict(gpu=second["gpu"], j=second["j_per_token"],
                                                              n_gpus=second["n_gpus"])))
                grid.append(row)
            out["cells"]["%s|%s" % (cname, sname)] = grid
    return out


def provisioning_example():
    """Open-loop example for LIMINAL_E.md: Qwen2.5-32B, prompt 1024, gen 256, TPOT <= 50 ms."""
    ex = {}
    for rate in (0.5, 5.0, 50.0):
        cand = le.provision_rate("Qwen/Qwen2.5-32B-Instruct", rate, 1024.0, 256.0, slo_tpot=0.05,
                                 gpu_names=["H200", "B200", "H100-SXM", "A100-80-SXM", "L40S"],
                                 time_model="eager")
        ex[str(rate)] = [dict((k, v) for k, v in c.items()) for c in cand[:6]]
    return ex


def ridges():
    out = {}
    for gname in ("H200", "B200", "H100-SXM", "A100-80-SXM", "L40S"):
        g = le.gpu(gname)
        r = le.ridges(g)
        r.update(le.power_limited_rates(g, "full"))
        for pf in ("kv_low", "kv_high", "runlevel"):
            if pf in le.PROFILES.get(gname, {}):
                r["energy_ridge_" + pf] = le.ridges(g, pf)["energy_ridge"]
        r["measured"] = g.get("measured", False)
        out[gname] = r
    return out


if __name__ == "__main__":
    data = dict(a=fig_a(), b=fig_b(), ridges=ridges())
    print("ridges:")
    for k, v in data["ridges"].items():
        print("  %-12s perf %.0f  energy(serving) %.0f  roof knee %.0f  static-equiv BW %.2f TB/s (%.0f%%)  BW_P %.0f%%  PHI_P %.0f%%" % (
            k, v["perf_ridge"], v["energy_ridge"], v["roof_knee"], v["static_equiv_bw"] / 1e12, 100 * v["static_equiv_frac"],
            100 * v["bw_frac"], 100 * v["flops_frac"]), {kk: round(vv, 1) for kk, vv in v.items()
                                                          if kk.startswith("energy_ridge_")})
    for k in ("H200", "B200"):
        print("  power roof", k, "full", {kk: round(vv, 3) if isinstance(vv, float) else vv
                                          for kk, vv in data["a"][k]["rates_full"].items()})
        print("  power roof", k, "cap ", {kk: round(vv, 3) if isinstance(vv, float) else vv
                                          for kk, vv in data["a"][k]["rates_cap"].items()})
    for k, v in data["b"].items():
        c = v["rows"]["canonical"]
        print("  fig b %-12s ridge B*=%.1f kv-ridge %.1f max_batch %4d first capped %s  J/tok b=1 %.3f  b=max %.4f  bounds %s" % (
            k, v["energy_ridge"], v["kv_ridge"], v["max_batch"], v["first_capped"], c[0]["total"], c[-1]["total"],
            [x["bound"][:3] for x in c]))
    data["c"] = fig_c()
    for key, grid in data["c"]["cells"].items():
        print("\n  fig c", key)
        for mname, row in zip(data["c"]["models"], grid):
            print("   %-14s" % mname + " ".join(
                ("%5s x%-2d %7.1f" % (x["gpu"][:5], x["n_gpus"], x["j"] * 1e3)) if x else "%17s" % "-"
                for x in row))
    data["provisioning_example"] = provisioning_example()
    for rate, cand in data["provisioning_example"].items():
        print("\n  32B open loop rate %s req/s:" % rate)
        for c in cand:
            print("    %-12s tp=%d R=%d (%d GPUs) batch %.1f  %.1f mJ/tok  TPOT %.1f ms  busy %.2f  P %.0f W%s" % (
                c["gpu"], c["tp"], c["replicas"], c["n_gpus"], c["batch"], c["j_per_token"] * 1e3,
                c["tpot"] * 1e3, c["busy"], c["power_total"], " CAP" if c["capped"] else ""))
    json.dump(data, open(os.path.join(HERE, "figs_data.json"), "w"), indent=1)
