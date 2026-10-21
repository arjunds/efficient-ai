#!/usr/bin/env python3
"""
figdata.py (v2, 4 measured GPUs) -- data behind the LIMINAL-E figures (container, no matplotlib):
  (a) rooflines with full-clock power roof and the measured AT-CAP roofline (A6000, A100, B200;
      H200 predicted), capped/uncapped microbench kernels and measured serving points
  (b) energy-per-token "roofline" vs batch by component, short and long context, 4 GPUs
  (c) provisioning maps + per-token ranking table over model size x concurrency
      (measured: A6000, A100-80-PCIe, H200, B200; priors: H100-SXM, L40S)
Writes liminal_energy/figs_data.json and ranking_table.csv; plot_figs.py draws the figures.
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

MEASURED = ["A6000", "A100-80-PCIe", "H200", "B200"]
PRIORS = ["H100-SXM", "L40S"]


def logspace(a, b, n):
    return [10 ** (a + (b - a) * i / (n - 1)) for i in range(n)]


def host_neutral():
    """Provisioning compares hardware on a common host: the A100 node's fitted host-overhead
    scale (1.92x, a node property) is reset to 1 (the H200/B200 prior).  PCIe parts without
    NVLink (A6000, L40S) get the A100 PCIe TP-sync fit as a prior."""
    g = le.gpu("A100-80-PCIe")
    g["calibrated"] = dict(g["calibrated"], ovh_scale=1.0)
    pcie = le.TP_SYNC.get("A100-80-PCIe")
    for name in ("A6000", "L40S"):
        if pcie:
            le.TP_SYNC[name] = dict(pcie, src="PRIOR: A100 PCIe fit")


def serving_points():
    """Measured serving operating points (AI, FLOP/s per GPU) from validation_v2_runs.csv."""
    pts = []
    for r in csv.DictReader(open(os.path.join(HERE, "validation_v2_runs.csv"))):
        if r["moe"] == "True":
            continue
        m = le.model(r["model"] if "/" in r["model"] else _hf(r["model"]))
        n, P, G, cf = float(r["n_meas"]), float(r["P"]), float(r["G"]), float(r["cf"])
        w = le.serving_work(m, n, P, G, cf)
        # run-level rate: FLOPs per generated token x measured tok/s (robust to sparse bins)
        flops = w["F"] / n * float(r["tok_s_meas"]) / float(r["tp"])
        pts.append(dict(gpu=r["gpu"], arm=r["arm"], I=le.intensity(w), flops=flops,
                        prefill="prefill" in r["arm"]))
    return pts


def _hf(short):
    for org in ("Qwen", "mistralai", "meta-llama", "google"):
        for mid in ("%s/%s" % (org, short),):
            try:
                le.model(mid)
                return mid
            except Exception:
                continue
    return short


def fig_a():
    out = {}
    Is = logspace(-0.5, 4.5, 200)
    for gname in MEASURED:
        g = le.gpu(gname)
        full = le.roofline_curves(g, Is, "full")
        cap = le.roofline_curves(g, Is, "cap")
        out[gname] = dict(I=Is, bw_roof=[r["bw_roof"] for r in full], compute=g["peak_flops"],
                          power_full=[r["power_roof"] for r in full],
                          power_cap=[r["power_roof"] for r in cap] if g.get("at_cap") else None,
                          rates_full=le.power_limited_rates(g, "full"),
                          rates_cap=le.power_limited_rates(g, "cap") if g.get("at_cap") else None,
                          bw=g["bw"], p_cap=g["p_cap"], p_static=g["p_static"],
                          micro_capped=[dict(I=p["flops_per_s"] / p["bytes_per_s"], flops=p["flops_per_s"])
                                        for p in (g.get("at_cap") or {}).get("points", []) if p["flops_per_s"] > 0],
                          micro_uncapped=[])
    js = json.load(open(os.path.join(HERE, "gpus_measured.json")))
    out["B200"]["micro_uncapped"] = [dict(I=p["flops_per_s"] / p["bytes_per_s"], flops=p["flops_per_s"])
                                     for p in js["B200"].get("uncapped_llmgemm", [])]
    out["serving"] = serving_points()
    return out


def fig_b():
    m = le.model("Qwen/Qwen2-7B-Instruct")
    batches = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    ctxs = {"short": (25.0, 147.0), "long": (8192.0, 512.0)}
    out = {}
    for gname in MEASURED:
        g = le.gpu(gname)
        e_w, e_kv, e_g, _, _ = le.coefficients(g)
        for cname, (P, G) in ctxs.items():
            cap_seqs = min(le.kv_capacity_seqs(g, m, 1, P + G), 1024)
            rows = []
            for b in batches:
                if b > cap_seqs:
                    continue
                r = le.serving(g, m, P, G, conc=b, time_model="eager")
                n = r["n"]
                rows.append(dict(batch=b, total=r["j_per_token"], static=(r["e_static"] + r["split"]["activity"]) / n,
                                 weights=r["split"]["weights"] / n, kv=r["split"]["kv"] / n,
                                 compute=r["split"]["compute"] / n, capped=r["capped"], bound=r["bound"],
                                 power=r["power"], t=r["t"]))
            W = le.rg.weight_params(m, 1) * le.DTYPE_B
            out["%s_%s" % (gname, cname)] = dict(
                rows=rows, P=P, G=G, max_batch=cap_seqs,
                energy_ridge=e_w * W / (e_g * 2.0 * m.active_params() * (1.0 + P / G)),
                first_capped=next((x["batch"] for x in rows if x["capped"]), None))
    return out


MODELS = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct",
          "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-32B-Instruct",
          "Qwen/Qwen2-72B-Instruct"]
CONCS = [1, 4, 16, 64, 256, 1024]
CTXS = {"short (P=256, G=256)": (256.0, 256.0), "long (P=8192, G=1024)": (8192.0, 1024.0)}


def fig_c():
    sets = {"measured": MEASURED, "with_priors": MEASURED + PRIORS}
    out = dict(models=[x.split("/")[-1].replace("-Instruct", "") for x in MODELS], concs=CONCS,
               cells={}, ranking=[])
    for cname, (P, G) in CTXS.items():
        for sname, names in sets.items():
            grid = []
            for mid in MODELS:
                row = []
                for c in CONCS:
                    cand = le.provision_conc(mid, c, P, G, slo_tpot=0.05, gpu_names=names, time_model="eager")
                    if not cand:
                        row.append(None)
                        continue
                    best_per = {}
                    for x in cand:
                        best_per.setdefault(x["gpu"], x)
                    b = cand[0]
                    row.append(dict(gpu=b["gpu"], tp=b["tp"], replicas=b["replicas"], n_gpus=b["n_gpus"],
                                    j=b["j_per_token"], capped=b["capped"],
                                    prior=b["gpu"] in PRIORS,
                                    per_gpu={k: dict(j=v["j_per_token"], n_gpus=v["n_gpus"], tp=v["tp"])
                                             for k, v in best_per.items()}))
                    if sname == "with_priors":
                        order = sorted(best_per.items(), key=lambda kv: kv[1]["j_per_token"])
                        out["ranking"].append(dict(context=cname, model=mid.split("/")[-1], conc=c,
                                                   order=[(k, v["j_per_token"], v["n_gpus"], v["tp"])
                                                          for k, v in order]))
                grid.append(row)
            out["cells"]["%s|%s" % (cname, sname)] = grid
    return out


def write_ranking(c):
    with open(os.path.join(HERE, "ranking_table.csv"), "w") as f:
        w = csv.writer(f)
        w.writerow(["context", "model", "concurrency", "rank", "gpu", "status", "mJ_per_token", "n_gpus", "tp",
                    "ratio_to_best"])
        for r in c["ranking"]:
            best = r["order"][0][1]
            for i, (g, j, ng, tp) in enumerate(r["order"]):
                w.writerow([r["context"], r["model"], r["conc"], i + 1, g, "prior" if g in PRIORS else "measured",
                            "%.2f" % (j * 1e3), ng, tp, "%.2f" % (j / best)])


def ridges():
    out = {}
    for gname in MEASURED + PRIORS:
        g = le.gpu(gname)
        r = le.ridges(g)
        r.update({"full_" + k: v for k, v in le.power_limited_rates(g, "full").items()})
        if g.get("at_cap"):
            r.update({"cap_" + k: v for k, v in le.power_limited_rates(g, "cap").items()})
        r["measured"] = gname in MEASURED
        r["p_static"], r["p0"] = g["p_static"], g.get("p0", 0.0)
        out[gname] = r
    return out


if __name__ == "__main__":
    host_neutral()
    data = dict(a=fig_a(), b=fig_b(), ridges=ridges())
    for k, v in data["ridges"].items():
        print("  %-13s perf %.0f energy %.0f | full-clock BW_P %.0f%% PHI_P %.0f%% | at-cap %s %s | P_s %.0f P0 %.0f" % (
            k, v["perf_ridge"], v["energy_ridge"], 100 * v["full_bw_frac"], 100 * v["full_flops_frac"],
            ("%.0f%%" % (100 * v["cap_bw_frac"])) if "cap_bw_frac" in v else "-",
            ("%.0f%%" % (100 * v["cap_flops_frac"])) if "cap_flops_frac" in v else "-", v["p_static"], v["p0"]))
    for k, v in data["b"].items():
        c = v["rows"]
        print("  fig b %-18s B*=%.0f maxB %4d first capped %s  J/tok b=1 %.3f  b=max %.4f" % (
            k, v["energy_ridge"], v["max_batch"], v["first_capped"], c[0]["total"], c[-1]["total"]))
    data["c"] = fig_c()
    write_ranking(data["c"])
    for key, grid in data["c"]["cells"].items():
        print("\n  fig c", key)
        for mname, row in zip(data["c"]["models"], grid):
            print("   %-14s" % mname + " ".join(
                ("%6s x%-2d %7.1f" % (x["gpu"][:6], x["n_gpus"], x["j"] * 1e3)) if x else "%18s" % "-"
                for x in row))
    json.dump(data, open(os.path.join(HERE, "figs_data.json"), "w"), indent=1)
