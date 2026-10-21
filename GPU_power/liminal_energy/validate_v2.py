#!/usr/bin/env python3
"""
validate_v2.py -- LIMINAL-E v2 against MEASURED serving runs on four GPUs.

  1. ZERO-SHOT A100 / A6000: microbench + datasheet + idle only (gpus_measured.json
     "zeroshot" + at-cap roofline; time = recommender pooled prior fitted on H200/B200 only,
     CUDA-graphs dtau from B200).  No A100/A6000 serving data is used.
  2. CALIBRATION (reported separately): per GPU a time scale on the uncapped time model
     (uncapped runs) and a scale on the at-cap roofline (capped runs); A100 PCIe TP sync
     (tau_ar, link rate) fitted on 14B TP=2, tested on 32B TP=2; H200 e_gemm refit with the
     KV prior e_kv = 1.25 e_w (B200 ratio).  Written to gpus_measured.json "calibrated".
  3. B200: settled-KV canonical coefficients; at-cap roofline vs time-stretch on capped runs.
Writes validation_v2.json, validation_v2_runs.csv.  Run in the container.
"""
import csv
import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(1, PARENT)
os.chdir(PARENT)

import liminal_e as le            # noqa: E402
import recommend_gpu as rg        # noqa: E402
import recommender_backtest as rb  # noqa: E402
import validate as v1             # noqa: E402

MJSON = os.path.join(HERE, "gpus_measured.json")


def gkey(name):
    n = name.upper()
    if "A6000" in n:
        return "A6000"
    return rb.gpu_key(name)


def load_dir_runs(pattern, arm_of, summary=None):
    out = []
    summ = {}
    if summary and os.path.exists(summary):
        summ = json.load(open(summary)).get("runs", {})
    for rd in sorted(glob.glob(pattern)):
        if not os.path.isdir(rd) or os.path.basename(rd) == "idle":
            continue
        if not os.path.exists(os.path.join(rd, "binned_table.csv")):
            print("[skip: no binned_table]", rd)
            continue
        r = rb.load_run(rd)
        # prefill-heavy runs have few, long steps -> sparse bins; run totals still valid
        if r is None or r["cover"] < (0.3 if "prefill" in rd else 0.9):
            continue
        meta = json.load(open(os.path.join(rd, "run_meta.json")))
        r["gpu"] = gkey(meta.get("gpu_name", ""))
        key = os.path.relpath(rd, os.path.dirname(summary)) if summary else rd
        s = summ.get(key)
        tp = int(meta.get("tp_degree") or 1)
        eager = meta.get("enforce_eager")
        r.update(set=r["gpu"], arm=arm_of(rd), key=rd, tp=tp, eager=(eager is not False),
                 j_meas=(s or {}).get("j_tok") or r["j_tok"], p_meas=(s or {}).get("p_avg") or r["power"],
                 tok_s_meas=(s or {}).get("tok_s") or r["tok_s"], cap_frac=(s or {}).get("cap_frac"),
                 idle_meas=meta.get("idle_power_w"))
        r["m"] = le.model(r["model"])
        out.append(r)
    return out


def load_all():
    runs = v1.load_sets()                                     # B200_prov + Sept H200/B200
    for r in runs:
        r["gpu"] = r["gpu"]
    a100 = load_dir_runs("logs/A100_prov/*/*/*", lambda rd: rd.split("/")[2],
                         "logs/A100_prov/provisioning_summary.json")
    a100 += load_dir_runs("logs/ragged_a100/*/*", lambda rd: "a100_sept_eager")
    a6k = load_dir_runs("logs/A6000_prov/*/*/*", lambda rd: rd.split("/")[2],
                        "logs/A6000_prov/provisioning_summary.json")
    return runs + a100 + a6k


def predict(r, **kw):
    g = le.gpu(r["gpu"])
    tm = "eager" if r["eager"] else "graphs"
    kw.setdefault("time_model", tm)
    return le.serving(g, r["m"], r["P"], r["G"], conc=r["conc"], rate=r.get("rate") if r["conc"] is None else None,
                      cached_frac=r["cf"], tp=r["tp"], **kw)


def ape(p, m):
    return 100.0 * abs(p - m) / m


def stats(rows, pk, mk):
    xs = [(x[pk], x[mk]) for x in rows if x.get(pk) is not None and x.get(mk)]
    if not xs:
        return None
    return dict(n=len(xs), mape=sum(ape(p, m) for p, m in xs) / len(xs),
                bias=sum(100.0 * (p / m - 1) for p, m in xs) / len(xs))


def golden(f, lo, hi, it=70):
    gr = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    for _ in range(it):
        c, d = b - gr * (b - a), a + gr * (b - a)
        if f(c) < f(d):
            b = d
        else:
            a = c
    return 0.5 * (a + b)


def set_cal(gname, **kv):
    g = le.gpu(gname)
    cal = dict(g.get("calibrated") or {})
    cal.update(kv)
    g["calibrated"] = cal
    for k in ("e_wbyte", "e_kvbyte", "e_gemm", "p_static", "p0"):
        if k in kv:
            g[k] = kv[k]


def main():
    runs = load_all()
    by_gpu = {}
    for r in runs:
        by_gpu.setdefault(r["gpu"], []).append(r)
    print("runs:", {k: len(v) for k, v in by_gpu.items()})
    # dense, non-MoE, single-GPU runs are the calibration/test population
    dense = lambda r: not rg.is_moe(r["m"])

    # start from a clean slate: drop any calibrated blocks for A100/A6000/H200
    js = json.load(open(MJSON))
    for k in ("A100-80-PCIe", "A6000", "H200"):
        if k in js:
            js[k].pop("calibrated", None)
            js[k].pop("tp_sync", None)
            if k == "H200" and len(js[k]) == 0:
                js.pop(k)
    json.dump(js, open(MJSON, "w"), indent=1)
    le._GPUS = None
    le.TP_SYNC.pop("A100-80-PCIe", None)

    # ---------------- 1. zero-shot predictions (A100 / A6000; B200 microbench path) -------
    rows = []
    for r in runs:
        zs = predict(r, profile="zeroshot") if r["gpu"] in ("A100-80-PCIe", "A6000", "B200") else None
        zg = predict(r, profile="zeroshot_gemv") if r["gpu"] in ("A100-80-PCIe", "A6000", "B200") else None
        lt = predict(r, time_model="liminal", power_model="tdp")
        row = dict(gpu=r["gpu"], set=r["set"], arm=r["arm"], key=r["key"], model=r["model"].split("/")[-1],
                   moe=rg.is_moe(r["m"]), task=r["task"], conc=r["conc"], tp=r["tp"], eager=r["eager"],
                   P=r["P"], G=r["G"], cf=r["cf"], n_meas=r["n_meas"], t_meas=r["t_iter"],
                   j_meas=r["j_meas"], p_meas=r["p_meas"], tok_s_meas=r["tok_s_meas"], cap_frac=r["cap_frac"],
                   idle_meas=r["idle_meas"],
                   j_liminal_tdp=lt["j_per_token"], tok_s_liminal=lt["tok_s"])
        if zs:
            zr = predict(r, profile="zeroshot", cap_mode="roof")
            row.update(j_zs_roof=zr["j_per_token"], tok_s_zs_roof=zr["tok_s"])
            row.update(j_zs=zs["j_per_token"], tok_s_zs=zs["tok_s"], p_zs=zs["power"], capped_zs=zs["capped"],
                       t_zs=zs["t"], j_zs_gemv=zg["j_per_token"], tok_s_zs_gemv=zg["tok_s"],
                       capped_zs_gemv=zg["capped"])
        row["_r"] = r
        rows.append(row)

    # ---------------- 2. calibration -----------------------------------------------------
    # regimes are classified by MEASURED power: capped if p_meas >= 0.93 * tp * P_cap
    for x in rows:
        g = le.gpu(x["gpu"])
        x["meas_capped"] = x["p_meas"] >= 0.93 * x["tp"] * g["p_cap"]
    cal_out = {}

    def tok_loss(sel, **kw):
        return sum(math.log(predict(x["_r"], **kw)["tok_s"] / x["tok_s_meas"]) ** 2 for x in sel)

    def energy_only_power(x, prof):
        r = x["_r"]
        pp = le.profile_params(le.gpu(r["gpu"]), prof)
        w = v1.work_at(r, r["n_meas"])
        e = pp["e_w"] * w["Wb"] + pp["e_kv"] * w["KVb"] + pp["e_g"] * (w["Fg"] + w["Fa_pf"])
        return r["tp"] * (pp["p_s"] + pp["p0"]) + e / r["t_iter"]

    for gname in ("A100-80-PCIe", "A6000"):
        g = le.gpu(gname)
        rs = [x for x in rows if x["gpu"] == gname and x["tp"] == 1 and not x["moe"] and x["arm"] != "a100_prefill"]
        unc = [x for x in rs if not x["meas_capped"]]
        cap = [x for x in rs if x["meas_capped"]]
        # (a) which microbench energy rule predicts power of measured-UNCAPPED runs (at measured t, n)?
        eo = {}
        for prof in ("zeroshot", "zeroshot_gemv"):
            errs = [100 * (energy_only_power(x, prof) / x["p_meas"] - 1) for x in unc]
            eo[prof] = dict(n=len(errs), mape=sum(abs(e) for e in errs) / len(errs) if errs else None,
                            bias=sum(errs) / len(errs) if errs else None)
        rule = "zeroshot"
        if eo["zeroshot_gemv"]["mape"] is not None and eo["zeroshot"]["mape"] is not None and \
                eo["zeroshot_gemv"]["mape"] < eo["zeroshot"]["mape"]:
            rule = "zeroshot_gemv"
        z = g["zeroshot"] if rule == "zeroshot" else g["zeroshot"]["alt_gemv"]
        set_cal(gname, e_wbyte=z["e_wbyte"], e_kvbyte=z["e_kvbyte"], e_gemm=z["e_gemm"], p_static=z["p_static"],
                p0=z.get("p0", 0.0), ovh_scale=1.0, cap_scale=1.0, cap_mode="roof_ovh", energy_rule=rule)
        # (b) host/launch overhead scale on measured-uncapped dense runs
        ovh = 1.0
        if len(unc) >= 3:
            def lo(sc):
                set_cal(gname, ovh_scale=sc)
                return tok_loss(unc, profile="canonical")
            ovh = golden(lo, 0.5, 3.0)
        set_cal(gname, ovh_scale=ovh)
        # (c) at-cap scale for each cap mode on measured-capped runs
        fits = {}
        for mode in ("roof", "roof_ovh"):
            set_cal(gname, cap_mode=mode)

            def lc(sc):
                set_cal(gname, cap_scale=sc)
                return tok_loss(cap, profile="canonical")
            sc = golden(lc, 0.6, 1.8)
            fits[mode] = dict(cap_scale=sc, rmsle=math.sqrt(lc(sc) / len(cap)),
                              rmsle_unscaled=math.sqrt(lc(1.0) / len(cap)))
        mode = min(fits, key=lambda k: fits[k]["rmsle"])
        set_cal(gname, cap_mode=mode, cap_scale=fits[mode]["cap_scale"],
                src="%s coefficients + serving-fitted ovh_scale (%d uncapped runs) and cap_scale (%s, %d capped "
                    "runs); validate_v2.py" % (rule, len(unc), mode, len(cap)))
        ac = g["at_cap"]
        cal_out[gname] = dict(energy_rule=rule, energy_only_uncapped=eo, ovh_scale=ovh, cap_fits=fits,
                              cap_mode=mode, n_uncapped=len(unc), n_capped=len(cap),
                              eps_b_micro=ac["eps_b"], bw_cap_micro=ac["bw_cap"])
        print("[cal] %s: energy rule %s (uncapped power MAPE zs %s / gemv %s); ovh_scale %.2f (n=%d); "
              "cap fits %s -> %s" % (
                  gname, rule, eo["zeroshot"]["mape"] and round(eo["zeroshot"]["mape"], 1),
                  eo["zeroshot_gemv"]["mape"] and round(eo["zeroshot_gemv"]["mape"], 1), ovh, len(unc),
                  {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in fits.items()}, mode))

    # transfer test: A100-fitted at-cap scale applied to A6000 and B200 capped runs (and vice versa)
    transfer = {}
    for src_g, dst_g in (("A100-80-PCIe", "A6000"), ("A6000", "A100-80-PCIe"), ("A100-80-PCIe", "B200")):
        cs = le.gpu(src_g)["calibrated"]
        dst = [x for x in rows if x["gpu"] == dst_g and x["tp"] == 1 and not x["moe"] and x["meas_capped"]
               and x["arm"] not in ("a100_prefill", "prefill")]
        if not dst:
            continue
        res = {}
        for label, kw in (("zero-shot roof_ovh", dict(profile="zeroshot", cap_mode="roof_ovh")),
                          ("transfer %s scale" % src_g, dict(profile="zeroshot", cap_mode=cs["cap_mode"]))):
            if "transfer" in label:
                gd = le.gpu(dst_g)
                saved = dict(gd["zeroshot"])
                # apply the source GPU's cap scale through a temporary calibrated block
                old = gd.get("calibrated")
                gd["calibrated"] = dict(old or {}, cap_scale=cs["cap_scale"], cap_mode=cs["cap_mode"])
                kw = dict(profile="canonical")
            ps = [predict(x["_r"], **kw) for x in dst]
            if "transfer" in label:
                gd["calibrated"] = old
            res[label] = dict(n=len(dst), tok_mape=sum(ape(p["tok_s"], x["tok_s_meas"]) for p, x in zip(ps, dst)) / len(dst),
                              j_mape=sum(ape(p["j_per_token"], x["j_meas"]) for p, x in zip(ps, dst)) / len(dst),
                              j_bias=sum(100 * (p["j_per_token"] / x["j_meas"] - 1) for p, x in zip(ps, dst)) / len(dst))
        transfer["%s->%s" % (src_g, dst_g)] = res
        print("[transfer] %s -> %s: %s" % (src_g, dst_g, {k: {kk: round(vv, 1) for kk, vv in v.items()}
                                                         for k, v in res.items()}))

    # A100 PCIe TP sync: fit tau_ar + link rate on 14B TP2, test 32B TP2
    tp2 = [x for x in rows if x["gpu"] == "A100-80-PCIe" and x["tp"] == 2]
    fit14 = [x for x in tp2 if "14B" in x["model"]]
    test32 = [x for x in tp2 if "32B" in x["model"]]

    def tp_loss(tau_ar, link, sel):
        le.TP_SYNC["A100-80-PCIe"] = dict(tau_ar=tau_ar, pen=1.0, link_bw=link)
        return sum(math.log(predict(x["_r"], profile="canonical")["tok_s"] / x["tok_s_meas"]) ** 2 for x in sel)
    best = None
    for tau in [i * 5e-6 for i in range(0, 41)]:
        for link in [l * 1e9 for l in (2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 50, 1e3)]:
            v = tp_loss(tau, link, fit14)
            if best is None or v < best[0]:
                best = (v, tau, link)
    _, tau_ar, link = best
    tau_only = golden(lambda t: tp_loss(t, 1e15, fit14), 0, 400e-6)
    tp_res = dict(tau_ar_us=tau_ar * 1e6, link_gbs=link / 1e9, tau_only_us=tau_only * 1e6, test=[])
    for x in tp2:
        le.TP_SYNC["A100-80-PCIe"] = dict(tau_ar=tau_ar, pen=1.0, link_bw=link)
        p_fit = predict(x["_r"], profile="canonical")
        le.TP_SYNC["A100-80-PCIe"] = dict(tau_ar=tau_only, pen=1.0)
        p_tau = predict(x["_r"], profile="canonical")
        le.TP_SYNC["A100-80-PCIe"] = dict(tau_ar=15.5e-6, pen=1.0)
        p_nvl = predict(x["_r"], profile="canonical")
        tp_res["test"].append(dict(key=x["key"], tok_s_meas=x["tok_s_meas"], j_meas=x["j_meas"],
                                   tok_s_fit=p_fit["tok_s"], j_fit=p_fit["j_per_token"],
                                   tok_s_tau_only=p_tau["tok_s"], tok_s_nvlink=p_nvl["tok_s"],
                                   j_nvlink=p_nvl["j_per_token"]))
    for k in ("tok_s_fit", "tok_s_tau_only", "tok_s_nvlink"):
        tp_res["test32_mape_" + k] = sum(ape(t[k], t["tok_s_meas"]) for t in tp_res["test"] if "32B" in t["key"]) / max(
            1, len(test32))
    le.TP_SYNC["A100-80-PCIe"] = dict(tau_ar=tau_ar, pen=1.0, link_bw=link)
    print("[tp] A100 PCIe: tau_ar %.0f us/layer + all-reduce at %.0f GB/s (14B fit); 32B held-out tok/s MAPE: "
          "fit %.1f%%, tau-only %.1f%%, B200-NVLink sync %.1f%%" % (
              tau_ar * 1e6, link / 1e9, tp_res["test32_mape_tok_s_fit"], tp_res["test32_mape_tok_s_tau_only"],
              tp_res["test32_mape_tok_s_nvlink"]))

    # H200: KV prior + e_gemm refit (run level, energy at measured operating point)
    h = [x["_r"] for x in rows if x["arm"] == "sept_H200"]
    gH = le.gpu("H200")
    e_w = gH["e_wbyte"]
    e_kv = 1.25 * e_w
    num = den = 0.0
    for r in h:
        w = v1.work_at(r, r["n_meas"])
        j0 = (gH["p_static"] + (e_w * w["Wb"] + e_kv * w["KVb"]) / r["t_iter"]) / r["tok_s_meas"]
        xF = ((w["Fg"] + w["Fa_pf"]) / r["t_iter"]) / r["tok_s_meas"]
        num += (xF / r["j_meas"]) * ((r["j_meas"] - j0) / r["j_meas"])
        den += (xF / r["j_meas"]) ** 2
    e_g_h = num / den
    print("[H200] e_kv prior 1.25*e_w = %.3e; refit e_gemm = %.3g pJ (was %.3g)" % (e_kv, e_g_h * 1e12,
                                                                                     gH["e_gemm"] * 1e12))

    # persist calibrated blocks
    js = json.load(open(MJSON))
    for gname in ("A100-80-PCIe", "A6000"):
        js[gname]["calibrated"] = le.gpu(gname)["calibrated"]
    js["A100-80-PCIe"]["tp_sync"] = dict(tau_ar=tau_ar, pen=1.0, link_bw=link,
                                         src="validate_v2.py: 14B TP=2 fit, 32B TP=2 held out")
    js.setdefault("H200", {})["calibrated"] = dict(
        e_wbyte=e_w, e_kvbyte=e_kv, e_gemm=e_g_h, p_static=gH["p_static"], p0=0.0,
        src="e_w Sept fit; e_kv = 1.25*e_w PRIOR (B200 settled ratio); e_gemm refit on 80 Sept runs (run level)")
    json.dump(js, open(MJSON, "w"), indent=1)
    le._GPUS = None

    # ---------------- 3. calibrated / canonical predictions for every GPU ----------------
    for row in rows:
        r = row["_r"]
        c = predict(r, profile="canonical")
        row.update(j_cal=c["j_per_token"], tok_s_cal=c["tok_s"], p_cal=c["power"], capped_cal=c["capped"],
                   bound_cal=c["bound"])
        if r["gpu"] in ("B200", "H200"):
            sp = predict(r, profile="sept_perbin")
            row.update(j_sept=sp["j_per_token"])
        if r["gpu"] == "B200":
            st = predict(r, profile="canonical", cap_mode="stretch")
            row.update(j_cal_stretch=st["j_per_token"], tok_s_cal_stretch=st["tok_s"])
        if r["arm"] in ("prefill", "a100_prefill"):
            row["prompt_ratio"] = r["G"] / r["P"]

    # ---------------- 4. summaries ------------------------------------------------------
    groups = [
        ("A6000 all (eager+graphs+highconc+longctx)", lambda x: x["gpu"] == "A6000"),
        ("A6000 c<=64 short ctx", lambda x: x["gpu"] == "A6000" and x["arm"] in ("gpu_eager", "gpu_graphs")),
        ("A6000 highconc", lambda x: x["gpu"] == "A6000" and x["arm"] == "gpu_highconc"),
        ("A6000 longctx", lambda x: x["gpu"] == "A6000" and x["arm"] == "gpu_longctx"),
        ("A100 dense TP1 (all arms)", lambda x: x["gpu"] == "A100-80-PCIe" and x["tp"] == 1 and not x["moe"]),
        ("A100 7-14B graphs/eager c<=64", lambda x: x["gpu"] == "A100-80-PCIe" and x["arm"] in ("a100_graphs", "a100_eager")),
        ("A100 highconc", lambda x: x["gpu"] == "A100-80-PCIe" and x["arm"] == "a100_highconc"),
        ("A100 longctx", lambda x: x["gpu"] == "A100-80-PCIe" and x["arm"] == "a100_longctx"),
        ("A100 ladder (0.5-14B, eager)", lambda x: x["gpu"] == "A100-80-PCIe" and x["arm"] == "a100_ladder_eager"),
        ("A100 ladder <=3B (uncapped)", lambda x: x["gpu"] == "A100-80-PCIe" and x["arm"] == "a100_ladder_eager"
         and any(s in x["model"] for s in ("0.5B", "1.5B", "3B"))),
        ("A100 Sept eager (ragged_a100)", lambda x: x["arm"] == "a100_sept_eager"),
        ("A100 prefill (per gen token)", lambda x: x["arm"] == "a100_prefill"),
        ("A100 MoE Qwen3-30B-A3B", lambda x: x["gpu"] == "A100-80-PCIe" and x["moe"]),
        ("A100 TP2 PCIe", lambda x: x["gpu"] == "A100-80-PCIe" and x["tp"] == 2),
        ("B200 Oct graphs/highconc/tp", lambda x: x["gpu"] == "B200" and x["arm"] in ("graphs", "highconc", "tp2", "tp1ref")),
        ("B200 Oct longctx (7B, 32B)", lambda x: x["gpu"] == "B200" and x["arm"] in ("longctx", "longctx32")),
        ("B200 Sept eager", lambda x: x["arm"] == "sept_B200"),
        ("B200 capped runs (cap>=50%)", lambda x: x["gpu"] == "B200" and (x["cap_frac"] or 0) >= 0.5),
        ("H200 Sept eager", lambda x: x["arm"] == "sept_H200"),
    ]
    summ = {}
    print("\n%-44s %4s | %-15s %-15s %-15s %-15s | %-15s %-15s | %-15s" % (
        "group", "n", "J zero-shot", "J zs-roof", "J zs-GEMV", "J calibrated", "tok/s zs", "tok/s cal", "J LIMINAL-TDP"))
    for gname, f in groups:
        sel = [x for x in rows if f(x)]
        if not sel:
            continue
        s = {k: stats(sel, p, m) for k, p, m in (
            ("j_zs", "j_zs", "j_meas"), ("j_zs_gemv", "j_zs_gemv", "j_meas"), ("j_cal", "j_cal", "j_meas"),
            ("tok_zs", "tok_s_zs", "tok_s_meas"), ("tok_cal", "tok_s_cal", "tok_s_meas"),
            ("j_tdp", "j_liminal_tdp", "j_meas"), ("j_sept", "j_sept", "j_meas"),
            ("j_cal_stretch", "j_cal_stretch", "j_meas"), ("tok_cal_stretch", "tok_s_cal_stretch", "tok_s_meas"),
            ("j_zs_roof", "j_zs_roof", "j_meas"), ("tok_zs_roof", "tok_s_zs_roof", "tok_s_meas"))}
        s["n"] = len(sel)
        s["capped_zs_frac"] = sum(1 for x in sel if x.get("capped_zs")) / float(len(sel))
        s["meas_capped_frac"] = sum(1 for x in sel if x.get("meas_capped")) / float(len(sel))
        summ[gname] = s
        fmt = lambda d: ("%5.1f (%+5.1f)" % (d["mape"], d["bias"])) if d else "      -        "
        print("%-44s %4d | %s %s %s %s | %s %s | %s" % (gname, len(sel), fmt(s["j_zs"]), fmt(s["j_zs_roof"]), fmt(s["j_zs_gemv"]),
                                                    fmt(s["j_cal"]), fmt(s["tok_zs"]), fmt(s["tok_cal"]),
                                                    fmt(s["j_tdp"])))
    b = summ.get("B200 capped runs (cap>=50%)")
    if b:
        print("B200 capped: roof J %.1f%% (%+.1f) tok/s %.1f%% | stretch J %.1f%% (%+.1f) tok/s %.1f%%" % (
            b["j_cal"]["mape"], b["j_cal"]["bias"], b["tok_cal"]["mape"], b["j_cal_stretch"]["mape"],
            b["j_cal_stretch"]["bias"], b["tok_cal_stretch"]["mape"]))
    for k in ("B200 Oct longctx (7B, 32B)", "B200 Sept eager", "H200 Sept eager"):
        if k in summ and summ[k]["j_sept"]:
            print("%s: settled/prior-KV J %.1f%% (%+.1f) vs Sept per-bin J %.1f%% (%+.1f)" % (
                k, summ[k]["j_cal"]["mape"], summ[k]["j_cal"]["bias"], summ[k]["j_sept"]["mape"],
                summ[k]["j_sept"]["bias"]))
    # prefill per prompt token
    for arm in ("prefill", "a100_prefill"):
        sel = [x for x in rows if x["arm"] == arm]
        if sel:
            jp = lambda x, k: x[k] * x["prompt_ratio"]
            print("%s per prompt token: meas %s | cal %s" % (arm, ["%.4f" % jp(x, "j_meas") for x in sel],
                                                             ["%.4f" % jp(x, "j_cal") for x in sel]))
    out = dict(calibration=cal_out, transfer=transfer, tp_a100=tp_res, h200_refit=dict(e_kvbyte=e_kv, e_gemm=e_g_h),
               summary=summ,
               notes=dict(zero_shot="A100/A6000 microbench + datasheet + idle only; uncapped time from the "
                                    "recommender prior (fitted on H200/B200 runs only); CUDA-graphs dtau from B200",
                          j_zs_gemv="zero-shot with e_w = GEMV duty-cycle J/B, P_s = vLLM idle, no P_0",
                          calibrated="zeroshot coefficients + 2 serving-fitted scales per GPU (A100/A6000)"))
    json.dump(out, open(os.path.join(HERE, "validation_v2.json"), "w"), indent=1)
    keys = []
    for x in rows:
        for k in x:
            if k != "_r" and k not in keys:
                keys.append(k)
    with open(os.path.join(HERE, "validation_v2_runs.csv"), "w") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for x in rows:
            wr.writerow({k: v for k, v in x.items() if k != "_r"})
    print("\nper-run A100/A6000 (meas | zero-shot | calibrated):")
    for x in rows:
        if x["gpu"] in ("A100-80-PCIe", "A6000"):
            print("  %-62s c=%3s tok/s %7.1f | %7.1f %s | %7.1f   J %7.4f | %7.4f | %7.4f  cap%%=%s" % (
                x["key"][5:], x["conc"], x["tok_s_meas"], x["tok_s_zs"], "C" if x["capped_zs"] else " ",
                x["tok_s_cal"], x["j_meas"], x["j_zs"], x["j_cal"],
                "%.2f" % x["cap_frac"] if x["cap_frac"] is not None else "-"))


if __name__ == "__main__":
    main()
