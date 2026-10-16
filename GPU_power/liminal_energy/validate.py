#!/usr/bin/env python3
"""
validate.py -- LIMINAL-E predictions vs MEASURED serving runs.

Measured sets (nothing here is re-measured):
  * logs/B200_prov (Oct 2026, NVML energy counter, smear-free): graphs, highconc, relog
    (eager), tp2, tp1ref, prefill.  J/token from logs/B200_prov/provisioning_summary.json.
  * Sept matched runs: balance_runs.csv valid rows (H200 logs/ragged + ragged_ladder,
    B200 logs/B200 + B200_32B + B200_large; GetPowerUsage window average), J/token as
    in balance_analysis.json.
Per-run prompt/gen lengths, measured prefix-cache fraction, time-weighted running batch
and mean iteration time come from recommender_backtest.load_run() (same as the backtest).

Writes liminal_energy/validation.json and liminal_energy/validation_runs.csv.
Run in the container (needs ~/models.py, python>=3.7).
"""
import csv
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

PROV = "logs/B200_prov/provisioning_summary.json"


def arm_of(key):
    return key.split("/")[0]


def load_sets():
    runs = []
    prov = json.load(open(PROV))
    for key, s in sorted(prov["runs"].items()):
        r = rb.load_run(s["run_dir"])
        if r is None:
            print("[skip]", key)
            continue
        r.update(set="B200_prov", arm=arm_of(key), key=key, tp=int(s.get("tp") or 1),
                 eager=bool(s.get("eager")), j_meas=s["j_tok"], p_meas=s["p_avg"],
                 tok_s_meas=s["tok_s"], cap_frac=s.get("cap_frac") or 0.0,
                 idle_meas=s.get("idle"))
        runs.append(r)
    for row in csv.DictReader(open("balance_runs.csv")):
        if row["valid"] != "True":
            continue
        r = rb.load_run(row["run_dir"])
        if r is None:
            print("[skip]", row["run_dir"])
            continue
        r.update(set="Sept", arm="sept_" + row["gpu"], key=row["run_dir"], tp=1, eager=True,
                 j_meas=float(row["j_tok"]), p_meas=float(row["p_avg_w"]),
                 tok_s_meas=float(row["tok_s"]), cap_frac=None, idle_meas=float(row["idle_w"]))
        runs.append(r)
    for r in runs:
        r["m"] = le.model(r["model"])
    return runs


def work_at(r, n):
    w = le.serving_work(r["m"], n, r["P"], r["G"], r["cf"])
    w["A"] = le.allreduce_bytes(r["m"], w["T"], r["tp"]) * r["tp"]
    w["tokens_out"] = n
    return w


def predict(r, **kw):
    g = le.gpu(r["gpu"])
    return le.serving(g, r["m"], r["P"], r["G"], conc=r["conc"], cached_frac=r["cf"], tp=r["tp"], **kw)


def energy_only(r, profile="canonical", cap_regime=True):
    """Energy model at the MEASURED operating point (running batch, step time): isolates
    the energy coefficients from the time model.  Returns predicted J/token."""
    g = le.gpu(r["gpu"])
    w = work_at(r, r["n_meas"])
    e_w, e_kv, e_g, p_s, _ = le.coefficients(g, profile)
    E_dyn = e_w * w["Wb"] + e_kv * w["KVb"] + e_g * (w["Fg"] + w["Fa_pf"]) + e_w * w["A"]
    budget = r["tp"] * (g["p_cap"] - p_s)
    if E_dyn / r["t_iter"] > budget:
        E_dyn = budget * r["t_iter"]           # measured time already includes throttling
    power = r["tp"] * p_s + E_dyn / r["t_iter"]
    return power / r["tok_s_meas"], power


def time_at_nmeas(r, time_model="eager", dtau=None):
    g = le.gpu(r["gpu"])
    w = work_at(r, r["n_meas"])
    if dtau is not None:
        saved = dict(le.GRAPHS_DTAU)
        le.GRAPHS_DTAU[g["name"]] = dtau
        t, _ = le.step_time(g, r["m"], w, r["tp"], "graphs")
        le.GRAPHS_DTAU.clear()
        le.GRAPHS_DTAU.update(saved)
        return t
    t, _ = le.step_time(g, r["m"], w, r["tp"], time_model)
    return t


def t0_variant(r, dt0):
    """Alternative graphs hypothesis: a per-STEP saving dt0 instead of per-layer."""
    g = le.gpu(r["gpu"])
    w = work_at(r, r["n_meas"])
    tp = dict(rg.TIME_CENTRAL[g["time_key"]])
    tp["t0"] = max(0.0, tp["t0"] - dt0)
    d = rg.central_draw(g)
    t, _ = rg.iter_time(g, tp, w, r["m"].L, rg.is_moe(r["m"]), d["mfu"], r["tp"],
                        d["tau_ar"] if r["tp"] > 1 else 0.0)
    return t * (d["tp_pen"] if r["tp"] > 1 else 1.0)


def fit_scalar(f, runs, lo, hi):
    """Golden-section search on the mean squared log error of step time."""
    def loss(x):
        return sum(math.log(f(r, x) / r["t_iter"]) ** 2 for r in runs) / len(runs)
    a, b = lo, hi
    gr = (math.sqrt(5) - 1) / 2
    for _ in range(80):
        c, d = b - gr * (b - a), a + gr * (b - a)
        if loss(c) < loss(d):
            b = d
        else:
            a = c
    x = 0.5 * (a + b)
    return x, loss(x)


def ape(p, m):
    return 100.0 * abs(p - m) / m


def mape(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def bias(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main():
    runs = load_sets()
    print("runs:", len(runs))

    # ---- 1. CUDA-graphs time correction: per-layer vs per-step, fitted on 7B ------------
    gr_all = [r for r in runs if r["set"] == "B200_prov" and not r["eager"] and r["arm"] in
              ("graphs", "highconc", "tp1ref")]
    fit7 = [r for r in gr_all if "7B" in r["model"] and r["conc"] <= 64]
    test = [r for r in gr_all if r not in fit7]
    dtau, l_tau = fit_scalar(lambda r, x: time_at_nmeas(r, dtau=x), fit7, 0.0, 78e-6)
    dt0, l_t0 = fit_scalar(t0_variant, fit7, 0.0, 1.419e-3)
    le.GRAPHS_DTAU["B200"] = dtau
    gf = dict(fit_runs=[r["key"] for r in fit7], dtau_us=dtau * 1e6, dt0_ms=dt0 * 1e3,
              fit_rmsle_tau=math.sqrt(l_tau), fit_rmsle_t0=math.sqrt(l_t0), test=[])
    for r in test:
        te = time_at_nmeas(r, "eager")
        tg = time_at_nmeas(r, dtau=dtau)
        t0v = t0_variant(r, dt0)
        gf["test"].append(dict(key=r["key"], L=r["m"].L, n_meas=r["n_meas"], t_meas_ms=r["t_iter"] * 1e3,
                               t_eager_ms=te * 1e3, t_graphs_tau_ms=tg * 1e3, t_graphs_t0_ms=t0v * 1e3))
    for k in ("t_eager_ms", "t_graphs_tau_ms", "t_graphs_t0_ms"):
        gf["test_mape_" + k] = mape([ape(x[k], x["t_meas_ms"]) for x in gf["test"]])
        lowc = [x for x in gf["test"] if x["n_meas"] <= 70]
        gf["test_mape_c<=64_" + k] = mape([ape(x[k], x["t_meas_ms"]) for x in lowc])
    # Sept-eager vs graphs step time at c=1, by depth (the per-layer signature)
    sept_c1 = {(r["model"]): r["t_iter"] for r in runs if r["set"] == "Sept" and r["gpu"] == "B200"
               and r["conc"] == 1 and r["task"] == "alpaca"}
    gr_c1 = {(r["model"]): r["t_iter"] for r in gr_all if r["conc"] == 1 and r["task"] == "alpaca"}
    gf["sept_eager_minus_graphs_c1"] = {
        mdl: dict(L=le.model(mdl).L, dt_ms=(sept_c1[mdl] - gr_c1[mdl]) * 1e3,
                  dt_per_layer_us=(sept_c1[mdl] - gr_c1[mdl]) / le.model(mdl).L * 1e6)
        for mdl in gr_c1 if mdl in sept_c1}
    print("\n[graphs] dtau %.1f us/layer (rmsle %.3f) | dt0 %.2f ms/step (rmsle %.3f)" % (
        dtau * 1e6, gf["fit_rmsle_tau"], dt0 * 1e3, gf["fit_rmsle_t0"]))
    for x in gf["test"]:
        print("   %-48s L=%2d n=%5.1f meas %6.2f  eager %6.2f  per-layer %6.2f  per-step %6.2f ms" % (
            x["key"], x["L"], x["n_meas"], x["t_meas_ms"], x["t_eager_ms"], x["t_graphs_tau_ms"],
            x["t_graphs_t0_ms"]))
    print("   held-out time MAPE: eager %.1f%%  per-layer %.1f%%  per-step %.1f%%" % (
        gf["test_mape_t_eager_ms"], gf["test_mape_t_graphs_tau_ms"], gf["test_mape_t_graphs_t0_ms"]))
    print("   c<=64 held-out: eager %.1f%%  per-layer %.1f%%  per-step %.1f%%" % (
        gf["test_mape_c<=64_t_eager_ms"], gf["test_mape_c<=64_t_graphs_tau_ms"],
        gf["test_mape_c<=64_t_graphs_t0_ms"]))
    print("   Sept eager - graphs at c=1:", {k: "%.2f ms (%.1f us/layer, L=%d)" % (
        v["dt_ms"], v["dt_per_layer_us"], v["L"]) for k, v in gf["sept_eager_minus_graphs_c1"].items()})

    # ---- 1b. TP=2 synchronization: per-layer all-reduce latency, fit on 72B, test 32B ---
    tp2 = [r for r in runs if r["arm"] == "tp2"]
    fit72 = [r for r in tp2 if "72B" in r["model"]]
    test32 = [r for r in tp2 if r not in fit72]

    def t_tp(r, tau_ar, pen=1.0):
        saved = dict(le.TP_SYNC)
        le.TP_SYNC["B200"] = dict(tau_ar=tau_ar, pen=pen)
        t = time_at_nmeas(r, "graphs")
        le.TP_SYNC.clear()
        le.TP_SYNC.update(saved)
        return t
    tau_ar, l_ar = fit_scalar(lambda r, x: t_tp(r, x), fit72, 0.0, 100e-6)
    le.TP_SYNC["B200"] = dict(tau_ar=tau_ar, pen=1.0)
    tpf = dict(fit_runs=[r["key"] for r in fit72], tau_ar_us=tau_ar * 1e6, fit_rmsle=math.sqrt(l_ar),
               test=[dict(key=r["key"], t_meas_ms=r["t_iter"] * 1e3, t_fit_ms=t_tp(r, tau_ar) * 1e3,
                          t_prior_ms=t_tp(r, 25e-6, 1.15) * 1e3, t_nosync_ms=t_tp(r, 0.0) * 1e3)
                     for r in tp2])
    for k in ("t_fit_ms", "t_prior_ms", "t_nosync_ms"):
        tpf["test32_mape_" + k] = mape([ape(x[k], x["t_meas_ms"]) for x in tpf["test"] if "32B" in x["key"]])
    print("\n[tp2] tau_ar %.1f us/layer (rmsle %.3f on 72B); 32B held-out time MAPE: fit %.1f%%, "
          "prior(25us,x1.15) %.1f%%, no sync %.1f%%" % (tau_ar * 1e6, tpf["fit_rmsle"], tpf["test32_mape_t_fit_ms"],
                                                       tpf["test32_mape_t_prior_ms"], tpf["test32_mape_t_nosync_ms"]))
    for x in tpf["test"]:
        print("   %-40s meas %6.2f fit %6.2f prior %6.2f nosync %6.2f" % (
            x["key"], x["t_meas_ms"], x["t_fit_ms"], x["t_prior_ms"], x["t_nosync_ms"]))

    # ---- 2. per-run predictions --------------------------------------------------------
    rows = []
    for r in runs:
        tm = "eager" if r["eager"] else "graphs"
        p_e = predict(r, time_model="eager")                          # recommender time (eager fit)
        p_m = predict(r, time_model=tm)                               # matched execution mode
        p_nc = predict(r, time_model=tm, cap_regime=True)             # cap with DVFS coefficients
        p_lt = predict(r, time_model="liminal")                       # LIMINAL time + our energy
        p_tdp = predict(r, time_model="liminal", power_model="tdp")   # LIMINAL as published
        p_tdp_ours = predict(r, time_model=tm, power_model="tdp")     # our time, TDP power
        prof = {pf: predict(r, time_model=tm, profile=pf)["j_per_token"]
                for pf in ("kv_high", "kv_low", "runlevel")}
        eo, eo_p = energy_only(r)
        eo_prof = {pf: energy_only(r, pf)[0] for pf in ("kv_high", "kv_low", "runlevel")}
        w = work_at(r, r["n_meas"])
        row = dict(set=r["set"], arm=r["arm"], key=r["key"], gpu=r["gpu"], model=r["model"].split("/")[-1],
                   task=r["task"], conc=r["conc"], tp=r["tp"], eager=r["eager"], P=r["P"], G=r["G"],
                   cf=r["cf"], n_meas=r["n_meas"], cap_frac=r["cap_frac"],
                   t_meas=r["t_iter"], j_meas=r["j_meas"], p_meas=r["p_meas"], tok_s_meas=r["tok_s_meas"],
                   ai_meas=le.intensity(w), flops_meas=w["F"] / r["t_iter"],
                   bytes_meas=(w["Wb"] + w["KVb"]) / r["t_iter"],
                   j_eager=p_e["j_per_token"], j_mode=p_m["j_per_token"], j_dvfs=p_nc["j_per_token"],
                   j_liminal_time=p_lt["j_per_token"], j_liminal_tdp=p_tdp["j_per_token"],
                   j_tdp_ourtime=p_tdp_ours["j_per_token"],
                   j_energy_only=eo, p_energy_only=eo_p,
                   t_mode=p_m["t"], t_eager=p_e["t"], t_liminal=p_lt["t"],
                   p_mode=p_m["power"], tok_s_mode=p_m["tok_s"], capped_pred=p_m["capped"],
                   bound_pred=p_m["bound"],
                   share_static=p_m["e_static"] / p_m["energy"],
                   share_kv=p_m["split"]["kv"] / p_m["energy"],
                   share_compute=p_m["split"]["compute"] / p_m["energy"])
        for pf, v in prof.items():
            row["j_" + pf] = v
        for pf, v in eo_prof.items():
            row["j_energy_only_" + pf] = v
        if r["arm"] == "prefill":
            # measured J per PROMPT token (provisioning_summary.json prefill section)
            pre = {d["key"]: d for d in json.load(open(PROV))["prefill"]}[r["key"]]
            row["j_prompt_meas"] = pre["j_per_prompt_tok"]
            ratio = r["G"] / r["P"]
            row["j_prompt_mode"] = p_m["j_per_token"] * ratio
            row["j_prompt_dvfs"] = p_nc["j_per_token"] * ratio
            row["j_prompt_energy_only"] = eo * ratio
        rows.append(row)

    # ---- 3. summaries ---------------------------------------------------------------
    groups = {
        "B200_prov graphs (c<=64)": lambda x: x["arm"] == "graphs",
        "B200_prov highconc (c 128-512)": lambda x: x["arm"] == "highconc",
        "B200_prov relog (eager)": lambda x: x["arm"] == "relog",
        "B200_prov tp1ref (72B)": lambda x: x["arm"] == "tp1ref",
        "B200_prov tp2": lambda x: x["arm"] == "tp2",
        "B200_prov prefill (per gen token)": lambda x: x["arm"] == "prefill",
        "Sept H200 (eager)": lambda x: x["arm"] == "sept_H200",
        "Sept B200 (eager)": lambda x: x["arm"] == "sept_B200",
    }
    preds = ["j_mode", "j_eager", "j_energy_only", "j_kv_high", "j_kv_low", "j_runlevel",
             "j_dvfs", "j_liminal_time", "j_liminal_tdp", "j_tdp_ourtime"]
    summ = {}
    for gname, f in groups.items():
        sel = [x for x in rows if f(x)]
        if not sel:
            continue
        s = dict(n=len(sel))
        for p in preds:
            s[p + "_mape"] = mape([ape(x[p], x["j_meas"]) for x in sel])
            s[p + "_bias"] = bias([100.0 * (x[p] / x["j_meas"] - 1) for x in sel])
        s["t_mode_mape"] = mape([ape(x["t_mode"], x["t_meas"]) for x in sel])
        s["t_mode_bias"] = bias([100.0 * (x["t_mode"] / x["t_meas"] - 1) for x in sel])
        s["t_liminal_mape"] = mape([ape(x["t_liminal"], x["t_meas"]) for x in sel])
        s["t_eager_mape"] = mape([ape(x["t_eager"], x["t_meas"]) for x in sel])
        s["t_liminal_bias"] = bias([100.0 * (x["t_liminal"] / x["t_meas"] - 1) for x in sel])
        s["p_mode_mape"] = mape([ape(x["p_mode"], x["p_meas"]) for x in sel])
        s["p_energy_only_mape"] = mape([ape(x["p_energy_only"], x["p_meas"]) for x in sel])
        summ[gname] = s
    pre = [x for x in rows if x["arm"] == "prefill"]
    summ["B200_prov prefill (per PROMPT token)"] = dict(
        n=len(pre),
        mode_mape=mape([ape(x["j_prompt_mode"], x["j_prompt_meas"]) for x in pre]),
        mode_bias=bias([100 * (x["j_prompt_mode"] / x["j_prompt_meas"] - 1) for x in pre]),
        dvfs_mape=mape([ape(x["j_prompt_dvfs"], x["j_prompt_meas"]) for x in pre]),
        dvfs_bias=bias([100 * (x["j_prompt_dvfs"] / x["j_prompt_meas"] - 1) for x in pre]),
        energy_only_mape=mape([ape(x["j_prompt_energy_only"], x["j_prompt_meas"]) for x in pre]))

    # decode-only overall (excludes prefill-dominated runs)
    dec = [x for x in rows if x["arm"] != "prefill"]
    summ["ALL decode-serving runs"] = dict(
        n=len(dec), **{p + "_mape": mape([ape(x[p], x["j_meas"]) for x in dec]) for p in preds})

    # TP premium: predicted vs measured J/tok(TP2)/J/tok(TP1)
    tp_rows = []
    by = {(x["arm"], x["model"], x["conc"]): x for x in rows}
    for x in rows:
        if x["arm"] != "tp2":
            continue
        ref = by.get(("tp1ref", x["model"], x["conc"])) or by.get(("graphs", x["model"], x["conc"])) \
            or by.get(("highconc", x["model"], x["conc"]))
        if ref is None:
            continue
        tp_rows.append(dict(model=x["model"], conc=x["conc"], meas=x["j_meas"] / ref["j_meas"],
                            pred=x["j_mode"] / ref["j_mode"],
                            speedup_meas=x["tok_s_meas"] / ref["tok_s_meas"],
                            speedup_pred=x["tok_s_mode"] / ref["tok_s_mode"]))
    summ["tp_premium"] = tp_rows

    # matched B200/H200 J/tok ratios (Sept): measured vs predicted, and LIMINAL-TDP
    sept = {(x["gpu"], x["model"], x["task"], x["conc"]): x for x in rows if x["set"] == "Sept"}
    mt = []
    for (gp, mdl, task, c), xb in sept.items():
        if gp != "B200":
            continue
        xh = sept.get(("H200", mdl, task, c))
        if xh is None:
            continue
        mt.append(dict(model=mdl, task=task, conc=c, meas=xb["j_meas"] / xh["j_meas"],
                       pred=xb["j_mode"] / xh["j_mode"],
                       pred_liminal_tdp=xb["j_liminal_tdp"] / xh["j_liminal_tdp"]))
    summ["matched_ratio"] = dict(
        n=len(mt), rows=mt,
        mape=mape([ape(x["pred"], x["meas"]) for x in mt]),
        mape_liminal_tdp=mape([ape(x["pred_liminal_tdp"], x["meas"]) for x in mt]),
        sign_agree=sum((x["pred"] > 1) == (x["meas"] > 1) for x in mt),
        sign_agree_liminal_tdp=sum((x["pred_liminal_tdp"] > 1) == (x["meas"] > 1) for x in mt))

    # capped runs: did the model predict the cap?
    capr = [x for x in rows if x["set"] == "B200_prov" and x["cap_frac"] is not None]
    summ["cap_detection"] = [dict(key=x["key"], cap_frac_meas=x["cap_frac"], capped_pred=x["capped_pred"],
                                  p_meas=x["p_meas"], p_pred=x["p_mode"])
                             for x in capr if x["cap_frac"] > 0.2 or x["capped_pred"]]

    out = dict(graphs_fit=gf, tp_fit=tpf, summary=summ,
               notes=dict(
                   j_mode="LIMINAL-E: canonical coefficients (gpu_coefficients.json), time model matched "
                          "to execution mode (eager fit; graphs = eager fit with per-layer floor - dtau)",
                   j_eager="eager-fitted time model for every run (= recommender, plus DVFS cap regime)",
                   j_energy_only="energy coefficients at the MEASURED running batch and step time",
                   j_dvfs="capped steps stretched with the DVFS capped-regime J/B, J/FLOP (B200 steady microbench)",
                   j_liminal_time="LIMINAL v2 time (peak BW, 114 us/layer) + our energy model",
                   j_liminal_tdp="LIMINAL as published: LIMINAL time, power = TDP",
                   j_tdp_ourtime="our time model, power = TDP"))
    json.dump(out, open(os.path.join(HERE, "validation.json"), "w"), indent=1)
    keys = list(rows[0].keys())
    for x in rows:
        for k in x:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(HERE, "validation_runs.csv"), "w") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for x in rows:
            wr.writerow(x)

    # ---- 4. print ------------------------------------------------------------------
    print("\nJ/token MAPE %% (bias %%)")
    hdr = ["j_mode", "j_eager", "j_energy_only", "j_kv_high", "j_kv_low", "j_dvfs",
           "j_liminal_time", "j_liminal_tdp"]
    print("%-36s %3s " % ("group", "n") + " ".join("%15s" % h[2:] for h in hdr) + "   t_mape(bias)  P_mape")
    for gname, s in summ.items():
        if "j_mode_mape" not in s:
            continue
        print("%-36s %3d " % (gname, s["n"]) + " ".join(
            "%7.1f(%+6.1f)" % (s[h + "_mape"], s.get(h + "_bias") or 0) for h in hdr) +
            "   %5.1f(%+5.1f)  %5.1f  t_lim %5.1f(%+5.1f)" % (
                s.get("t_mode_mape") or 0, s.get("t_mode_bias") or 0, s.get("p_mode_mape") or 0,
                s.get("t_liminal_mape") or 0, s.get("t_liminal_bias") or 0))
    print("\nprefill per prompt token:", summ["B200_prov prefill (per PROMPT token)"])
    print("\nTP premium (J/tok TP2 / TP1): ")
    for x in tp_rows:
        print("   %-22s c=%3d meas %.2f pred %.2f | speedup meas %.2f pred %.2f" % (
            x["model"], x["conc"], x["meas"], x["pred"], x["speedup_meas"], x["speedup_pred"]))
    mr = summ["matched_ratio"]
    print("\nmatched B200/H200 ratio: n=%d MAPE %.1f%% (LIMINAL-TDP %.1f%%); sign agree %d/%d (TDP %d/%d)" % (
        mr["n"], mr["mape"], mr["mape_liminal_tdp"], mr["sign_agree"], mr["n"],
        mr["sign_agree_liminal_tdp"], mr["n"]))
    print("\ncap detection:")
    for x in summ["cap_detection"]:
        print("   ", x)
    print("\nper-run (B200_prov):")
    for x in rows:
        if x["set"] != "B200_prov":
            continue
        print("   %-46s n=%5.1f t %6.2f/%6.2f ms  P %4.0f/%4.0f W  J %8.4f / mode %8.4f eo %8.4f lim %8.4f  %s%s" % (
            x["key"], x["n_meas"], x["t_meas"] * 1e3, x["t_mode"] * 1e3, x["p_meas"], x["p_mode"],
            x["j_meas"], x["j_mode"], x["j_energy_only"], x["j_liminal_tdp"], x["bound_pred"],
            "/CAP" if x["capped_pred"] else ""))


if __name__ == "__main__":
    main()
