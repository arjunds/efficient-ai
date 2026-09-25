#!/usr/bin/env python3
"""
recommender_backtest.py — fit the realized-utilization (iteration-time) model and
BACKTEST recommend_gpu.py against every measured serving run.

For each run the recommender is given ONLY: model id, GPU, the workload's mean
prompt length P and generation length G (measured from the request stream), and
the offered load (closed-loop concurrency c, or Poisson rate λ). It predicts
tok/s, average power and J/token, compared with results.json.

Modes (run in the container; numpy/scipy needed for --fit only):
  python3 recommender_backtest.py --fit        # fit time model, patch TIME_FIT in recommend_gpu.py
  python3 recommender_backtest.py              # backtest table (v2 in-sample, v2 LOMO, v1 baseline)
  python3 recommender_backtest.py --figure     # + plots_proposal/fig8_recommender_v2.png (needs matplotlib)
  python3 recommender_backtest.py --winmap     # print H200-vs-B200 win-map summary

Run discovery is automatic: logs/ragged*, logs/B200*, and any logs/*A5000* dir.
Runs whose binned window covers <95% of the measurement window (crash-truncated
B200 sharegpt runs) are excluded and listed.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import recommend_gpu as rg  # noqa: E402

ROOTS = ["logs/ragged", "logs/ragged_ladder", "logs/ragged_moe", "logs/B200", "logs/B200_32B",
         "logs/B200_large", "logs/ragged_a100"]
FORMS = {"additive(k=1)": 1.0, "soft(k=3)": 3.0, "max(k=50)": 50.0}
# central form per GPU is chosen by leave-one-model-out t_iter error (see do_fit)
FIT_GPUS_MIN_MODELS = 2          # a GPU needs >=2 dense models to get its own time fit


def gpu_key(name):
    n = name.upper()
    if "H200" in n: return "H200"
    if "B200" in n: return "B200"
    if "A100" in n and "PCIE" in n.replace(" ", ""): return "A100-80-PCIe"
    if "A100" in n: return "A100-80-SXM"
    if "A5000" in n: return "A5000"
    if "L40S" in n: return "L40S"
    if "H100" in n: return "H100-SXM"
    return name


def group_of(rd, gpu):
    if "ragged_ladder" in rd: return gpu + " ladder"
    if "ragged_moe" in rd: return gpu + " MoE"
    if "B200_32B" in rd: return "B200 32B"
    if "B200_large" in rd: return "B200 72B"
    if gpu == "B200": return "B200 7B"
    if gpu == "H200": return "H200 dense-7B"
    return gpu


# ---------------------------------------------------------------------------
def load_run(rd):
    try:
        meta = json.load(open(os.path.join(rd, "run_meta.json")))
        res = json.load(open(os.path.join(rd, "results.json")))
    except Exception:
        return None
    t0, t1 = meta.get("window_wall_t0"), meta.get("window_wall_t1")
    pref = dec = busy = nr = 0.0
    for r in csv.DictReader(open(os.path.join(rd, "iter_log.csv"))):
        try:
            a, b = float(r["t_start"]), float(r["t_end"])
        except (TypeError, ValueError):
            continue
        if t0 is not None and (b < t0 or a > t1):
            continue
        pref += float(r.get("prefill_tokens") or 0)
        dec += float(r.get("decode_tokens") or 0)
        busy += b - a
        nr += float(r.get("n_running") or 0) * (b - a)
    dt = nit = p_log = p_comp = 0.0
    has_comp = False
    for r in csv.DictReader(open(os.path.join(rd, "binned_table.csv"))):
        dt += float(r["dt_s"]); nit += float(r.get("n_iters") or 0)
        if r.get("prefill_tokens_computed") not in (None, ""):
            has_comp = True
            p_log += float(r.get("prefill_tokens") or 0)
            p_comp += float(r["prefill_tokens_computed"])
    # measured prefix-cache hit fraction of prompt tokens (controls/prefix_cache_refit.py
    # columns); runs without those columns (MoE, A100) -> 0 and flagged
    cf = (1.0 - p_comp / p_log) if (has_comp and p_log > 0) else 0.0
    win = res.get("window_duration_s") or (t1 - t0)
    comp = res.get("completed_requests") or 0
    gen = res.get("generated_tokens_window") or 0
    if not comp or not dec or not nit:
        return None
    G = gen / float(comp)
    P = pref / dec * G
    rate = meta.get("arrival_rate") or 0.0
    # Poisson runs: use the REALIZED request rate (completed/window). The harness's
    # client delivers only ~0.83-0.95x the nominal rate, so the nominal λ is not the
    # load the GPU actually saw. (tok/s for these runs is then ~input-determined;
    # power and J/token still test the model via the Little's-law batch size.)
    rate_nominal = rate
    if rate:
        rate = comp / float(win)
    conc = meta.get("concurrency_or_rate")
    gk = gpu_key(meta.get("gpu_name", ""))
    return dict(rd=os.path.relpath(rd, HERE), gpu=gk, group=group_of(rd, gk),
                model=meta["model"], task=meta.get("task"),
                conc=None if rate else int(conc), rate=float(rate) if rate else None,
                rate_nominal=rate_nominal,
                P=P, G=G, n_meas=nr / busy if busy else None, t_iter=dt / nit,
                cf=max(0.0, cf), cf_known=has_comp,
                cover=dt / win if win else 0, idle=meta.get("idle_power_w"),
                tok_s=res["aggregate_tokens_per_sec"], power=res["avg_power_window_w"],
                j_tok=res["avg_power_window_w"] / res["aggregate_tokens_per_sec"])


def excluded_gpus():
    try:
        return set((json.load(open(rg.COEFF_JSON)).get("_excluded") or {}).keys())
    except Exception:
        return set()


def discover(roots=None):
    roots = list(roots or ROOTS)
    excl = excluded_gpus()
    if "A5000" not in excl:      # auto-discover A5000 runs unless flagged invalid
        roots += sorted(d for d in glob.glob(os.path.join(HERE, "logs", "*"))
                        if re.search("a5000", d, re.I) and os.path.isdir(d))
    runs, dropped = [], []
    for root in roots:
        base = root if os.path.isabs(root) else os.path.join(HERE, root)
        for rd in sorted(glob.glob(os.path.join(base, "*", "*"))):
            if not os.path.isdir(rd) or os.path.basename(rd) == "idle":
                continue
            r = load_run(rd)
            if r is None or r["gpu"] in excl:
                continue
            (runs if r["cover"] >= 0.95 else dropped).append(r)
    models = {}
    for r in runs + dropped:
        if r["model"] not in models:
            models[r["model"]] = rg.resolve_model(r["model"])
        r["m"] = models[r["model"]]
    return runs, dropped


# ---------------------------------------------------------------------------
# time-model fitting
# ---------------------------------------------------------------------------
def _tp(p, k, tau_moe=None):
    return dict(t0=p[0] * 1e-3, eta=p[1], eta_kv=p[2], mu=p[3], tau=p[4] * 1e-6, k=k,
                tau_moe=tau_moe)


def _pred_t(g, tp, r, n=None):
    w = rg.serving_iter_work(r["m"], n if n is not None else r["n_meas"], r["P"], r["G"], r["cf"])
    # time model only (no cap, no compute ceiling -> the fitted domain)
    t, _ = rg.iter_time(g, tp, w, r["m"].L, rg.is_moe(r["m"]), mfu_pf=1e9)
    return t


def fit_form(g, runs, k, fixed=None):
    """Fit p=[t0_ms, eta, eta_kv, mu, tau_us] for softmax exponent k. `fixed` maps
    parameter index -> value (used for single-model GPUs)."""
    import numpy as np
    from scipy.optimize import least_squares
    fixed = fixed or {}
    free = [i for i in range(5) if i not in fixed]
    def full(q):
        p = [0.0] * 5
        for i, v in fixed.items():
            p[i] = v
        for i, v in zip(free, q):
            p[i] = v
        return p
    y = np.array([r["t_iter"] for r in runs])
    f = lambda q: np.array([_pred_t(g, _tp(full(q), k), r) for r in runs]) / y - 1.0
    lo, hi = [0, 0.05, 0.05, 0.05, 0], [20, 0.95, 1.0, 5.0, 1000]
    best = None
    for p0 in ([1.5, 0.7, 0.3, 0.7, 100.0], [3.0, 0.8, 0.5, 1.5, 30.0], [0.5, 0.6, 0.2, 0.5, 150.0]):
        res = least_squares(f, [p0[i] for i in free], bounds=([lo[i] for i in free], [hi[i] for i in free]))
        if best is None or res.cost < best.cost:
            best = res
    return _tp([float(v) for v in full(best.x)], k)


def fit_tau_moe(g, tp, moe_runs):
    import numpy as np
    from scipy.optimize import minimize_scalar
    if not moe_runs:
        return None
    def loss(tau_us):
        q = dict(tp, tau_moe=tau_us * 1e-6)
        return sum((_pred_t(g, q, r) / r["t_iter"] - 1) ** 2 for r in moe_runs)
    return minimize_scalar(loss, bounds=(1, 3000), method="bounded").x * 1e-6


def do_fit(runs):
    gpus = rg.load_gpus()
    out, single = {}, []
    for gk in sorted(set(r["gpu"] for r in runs)):
        if gk.startswith("A100"):       # cap-throttled: timing reflects the cap, not the GPU
            continue
        dense = [r for r in runs if r["gpu"] == gk and not rg.is_moe(r["m"])]
        moe = [r for r in runs if r["gpu"] == gk and rg.is_moe(r["m"])]
        models = sorted(set(r["model"] for r in dense))
        if len(models) < FIT_GPUS_MIN_MODELS:
            single.append((gk, dense, moe))
            continue
        g = gpus[gk]
        ens, cand = [], {}
        for fname, k in FORMS.items():
            tp = fit_form(g, dense, k)
            tp["tau_moe"] = fit_tau_moe(g, tp, moe)
            tp["form"] = fname; tp["held_out"] = None
            ens.append(tp)
            lomo, errs = {}, []
            for mm in models:
                q = fit_form(g, [r for r in dense if r["model"] != mm], k)
                q["tau_moe"] = fit_tau_moe(g, q, moe)
                q["form"] = fname; q["held_out"] = mm
                ens.append(q); lomo[mm] = q
                errs += [abs(_pred_t(g, q, r) / r["t_iter"] - 1) for r in dense if r["model"] == mm]
            ins = [abs(_pred_t(g, tp, r) / r["t_iter"] - 1) for r in dense]
            cand[fname] = (100 * sum(errs) / len(errs), tp, lomo, 100 * sum(ins) / len(ins))
            print("      form %-14s t0=%.2fms tau=%3.0fus eta=%.2f eta_kv=%.2f mu=%.2f  "
                  "t_iter MAPE in-sample %.1f%%  LOMO %.1f%%" % (
                      fname, tp["t0"] * 1e3, tp["tau"] * 1e6, tp["eta"], tp["eta_kv"], tp["mu"],
                      cand[fname][3], cand[fname][0]))
        # central form chosen by leave-one-model-out error (model selection by CV)
        best = min(cand, key=lambda f: cand[f][0])
        central, lomo = cand[best][1], cand[best][2]
        central["lomo_mape_t"] = cand[best][0]
        out[gk] = dict(central=central, ensemble=ens, lomo=lomo)
        print("[fit] %s central = %s (lowest LOMO): t0=%.2fms tau=%.0fus eta=%.2f eta_kv=%.2f "
              "mu=%.2f tau_moe=%s" % (gk, best, central["t0"] * 1e3, central["tau"] * 1e6,
                                      central["eta"], central["eta_kv"], central["mu"],
                                      "%.0fus" % (central["tau_moe"] * 1e6) if central["tau_moe"] else "n/a"))
    # GPUs with ONE dense model (e.g. a first A5000 sweep): with a single layer size,
    # tau / eta_kv / mu are not separable from t0 and eta, so fix them at the median of
    # the measured GPUs' centrals and fit only t0 and eta (for each form).
    for gk, dense, moe in single:
        if not dense or not out:
            continue
        med = lambda key: sorted(d["central"][key] for d in out.values())[len(out) // 2]
        fixed = {2: med("eta_kv"), 3: med("mu"), 4: med("tau") * 1e6}
        g = gpus[gk]; ens = []
        for fname, k in FORMS.items():
            tp = fit_form(g, dense, k, fixed=fixed)
            tp["tau_moe"] = None; tp["form"] = fname; tp["held_out"] = None
            tp["note"] = "single-model fit: tau, eta_kv, mu fixed at measured-GPU medians"
            ens.append(tp)
        errs = {q["form"]: sum(abs(_pred_t(g, q, r) / r["t_iter"] - 1) for r in dense) for q in ens}
        central = [q for q in ens if q["form"] == min(errs, key=errs.get)][0]
        if errs[central["form"]] / len(dense) > 0.15:
            print("[fit] %s single-model fit is poor (%.0f%% t_iter MAPE) — keeping the pooled prior"
                  % (gk, 100 * errs[central["form"]] / len(dense)))
            continue
        out[gk] = dict(central=central, ensemble=ens, lomo={})
        print("[fit] %s (single model) central=%s t0=%.2fms eta=%.2f (in-sample t_iter MAPE %.1f%%)" % (
            gk, central["form"], central["t0"] * 1e3, central["eta"],
            100 * errs[central["form"]] / len(dense)))
    # GPUs without MoE data: scale H200's MoE layer floor by the tau ratio (flagged)
    ref = out.get("H200")
    for gk, d in out.items():
        if d["central"]["tau_moe"] is None and ref and ref["central"]["tau_moe"]:
            for q in [d["central"]] + d["ensemble"] + list(d["lomo"].values()):
                q["tau_moe"] = ref["central"]["tau_moe"] * q["tau"] / ref["central"]["tau"]
                q["tau_moe_note"] = "scaled from H200 (no MoE run on this GPU)"
    return out


def patch_recommender(fit):
    """Write the fitted ensemble into recommend_gpu.py (TIME_FIT = ...)."""
    slim = {}
    for gk, d in fit.items():
        rnd = lambda q: {k: (round(v, 9) if isinstance(v, float) else v) for k, v in q.items()}
        slim[gk] = dict(central=rnd(d["central"]), ensemble=[rnd(q) for q in d["ensemble"]],
                        lomo={m: rnd(q) for m, q in d["lomo"].items()})
    path = os.path.join(HERE, "recommend_gpu.py")
    src = open(path).read()
    new = "TIME_FIT = " + json.dumps(slim, indent=None, sort_keys=True).replace("null", "None") + \
          "  # generated by recommender_backtest.py --fit"
    src = re.sub(r"^TIME_FIT = .*$", lambda _m: new, src, count=1, flags=re.M)
    open(path, "w").write(src)
    print("[fit] patched TIME_FIT in recommend_gpu.py (%d GPUs)" % len(slim))


# ---------------------------------------------------------------------------
# v1 baseline (the old datasheet recommender), reproduced for comparison
# ---------------------------------------------------------------------------
V1_H200 = dict(e_w=1.077e-10, e_kv=3.38e-10, e_g=0.673e-12, bw=4.8e12, pk=9.9e14)


def v1_predict(gk, r, g):
    bw, pk = g["bw"], g["peak_flops"]
    s_b, s_f = V1_H200["bw"] / bw, V1_H200["pk"] / pk
    e_w, e_kv, e_g = V1_H200["e_w"] * s_b, V1_H200["e_kv"] * s_b, V1_H200["e_g"] * s_f
    n = r["conc"] if r["conc"] else max(1.0, r["n_meas"])   # v1 had no queueing model: generous
    w = rg.serving_iter_work(r["m"], n, r["P"], r["G"])
    t = max((w["Wb"] + w["KVb"]) / (bw * 0.7), w["Fg"] / (pk * 0.7))
    e = e_w * w["Wb"] + e_kv * w["KVb"] + e_g * w["Fg"]
    p_s = r["idle"] if r["idle"] else g["p_static"]
    if e / t + p_s > g["p_cap"]:
        t = e / (g["p_cap"] - p_s)
    tok_s = n / t
    if r["rate"]:
        tok_s = min(tok_s, r["rate"] * r["G"])
    p = e / t + p_s
    return dict(tok_s=tok_s, power=p, j_tok=p / tok_s)


# ---------------------------------------------------------------------------
OLD_FIT = None   # v2.0 TIME_FIT (pre prefix-cache fix), loaded with --old-fit


def backtest(runs):
    gpus = rg.load_gpus()
    gpus_old = rg.load_gpus(uncorrected=True)
    rows = []
    for r in runs:
        g = gpus.get(r["gpu"])
        if g is None:
            continue
        cf = r["cf"]
        sp = lambda gg, d: rg.predict_serving(gg, r["m"], r["P"], r["G"], r["conc"], r["rate"], d,
                                              cached_frac=cf)
        d_in = rg.central_draw(g)
        preds = dict(v2=sp(g, d_in))
        # LOMO: time model refit without this model (energy coefficients unchanged)
        fitd = (rg.TIME_FIT or {}).get(g["time_key"], {})
        lomo_tp = fitd.get("lomo", {}).get(r["model"])
        preds["v2_lomo"] = sp(g, dict(d_in, tp=lomo_tp)) if lomo_tp is not None else preds["v2"]
        if rg.is_moe(r["m"]):   # zero-shot MoE: dense layer floor, no MoE calibration
            preds["v2_lomo"] = sp(g, dict(d_in, tp=dict(d_in["tp"], tau_moe=None)))
        # v2.0 (committed 84fea1a): logged-prefill (uncorrected) coefficients, no cache term,
        # and its own time fit if --old-fit is given
        go = gpus_old[r["gpu"]]
        d_old = rg.central_draw(go)
        if OLD_FIT and go["time_key"] in OLD_FIT:
            d_old = dict(d_old, tp=OLD_FIT[go["time_key"]]["central"])
        preds["v2_0"] = rg.predict_serving(go, r["m"], r["P"], r["G"], r["conc"], r["rate"], d_old)
        preds["v1"] = v1_predict(r["gpu"], r, g)
        if not g["measured"]:   # truly zero-shot variant: pretend we only had the HBM3e band
            pr = rg.MEM_PRIOR["HBM3e"]
            d = dict(d_in, coef=(pr["e_wbyte"][0], pr["e_kvbyte"][0], d_in["coef"][2], d_in["coef"][3]))
            preds["v2_hbm3e"] = sp(g, d)
        rows.append(dict(r, preds=preds, conf=g["confidence"]))
    return rows


def mape(xs):
    return 100.0 * sum(xs) / len(xs) if xs else float("nan")


def summarize(rows, dropped):
    Q = [("tok_s", "tok/s"), ("power", "power"), ("j_tok", "J/tok")]
    order = ["H200 dense-7B", "H200 ladder", "H200 MoE", "B200 7B", "B200 32B", "B200 72B"]
    groups = order + sorted(set(r["group"] for r in rows) - set(order))
    print("\nBacktest: MAPE (%) of predicted vs measured, per run group.")
    print("  v2 = shipped recommender (time model fit on all runs of that GPU = in-sample for time)")
    print("  LOMO = time model refit WITHOUT that model (MoE: zero-shot dense layer floor)")
    print("  v1 = old datasheet recommender (1/BW-scaled coefficients, flat 70% of peak)")
    hdr = "%-16s%4s " % ("group", "n") + "".join("%9s%9s%8s |" % (q + " v2", "LOMO", "v1") for _, q in Q)
    print(hdr)
    table = []
    for grp in groups + ["ALL H200", "ALL B200"]:
        if grp.startswith("ALL"):
            sel = [r for r in rows if r["gpu"] == grp.split()[1]]
        else:
            sel = [r for r in rows if r["group"] == grp]
        if not sel:
            continue
        line = "%-16s%4d " % (grp, len(sel))
        rec = dict(group=grp, n=len(sel))
        for k, q in Q:
            vals = []
            for v in ("v2", "v2_lomo", "v1"):
                e = mape([abs(r["preds"][v][k] / r[k] - 1) for r in sel])
                vals.append(e); rec["%s_%s" % (k, v)] = e
            line += "%9.1f%9.1f%8.1f |" % tuple(vals)
        # signed bias of v2 J/tok
        rec["j_bias_v2"] = 100 * sum(r["preds"]["v2"]["j_tok"] / r["j_tok"] - 1 for r in sel) / len(sel)
        print(line + " bias(J/tok v2)=%+.1f%%" % rec["j_bias_v2"])
        table.append(rec)
    for grp in sorted(set(r["group"] for r in rows if "v2_hbm3e" in r["preds"])):
        sel = [r for r in rows if r["group"] == grp]
        print("%-16s%4d  zero-shot with HBM3e-band e_byte instead of its own tech prior: "
              "tok/s %.1f%%  power %.1f%%  J/tok %.1f%%  (J/tok bias %+.1f%%)" % (
                  grp, len(sel), *[mape([abs(r["preds"]["v2_hbm3e"][k] / r[k] - 1) for r in sel])
                                   for k in ("tok_s", "power", "j_tok")],
                  100 * sum(r["preds"]["v2_hbm3e"]["j_tok"] / r["j_tok"] - 1 for r in sel) / len(sel)))
    for kind in ("closed", "poisson"):
        sel = [r for r in rows if (r["rate"] is None) == (kind == "closed") and r["gpu"] in ("H200", "B200")]
        if sel:
            print("H200+B200 %-8s n=%3d  v2 MAPE tok/s %.1f%%  power %.1f%%  J/tok %.1f%%" % (
                kind, len(sel), *[mape([abs(r["preds"]["v2"][k] / r[k] - 1) for r in sel])
                                  for k in ("tok_s", "power", "j_tok")]))
    print("\nOld vs new (prefix-cache fix): J/tok and power MAPE, v2.0 (uncorrected coeffs, "
          "logged prefill FLOPs%s) -> v2.1 (corrected coeffs + measured cached_frac)" % (
              ", v2.0 time fit" if OLD_FIT else ", CURRENT time fit"))
    for grp in groups + ["ALL H200", "ALL B200"]:
        sel = [r for r in rows if (r["gpu"] == grp.split()[1] if grp.startswith("ALL") else r["group"] == grp)]
        if not sel:
            continue
        f = lambda v, k: mape([abs(r["preds"][v][k] / r[k] - 1) for r in sel])
        cfs = [r["cf"] for r in sel if r["cf_known"]]
        print("  %-16s n=%3d  cached_frac %s  J/tok %5.1f -> %5.1f   power %5.1f -> %5.1f   tok/s %5.1f -> %5.1f" % (
            grp, len(sel), ("%.2f-%.2f" % (min(cfs), max(cfs))) if cfs else "  n/a   ",
            f("v2_0", "j_tok"), f("v2", "j_tok"), f("v2_0", "power"), f("v2", "power"),
            f("v2_0", "tok_s"), f("v2", "tok_s")))
    if dropped:
        print("excluded (crash-truncated, <95% window coverage): " +
              ", ".join(r["rd"].replace("logs/", "") for r in dropped))
    worst = sorted(rows, key=lambda r: -abs(r["preds"]["v2"]["j_tok"] / r["j_tok"] - 1))[:8]
    print("\nworst v2 J/token errors:")
    for r in worst:
        p = r["preds"]["v2"]
        print("  %-52s meas %.3f J/tok pred %.3f (%+.0f%%)  tok/s %+.0f%%  P %+.0f%%" % (
            r["rd"].replace("logs/", ""), r["j_tok"], p["j_tok"], 100 * (p["j_tok"] / r["j_tok"] - 1),
            100 * (p["tok_s"] / r["tok_s"] - 1), 100 * (p["power"] / r["power"] - 1)))
    return table


# ---------------------------------------------------------------------------
def measured_mbu(r, g):
    w = rg.serving_iter_work(r["m"], r["n_meas"], r["P"], r["G"])
    return (w["Wb"] + w["KVb"]) / (g["bw"] * r["t_iter"])


def make_figure(rows, wm, path):
    try:
        import matplotlib
    except ImportError:   # matplotlib lives in pydeps. transformers/huggingface-hub are
        # already imported (models resolved), so prepending pydeps now is safe.
        sys.path.insert(0, "/shared_data0/adsampat/pydeps")
        import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
    import numpy as np
    C = {"H200": "#2a78d6", "B200": "#eb6834", "A100-80-PCIe": "#1baf7a", "A5000": "#4a3aa7"}
    INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    gpus = rg.load_gpus()
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 9.2), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")

    # (a) utilization model: measured c=1 MBU vs weight bytes per layer
    a = ax[0, 0]
    for gk in ("H200", "B200", "A5000"):
        sel = [r for r in rows if r["gpu"] == gk and r["conc"] == 1 and not rg.is_moe(r["m"])]
        if not sel or gk not in gpus:
            continue
        g = gpus[gk]
        xs = [r["m"].active_params() * 2 / r["m"].L / 1e9 for r in sel]
        a.scatter(xs, [100 * measured_mbu(r, g) for r in sel], s=34, color=C[gk],
                  edgecolor="white", linewidth=0.8, zorder=3, label="%s measured (c=1)" % gk)
        tk = g["time_key"]
        if tk in rg.TIME_CENTRAL:
            wl = np.logspace(np.log10(0.02), np.log10(3.0), 80)
            for L, ls in ((28, "-"), (64, "--")):
                ys = []
                for x in wl:
                    tp = rg.TIME_CENTRAL[tk]
                    t = tp["t0"] + L * rg.softmax_k(tp["tau"], x * 1e9 / (g["bw"] * tp["eta"]), tp["k"])
                    ys.append(100 * L * x * 1e9 / (g["bw"] * t))
                a.plot(wl, ys, ls, color=C[gk], lw=1.6, alpha=0.9,
                       label="%s model, L=%d layers" % (gk, L))
            a.axhline(100 * rg.TIME_CENTRAL[tk]["eta"], color=C[gk], lw=0.8, ls=":", alpha=0.7)
    a.set_xscale("log"); a.set_ylim(0, 100)
    a.set_xlabel("weight bytes per layer (GB)  [known before deployment]")
    a.set_ylabel("realized MBU  (% of datasheet HBM bandwidth)")
    a.set_title("(a) Utilization model: MBU rises with layer size,\n"
                "saturating at η (dotted) — datasheet peak is never reached", loc="left")
    a.grid(alpha=0.6, color=GRID, lw=0.6); a.legend(fontsize=7, frameon=False, loc="upper left")

    # (b) backtest: predicted vs measured J/token, v1 vs v2
    b = ax[0, 1]
    for gk in ("H200", "B200", "A100-80-PCIe", "A5000"):
        sel = [r for r in rows if r["gpu"] == gk]
        if not sel:
            continue
        b.scatter([r["j_tok"] for r in sel], [r["preds"]["v1"]["j_tok"] for r in sel], s=16,
                  marker="x", color=C[gk], alpha=0.45, linewidth=0.9, zorder=2)
        err = mape([abs(r["preds"]["v2"]["j_tok"] / r["j_tok"] - 1) for r in sel])
        err1 = mape([abs(r["preds"]["v1"]["j_tok"] / r["j_tok"] - 1) for r in sel])
        b.scatter([r["j_tok"] for r in sel], [r["preds"]["v2"]["j_tok"] for r in sel], s=26,
                  color=C[gk], edgecolor="white", linewidth=0.7, zorder=3,
                  label="%s: v2 %.0f%% | v1 %.0f%% MAPE%s" % (
                      gk, err, err1, "\n   (cap-bound; its HBM2e prior was set from these runs)"
                      if gk.startswith("A100") else ""))
    lim = [0.01, 30]
    b.plot(lim, lim, color=INK2, lw=0.8); b.fill_between(lim, [x * 0.85 for x in lim],
                                                          [x * 1.15 for x in lim], color=GRID, alpha=0.6)
    b.set_xscale("log"); b.set_yscale("log"); b.set_xlim(lim); b.set_ylim(lim)
    b.set_xlabel("measured J / generated token"); b.set_ylabel("predicted J / token")
    b.set_title("(b) Backtest on every serving run: v2 (dots) vs v1 datasheet (×)\n"
                "band = ±15%", loc="left")
    b.grid(alpha=0.6, color=GRID, lw=0.6, which="major"); b.legend(fontsize=7, frameon=False, loc="upper left")

    # (c,d) win-maps: log2(E_B200/E_H200), hatched where P(winner) < 0.9
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
    for axx, phase in ((ax[1, 0], "decode"), (ax[1, 1], "prefill")):
        rws = wm[phase]
        if phase == "decode":
            axx.axvline(6.5, color=INK, lw=0.9, ls="--")
        models = list(dict.fromkeys(r["model"] for r in rws))
        bs = sorted(set(r["batch"] for r in rws))
        Z = np.full((len(models), len(bs)), np.nan)
        conf = np.zeros_like(Z, dtype=bool); tp2 = np.zeros_like(conf)
        for r in rws:
            i, j = models.index(r["model"]), bs.index(r["batch"])
            res = r["res"]
            if "H200" in res and "B200" in res:
                Z[i, j] = np.log2(res["B200"]["j_tok"] / res["H200"]["j_tok"])
                conf[i, j] = max(res["B200"]["p_best"], res["H200"]["p_best"]) >= 0.9
                tp2[i, j] = res["H200"]["n_gpu"] > 1 or res["B200"]["n_gpu"] > 1
        im = axx.imshow(Z, cmap=cmap, norm=TwoSlopeNorm(0, -1.0, 1.0), aspect="auto", origin="lower")
        for i in range(len(models)):
            for j in range(len(bs)):
                if np.isnan(Z[i, j]):
                    axx.text(j, i, "OOM", ha="center", va="center", fontsize=6, color=INK2)
                    continue
                ratio = 2 ** Z[i, j]
                axx.text(j, i, "%.2f" % ratio, ha="center", va="center", fontsize=6.5,
                         color=INK, fontweight="bold" if conf[i, j] else "normal")
                if not conf[i, j]:
                    axx.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, hatch="////",
                                                edgecolor=INK2, lw=0, alpha=0.45))
                if tp2[i, j]:
                    axx.add_patch(plt.Rectangle((j - .48, i - .48), .96, .96, fill=False,
                                                edgecolor=INK, lw=1.0, ls=":"))
        axx.set_xticks(range(len(bs))); axx.set_xticklabels(bs)
        axx.set_yticks(range(len(models)))
        axx.set_yticklabels([m.split("/")[-1].replace("-Instruct", "") for m in models])
        axx.set_xlabel("batch (concurrent sequences);  dashed line: measured \u2264 64 | extrapolated \u2192"
                       if phase == "decode"
                       else "batch of 1024-token prompts")
        axx.set_title("(%s) %s: E(B200)/E(H200) per token  [blue: H200 wins, red: B200 wins]\n"
                      "hatched = P(winner)<0.9; dotted box = one GPU needs TP>1 (extrapolated)"
                      % ("c" if phase == "decode" else "d", phase.upper()), loc="left", fontsize=8.5)
        cb = fig.colorbar(im, ax=axx, fraction=0.04, pad=0.02, ticks=[-1, -0.5, 0, 0.5, 1])
        cb.ax.set_yticklabels(["0.5", "0.71", "1", "1.41", "2"]); cb.set_label("B200 / H200", fontsize=8)
    fig.suptitle("Recommender v2.2: measured (prefix-cache-corrected) coefficients + realized utilization (vLLM 0.10.2, fp16); win-maps at cached_frac=0",
                 x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, facecolor=fig.get_facecolor())
    print("wrote", path)


def reversal(draws=400, seed=3):
    """Old v1 story: 'L40S for prefill, A100 for decode'. Joint Monte Carlo (same
    draw for both phases of a GPU) -> P(L40S wins prefill AND A100 wins decode)."""
    import random
    gpus = rg.load_gpus()
    rnd = random.Random(seed)
    print("\nPhase-reversal test (v1 claim: L40S wins prefill, A100 wins decode); "
          "prompt 2048 / gen 256 / batch 32. Both GPUs are PRIOR-only.")
    for a100 in ("A100-80-SXM", "A100-80-PCIe"):
        for mid in ("Qwen/Qwen2-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct"):
            m = rg.resolve_model(mid)
            cen = {n: {ph: rg.phase_predict(gpus[n], m, ph, 2048, 256, 32, rg.central_draw(gpus[n]))
                       for ph in ("prefill", "decode")} for n in ("L40S", a100)}
            wp = wd = both = 0
            for _ in range(draws):
                d = {n: rg.random_draw(gpus[n], rnd) for n in ("L40S", a100)}
                e = {n: {ph: rg.phase_predict(gpus[n], m, ph, 2048, 256, 32, d[n])["j_tok"]
                         for ph in ("prefill", "decode")} for n in ("L40S", a100)}
                p_ok = e["L40S"]["prefill"] < e[a100]["prefill"]
                d_ok = e[a100]["decode"] < e["L40S"]["decode"]
                wp += p_ok; wd += d_ok; both += p_ok and d_ok
            print("  %-11s vs L40S  %-16s central E(L40S)/E(A100): prefill %.2f decode %.2f | "
                  "P(L40S wins prefill)=%.0f%%  P(A100 wins decode)=%.0f%%  P(reversal)=%.0f%%" % (
                      a100, mid.split("/")[-1], cen["L40S"]["prefill"]["j_tok"] / cen[a100]["prefill"]["j_tok"],
                      cen["L40S"]["decode"]["j_tok"] / cen[a100]["decode"]["j_tok"],
                      100.0 * wp / draws, 100.0 * wd / draws, 100.0 * both / draws))
    # the same question on the only credible pair
    print("  H200 vs B200 (measured): see win-maps — B200 wins prefill, H200 wins decode (<=64 seqs).")


def run_winmap(draws=200):
    wm = {}
    for phase in ("decode", "prefill"):
        wm[phase] = rg.winmap(phase=phase, prompt=1024, gen=256, draws=draws)
    g = rg.load_gpus()
    if g.get("A5000", {}).get("measured"):    # auto-include once the A5000 entry exists
        for phase in ("decode", "prefill"):
            wm["A5000_" + phase] = rg.winmap(("H200", "A5000"), phase=phase, prompt=1024, gen=256,
                                             models=rg.WINMAP_MODELS[:4], draws=draws)
    return wm


def print_winmap(wm):
    for phase, rws in wm.items():
        if phase.startswith("A5000_"):
            print("\nA5000 vs H200 %s: E(A5000)/E(H200) per token" % phase[6:].upper())
            for r in rws:
                res = r["res"]
                if "A5000" in res and "H200" in res:
                    print("  %-24s b=%4d  %.2f  P(A5000 best)=%.0f%%" % (
                        r["model"].split("/")[-1], r["batch"],
                        res["A5000"]["j_tok"] / res["H200"]["j_tok"], 100 * res["A5000"]["p_best"]))
                else:
                    print("  %-24s b=%4d  OOM on A5000" % (r["model"].split("/")[-1], r["batch"]))
            continue
        print("\n%s win-map: E(B200)/E(H200) per token (median draw-central); "
              "* = P(winner)>=0.9; ~ = TP>1 somewhere" % phase.upper())
        bs = sorted(set(r["batch"] for r in rws))
        print("%-12s" % "" + "".join("%8d" % b for b in bs))
        for mid in dict.fromkeys(r["model"] for r in rws):
            line = "%-12s" % mid.split("/")[-1].replace("-Instruct", "")[:12]
            pb = []
            for b in bs:
                res = [x for x in rws if x["model"] == mid and x["batch"] == b][0]["res"]
                if "H200" in res and "B200" in res:
                    ratio = res["B200"]["j_tok"] / res["H200"]["j_tok"]
                    c = max(res["B200"]["p_best"], res["H200"]["p_best"]) >= 0.9
                    tp = res["H200"]["n_gpu"] > 1 or res["B200"]["n_gpu"] > 1
                    line += "%6.2f%s%s" % (ratio, "*" if c else " ", "~" if tp else " ")
                    pb.append("%7.0f%%" % (100 * res["B200"]["p_best"]))
                else:
                    line += "%8s" % "OOM"; pb.append("%8s" % "")
            print(line)
            print("%-12s" % "  P(B200)" + "".join(pb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--winmap", action="store_true")
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--dump", default=None, help="write per-run predictions CSV here")
    ap.add_argument("--old-fit", default=None, help="python-literal file with the v2.0 TIME_FIT")
    a = ap.parse_args()
    if a.old_fit:
        import ast
        global OLD_FIT
        OLD_FIT = ast.literal_eval(open(a.old_fit).read())
    runs, dropped = discover()
    if excluded_gpus():
        print("GPUs excluded via gpu_coefficients.json _excluded: %s" % sorted(excluded_gpus()))
    print("runs: %d usable, %d excluded; groups: %s" % (
        len(runs), len(dropped), sorted(set(r["group"] for r in runs))))
    if a.fit:
        fit = do_fit(runs)
        patch_recommender(fit)
        return
    rows = backtest(runs)
    summarize(rows, dropped)
    # MBU table
    gpus = rg.load_gpus()
    print("\nRealized MBU at c=1 (measured vs v2 model):")
    for r in rows:
        if r["conc"] == 1 and r["task"] == "alpaca":
            print("  %-10s %-26s L=%2d W/L=%.3fGB  MBU meas %4.1f%%  pred %4.1f%%" % (
                r["gpu"], r["model"].split("/")[-1], r["m"].L,
                r["m"].active_params() * 2 / r["m"].L / 1e9,
                100 * measured_mbu(r, gpus[r["gpu"]]), 100 * r["preds"]["v2"]["mbu"]))
    if a.dump:
        with open(a.dump, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run", "gpu", "group", "P", "G", "cached_frac", "conc", "rate", "tok_s", "power", "j_tok",
                        "v2_tok_s", "v2_power", "v2_j_tok", "lomo_tok_s", "lomo_j_tok",
                        "v1_tok_s", "v1_power", "v1_j_tok"])
            for r in rows:
                p = r["preds"]
                w.writerow([r["rd"], r["gpu"], r["group"], "%.1f" % r["P"], "%.1f" % r["G"], "%.3f" % r["cf"], r["conc"],
                            r["rate"], "%.2f" % r["tok_s"], "%.1f" % r["power"], "%.5f" % r["j_tok"],
                            "%.2f" % p["v2"]["tok_s"], "%.1f" % p["v2"]["power"], "%.5f" % p["v2"]["j_tok"],
                            "%.2f" % p["v2_lomo"]["tok_s"], "%.5f" % p["v2_lomo"]["j_tok"],
                            "%.2f" % p["v1"]["tok_s"], "%.1f" % p["v1"]["power"], "%.5f" % p["v1"]["j_tok"]])
    if a.winmap or a.figure:
        wm = run_winmap(a.draws)
        print_winmap(wm)
        reversal()
        if a.figure:
            make_figure(rows, wm, os.path.join(HERE, "plots_proposal", "fig8_recommender_v2.png"))


if __name__ == "__main__":
    main()
