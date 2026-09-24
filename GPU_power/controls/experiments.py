"""
controls/experiments.py -- scenarios, baselines tuning, experiment matrix, figures.

  python3 -m controls.experiments --exp validate|main|robust|hetero|park|sizes|
                                        sens|pareto|plantcal|figs|all  [--reps 5]
Results -> controls/results/<exp>.json ; figures -> plots_controls/*.png
Run inside the container on a compute node (numpy/scipy); figures need pydeps.
"""
import argparse
import copy
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from . import controllers as C
from .sim import run_episode

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "..", "plots_controls")
os.makedirs(RES, exist_ok=True)

MIX4 = ["H200", "B200", "H200", "B200"]


# ----------------------------------------------------------------------------
# scenarios
# ----------------------------------------------------------------------------
def base(model="7B", gpus=MIX4, T=1200):
    lam = dict(zip(("7B", "32B"), (40.0, 12.0)))[model] * len(gpus) / 4
    return dict(gpus=list(gpus), model=model, T=T, dt=1.0, lp=50, gl=153, slo_q=2.0,
                slo_tpot=0.1, lam=lam)


def budget_levels(scn):
    """Budgets relative to the fleet (sum of TDP / sum of floors)."""
    from .plant_fp import load_gpu_coeffs
    g = [load_gpu_coeffs(x) for x in scn["gpus"]]
    tdp = sum(x["p_cap"] for x in g)
    floors = sum(x["p_cap_min"] for x in g)
    ps = sum(x["p_static"] for x in g)
    return tdp, floors, ps


def scenario(name, model="7B", gpus=MIX4, T=1200):
    s = base(model, gpus, T)
    tdp, floors, ps = budget_levels(s)
    lam = s["lam"]
    steady = dict(kind="poisson", rate=lam)
    bursty = dict(kind="diurnal", rate=lam, burst_x=2.5, durs=(60.0, 15.0), period=600, amp=0.5)
    lvl = 0.47 * tdp if model == "7B" else 0.55 * tdp
    dr_events = [(250, 90, 0.60), (700, 150, 0.75), (1000, 60, 0.55)]
    if name == "flat_steady":
        s.update(arrivals=steady, budget=dict(kind="flat", level=lvl))
    elif name == "flat_bursty":
        s.update(arrivals=bursty, budget=dict(kind="flat", level=lvl))
    elif name == "dr_bursty":          # announced demand-response budget dips
        s.update(arrivals=bursty, budget=dict(kind="dr", level=lvl, events=dr_events))
    elif name == "dr_steady":
        s.update(arrivals=steady, budget=dict(kind="dr", level=lvl, events=dr_events))
    elif name == "oversub_bursty":     # unannounced: shared feed minus other tenants (OU)
        s.update(arrivals=bursty, budget=dict(kind="oversub", level=lvl + 0.12 * tdp,
                                              other_mean=0.12 * tdp, other_sd=0.06 * tdp,
                                              tau=60.0, jump_p=0.004, jump=0.1 * tdp,
                                              min_level=0.8 * floors))
    elif name == "park_diurnal":       # low load, flat generous budget, parking allowed
        s.update(arrivals=dict(kind="diurnal", rate=0.4 * lam, burst_x=2.0, durs=(90.0, 15.0),
                               period=600, amp=0.8),
                 budget=dict(kind="flat", level=tdp),
                 park=dict(p_park_frac=0.3, t_wake=30.0))
    else:
        raise ValueError(name)
    s["name"] = name
    return s


# ----------------------------------------------------------------------------
# controllers by name
# ----------------------------------------------------------------------------
def make_ctrl(name, scn, tuned=None, **kw):
    tuned = tuned or {}
    if name == "Uncapped":
        return C.Uncapped()
    if name == "POLCA":
        return C.POLCA()
    if name == "PI-adm":
        return C.PIAdmission()
    if name == "PI-cap":
        return C.PICap(**kw)
    if name == "PI-cap-safe":
        c = C.PICap(tpot_frac=0.5, **kw)
        c.name = "PI-cap-safe"
        return c
    if name == "StaticOpt":
        p = tuned.get(scn["name"] + "|" + scn["model"] + "|" + "-".join(scn["gpus"]))
        return C.StaticOpt(**p) if p else C.StaticOpt(phi={"H200": 0.1, "B200": 0.1})
    if name == "MPC":
        return C.EconMPC(forecast="ewma", preview=True, adapt=True, **kw)
    if name == "MPC-LP":
        return C.EconMPC(binaries=False, name="MPC-LP", **kw)
    if name == "MPC-blind":
        return C.EconMPC(blind=True, name="MPC-blind", **kw)
    if name == "MPC-persist":
        return C.EconMPC(forecast="persistence", preview=False, name="MPC-persist", **kw)
    if name == "MPC-holt":
        return C.EconMPC(forecast="holt", name="MPC-holt", **kw)
    if name == "MPC-oracle":
        return C.EconMPC(forecast="oracle", name="MPC-oracle", **kw)
    if name == "MPC-oracle+30%":
        return C.EconMPC(forecast="oracle", fc_noise=0.3, name="MPC-oracle+30%", **kw)
    if name == "MPC-noadapt":
        return C.EconMPC(adapt=False, name="MPC-noadapt", **kw)
    if name == "MPC-nopreview":
        return C.EconMPC(preview=False, name="MPC-nopreview", **kw)
    if name.startswith("PARK:"):
        _, inner, mode = name.split(":")
        return C.ParkingLoop(make_ctrl(inner, scn, tuned), aware=(mode == "aware"),
                             p_park_frac=scn.get("park", {}).get("p_park_frac", 0.3),
                             name=f"{inner}+{mode}park")
    raise ValueError(name)


def _job(args):
    scn, cname, seed, plant, mismatch, tuned, kw, trace = args
    ctrl = make_ctrl(cname, scn, tuned, **kw)
    t0 = time.time()
    m = run_episode(scn, ctrl, seed=seed, plant=plant, mismatch=mismatch, keep_trace=trace)
    m.update(scenario=scn["name"], model=scn["model"], gpus="-".join(scn["gpus"]),
             ctrl=ctrl.name, seed=seed, plant=plant, mismatch=mismatch, kw=kw,
             wall_s=time.time() - t0, n_fallback=getattr(ctrl, "n_fallback", 0))
    if trace:
        tr = m.pop("trace")
        tok = np.asarray(tr["tok"])                      # (T, G)
        pw = np.asarray(tr["pw"])
        m["tok_share_by_gpu"] = (tok.sum(0) / max(tok.sum(), 1e-9)).tolist()
        m["power_by_gpu"] = pw.mean(0).tolist()
        m["cap_by_gpu"] = np.asarray(tr["caps"]).mean(0).tolist()
        m["idle_frac_by_gpu"] = (tok < 1e-6).mean(0).tolist()
        if trace == "full":
            m["trace"] = {k: v.tolist() for k, v in tr.items()}
    return m


def pool_map(jobs, procs):
    if procs <= 1:
        return [_job(j) for j in jobs]
    with Pool(procs) as p:
        return p.map(_job, jobs, chunksize=1)


def save(name, rows):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    with open(os.path.join(RES, name + ".json"), "w") as f:
        json.dump(rows, f, default=conv)


def load(name):
    p = os.path.join(RES, name + ".json")
    return json.load(open(p)) if os.path.exists(p) else []


# ----------------------------------------------------------------------------
# static-optimal tuning (grid search on a TRAINING seed, then frozen)
# ----------------------------------------------------------------------------
PHI_GRID = [0.0, 0.03, 0.06, 0.1, 0.15, 0.25, 0.4, 0.7, 1.0]
SHARE_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def tune_static(scn, procs, seed=991, T=600):
    s = dict(copy.deepcopy(scn), T=T)
    types = sorted(set(s["gpus"]))
    combos = []
    for pH in PHI_GRID:
        for pB in (PHI_GRID if len(types) > 1 else [None]):
            for sh in (SHARE_GRID if len(types) > 1 else [None]):
                phi = {types[0]: pH} if pB is None else {"B200": pB, "H200": pH}
                share = None if sh is None else {"H200": sh, "B200": 1 - sh}
                combos.append(dict(phi=phi, share=share))
    jobs = []
    for p in combos:
        key = "tmp"
        jobs.append((s, "StaticOpt", seed, "fp", 0.0,
                     {s["name"] + "|" + s["model"] + "|" + "-".join(s["gpus"]): p}, {}, False))
    rows = pool_map(jobs, procs)
    viol = np.array([r["q_viol_frac"] + r["tpot_viol_frac"] + r["budget_viol_frac"] for r in rows])
    E = np.array([r["energy_kj"] for r in rows])
    ok = viol <= viol.min() + 0.01
    best = int(np.argmin(np.where(ok, E, np.inf)))
    return combos[best], dict(viol=float(viol[best]), E=float(E[best]), n=len(combos),
                              viol_min=float(viol.min()))


def get_tuned(scns, procs):
    path = os.path.join(RES, "static_tuned.json")
    tuned = json.load(open(path)) if os.path.exists(path) else {}
    for s in scns:
        key = s["name"] + "|" + s["model"] + "|" + "-".join(s["gpus"])
        if key not in tuned:
            p, info = tune_static(s, procs)
            tuned[key] = p
            print("tuned", key, p, info, flush=True)
            json.dump(tuned, open(path, "w"), indent=1)
    return tuned


# ----------------------------------------------------------------------------
# experiments
# ----------------------------------------------------------------------------
MAIN_CTRLS = ["Uncapped", "POLCA", "PI-adm", "PI-cap", "PI-cap-safe", "StaticOpt", "MPC-LP", "MPC",
              "MPC-blind", "MPC-persist", "MPC-oracle"]


def exp_validate(procs):
    """plant_fp vs measured runs (throughput, power)."""
    from .plant_fp import Plant
    meas = {("H200", 1): (166.4, 371), ("H200", 4): (652.6, 375), ("H200", 16): (2481.7, 401),
            ("H200", 64): (8572.7, 491), ("H200", "poisson8"): (1169.9, 385),
            ("B200", 1): (182.8, 561), ("B200", 4): (668.7, 550), ("B200", 16): (2567, 563),
            ("B200", 64): (9312, 642)}
    rows = []
    for (gpu, c), (tps, pw) in meas.items():
        p = Plant(gpu, "7B", dt=0.1, nsub=5, prompt_len=50, gen_len=153)
        x = p.reset()
        E = tok = T = 0.0
        for k in range(600):
            a = 0.8 if c == "poisson8" else max(0.0, c - x[1] - x[0])
            x, y = p.step(x, {"max_running": 256}, {"arrivals": a})
            if k >= 200:
                E += y["energy_j"]; tok += y["tokens_out"]; T += 0.1
        rows.append(dict(gpu=gpu, load=str(c), sim_tps=tok / T, meas_tps=tps, sim_w=E / T,
                         meas_w=pw, err_tps=(tok / T) / tps - 1, err_w=(E / T) / pw - 1))
        print(rows[-1], flush=True)
    save("validate", rows)


def exp_main(procs, reps, plant="fp", tag="main"):
    names = ["flat_steady", "flat_bursty", "dr_bursty", "dr_steady", "oversub_bursty"]
    scns = [scenario(n) for n in names]
    scns += [scenario("dr_bursty", gpus=["H200"] * 4), scenario("flat_bursty", gpus=["H200"] * 4)]
    scns += [scenario("dr_bursty", model="32B"), scenario("flat_bursty", model="32B")]
    for s in scns:                                  # unique names for the variants
        if s["gpus"] != MIX4:
            s["name"] += "_4xH200"
        if s["model"] == "32B":
            s["name"] += "_32B"
    tuned = get_tuned(scns, procs)
    jobs = [(s, c, seed, plant, 0.0, tuned, {}, "summary")
            for s in scns for c in MAIN_CTRLS for seed in range(reps)]
    rows = pool_map(jobs, procs)
    save(tag, rows)
    return rows


def exp_traces(procs):
    s = scenario("dr_bursty")
    tuned = get_tuned([s], procs)
    jobs = [(s, c, 0, "fp", 0.0, tuned, {}, "full") for c in ["POLCA", "PI-cap", "StaticOpt", "MPC"]]
    save("traces", pool_map(jobs, procs))


def exp_robust(procs, reps):
    s = scenario("dr_bursty")
    tuned = get_tuned([s], procs)
    jobs = []
    for mm in [0.0, 0.1, 0.2]:
        for c in ["PI-cap", "PI-cap-safe", "StaticOpt", "MPC", "MPC-noadapt"]:
            for seed in range(reps):
                jobs.append((s, c, seed, "fp", mm, tuned, {}, False))
    for c in ["MPC-persist", "MPC-nopreview", "MPC", "MPC-holt", "MPC-oracle", "MPC-oracle+30%"]:
        for seed in range(reps):
            jobs.append((s, c, seed, "fp", 0.0, tuned, {}, False))
    save("robust", pool_map(jobs, procs))


def exp_pareto(procs, reps):
    s = scenario("dr_bursty")
    tuned = get_tuned([s], procs)
    jobs = []
    for tf in [0.3, 0.5, 0.8, 1.0]:
        for seed in range(reps):
            jobs.append((s, "PI-cap", seed, "fp", 0.0, tuned, dict(tpot_frac=tf), False))
            jobs.append((s, "MPC", seed, "fp", 0.0, tuned, dict(T_frac=tf), False))
    for wf in [0.1, 0.5]:
        for seed in range(reps):
            jobs.append((s, "MPC", seed, "fp", 0.0, tuned, dict(W_frac=wf), False))
            jobs.append((s, "PI-cap", seed, "fp", 0.0, tuned, dict(q_frac=wf), False))
    save("pareto", pool_map(jobs, procs))


def exp_hetero(procs, reps):
    """Routing split + energy of hetero-aware vs blind, and B200-floor sensitivity."""
    jobs = []
    for model in ["7B", "32B"]:
        for name in ["flat_bursty", "dr_bursty"]:
            s = scenario(name, model=model)
            s["name"] += "" if model == "7B" else "_32B"
            for c in ["PI-cap", "MPC", "MPC-blind"]:
                for seed in range(reps):
                    jobs.append((s, c, seed, "fp", 0.0, {}, {}, "summary"))
    save("hetero", pool_map(jobs, procs))


def exp_sens(procs, reps):
    """Sensitivity of the heterogeneous routing to the ASSUMED B200 cap floor."""
    from . import plant_fp
    rows = []
    for fl in [300.0, 400.0, 500.0]:
        plant_fp._EXTRA["B200"]["p_cap_min"] = fl
        s = scenario("flat_bursty")
        s["name"] += f"_b200floor{int(fl)}"
        jobs = [(s, c, seed, "fp", 0.0, {}, {}, "summary") for c in ["PI-cap", "MPC", "MPC-blind"]
                for seed in range(reps)]
        r = pool_map(jobs, procs)
        for x in r:
            x["b200_floor"] = fl
        rows += r
    plant_fp._EXTRA["B200"]["p_cap_min"] = 300.0
    save("sens", rows)


def exp_park(procs, reps):
    s = scenario("park_diurnal")
    jobs = []
    for c in ["PI-cap", "MPC", "PARK:PI-cap:blind", "PARK:PI-cap:aware", "PARK:MPC:blind",
              "PARK:MPC:aware"]:
        for seed in range(reps):
            jobs.append((s, c, seed, "fp", 0.0, {}, {}, "full" if seed == 0 else "summary"))
    save("park", pool_map(jobs, procs))


def exp_sizes(procs, reps):
    rows = []
    for G in [2, 4, 8]:
        gpus = (["H200", "B200"] * 4)[:G]
        s = scenario("dr_bursty", gpus=gpus)
        s["name"] += f"_G{G}"
        tuned = get_tuned([s], procs)
        jobs = [(s, c, seed, "fp", 0.0, tuned, {}, False)
                for c in ["PI-cap", "StaticOpt", "MPC", "MPC-LP"] for seed in range(reps)]
        rows += pool_map(jobs, procs)
    save("sizes", rows)


def exp_horizon(procs, reps):
    """Is the *predictive* part doing anything?  Same MPC with N = 1 (myopic
    model-based optimisation), 3, 10, 30, with oracle vs EWMA forecasts."""
    jobs = []
    for nm in ["dr_bursty", "flat_bursty"]:
        s = scenario(nm)
        for N in [1, 3, 10, 30]:
            for fc in ["ewma", "oracle"]:
                for seed in range(reps):
                    jobs.append((s, "MPC-oracle" if fc == "oracle" else "MPC", seed, "fp", 0.0,
                                 {}, dict(N=N), "summary"))
    save("horizon", pool_map(jobs, procs))


def exp_park2(procs, reps):
    """Parking with the fleet listed B200-first: blind index-order parking now
    keeps a B200 awake (the ordering is arbitrary to a blind policy)."""
    s = scenario("park_diurnal", gpus=["B200", "H200", "B200", "H200"])
    s["name"] += "_B200first"
    jobs = [(s, c, seed, "fp", 0.0, {}, {}, "summary")
            for c in ["PARK:PI-cap:blind", "PARK:PI-cap:aware", "PARK:MPC:blind", "PARK:MPC:aware"]
            for seed in range(reps)]
    save("park2", pool_map(jobs, procs))


def exp_robust32(procs, reps):
    """Mismatch where the energy model actually binds decisions (32B: caps are
    above the floors, so e_* and MBU errors move the operating point)."""
    jobs = []
    for nm in ["flat_bursty", "dr_bursty"]:
        s = scenario(nm, model="32B")
        s["name"] += "_32B"
        for mm in [0.0, 0.2]:
            for c in ["PI-cap-safe", "MPC", "MPC-noadapt"]:
                for seed in range(reps):
                    jobs.append((s, c, seed, "fp", mm, {}, {}, "summary"))
    save("robust32", pool_map(jobs, procs))


def exp_plantcal(procs, reps):
    """Re-run key comparisons on the calibrated plant (controls/plant.py) if present."""
    try:
        from . import plant  # noqa
    except Exception as e:
        print("controls/plant.py not importable:", e)
        save("plantcal", [dict(error=str(e))])
        return
    exp_main(procs, reps, plant="cal", tag="plantcal")


# ----------------------------------------------------------------------------
# summary helpers
# ----------------------------------------------------------------------------
def ci(v):
    v = np.asarray(v, float)
    if len(v) < 2:
        return float(v.mean()), 0.0
    return float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(len(v)))


def summarize(rows, keys=("scenario", "ctrl"), metrics=("energy_kj", "j_per_tok",
              "q_viol_frac", "tpot_viol_frac", "budget_viol_frac", "solve_ms_mean",
              "solve_ms_max", "throughput_tps")):
    out = {}
    for r in rows:
        if "error" in r:
            continue
        k = tuple(r[x] for x in keys)
        out.setdefault(k, []).append(r)
    table = {}
    for k, rs in out.items():
        table[k] = {m: ci([r[m] for r in rs]) for m in metrics}
        table[k]["n"] = len(rs)
    return table


# ----------------------------------------------------------------------------
# figures + markdown tables
# ----------------------------------------------------------------------------
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MUTED = "#a8a7a1"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
                         "grid.linewidth": 0.6, "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 9, "figure.dpi": 130,
                         "axes.titlesize": 10, "axes.titlecolor": INK, "legend.frameon": False})
    return plt


def md_table(table, rows_order, cols, fmt, header):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for k in rows_order:
        if k not in table:
            continue
        t = table[k]
        cells = [str(x) for x in (k if isinstance(k, tuple) else (k,))]
        for c, f in zip(cols, fmt):
            m, h = t[c]
            cells.append(f.format(m, h))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


METRIC_COLS = ["energy_kj", "j_per_tok", "q_viol_frac", "tpot_viol_frac", "budget_viol_frac",
               "solve_ms_mean", "solve_ms_max"]
METRIC_FMT = ["{:.0f} ± {:.0f}", "{:.4f} ± {:.4f}", "{:.2%} ± {:.2%}", "{:.2%} ± {:.2%}",
              "{:.2%} ± {:.2%}", "{:.1f}", "{:.0f}"]
METRIC_HDR = ["energy kJ", "J/token", "TTFT-queue viol", "TPOT viol", "budget viol",
              "ctrl ms (mean)", "ctrl ms (max)"]


def exp_figs():
    plt = _plt()
    os.makedirs(FIG, exist_ok=True)
    md = []
    # ---- main table
    rows = load("main")
    if rows:
        T = summarize(rows)
        scns = list(dict.fromkeys(r["scenario"] for r in rows))
        for sc in scns:
            md.append(f"\n#### {sc}\n")
            md.append(md_table(T, [(sc, c) for c in MAIN_CTRLS], METRIC_COLS, METRIC_FMT,
                               ["scenario", "controller"] + METRIC_HDR))
        # relative-energy dot plot vs PI-cap (energy) + violation panel
        show = ["POLCA", "PI-adm", "PI-cap", "StaticOpt", "MPC-LP", "MPC"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 0.42 * len(scns) + 1.4), sharey=True)
        ys = np.arange(len(scns))[::-1]
        for ci_, c in enumerate(show):
            col = SER[ci_]
            e = [T[(sc, c)]["energy_kj"][0] / T[(sc, "PI-cap")]["energy_kj"][0] - 1
                 if (sc, c) in T else np.nan for sc in scns]
            v = [T[(sc, c)]["q_viol_frac"][0] + T[(sc, c)]["tpot_viol_frac"][0]
                 + T[(sc, c)]["budget_viol_frac"][0] if (sc, c) in T else np.nan for sc in scns]
            off = (ci_ - len(show) / 2) * 0.1
            axes[0].plot(np.array(e) * 100, ys + off, "o", ms=5, color=col, label=c,
                         mec="white", mew=0.8)
            axes[1].plot(np.array(v) * 100, ys + off, "o", ms=5, color=col, mec="white", mew=0.8)
        axes[0].axvline(0, color=INK2, lw=0.8)
        axes[0].set_yticks(ys)
        axes[0].set_yticklabels(scns)
        axes[0].set_xlabel("energy vs PI-cap (%)")
        axes[1].set_xlabel("TTFT-queue + TPOT + budget violations (%)")
        axes[0].set_title("Energy (lower is better)", loc="left")
        axes[1].set_title("Constraint violations (lower is better)", loc="left")
        axes[0].legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(1.0, -0.09 * 8 / max(len(scns), 1) - 0.08))
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "main_energy_violations.png"), bbox_inches="tight")
        plt.close(fig)
    # ---- DR traces
    tr = load("traces")
    if tr:
        fig, axes = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True)
        B = np.array(tr[0]["trace"]["B"])
        t = np.arange(len(B))
        axes[0].plot(t, B, color=INK, lw=1.5, ls="--", label="budget")
        for k_, r in enumerate(tr):
            axes[0].plot(t, r["trace"]["P"], lw=1.1, color=SER[k_], label=r["ctrl"])
        axes[0].set_ylabel("fleet power (W)")
        axes[0].set_title("Announced demand-response budget, bursty diurnal load (2xH200 + 2xB200, 7B)", loc="left")
        axes[0].legend(ncol=5, fontsize=8, loc="upper right")
        axes[1].plot(t, tr[0]["trace"]["arr"], color=MUTED, lw=0.8)
        axes[1].set_ylabel("arrivals (req/s)")
        axes[1].set_xlabel("time (s)")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "dr_trace.png"), bbox_inches="tight")
        plt.close(fig)
        # MPC per-GPU routing during DR
        r = [x for x in tr if x["ctrl"] == "MPC"][0]
        tok = np.array(r["trace"]["tok"])
        names = r["gpus"].split("-")
        fig, ax = plt.subplots(figsize=(10, 2.8))
        base_ = np.zeros(len(t))
        cols = {"H200": [SER[0], "#7fb0ea"], "B200": [SER[1], "#f4a582"]}
        seen = {}
        for i, nm in enumerate(names):
            c = cols[nm][seen.get(nm, 0) % 2]
            seen[nm] = seen.get(nm, 0) + 1
            sm = np.convolve(tok[:, i], np.ones(10) / 10, mode="same")
            ax.fill_between(t, base_, base_ + sm, color=c, lw=0, label=f"GPU{i} {nm}")
            base_ = base_ + sm
        ax2 = ax.twinx()
        ax2.grid(False)
        ax.set_ylabel("tokens/s served (10 s avg)")
        ax.set_title("MPC token allocation per GPU (stacked) under the DR budget", loc="left")
        ax.legend(ncol=4, fontsize=8, loc="upper left")
        ax2.set_yticks([])
        ax2.spines["right"].set_visible(False)
        ax.set_xlabel("time (s)")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "mpc_allocation.png"), bbox_inches="tight")
        plt.close(fig)
    # ---- pareto
    pr = load("pareto")
    if pr:
        fig, ax = plt.subplots(figsize=(6, 4))
        for k_, c in enumerate(["PI-cap", "MPC"]):
            pts = {}
            for r in pr:
                if r["ctrl"] == c:
                    pts.setdefault(json.dumps(r["kw"], sort_keys=True), []).append(r)
            xs, ys_ = [], []
            for kw, rs in pts.items():
                xs.append(np.mean([x["q_viol_frac"] + x["tpot_viol_frac"] + x["budget_viol_frac"] for x in rs]) * 100)
                ys_.append(np.mean([x["energy_kj"] for x in rs]))
            ax.plot(xs, ys_, "o", color=SER[k_], ms=7, label=c, mec="white", mew=0.8)
        main = load("main")
        st = [r for r in main if r["scenario"] == "dr_bursty" and r["ctrl"] == "StaticOpt"]
        if st:
            ax.plot([np.mean([x["q_viol_frac"] + x["tpot_viol_frac"] + x["budget_viol_frac"] for x in st]) * 100],
                    [np.mean([x["energy_kj"] for x in st])], "s", color=SER[2], ms=7, label="StaticOpt (tuned)")
        ax.set_xlabel("total violations (%)")
        ax.set_ylabel("energy (kJ, 1200 s)")
        ax.set_title("Energy vs violations (dr_bursty); each dot = one setpoint\n(PI-cap tpot_frac/q_frac, MPC T_frac/W_frac sweeps)", loc="left")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "pareto.png"), bbox_inches="tight")
        plt.close(fig)
    # ---- robustness
    rb = load("robust")
    if rb:
        T = summarize(rb, keys=("ctrl", "mismatch"))
        md.append("\n#### robustness (dr_bursty): model mismatch and forecast\n")
        order = sorted(T.keys(), key=lambda k: (k[1], k[0]))
        md.append(md_table(T, order, METRIC_COLS, METRIC_FMT, ["controller", "mismatch"] + METRIC_HDR))
    # ---- hetero
    for nm in ["hetero", "sens", "park", "park2"]:
        hz = [r for r in load(nm) if "tok_share_by_gpu" in r]
        if not hz:
            continue
        agg = {}
        for r in hz:
            k = (r["scenario"], r["ctrl"])
            names = r["gpus"].split("-")
            sh = {"H200": 0.0, "B200": 0.0}
            pw = {"H200": 0.0, "B200": 0.0}
            idle = {"H200": [], "B200": []}
            for i, g in enumerate(names):
                sh[g] += r["tok_share_by_gpu"][i]
                pw[g] += r["power_by_gpu"][i]
                idle[g].append(r["idle_frac_by_gpu"][i])
            agg.setdefault(k, []).append((sh["B200"], pw["H200"], pw["B200"], r["energy_kj"],
                                          np.mean(idle["H200"]), np.mean(idle["B200"]),
                                          r["q_viol_frac"] + r["tpot_viol_frac"] + r["budget_viol_frac"]))
        md.append(f"\n#### {nm}: where the load goes (B200 share of tokens; per-type mean power)\n")
        md.append("| scenario | controller | B200 token share | H200 W (sum) | B200 W (sum) | energy kJ | H200 idle frac | B200 idle frac | total viol |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for k, v in agg.items():
            v = np.array(v)
            m_ = v.mean(0)
            md.append(f"| {k[0]} | {k[1]} | {m_[0]:.2f} | {m_[1]:.0f} | {m_[2]:.0f} | {m_[3]:.0f} ± {ci(v[:,3])[1]:.0f} | {m_[4]:.2f} | {m_[5]:.2f} | {m_[6]:.2%} |")
        if nm == "hetero":
            fig, ax = plt.subplots(figsize=(7, 3))
            keys = list(agg.keys())
            ys = np.arange(len(keys))[::-1]
            vals = [np.mean([x[0] for x in agg[k]]) for k in keys]
            ax.barh(ys, vals, color=SER[1], height=0.6)
            for y_, v_ in zip(ys, vals):
                ax.text(v_ + 0.01, y_, f"{v_:.2f}", va="center", fontsize=7, color=INK2)
            ax.axvline(0.5, color=INK2, lw=0.8, ls="--")
            ax.set_yticks(ys)
            ax.set_yticklabels([f"{a} | {b}" for a, b in keys], fontsize=7)
            ax.set_xlabel("fraction of tokens served by the two B200s (0.5 = even split)")
            ax.set_title("Heterogeneity: B200 share of served tokens", loc="left")
            fig.tight_layout()
            fig.savefig(os.path.join(FIG, "hetero_share.png"), bbox_inches="tight")
            plt.close(fig)
    # ---- park
    pk = load("park")
    if pk:
        T = summarize(pk)
        md.append("\n#### parking (low-load diurnal, P_park = 0.3 P_static, 30 s wake)\n")
        md.append(md_table(T, sorted(T.keys()), METRIC_COLS, METRIC_FMT, ["scenario", "controller"] + METRIC_HDR))
        fig, ax = plt.subplots(figsize=(6.5, 2.8))
        ks = sorted(T.keys(), key=lambda k: -T[k]["energy_kj"][0])
        ax.barh(range(len(ks)), [T[k]["energy_kj"][0] for k in ks],
                xerr=[T[k]["energy_kj"][1] for k in ks], color=[SER[0] if "aware" in k[1] else MUTED for k in ks], height=0.6)
        ax.set_yticks(range(len(ks)))
        ax.set_yticklabels([k[1] for k in ks])
        ax.set_xlabel("energy (kJ, 1200 s)")
        ax.set_title("Parking: idle-power-aware (blue) vs blind/none (gray)", loc="left")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "parking.png"), bbox_inches="tight")
        plt.close(fig)
    # ---- sizes
    sz = load("sizes")
    if sz:
        T = summarize(sz)
        md.append("\n#### fleet size (dr_bursty)\n")
        md.append(md_table(T, sorted(T.keys()), METRIC_COLS, METRIC_FMT, ["scenario", "controller"] + METRIC_HDR))
        fig, ax = plt.subplots(figsize=(5, 3))
        for k_, c in enumerate(["MPC", "MPC-LP"]):
            Gs, mean_, mx = [], [], []
            for G in [2, 4, 8]:
                k = (f"dr_bursty_G{G}", c)
                if k in T:
                    Gs.append(G); mean_.append(T[k]["solve_ms_mean"][0]); mx.append(T[k]["solve_ms_max"][0])
            ax.plot(Gs, mean_, "o-", color=SER[k_], lw=2, label=f"{c} mean")
            ax.plot(Gs, mx, "o--", color=SER[k_], lw=1, label=f"{c} max")
        ax.set_yscale("log")
        ax.set_xlabel("GPUs in fleet")
        ax.set_ylabel("controller time per step (ms)")
        ax.axhline(1000, color=INK2, lw=0.8)
        ax.text(2, 700, "control interval = 1000 ms", fontsize=7, color=INK2)
        ax.legend(fontsize=7)
        ax.set_title("MPC solve time (HiGHS)", loc="left")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "solve_time.png"), bbox_inches="tight")
        plt.close(fig)
    hz = load("horizon")
    if hz:
        for r in hz:
            r["N"] = r["kw"].get("N", 10) if isinstance(r["kw"], dict) else 10
        T = summarize(hz, keys=("scenario", "ctrl", "N"))
        md.append("\n#### horizon length N (does prediction matter?)\n")
        md.append(md_table(T, sorted(T.keys()), METRIC_COLS, METRIC_FMT,
                           ["scenario", "controller", "N"] + METRIC_HDR))
        fig, ax = plt.subplots(figsize=(5.5, 3))
        for k_, (sc, c) in enumerate([("dr_bursty", "MPC"), ("dr_bursty", "MPC-oracle"),
                                      ("flat_bursty", "MPC"), ("flat_bursty", "MPC-oracle")]):
            Ns = [1, 3, 10, 30]
            ys_ = [T[(sc, c, N)]["energy_kj"][0] for N in Ns if (sc, c, N) in T]
            es_ = [T[(sc, c, N)]["energy_kj"][1] for N in Ns if (sc, c, N) in T]
            ax.errorbar(Ns[:len(ys_)], ys_, yerr=es_, marker="o", color=SER[k_], lw=1.5,
                        capsize=2, label=f"{sc} {c}")
        ax.set_xscale("log")
        ax.set_xticks([1, 3, 10, 30])
        ax.set_xticklabels(["1", "3", "10", "30"])
        ax.set_xlabel("MPC horizon N (steps of 1 s)")
        ax.set_ylabel("energy (kJ, 1200 s)")
        ax.set_title("Horizon sweep: myopic (N=1) vs predictive", loc="left")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "horizon.png"), bbox_inches="tight")
        plt.close(fig)
    for nm, keys in [("robust32", ("scenario", "ctrl", "mismatch")), ("park2", ("scenario", "ctrl"))]:
        rr = load(nm)
        if rr:
            T = summarize(rr, keys=keys)
            md.append(f"\n#### {nm}\n")
            md.append(md_table(T, sorted(T.keys()), METRIC_COLS, METRIC_FMT,
                               list(keys) + METRIC_HDR))
    for nm in ["plantcal"]:
        pc = load(nm)
        if pc and "error" not in pc[0]:
            T = summarize(pc)
            md.append(f"\n#### {nm}: same matrix on controls/plant.py\n")
            scns = list(dict.fromkeys(r["scenario"] for r in pc))
            for sc in scns:
                md.append(md_table(T, [(sc, c) for c in MAIN_CTRLS], METRIC_COLS, METRIC_FMT,
                                   ["scenario", "controller"] + METRIC_HDR))
    val = load("validate")
    if val:
        md.append("\n#### plant_fp validation vs measured vLLM runs (Qwen2-7B)\n")
        md.append("| GPU | load | sim tok/s | meas tok/s | err | sim W | meas W | err |")
        md.append("|---|---|---|---|---|---|---|---|")
        for r in val:
            md.append(f"| {r['gpu']} | {r['load']} | {r['sim_tps']:.0f} | {r['meas_tps']:.0f} | {r['err_tps']:+.1%} | {r['sim_w']:.0f} | {r['meas_w']:.0f} | {r['err_w']:+.1%} |")
    with open(os.path.join(RES, "tables.md"), "w") as f:
        f.write("\n".join(md))
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--procs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    a = ap.parse_args()
    exps = a.exp.split(",")
    allx = ["validate", "main", "traces", "robust", "pareto", "hetero", "sens", "park",
            "sizes", "plantcal"]
    if "all" in exps:
        exps = allx
    for e in exps:
        t0 = time.time()
        print("=== exp", e, flush=True)
        if e == "validate":
            exp_validate(a.procs)
        elif e == "main":
            exp_main(a.procs, a.reps)
        elif e == "traces":
            exp_traces(a.procs)
        elif e == "robust":
            exp_robust(a.procs, a.reps)
        elif e == "pareto":
            exp_pareto(a.procs, a.reps)
        elif e == "hetero":
            exp_hetero(a.procs, a.reps)
        elif e == "sens":
            exp_sens(a.procs, a.reps)
        elif e == "park":
            exp_park(a.procs, a.reps)
        elif e == "sizes":
            exp_sizes(a.procs, a.reps)
        elif e == "plantcal":
            exp_plantcal(a.procs, a.reps)
        elif e == "park2":
            exp_park2(a.procs, a.reps)
        elif e == "robust32":
            exp_robust32(a.procs, a.reps)
        elif e == "horizon":
            exp_horizon(a.procs, a.reps)
        elif e == "figs":
            exp_figs()
        print(f"=== exp {e} done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
