"""
controls/sim.py -- fleet simulator: arrivals, budgets, routing, parking, metrics.

run_episode(scn, controller, seed, plant="fp", mismatch=0.0) -> metrics dict

Scenario dict keys:
  gpus        list of GPU names, e.g. ["H200","H200","B200","B200"]
  model       "7B" | "32B"
  T           episode length (s);  dt control interval (s, default 1)
  arrivals    dict(kind=poisson|mmpp|diurnal, ...)  (per-interval counts)
  budget      dict(kind=flat|dr|oversub, ...)
  lp, gl      mean prompt / output tokens
  slo_q       queue-delay (TTFT proxy) SLO, s     slo_tpot  TPOT SLO, s
  park        optional dict(p_park_frac, t_wake) enables parking

Queue delay is measured exactly on per-GPU FIFOs of (arrival interval, count)
cohorts from each plant's admissions, so it works with any plant implementing the
interface (admitted = q_prev + arrivals - q_next).
"""
import collections
import importlib
import time

import numpy as np

from .controllers import FleetInfo


# ----------------------------------------------------------------------------
# exogenous signals
# ----------------------------------------------------------------------------
def gen_arrivals(spec, T, dt, rng):
    n = int(T / dt)
    t = np.arange(n) * dt
    kind = spec["kind"]
    if kind == "poisson":
        lam = np.full(n, spec["rate"])
    elif kind in ("mmpp", "diurnal"):
        lo, hi = spec.get("rates", (spec["rate"], spec["rate"] * spec.get("burst_x", 3.0)))
        d_lo, d_hi = spec.get("durs", (60.0, 15.0))
        state, lam = 0, np.empty(n)
        for k in range(n):
            lam[k] = hi if state else lo
            if rng.random() < dt / (d_hi if state else d_lo):
                state = 1 - state
        if kind == "diurnal":
            per = spec.get("period", 600.0)
            lam = lam * (1 + spec.get("amp", 0.5) * np.sin(2 * np.pi * t / per - np.pi / 2))
    elif kind == "trace":
        base = np.asarray(spec["counts"], float)
        lam = np.resize(base, n) / dt
    else:
        raise ValueError(kind)
    return rng.poisson(np.maximum(lam, 0) * dt).astype(float), lam


def gen_budget(spec, T, dt, rng):
    n = int(T / dt)
    t = np.arange(n) * dt
    kind = spec["kind"]
    B = np.full(n, float(spec["level"]))
    announced = False
    if kind == "flat":
        pass
    elif kind == "dr":             # demand-response events, announced ahead
        announced = True
        ramp = spec.get("ramp", 10.0)
        for (t0, dur, frac) in spec["events"]:
            drop = (1 - frac) * spec["level"]
            up = np.clip((t - t0) / ramp, 0, 1)
            down = np.clip((t0 + dur - t) / ramp, 0, 1)
            B -= drop * np.minimum(up, down)
    elif kind == "oversub":        # shared feed; other tenants' draw is an OU process
        mu, sig, tau = spec["other_mean"], spec["other_sd"], spec.get("tau", 60.0)
        o = np.empty(n)
        o[0] = mu
        for k in range(1, n):
            o[k] = o[k - 1] + (mu - o[k - 1]) * dt / tau + sig * np.sqrt(2 * dt / tau) * rng.standard_normal()
            if rng.random() < spec.get("jump_p", 0.003):
                o[k] += spec.get("jump", 0.0)
        B = B - np.clip(o, 0, None)
    else:
        raise ValueError(kind)
    return np.maximum(B, spec.get("min_level", 0.0)), announced


# ----------------------------------------------------------------------------
# plants
# ----------------------------------------------------------------------------
def plant_class(which):
    mod = importlib.import_module("controls.plant_fp" if which == "fp" else "controls.plant")
    return mod.Plant


MISMATCH_KEYS = ("e_wbyte", "e_kvbyte", "e_gemm", "mbu1")


def make_fleet(scn, plant="fp", mismatch=0.0, rng=None):
    """Truth plants (possibly perturbed) + controller-side nominal FleetInfo."""
    from .plant_fp import Plant as FPPlant
    P = plant_class(plant)
    plants, nominal, applied = [], [], []
    for g in scn["gpus"]:
        nom = FPPlant(g, scn["model"], dt=scn.get("dt", 1.0), prompt_len=scn["lp"],
                      gen_len=scn["gl"])
        nominal.append(nom.gpu_params)
        ov = {}
        if mismatch > 0:
            for k in MISMATCH_KEYS:
                ov[k] = nom.gpu_params[k] * (1 + rng.uniform(-mismatch, mismatch))
        if plant == "fp":
            p = P(g, scn["model"], dt=scn.get("dt", 1.0), prompt_len=scn["lp"],
                  gen_len=scn["gl"], **ov)
        else:
            from .plant_fp import MODEL_ALIASES
            mname = MODEL_ALIASES.get(scn["model"], scn["model"])
            ov2 = {("u_w" if k == "mbu1" else k): v for k, v in ov.items()}
            if "u_w" in ov2:
                ov2["u_w"] = None   # filled below (relative perturbation of the plant's own u_w)
            p = P(g, mname, dt=scn.get("dt", 1.0), prompt_len=scn["lp"], gen_len=scn["gl"])
            if ov:
                base_gp = p.gpu_params
                ov3 = {k: (base_gp["u_w"] * ov["mbu1"] / nom.gpu_params["mbu1"] if k == "u_w"
                           else v) for k, v in ov2.items()}
                p = P(g, mname, dt=scn.get("dt", 1.0), prompt_len=scn["lp"], gen_len=scn["gl"],
                      **ov3)
            # the controller knows the deployment config of the real plant
            nom.gpu_params["max_num_seqs"] = int(p.gpu_params["max_num_seqs"])
            nom.gpu_params["kv_max_tokens"] = float(p.gpu_params["kv_capacity"])
            p._is_cal = True
        plants.append(p)
        applied.append(ov)
    fleet = FleetInfo(nominal, list(scn["gpus"]), scn["lp"], scn["gl"],
                      scn.get("slo_q", 2.0), scn.get("slo_tpot", 0.1), scn.get("dt", 1.0))
    return plants, fleet, applied


# ----------------------------------------------------------------------------
# episode
# ----------------------------------------------------------------------------
def run_episode(scn, ctrl, seed=0, plant="fp", mismatch=0.0, keep_trace=False):
    rng = np.random.default_rng(seed)
    dt = scn.get("dt", 1.0)
    T = scn["T"]
    n = int(T / dt)
    arr, lam_true = gen_arrivals(scn["arrivals"], T, dt, np.random.default_rng(seed + 1000))
    bud, announced = gen_budget(scn["budget"], T, dt, np.random.default_rng(seed + 2000))
    plants, fleet, applied = make_fleet(scn, plant, mismatch, np.random.default_rng(seed + 3000))
    G = len(plants)
    ctrl.reset(fleet)
    xs = [p.reset() for p in plants]
    ys = None
    fifo = [collections.deque() for _ in range(G)]
    park = scn.get("park")
    mode = np.zeros(G, int)            # 0 awake, 1 parked, 2 waking (timer)
    wake_t = np.zeros(G)
    ps_true = np.array([p.gpu_params["p_static"] for p in plants])
    slo_q, slo_tp = fleet.slo_q, fleet.slo_tpot
    N_H = getattr(ctrl, "N", 10)

    E = tok = comp = 0.0
    req_adm = req_viol = 0.0
    delay_sum = 0.0
    tok_viol = 0.0
    bud_viol_n = 0
    bud_over_j = 0.0
    act_t = []
    tr = collections.defaultdict(list)
    for k in range(n):
        obs = dict(k=k, dt=dt, x=xs, y=ys, budget=bud[k], budget_future=bud[k:k + N_H],
                   budget_announced=announced, arr_hist=list(arr[max(0, k - 120):k]),
                   arr_future=arr[k:k + N_H], lp=scn["lp"], gl=scn["gl"],
                   mode=mode.copy())
        t0 = time.perf_counter()
        u = ctrl.act(obs)
        act_t.append(time.perf_counter() - t0)
        route = np.maximum(np.asarray(u["route"], float), 0)
        # parking state machine (slow actuator)
        if park is not None:
            want = np.asarray(u.get("park", np.zeros(G, bool)), bool)
            for i in range(G):
                if mode[i] == 0 and want[i] and xs[i][0] + xs[i][1] < 0.5:
                    mode[i] = 1
                elif mode[i] == 1 and not want[i]:
                    mode[i], wake_t[i] = 2, park.get("t_wake", 30.0)
                elif mode[i] == 2:
                    wake_t[i] -= dt
                    if wake_t[i] <= 0:
                        mode[i] = 0
            route = np.where((mode == 0) & ~want, route, 0.0)
            if route.sum() <= 0:
                route = (mode == 0).astype(float)
                if route.sum() == 0:
                    route = np.ones(G)
        route = route / route.sum()
        P_tot = 0.0
        new_ys = []
        for i, p in enumerate(plants):
            a_i = arr[k] * route[i]
            if park is not None and mode[i] != 0 and xs[i][0] + xs[i][1] < 1e-6 and a_i == 0:
                pw = ps_true[i] * (park.get("p_park_frac", 0.3) if mode[i] == 1 else 1.0)
                y = dict(power_w=pw, energy_j=pw * dt, tokens_out=0.0, completions=0.0,
                         t_iter_s=0.0, queue_delay_s=0.0, throttled=0.0, admitted=0.0,
                         busy_frac=0.0, kv_frac=0.0)
                xn = xs[i]
            else:
                um = dict(max_running=int(max(1, round(u["max_running"][i]))),
                          power_cap_w=float(u["caps"][i]))
                xn, y = p.step(xs[i], um, dict(arrivals=a_i, prompt_len=scn["lp"],
                                               gen_len=scn["gl"]))
                # (plant.py now reports throttled t_iter_s itself; the earlier
                #  max(t_iter_s, dt/iterations) adapter was removed 2026-09-24)
            if a_i > 0:
                fifo[i].append([k, a_i])
            adm = y.get("admitted", xs[i][0] + a_i - xn[0])
            while adm > 1e-12 and fifo[i]:
                k0, c0 = fifo[i][0]
                take = min(c0, adm)
                d = (k - k0) * dt
                delay_sum += take * d
                req_adm += take
                if d > slo_q:
                    req_viol += take
                adm -= take
                fifo[i][0][1] -= take
                if fifo[i][0][1] <= 1e-12:
                    fifo[i].popleft()
            if y["t_iter_s"] > slo_tp:
                tok_viol += y["tokens_out"]
            E += y["energy_j"]
            tok += y["tokens_out"]
            comp += y["completions"]
            P_tot += y["power_w"]
            xs[i] = xn
            new_ys.append(y)
        ys = new_ys
        if P_tot > bud[k] * 1.005:
            bud_viol_n += 1
            bud_over_j += (P_tot - bud[k]) * dt
        if keep_trace:
            tr["P"].append(P_tot)
            tr["B"].append(bud[k])
            tr["arr"].append(arr[k])
            tr["caps"].append(np.asarray(u["caps"], float).copy())
            tr["route"].append(route.copy())
            tr["pw"].append([y["power_w"] for y in ys])
            tr["q"].append([x[0] for x in xs])
            tr["nrun"].append([x[1] for x in xs])
            tr["tpot"].append([y["t_iter_s"] for y in ys])
            tr["tok"].append([y["tokens_out"] for y in ys])
            tr["mode"].append(mode.copy())
    # requests still waiting at the end: count those already older than SLO
    for i in range(G):
        for k0, c0 in fifo[i]:
            if (n - k0) * dt > slo_q:
                req_viol += c0
            req_adm += c0
    backlog = sum(x[0] + x[1] for x in xs)
    st = np.asarray(getattr(ctrl, "solve_times", []) or act_t)
    m = dict(energy_kj=E / 1e3, tokens=tok, j_per_tok=E / max(tok, 1), completions=comp,
             throughput_tps=tok / T, q_viol_frac=req_viol / max(req_adm, 1),
             mean_qdelay_s=delay_sum / max(req_adm, 1),
             tpot_viol_frac=tok_viol / max(tok, 1),
             slo_viol_frac=min(1.0, req_viol / max(req_adm, 1) + tok_viol / max(tok, 1)),
             budget_viol_frac=bud_viol_n / n, budget_over_kj=bud_over_j / 1e3,
             backlog_req=backlog, arrivals=float(arr.sum()),
             ctrl_ms_mean=1e3 * float(np.mean(act_t)), ctrl_ms_p99=1e3 * float(np.percentile(act_t, 99)),
             solve_ms_mean=1e3 * float(st.mean()) if len(st) else 0.0,
             solve_ms_max=1e3 * float(st.max()) if len(st) else 0.0,
             mean_power_w=E / T, mean_budget_w=float(bud.mean()))
    if keep_trace:
        m["trace"] = {k: np.asarray(v) for k, v in tr.items()}
    return m
