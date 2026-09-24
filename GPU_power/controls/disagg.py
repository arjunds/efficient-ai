"""
controls/disagg.py -- disaggregated prefill/decode pools on a heterogeneous
H200 + B200 fleet under a shared power budget (the arXiv 2609.11133 setting,
extended to heterogeneous GPUs).

  python3 -m controls.disagg --exp steady|dynamic|sens|figs|all [--procs 8]
  (coefficients come from gpu_coefficients.json at run time, so the whole study
   re-runs unchanged after the e_gemm prefix-cache correction; --egemm_scale /
   --b200_floor give the sensitivities.)

Per-GPU physics (our 3-term energy model + roofline, same as plant_fp):
  PREFILL GPU, chunk C tokens per iteration:
      e_p  = e_gemm*2P + e_kvbyte*1.5*kvb + e_wbyte*W/C        [J per prompt token]
      R_p  = min( C / t_roof(C), (TDP-P_static)/e_p )           [max prompt tok/s]
      drawn power = P_static + e_p * r   (work-proportional; the cap bounds r)
  DECODE GPU (TPOT target T, context lp + g/2, prefill work excluded):
      pi >= max(floor, a/T + b D, (a/n_max + b) D),   D <= D_max
      a = e_wbyte*W,  b = e_kvbyte*ctx*kvb + e_gemm*2P
  COLOCATED GPU: both, sharing time (r/R_p + D/D_max <= 1) and power.
KV transfer prefill->decode is assumed free (NVLink/IB energy not modelled).
"""
import argparse
import copy
import itertools
import json
import math
import os
import time
from multiprocessing import Pool

import numpy as np
from scipy.optimize import milp, linprog, LinearConstraint, Bounds

from . import plant_fp
from .plant_fp import MODEL_TABLE, MODEL_ALIASES, MBU1, load_gpu_coeffs

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "..", "plots_controls")
CHUNK = 8192
MFU_P = 0.5            # realized prefill MFU (assumed, both GPUs); sensitivity below

WORKLOADS = {"chat": dict(lp=512, gl=256), "rag": dict(lp=3000, gl=150)}


# ----------------------------------------------------------------------------
# per-GPU steady-state models
# ----------------------------------------------------------------------------
class GpuPD:
    def __init__(self, gpu, model, lp, gl, T=0.08, egemm_scale=1.0, b200_floor=None,
                 mfu_p=MFU_P, blind_params=None):
        model = MODEL_ALIASES.get(model, model)
        g = dict(load_gpu_coeffs(gpu))
        if blind_params is not None:
            g.update(blind_params)
        g["e_gemm"] *= egemm_scale
        if gpu == "B200" and b200_floor is not None:
            g["p_cap_min"] = b200_floor
        mt = MODEL_TABLE[model]
        self.name, self.gpu = gpu, g
        P, W, kvb = mt["active"], 2.0 * mt["active"], mt["kvb_tok"]
        self.ps, self.tdp, self.cap_min = g["p_static"], g["p_cap"], g["p_cap_min"]
        self.floor = self.cap_min - self.ps
        mbu = g.get("mbu1_override") or MBU1.get((gpu, model), 0.45)
        # prefill
        self.e_p = g["e_gemm"] * 2 * P + g["e_kvbyte"] * 1.5 * kvb + g["e_wbyte"] * W / CHUNK
        t_roof = max((W + CHUNK * 1.5 * kvb) / (g["bw"] * mbu),
                     2 * P * CHUNK / (g["peak_flops"] * mfu_p))
        self.R_p = min(CHUNK / t_roof, (self.tdp - self.ps) / self.e_p)
        # decode
        ctx = lp + 0.5 * gl
        kv_max = (0.9 * g["mem_bytes"] - 2.0 * mt["total"] - 2e9) / kvb
        self.fits = kv_max > 0
        self.a = g["e_wbyte"] * W
        self.b = g["e_kvbyte"] * ctx * kvb + g["e_gemm"] * 2 * P
        self.n_max = float(min(256, 0.9 * max(kv_max, 1) / (lp + gl)))
        self.c = self.a / self.n_max + self.b
        self.T = T
        n = self.n_max
        mb = mbu * max(0.6, 1 - 0.037 * math.log(max(n, 1)))
        t_r = max((W + n * ctx * kvb) / (g["bw"] * mb), 2 * P * n / (g["peak_flops"] * 0.6))
        t_tdp = (self.a + self.b * n) / (self.tdp - self.ps)
        self.d_max = n / max(t_r, t_tdp)


def fleet_models(gpus, model, wl, **kw):
    return [GpuPD(g, model, wl["lp"], wl["gl"], **kw) for g in gpus]


# ----------------------------------------------------------------------------
# steady-state MILP: min power (or max lambda) for a pool assignment
# ----------------------------------------------------------------------------
def solve_steady(M, roles, lam, lp, gl, budget=None, maximize_lambda=False, headroom=0.7):
    """roles[i] in {'P','D','C'} (prefill / decode / colocated).
    Variables per GPU: r (prompt tok/s), D (decode tok/s), pi (dyn W), z (on);
    plus lambda if maximize_lambda. Budget (if given) is on sum of CAPS:
    prefill cap >= Ps + e_p r / headroom (queueing headroom), decode cap = Ps + pi.
    Returns dict or None if infeasible."""
    G = len(M)
    nv = 5  # r, D, pi, z, cap
    n = nv * G + 1
    idx = lambda v, i: i * nv + v
    L = nv * G
    c = np.zeros(n)
    lb, ub = np.zeros(n), np.full(n, np.inf)
    integ = np.zeros(n)
    rows, lo, hi = [], [], []

    def add(co, l, h):
        r = np.zeros(n)
        for k, v in co:
            r[k] += v
        rows.append(r); lo.append(l); hi.append(h)

    for i, m in enumerate(M):
        r, D, P, Z, K = (idx(v, i) for v in range(nv))
        integ[Z] = 1
        ub[Z] = 1
        ub[P] = m.tdp - m.ps
        ub[K] = m.tdp
        if not maximize_lambda:
            c[P] = 1.0
        role = roles[i]
        if role in ("P", "C"):
            ub[r] = headroom * m.R_p
        else:
            ub[r] = 0
        if role in ("D", "C") and m.fits:
            ub[D] = m.d_max
        else:
            ub[D] = 0
        # power: prefill work + decode envelope
        add([(P, 1), (r, -m.e_p), (Z, -m.a / m.T), (D, -m.b)], 0, np.inf) if role != "P" else \
            add([(P, 1), (r, -m.e_p)], 0, np.inf)
        if role != "P":
            add([(P, 1), (r, -m.e_p), (D, -m.c)], 0, np.inf)
            add([(P, 1), (r, -m.e_p), (Z, -m.floor)], 0, np.inf)
        add([(D, 1), (Z, -m.d_max)], -np.inf, 0)
        add([(r, 1), (Z, -m.R_p)], -np.inf, 0)
        if role == "C":      # time sharing on a colocated GPU
            add([(r, 1.0 / m.R_p), (D, 1.0 / max(m.d_max, 1e-9))], -np.inf, 1.0)
        # cap: decode/colocated cap = Ps + pi (>= cap_min if on); prefill needs headroom
        add([(K, 1), (P, -1), (Z, -m.ps)], 0, np.inf)
        add([(K, 1), (Z, -m.cap_min)], 0, np.inf)
        if role in ("P", "C"):
            add([(K, 1), (r, -m.e_p / headroom), (Z, -m.ps)], 0, np.inf)
        add([(K, 1), (Z, -m.tdp)], -np.inf, 0)
        lb[K] = 0.0
    if maximize_lambda:
        c[L] = -1.0
        lamv = [(L, 1.0)]
        ub[L] = np.inf
        add([(idx(0, i), 1) for i in range(G)] + [(L, -lp)], 0, 0)
        add([(idx(1, i), 1) for i in range(G)] + [(L, -gl)], 0, 0)
    else:
        lb[L] = ub[L] = 0
        add([(idx(0, i), 1) for i in range(G)], lam * lp, lam * lp)
        add([(idx(1, i), 1) for i in range(G)], lam * gl, lam * gl)
    if budget is not None:
        # caps of GPUs that are off still sit at their floor (hardware guarantee):
        # count cap_min for off GPUs:  sum_i [K_i + (1 - z_i) cap_min_i] <= B
        add([(idx(4, i), 1) for i in range(G)] + [(idx(3, i), -M[i].cap_min) for i in range(G)],
            -np.inf, budget - sum(m.cap_min for m in M))
    res = milp(c, constraints=LinearConstraint(np.array(rows), lo, hi), bounds=Bounds(lb, ub),
               integrality=integ, options=dict(time_limit=5.0))
    if res.x is None:
        return None
    x = res.x
    r_ = np.array([x[idx(0, i)] for i in range(G)])
    D_ = np.array([x[idx(1, i)] for i in range(G)])
    P_ = np.array([x[idx(2, i)] for i in range(G)])
    K_ = np.array([x[idx(4, i)] for i in range(G)])
    lam_ = x[L] if maximize_lambda else lam
    ps = sum(m.ps for m in M)
    return dict(lam=float(lam_), power=float(ps + P_.sum()), dyn=float(P_.sum()),
                j_per_req=float((ps + P_.sum()) / max(lam_, 1e-9)), r=r_.tolist(), D=D_.tolist(),
                pi=P_.tolist(), caps=K_.tolist(), roles=list(roles))


def assignments(gpus):
    """All distinct prefill/decode splits (by GPU type counts) + colocated."""
    types = sorted(set(gpus))
    cnt = {t: gpus.count(t) for t in types}
    out = [("colocated", ["C"] * len(gpus))]
    for combo in itertools.product(*[range(cnt[t] + 1) for t in types]):
        npf = dict(zip(types, combo))
        tot = sum(combo)
        if tot == 0 or tot == len(gpus):
            continue
        roles, used = [], {t: 0 for t in types}
        for g in gpus:
            if used[g] < npf[g]:
                roles.append("P")
                used[g] += 1
            else:
                roles.append("D")
        label = "P:" + "+".join(f"{npf[t]}{t}" for t in types if npf[t]) + " | D:" + \
                "+".join(f"{cnt[t]-npf[t]}{t}" for t in types if cnt[t] - npf[t])
        out.append((label, roles))
    return out


def steady_case(args):
    gpus, model, wlname, lam_frac, budget_frac, kw = args
    wl = WORKLOADS[wlname]
    M = fleet_models(gpus, model, wl, **kw)
    tdp = sum(m.tdp for m in M)
    rows = []
    # reference capacity: best assignment, no budget
    cap_ref = max((solve_steady(M, roles, 0, wl["lp"], wl["gl"], maximize_lambda=True) or
                   {"lam": 0})["lam"] for _, roles in assignments(gpus))
    lam = lam_frac * cap_ref
    for label, roles in assignments(gpus):
        e = solve_steady(M, roles, lam, wl["lp"], wl["gl"])
        t = solve_steady(M, roles, 0, wl["lp"], wl["gl"], budget=budget_frac * tdp,
                         maximize_lambda=True)
        rows.append(dict(gpus="-".join(gpus), model=model, wl=wlname, lam_frac=lam_frac,
                         lam=lam, cap_ref=cap_ref, budget_w=budget_frac * tdp, assign=label,
                         power_w=e["power"] if e else None,
                         j_per_req=e["j_per_req"] if e else None,
                         lam_max_budget=t["lam"] if t else 0.0,
                         pi=e["pi"] if e else None, r=e["r"] if e else None,
                         D=e["D"] if e else None, kw={k: v for k, v in kw.items()}))
    return rows


# ----------------------------------------------------------------------------
# dynamic: fluid prefill queues + plant_fp decode GPUs, controllers
# ----------------------------------------------------------------------------
class PDController:
    name = "base"

    def reset(self, M, roles, wl, slo):
        self.M, self.roles, self.wl, self.slo = M, roles, wl, slo
        self.P = [i for i, r in enumerate(roles) if r == "P"]
        self.D = [i for i, r in enumerate(roles) if r == "D"]
        self.solve_times = []


def _project(caps, M, budget):
    caps = np.clip(np.asarray(caps, float), [m.cap_min for m in M], [m.tdp for m in M])
    if caps.sum() <= budget:
        return caps
    cmin = np.array([m.cap_min for m in M])
    room = budget - cmin.sum()
    if room <= 0:
        return cmin
    return cmin + (caps - cmin) * room / (caps - cmin).sum()


class StaticSplit(PDController):
    """Budget split in proportion to TDP (heterogeneity-blind); equal routing."""
    name = "static-split"

    def act(self, obs):
        M = self.M
        caps = _project([m.tdp for m in M], M, obs["budget"])
        rp = np.array([1.0 if i in self.P else 0 for i in range(len(M))])
        rd = np.array([1.0 if i in self.D else 0 for i in range(len(M))])
        return dict(caps=caps, route_p=rp / rp.sum(), route_d=rd / rd.sum())


class PIPools(PDController):
    """Per-GPU PI: prefill caps on prefill queue delay, decode caps on TPOT/queue
    (the fleet-study PI-cap); JSQ routing; budget projection."""
    name = "PI-pools"

    def reset(self, *a):
        super().reset(*a)
        self.u = np.full(len(self.M), 0.5)
        self.ep = np.zeros(len(self.M))

    def act(self, obs):
        M, s = self.M, self.slo
        e = np.zeros(len(M))
        for i in self.P:
            e[i] = obs["pf_wait"][i] / (0.3 * s["ttft"]) - 1.0
        for i in self.D:
            y = obs["y"][i] if obs["y"] else None
            if y is not None and y.get("busy_frac", 0) > 0.05:
                e[i] = y["t_iter_s"] / (0.8 * s["tpot"]) - 1.0 + y["queue_delay_s"] / 0.5
            else:
                e[i] = -0.2
        e = np.clip(e, -1, 3)
        self.u = np.clip(self.u + 0.15 * (e - self.ep) + 0.08 * e, 0, 1)
        self.ep = e
        caps = [m.cap_min + u * (m.tdp - m.cap_min) for m, u in zip(M, self.u)]
        caps = _project(caps, M, obs["budget"])
        qp = np.array([obs["pf_queue"][i] + 1.0 if i in self.P else 0 for i in range(len(M))])
        rp = np.array([(1.0 / qp[i]) if i in self.P else 0 for i in range(len(M))])
        od = np.array([(1.0 / (1 + obs["x"][i][0] + obs["x"][i][1])) if i in self.D else 0
                       for i in range(len(M))])
        return dict(caps=caps, route_p=rp / rp.sum(), route_d=od / od.sum())


class PoolMPC(PDController):
    """Economic MPC (LP, HiGHS) over both pools with a shared budget.
    Per step j: prefill GPU i: r_ij <= R_p, power e_p r_ij; prefill queue Qp_j
    (tokens, pool level) with Little's law Qp <= W_ttft * sum r; decode demand
    = prefill completions * gl/lp (tokens/s) into a decode backlog Bd_j; decode
    envelope per GPU; sum(P_static) + sum(power) <= budget_j (preview).
    blind=True: every GPU modelled as the fleet-average GPU."""
    name = "MPC-pools"

    def __init__(self, N=10, blind=False, name=None):
        self.N, self.blind = N, blind
        if name:
            self.name = name

    def reset(self, M, roles, wl, slo):
        super().reset(M, roles, wl, slo)
        if self.blind:
            keys = ["e_p", "R_p", "a", "b", "c", "d_max", "ps", "floor", "tdp", "cap_min"]
            avg = {k: float(np.mean([getattr(m, k) for m in M])) for k in keys}
            self.Mc = []
            for m in M:
                mm = copy.copy(m)
                for k, v in avg.items():
                    setattr(mm, k, v)
                self.Mc.append(mm)
        else:
            self.Mc = M
        self.lam = None

    def act(self, obs):
        t0 = time.perf_counter()
        M, N, dt = self.Mc, self.N, obs["dt"]
        lp, gl = self.wl["lp"], self.wl["gl"]
        a = obs["arr_hist"][-1] if obs["arr_hist"] else 0.0
        self.lam = a if self.lam is None else 0.25 * a + 0.75 * self.lam
        bud = np.asarray(obs["budget_future"][:N], float)
        if len(bud) < N:
            bud = np.concatenate([bud, np.full(N - len(bud), bud[-1] if len(bud) else obs["budget"])])
        Pn, Dn = self.P, self.D
        # variables: per j: r_i (P), pp_i (P power), D_i (D), pd_i (D power), Qp, Bd, s1, s2
        nP, nD = len(Pn), len(Dn)
        per = 2 * nP + 2 * nD + 4
        n = per * N
        o = lambda j: j * per
        c = np.zeros(n)
        lb, ub = np.zeros(n), np.full(n, np.inf)
        A, lo, hi = [], [], []

        def add(co, l, h):
            r = np.zeros(n)
            for k, v in co:
                r[k] += v
            A.append(r); lo.append(l); hi.append(h)
        Qp0 = float(sum(obs["pf_queue"][i] for i in Pn))
        Bd0 = float(sum(obs["x"][i][0] for i in Dn)) * gl
        Wt = 0.3 * self.slo["ttft"]
        for j in range(N):
            b0 = o(j)
            ri = lambda k: b0 + k
            ppi = lambda k: b0 + nP + k
            Di = lambda k: b0 + 2 * nP + k
            pdi = lambda k: b0 + 2 * nP + nD + k
            Qp, Bd, s1, s2 = b0 + per - 4, b0 + per - 3, b0 + per - 2, b0 + per - 1
            for k, i in enumerate(Pn):
                m = M[i]
                ub[ri(k)] = 0.95 * m.R_p
                ub[ppi(k)] = m.tdp - m.ps
                c[ppi(k)] += dt
                add([(ppi(k), 1), (ri(k), -m.e_p)], 0, np.inf)
            for k, i in enumerate(Dn):
                m = M[i]
                ub[Di(k)] = m.d_max
                ub[pdi(k)] = m.tdp - m.ps
                c[pdi(k)] += dt
                add([(pdi(k), 1), (Di(k), -m.b)], m.a / m.T, np.inf)
                add([(pdi(k), 1), (Di(k), -m.c)], 0, np.inf)
                lb[pdi(k)] = m.floor
            # prefill queue: Qp_{j+1} = Qp_j + dt*(lam*lp - sum r)
            prevQ = [] if j == 0 else [(o(j - 1) + per - 4, -1)]
            add([(Qp, 1)] + [(ri(k), dt) for k in range(nP)] + prevQ,
                (Qp0 if j == 0 else 0) + dt * self.lam / dt * lp, (Qp0 if j == 0 else 0) + dt * self.lam / dt * lp)
            # decode backlog: Bd_{j+1} = Bd_j + dt*(sum r * gl/lp - sum D)
            prevB = [] if j == 0 else [(o(j - 1) + per - 3, -1)]
            add([(Bd, 1)] + [(ri(k), -dt * gl / lp) for k in range(nP)] +
                [(Di(k), dt) for k in range(nD)] + prevB, Bd0 if j == 0 else 0, Bd0 if j == 0 else 0)
            # SLOs via Little's law (soft)
            add([(Qp, 1)] + [(ri(k), -Wt) for k in range(nP)] + [(s1, -1)], -np.inf, 0)
            add([(Bd, 1)] + [(Di(k), -0.5) for k in range(nD)] + [(s2, -1)], -np.inf, 0)
            c[s1] += 0.5
            c[s2] += 1.0
            c[Qp] += 1e-4
            c[Bd] += 1e-4
            if j == N - 1:
                c[Qp] += 2 * max(m.e_p for m in M)
                c[Bd] += 2 * max(m.c for m in M)
            # budget
            add([(ppi(k), 1) for k in range(nP)] + [(pdi(k), 1) for k in range(nD)],
                -np.inf, bud[j] - sum(m.ps for m in M))
        res = linprog(c, bounds=list(zip(lb, ub)), method="highs",
                      **_split(np.array(A), np.array(lo), np.array(hi)))
        Mt = self.M
        if res.x is None or res.status != 0:
            caps = _project([m.tdp for m in Mt], Mt, obs["budget"])
            out = dict(caps=caps, route_p=np.array([1.0 if i in Pn else 0 for i in range(len(Mt))]) / nP,
                       route_d=np.array([1.0 if i in Dn else 0 for i in range(len(Mt))]) / nD)
        else:
            x = res.x
            caps = np.zeros(len(Mt))
            rp = np.zeros(len(Mt))
            rd = np.zeros(len(Mt))
            for k, i in enumerate(Pn):
                r0 = x[k]
                rp[i] = r0 + 1e-6
                # prefill cap = rate target (+10% so the queue drains) as power
                caps[i] = Mt[i].ps + max(Mt[i].floor, 1.1 * Mt[i].e_p * r0)
            for k, i in enumerate(Dn):
                rd[i] = x[2 * nP + k] + 1e-6
                caps[i] = Mt[i].ps + max(Mt[i].floor, x[2 * nP + nD + k])
            caps = _project(caps, Mt, obs["budget"])
            out = dict(caps=caps, route_p=rp / rp.sum(), route_d=rd / rd.sum())
        self.solve_times.append(time.perf_counter() - t0)
        return out


def _split(A, lo, hi):
    """two-sided rows -> A_ub/b_ub + A_eq/b_eq for linprog."""
    eq = np.isclose(lo, hi)
    Aub, bub = [], []
    for r, l, h, e in zip(A, lo, hi, eq):
        if e:
            continue
        if np.isfinite(h):
            Aub.append(r); bub.append(h)
        if np.isfinite(l):
            Aub.append(-r); bub.append(-l)
    return dict(A_ub=np.array(Aub), b_ub=np.array(bub), A_eq=A[eq], b_eq=lo[eq])


def gen_budget_arr(T, level, events, ramp=10.0):
    t = np.arange(T)
    B = np.full(T, float(level))
    for (t0, dur, frac) in events:
        drop = (1 - frac) * level
        B -= drop * np.minimum(np.clip((t - t0) / ramp, 0, 1), np.clip((t0 + dur - t) / ramp, 0, 1))
    return B


def run_dynamic(args):
    """One episode: fleet, roles, controller name, seed, kw -> metrics."""
    gpus, model, wlname, roles, cname, seed, kw, T = args
    from .sim import gen_arrivals
    wl = WORKLOADS[wlname]
    lp, gl = wl["lp"], wl["gl"]
    M = fleet_models(gpus, model, wl, **kw)
    slo = dict(ttft=2.0 if wlname == "chat" else 4.0, tpot=0.1)
    # load: 45% of the best steady-state capacity of this assignment (no budget)
    cap = solve_steady(M, roles, 0, lp, gl, maximize_lambda=True)["lam"]
    rng = np.random.default_rng(seed + 1000)
    arr, _ = gen_arrivals(dict(kind="diurnal", rate=0.45 * cap, burst_x=2.0, durs=(60.0, 15.0),
                               period=600, amp=0.4), T, 1.0, rng)
    tdp = sum(m.tdp for m in M)
    bud = gen_budget_arr(T, 0.55 * tdp, [(250, 90, 0.7), (700, 150, 0.8), (1000, 60, 0.65)])
    ctrl = {"static-split": StaticSplit, "PI-pools": PIPools, "MPC-pools": PoolMPC}[cname.split(":")[0]]()
    if cname == "MPC-pools:blind":
        ctrl = PoolMPC(blind=True, name="MPC-pools-blind")
    ctrl.reset(M, roles, wl, slo)
    G = len(M)
    Pn = [i for i, r in enumerate(roles) if r == "P"]
    Dn = [i for i, r in enumerate(roles) if r == "D"]
    plants = {i: plant_fp.Plant(gpus[i], model, dt=1.0, nsub=10, prompt_len=lp, gen_len=gl,
                                **({"e_gemm": M[i].gpu["e_gemm"], "p_cap_min": M[i].cap_min}))
              for i in Dn}
    xs = {i: plants[i].reset() for i in Dn}
    pf_fifo = {i: [] for i in Pn}          # [t_arr, tokens_remaining]
    ttft = []
    E = tok = tok_viol = 0.0
    bviol = 0
    ys = None
    arr_hist = []
    for k in range(T):
        obs = dict(k=k, dt=1.0, budget=bud[k], budget_future=bud[k:k + 10],
                   arr_hist=arr_hist[-60:],
                   pf_queue={i: sum(c[1] for c in pf_fifo[i]) for i in Pn},
                   pf_wait={i: (k - pf_fifo[i][0][0]) if pf_fifo[i] else 0.0 for i in Pn},
                   x={i: xs[i] for i in Dn}, y=ys)
        u = ctrl.act(obs)
        caps = np.asarray(u["caps"], float)
        P_tot = 0.0
        # prefill
        new_dec = 0.0
        for i in Pn:
            if arr[k] * u["route_p"][i] > 0:
                pf_fifo[i].append([k, arr[k] * u["route_p"][i] * lp])
            m = M[i]
            rate = min(m.R_p, max(caps[i] - m.ps, 0) / m.e_p)
            budget_tok = rate * 1.0
            done_tok = 0.0
            while pf_fifo[i] and budget_tok > 1e-9:
                c = pf_fifo[i][0]
                take = min(c[1], budget_tok)
                c[1] -= take
                budget_tok -= take
                done_tok += take
                if c[1] <= 1e-9:
                    pf_fifo[i].pop(0)
                    nreq_tok = take
                # TTFT accounting per token cohort (fraction of the cohort finished)
                ttft.append((k + 1 - c[0], take / lp))
            new_dec += done_tok / lp
            P_i = m.ps + m.e_p * done_tok
            E += P_i
            P_tot += P_i
        # decode
        new_ys = {}
        for i in Dn:
            a_i = new_dec * u["route_d"][i]
            xn, y = plants[i].step(xs[i], dict(max_running=256, power_cap_w=float(caps[i])),
                                   dict(arrivals=a_i, prompt_len=lp, gen_len=gl, prefilled=True))
            xs[i] = xn
            new_ys[i] = y
            E += y["energy_j"]
            P_tot += y["power_w"]
            tok += y["tokens_out"]
            if y["t_iter_s"] > slo["tpot"]:
                tok_viol += y["tokens_out"]
        ys = new_ys
        arr_hist.append(arr[k])
        if P_tot > bud[k] * 1.005:
            bviol += 1
    w = np.array([x[1] for x in ttft])
    d = np.array([x[0] for x in ttft])
    ttft_viol = float((w * (d > slo["ttft"])).sum() / max(w.sum(), 1e-9))
    st = np.array(ctrl.solve_times) if ctrl.solve_times else np.zeros(1)
    return dict(gpus="-".join(gpus), model=model, wl=wlname, roles="".join(roles),
                ctrl=ctrl.name, seed=seed, kw=kw, energy_kj=E / 1e3, tokens=tok,
                j_per_tok=E / max(tok, 1), ttft_viol=ttft_viol, tpot_viol=tok_viol / max(tok, 1),
                budget_viol=bviol / T, mean_power=E / T,
                solve_ms_mean=1e3 * float(st.mean()), solve_ms_max=1e3 * float(st.max()),
                backlog=float(sum(xs[i][0] + xs[i][1] for i in Dn) +
                              sum(sum(c[1] for c in pf_fifo[i]) for i in Pn) / lp))


# ----------------------------------------------------------------------------
# experiments
# ----------------------------------------------------------------------------
def pmap(f, jobs, procs):
    if procs <= 1:
        return [f(j) for j in jobs]
    with Pool(procs) as p:
        return p.map(f, jobs, chunksize=1)


FLEET = ["H200", "H200", "B200", "B200"]


def exp_steady(procs, tag="disagg_steady", kw=None):
    kw = kw or {}
    jobs = [(FLEET, model, wl, lf, 0.45, kw) for model in ["7B", "32B"]
            for wl in ["chat", "rag"] for lf in [0.3, 0.7]]
    rows = [r for rs in pmap(steady_case, jobs, procs) for r in rs]
    json.dump(rows, open(os.path.join(RES, tag + ".json"), "w"), indent=0, default=float)
    return rows


def exp_sens(procs):
    rows = []
    for es in [0.7, 0.8, 1.0, 1.2, 1.3]:
        for fl in [300.0, 400.0, 500.0]:
            kw = dict(egemm_scale=es, b200_floor=fl)
            jobs = [(FLEET, model, wl, 0.3, 0.45, kw) for model in ["7B", "32B"]
                    for wl in ["chat", "rag"]]
            rows += [r for rs in pmap(steady_case, jobs, procs) for r in rs]
    json.dump(rows, open(os.path.join(RES, "disagg_sens.json"), "w"), indent=0, default=float)
    return rows


def best_roles(model, wl, kw=None):
    """Energy-optimal disaggregated assignment at 30% load (from the steady LP)."""
    kw = kw or {}
    rows = steady_case((FLEET, model, wl, 0.3, 0.45, kw))
    dis = [r for r in rows if r["assign"] != "colocated" and r["j_per_req"] is not None]
    b = min(dis, key=lambda r: r["j_per_req"])
    worst = max(dis, key=lambda r: r["j_per_req"])
    lab = dict(assignments(FLEET))
    return b["assign"], lab[b["assign"]], worst["assign"], lab[worst["assign"]]


def exp_dynamic(procs, reps=3, T=1200):
    jobs = []
    meta = {}
    for model in ["7B", "32B"]:
        for wl in ["chat", "rag"]:
            bl, br, wlb, wr = best_roles(model, wl)
            meta[f"{model}|{wl}"] = dict(best=bl, worst=wlb)
            for roles, which in [(br, "best"), (wr, "worst")]:
                for cname in ["static-split", "PI-pools", "MPC-pools", "MPC-pools:blind"]:
                    if which == "worst" and cname not in ("MPC-pools",):
                        continue
                    for seed in range(reps):
                        jobs.append((FLEET, model, wl, roles, cname, seed, {}, T))
    rows = pmap(run_dynamic, jobs, procs)
    json.dump(dict(meta=meta, rows=rows), open(os.path.join(RES, "disagg_dynamic.json"), "w"),
              indent=0, default=float)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all")
    ap.add_argument("--procs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    ex = a.exp.split(",")
    if "all" in ex:
        ex = ["steady", "sens", "dynamic"]
    for e in ex:
        t0 = time.time()
        if e == "steady":
            exp_steady(a.procs)
        elif e == "sens":
            exp_sens(a.procs)
        elif e == "dynamic":
            exp_dynamic(a.procs, a.reps)
        print(f"=== {e} done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
