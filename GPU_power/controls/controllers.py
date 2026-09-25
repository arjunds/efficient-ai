"""
controls/controllers.py -- fleet power/admission controllers.

Every controller implements
    reset(fleet)                    fleet = FleetInfo (nominal per-GPU params, workload)
    act(obs) -> dict(caps=[W]*G, max_running=[int]*G, route=[w]*G, park=[bool]*G opt.)
obs (built by sim.py each interval k):
    k, dt, x (list of plant states), y (list of last plant outputs, None at k=0),
    budget (current W), budget_future (true next-N budget, only meaningful if the
    scenario *announces* it -- controllers decide whether to trust it),
    budget_announced (bool), arr_hist (past per-interval arrival counts),
    arr_future (true future counts; only the oracle uses it), lp, gl (mean prompt /
    output tokens).

Controllers:
  Uncapped      caps = TDP, JSQ routing (reference, ignores budget)
  StaticOpt     fixed routing split + fixed per-GPU-type cap fraction (tuned offline
                by grid search on a *training* seed); scaled to fit the budget
  POLCA         reactive two-threshold power-cap throttling on measured fleet power
  PIAdmission   budget-proportional caps + PI on max_running holding queue delay
  PICap         per-GPU PI on the power cap holding TPOT/queue-delay setpoints
                (feedback "lowest cap that meets the SLO") + budget projection
  EconMPC       economic MPC (MILP, HiGHS) over per-GPU caps, routing, on/off,
                horizon N, forecast (persistence / EWMA / Holt / oracle), with
                optional online gain adaptation and heterogeneity-blind mode.
"""
import math
import time

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


# ----------------------------------------------------------------------------
# controller-side (nominal) steady-state model of one GPU
# ----------------------------------------------------------------------------
class GpuModel:
    """Steady-state service/power envelope used by the MPC (and the static tuner).

    Operating a GPU at iteration time T with n = D*T running requests costs
        E_iter(n) = a + b*n      (a = e_w*W ; b = per-output-token KV+GEMM+prefill)
        pi(D)     = E_iter/T = a/T + b*D           [throttled at iteration time T]
    and because n <= n_max:  pi >= (a/n_max + b) * D.  A loaded GPU never draws
    less than its settable floor (p_cap_min - P_static).  Max throughput D_max is
    set by the roofline (and TDP) at n_max.  All of this is linear -> MILP."""

    def __init__(self, gp, lp, gl, T):
        self.update(gp, lp, gl, T)

    def update(self, gp, lp, gl, T):
        self.gp = gp
        ctx = lp + 0.5 * gl
        W = gp["w_bytes"]
        self.ps = gp["p_static"]
        self.tdp = gp["p_cap"]
        self.cap_min = gp["p_cap_min"]
        self.floor = self.cap_min - self.ps
        self.a = gp["e_wbyte"] * W
        self.b = (gp["e_kvbyte"] * ctx * gp["kvb_tok"]
                  + gp["e_gemm"] * 2.0 * gp["p_active"] * (1.0 + lp / max(gl, 1.0)))
        self.n_max = float(min(gp["max_num_seqs"], 0.9 * gp["kv_max_tokens"] / (lp + gl)))
        self.c = self.a / self.n_max + self.b
        self.T = T
        n = self.n_max
        mbu = gp["mbu1"] * max(0.6, 1 - gp["mbu_kappa"] * math.log(n))
        byt = W + n * ctx * gp["kvb_tok"]
        flops = 2.0 * gp["p_active"] * n * (1 + lp / max(gl, 1.0))
        t_roof = max(byt / (gp["bw"] * mbu), flops / (gp["peak_flops"] * gp["mfu"]))
        t_tdp = (self.a + self.b * n) / (self.tdp - self.ps)
        self.t_min = max(t_roof, t_tdp)
        self.d_max = n / self.t_min
        # natural (uncapped) iteration time at batch 1 -- used by PI heuristics
        self.t1 = W / (gp["bw"] * gp["mbu1"])

    def pi_needed(self, D, T=None):
        T = T or self.T
        if D <= 0:
            return 0.0
        return max(self.floor, self.a / T + self.b * D, self.c * D)


class FleetInfo:
    def __init__(self, gpu_params, gpu_names, lp, gl, slo_q=2.0, slo_tpot=0.1, dt=1.0):
        self.gp = gpu_params            # nominal (controller-side) params, list
        self.names = gpu_names
        self.G = len(gpu_params)
        self.lp, self.gl = lp, gl
        self.slo_q, self.slo_tpot, self.dt = slo_q, slo_tpot, dt

    def models(self, T):
        return [GpuModel(g, self.lp, self.gl, T) for g in self.gp]


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def jsq_weights(x_list, kappa=None):
    """Fluid join-shortest-queue: split arrivals to equalize outstanding/kappa."""
    o = np.array([x[0] + x[1] for x in x_list], float)
    G = len(o)
    kappa = np.ones(G) if kappa is None else np.asarray(kappa, float)
    A = max(1.0, o.sum() * 0.05 + 1.0)
    # water-filling for a nominal batch A of arrivals (weights are what matter)
    lo, hi = 0.0, (o / kappa).max() + A / kappa.min() + 1
    for _ in range(50):
        L = 0.5 * (lo + hi)
        fill = np.maximum(0, L * kappa - o).sum()
        lo, hi = (L, hi) if fill < A else (lo, L)
    w = np.maximum(0, hi * kappa - o)
    return w / w.sum() if w.sum() > 0 else np.ones(G) / G


def project_budget(caps, fleet_models, budget, n_run=None):
    """Scale dynamic headroom (cap - P_static) proportionally so sum(caps) <= budget,
    respecting settable floors (a GPU cannot be capped below its floor)."""
    caps = np.asarray(caps, float).copy()
    ps = np.array([m.ps for m in fleet_models])
    cmin = np.array([m.cap_min for m in fleet_models])
    tdp = np.array([m.tdp for m in fleet_models])
    caps = np.clip(caps, cmin, tdp)
    if caps.sum() <= budget:
        return caps
    head = caps - cmin
    room = budget - cmin.sum()
    if room <= 0:
        return cmin                     # infeasible: floors exceed budget
    return cmin + head * (room / head.sum())


def shed_and_project(caps, fleet_models, budget, active=None, order=None):
    """Budget projection that is aware of settable floors.  If the floors of all
    serving GPUs exceed the budget, stop routing to GPUs (in `order`, default:
    highest index first -- heterogeneity-blind) until feasible; a GPU with no load
    draws only P_static.  Returns (caps, active_mask)."""
    G = len(fleet_models)
    act = np.ones(G, bool) if active is None else np.asarray(active, bool).copy()
    cmin = np.array([m.cap_min for m in fleet_models])
    ps = np.array([m.ps for m in fleet_models])
    order = list(range(G))[::-1] if order is None else list(order)
    for i in order:
        if (cmin[act].sum() + ps[~act].sum()) <= budget or act.sum() <= 1:
            break
        act[i] = False
    caps = np.asarray(caps, float).copy()
    sub = [m for m, a in zip(fleet_models, act) if a]
    caps[act] = project_budget(caps[act], sub, budget - ps[~act].sum())
    caps[~act] = cmin[~act]
    return caps, act


def masked(route, act):
    r = np.asarray(route, float) * act
    return r / r.sum() if r.sum() > 0 else act / max(act.sum(), 1)


class Forecaster:
    """Per-interval arrival-count forecaster: persistence | ewma | holt | oracle."""

    def __init__(self, kind="ewma", alpha=0.25, beta=0.05):
        self.kind, self.alpha, self.beta = kind, alpha, beta
        self.level = None
        self.trend = 0.0

    def update(self, a):
        if self.level is None:
            self.level = float(a)
            return
        if self.kind == "holt":
            prev = self.level
            self.level = self.alpha * a + (1 - self.alpha) * (self.level + self.trend)
            self.trend = self.beta * (self.level - prev) + (1 - self.beta) * self.trend
        else:
            self.level = self.alpha * a + (1 - self.alpha) * self.level

    def predict(self, N, hist, future=None, noise=0.0, rng=None):
        if self.kind == "oracle" and future is not None:
            f = np.asarray(future[:N], float)
            if len(f) < N:
                f = np.concatenate([f, np.full(N - len(f), f[-1] if len(f) else 0)])
            if noise > 0 and rng is not None:
                f = f * np.maximum(0.0, 1 + noise * rng.standard_normal(N))
            return f
        if not hist:
            return np.zeros(N)
        if self.kind == "persistence":
            return np.full(N, float(hist[-1]))
        if self.kind == "holt":
            tr = np.clip(self.trend, -0.1 * self.level, 0.1 * self.level)
            return np.maximum(0, self.level + tr * np.arange(1, N + 1))
        return np.full(N, self.level)


# ----------------------------------------------------------------------------
# baselines
# ----------------------------------------------------------------------------
class Controller:
    name = "base"
    uses_park = False

    def reset(self, fleet: FleetInfo):
        self.fleet = fleet
        self.M = fleet.models(0.8 * fleet.slo_tpot)
        self.solve_times = []

    def _out(self, caps, maxrun=None, route=None, park=None):
        G = self.fleet.G
        return dict(caps=np.asarray(caps, float),
                    max_running=(np.array([m.n_max for m in self.M]) if maxrun is None
                                 else np.asarray(maxrun, float)),
                    route=(np.ones(G) / G if route is None else np.asarray(route, float)),
                    park=(np.zeros(G, bool) if park is None else np.asarray(park, bool)))


class Uncapped(Controller):
    name = "Uncapped"

    def act(self, obs):
        return self._out([m.tdp for m in self.M], route=jsq_weights(obs["x"]))


class StaticOpt(Controller):
    """Fixed setting: route share per GPU (by type) + cap = floor + phi_type*(TDP-floor).
    Params tuned offline by experiments.tune_static() on a training seed."""
    name = "StaticOpt"

    def __init__(self, share=None, phi=None):
        self.share = share   # dict type -> total share of arrivals for that type
        self.phi = phi       # dict type -> fraction of (TDP - cap_min)

    def act(self, obs):
        names = self.fleet.names
        types = sorted(set(names))
        share = self.share or {t: 1.0 / len(types) for t in types}
        phi = self.phi or {t: 1.0 for t in types}
        cnt = {t: names.count(t) for t in types}
        route = np.array([share[n] / cnt[n] for n in names])
        route = route / route.sum() if route.sum() > 0 else np.ones(len(names)) / len(names)
        caps = [m.cap_min + phi[n] * (m.tdp - m.cap_min) for m, n in zip(self.M, names)]
        caps, act = shed_and_project(caps, self.M, obs["budget"])
        return self._out(caps, route=masked(route, act))


class POLCA(Controller):
    """POLCA-style reactive oversubscription control (arXiv 2308.12908 spirit):
    uncapped until measured fleet power crosses T_hi*budget, then cap all GPUs
    uniformly (fraction of dynamic range); release slowly below T_lo*budget.
    Reacts to the *previous* interval's measurement (no prediction)."""
    name = "POLCA"

    def __init__(self, t_hi=0.95, t_lo=0.80, release=0.05):
        self.t_hi, self.t_lo, self.release = t_hi, t_lo, release

    def reset(self, fleet):
        super().reset(fleet)
        self.frac = 1.0

    def act(self, obs):
        if obs["y"] is not None:
            P = sum(y["power_w"] for y in obs["y"])
            B = obs["budget"]
            if P > self.t_hi * B:
                ps = sum(m.ps for m in self.M)
                dyn = max(P - ps, 1.0)
                self.frac = max(0.0, self.frac * min(1.0, (self.t_hi * B - ps) / dyn))
            elif P < self.t_lo * B:
                self.frac = min(1.0, self.frac + self.release)
        caps = [m.cap_min + self.frac * (m.tdp - m.cap_min) for m in self.M]
        return self._out(caps, route=jsq_weights(obs["x"]))


class PIAdmission(Controller):
    """Budget-proportional caps (always feasible) + PI on each GPU's admission cap
    (max_running) holding the queue delay at a setpoint.  Heterogeneity-blind."""
    name = "PI-adm"

    def __init__(self, setpoint_frac=0.2, kp=20.0, ki=10.0, m_min=8):
        self.sp, self.kp, self.ki, self.m_min = setpoint_frac, kp, ki, m_min

    def reset(self, fleet):
        super().reset(fleet)
        self.m = np.array([mm.n_max for mm in self.M])
        self.e_prev = np.zeros(fleet.G)

    def act(self, obs):
        G = self.fleet.G
        B = obs["budget"]
        ps = np.array([m.ps for m in self.M])
        rng_ = np.array([m.tdp - m.ps for m in self.M])
        caps = ps + max(B - ps.sum(), 0) * rng_ / rng_.sum()
        caps, act = shed_and_project(caps, self.M, B)
        if obs["y"] is not None:
            e = np.array([y["queue_delay_s"] for y in obs["y"]]) - self.sp * self.fleet.slo_q
            kv = np.array([y.get("kv_frac", 0) for y in obs["y"]])
            self.m = self.m + self.kp * (e - self.e_prev) + self.ki * e
            self.m = np.where(kv > 0.92, np.minimum(self.m, [x[1] for x in obs["x"]]), self.m)
            self.m = np.clip(self.m, self.m_min, [mm.n_max for mm in self.M])
            self.e_prev = e
        return self._out(caps, maxrun=self.m, route=masked(jsq_weights(obs["x"]), act))


class PICap(Controller):
    """Per-GPU PI on the power cap: error = TPOT/T* - 1 + queue_delay/W*  (>0 raises
    the cap).  Drives each GPU toward the *lowest cap that meets the SLO*, then
    projects onto the budget.  Local feedback, no model, no forecast."""
    name = "PI-cap"

    def __init__(self, tpot_frac=0.8, q_frac=0.25, kp=0.15, ki=0.08):
        self.tf, self.qf, self.kp, self.ki = tpot_frac, q_frac, kp, ki

    def reset(self, fleet):
        super().reset(fleet)
        self.u = np.ones(fleet.G) * 0.5          # fraction of (TDP - cap_min)
        self.e_prev = np.zeros(fleet.G)

    def act(self, obs):
        f = self.fleet
        if obs["y"] is not None:
            tp = np.array([y["t_iter_s"] for y in obs["y"]])
            qd = np.array([y["queue_delay_s"] for y in obs["y"]])
            busy = np.array([y.get("busy_frac", 1.0) for y in obs["y"]])
            e = np.where(busy > 0.05, tp / (self.tf * f.slo_tpot) - 1.0, -0.2)
            e = np.clip(e, -1.0, 2.0) + np.clip(qd / (self.qf * f.slo_q), 0, 4.0)
            self.u = np.clip(self.u + self.kp * (e - self.e_prev) + self.ki * e, 0.0, 1.0)
            self.e_prev = e
        caps = [m.cap_min + u * (m.tdp - m.cap_min) for m, u in zip(self.M, self.u)]
        caps, act = shed_and_project(caps, self.M, obs["budget"])
        return self._out(caps, route=masked(jsq_weights(obs["x"]), act))


# ----------------------------------------------------------------------------
# economic MPC
# ----------------------------------------------------------------------------
class EconMPC(Controller):
    """Receding-horizon economic MPC, solved as a small MILP with HiGHS.

    Per GPU i and step j (j = 0..N-1), variables
        A_ij  routed demand (tok/s)   D_ij service (tok/s)   pi_ij dynamic power (W)
        B_i,j+1 waiting backlog (tok)  s_ij SLO slack (tok)    z_ij on/serving (binary)
    Dynamics  B_i,j+1 = B_ij + dt*(A_ij - D_ij),  B_i0 = measured waiting tokens
    Coupling  sum_i A_ij = lambda_hat_j * Gbar   (forecast)
              sum_i (P_static_i + pi_ij) <= budget_j   (preview if announced)
    Service   pi >= a/T z + b D ;  pi >= c D ;  pi >= floor z ;  pi <= (TDP-Ps) z ;
              D <= D_max z ;  A <= Amax z
    SLO       B_i,j+1 <= W_target * D_ij + s_ij          (Little's law on the queue)
    Cost      sum_j dt*sum_i pi_ij + rho_s*sum s + rho_T*sum_i B_i,N
    First-step caps = P_static + max(pi_i0, floor) (+ adaptive gain); routing ~ A_i0.
    """
    name = "MPC"

    def __init__(self, N=10, forecast="ewma", preview=True, blind=False, adapt=True,
                 binaries=True, T_frac=0.8, W_frac=0.25, rho_s=1.0, fc_noise=0.0,
                 seed=0, time_limit=0.3, drain_aware=True, du_w=0.3, name=None):
        self.du_w = du_w
        self.drain_aware = drain_aware
        self.N, self.fc_kind, self.preview, self.blind = N, forecast, preview, blind
        self.adapt, self.binaries = adapt, binaries
        self.T_frac, self.W_frac, self.rho_s = T_frac, W_frac, rho_s
        self.fc_noise, self.time_limit = fc_noise, time_limit
        self.rng = np.random.default_rng(seed)
        if name:
            self.name = name

    def reset(self, fleet):
        self.fleet = fleet
        T = self.T_frac * fleet.slo_tpot
        gps = fleet.gp
        if self.blind:   # heterogeneity-blind: every GPU = the fleet-average GPU
            keys = [k for k, v in gps[0].items() if isinstance(v, (int, float))]
            avg = {k: float(np.mean([g[k] for g in gps])) for k in keys}
            gps = [dict(g, **avg) for g in gps]
        self.gp_ctrl = gps
        self.M = [GpuModel(g, fleet.lp, fleet.gl, T) for g in gps]
        self.T = T
        self.theta = np.ones(fleet.G)          # online energy-gain estimate
        self.kappa = np.ones(fleet.G)          # offset-free TPOT correction (T_i = kappa_i*T)
        self.z_last = None
        self.p_last = np.zeros(fleet.G)
        self.fc = Forecaster(self.fc_kind)
        self.solve_times = []
        self.last_status = None
        self._last = None

    # -- online adaptation: measured dyn energy vs model at observed operating point
    def _adapt(self, obs):
        if not self.adapt or obs["y"] is None:
            return
        # offset-free TPOT: shrink the iteration-time target where measured TPOT
        # overshoots it (model/plant mismatch, bursts); relax slowly otherwise
        tgt = self.T
        for i, y in enumerate(obs["y"]):
            if y.get("busy_frac", 1.0) > 0.2 and y["t_iter_s"] > 0:
                r = y["t_iter_s"] / tgt
                if r > 1.0:
                    self.kappa[i] = max(0.4, self.kappa[i] * (1 - 0.3 * min(r - 1, 1)))
                else:
                    self.kappa[i] = min(1.0, self.kappa[i] + 0.01)
        for i, (y, m) in enumerate(zip(obs["y"], self.M)):
            if y.get("busy_frac", 1.0) < 0.2 or y["t_iter_s"] <= 0:
                continue
            n_iter = y.get("iterations", y.get("busy_frac", 1.0) * obs["dt"] / y["t_iter_s"])
            e_model = n_iter * m.a + y["tokens_out"] * m.b
            e_meas = y["energy_j"] - m.ps * obs["dt"]
            if e_model > 1.0 and e_meas > 0:
                r = float(np.clip(e_meas / e_model, 0.5, 2.0))
                self.theta[i] = 0.9 * self.theta[i] + 0.1 * r

    def act(self, obs):
        t0 = time.perf_counter()
        f, N, G, dt = self.fleet, self.N, self.fleet.G, obs["dt"]
        if obs["arr_hist"]:
            self.fc.update(obs["arr_hist"][-1])
        self._adapt(obs)
        lam = self.fc.predict(N, obs["arr_hist"], obs.get("arr_future"),
                              self.fc_noise, self.rng) / dt           # req/s
        dem = lam * f.gl                                                # tok/s
        if self.preview and obs.get("budget_announced", False):
            bud = np.asarray(obs["budget_future"][:N], float)
            if len(bud) < N:
                bud = np.concatenate([bud, np.full(N - len(bud), bud[-1])])
        else:
            bud = np.full(N, obs["budget"])
        B0 = np.array([x[0] for x in obs["x"]]) * f.gl
        nrun = np.array([x[1] for x in obs["x"]])
        self._avail = obs.get("avail", np.ones(G, bool))
        self._idle_w = obs.get("idle_w", None)
        sol = self._solve(dem, bud, B0, nrun, dt)
        self.solve_times.append(time.perf_counter() - t0)
        return sol

    def _solve(self, dem, bud, B0, nrun, dt):
        from scipy.sparse import coo_matrix
        f, N, G = self.fleet, self.N, self.fleet.G
        M, th = self.M, self.theta
        nv = 6  # per (i,j): A, D, pi, Bn, s1, s2 ; per i: z (on over horizon), u (turn-on)
        n = nv * G * N + 2 * G + 2 * G + N   # + move-suppression |dpi0| (2G) + budget slack (N)
        idx = lambda v, i, j: (j * G + i) * nv + v
        zi = lambda i: nv * G * N + i
        ui = lambda i: nv * G * N + G + i
        mpi = lambda i: nv * G * N + 2 * G + i          # (pi_i0 - pi_last)^+
        mni = lambda i: nv * G * N + 3 * G + i          # (pi_i0 - pi_last)^-
        bsl = lambda j: nv * G * N + 4 * G + j          # budget slack (W), huge cost
        Wt = self.W_frac * f.slo_q
        c = np.zeros(n)
        lb = np.zeros(n)
        ub = np.full(n, np.inf)
        integ = np.zeros(n)
        ri, ci, vv, lo_, hi_ = [], [], [], [], []

        def add(coefs, lo, hi):
            r = len(lo_)
            for k, v in coefs:
                ri.append(r); ci.append(k); vv.append(v)
            lo_.append(lo); hi_.append(hi)

        amax = max(dem.max(), 1.0) * 1.5 + B0.sum() / dt
        cmax = max(m.c for m in M)
        s1max = max(dem.mean(), 1.0) * f.slo_q / G      # tier-1 slack per GPU (tok)
        # previous on/off = the controller's own last decision (a GPU it switched off
        # but that is still draining must pay the turn-on cost to be reused)
        z_prev = ((nrun > 0.5) if self.z_last is None else self.z_last).astype(float)
        # drain time if switched off now: residual tokens x iteration time at floor
        kdr = np.zeros(G, int)
        if self.drain_aware:
            for i, m in enumerate(M):
                if nrun[i] > 0.5:
                    nb = min(nrun[i], m.n_max)
                    t_fl = (m.a + m.b * nb) / max(m.floor, 1.0)
                    kdr[i] = int(np.ceil(0.6 * f.gl * t_fl / dt))
        self._kdr = kdr
        ps = (sum(m.ps for m in M) if self._idle_w is None else float(np.sum(self._idle_w)))
        for i, m in enumerate(M):
            Z, U = zi(i), ui(i)
            ub[Z] = 1.0
            if self.binaries:
                integ[Z] = 1
            else:
                lb[Z] = 1.0
            if not self._avail[i]:
                ub[Z] = 0.0                 # parked / waking / draining-to-park
            elif self.drain_aware is False and nrun[i] > 0.5:
                lb[Z] = 1.0                 # (legacy) running work => stays on
            c[U] += 2.0 * m.floor * dt      # turn-on cost: damps churn
            add([(U, 1.0), (Z, -1.0)], -z_prev[i], np.inf)
            # switching off a GPU that holds work still costs its floor while draining
            if kdr[i] > 0:
                c[Z] -= (m.cap_min - m.ps) * min(kdr[i], N) * dt
            # move suppression on the applied power (Delta-u penalty)
            c[mpi(i)] += self.du_w
            c[mni(i)] += self.du_w
            add([(mpi(i), 1.0), (mni(i), -1.0), (idx(2, i, 0), -1.0)], -self.p_last[i], -self.p_last[i])
        for j in range(N):
            for i, m in enumerate(M):
                A, D, P, Bn, S1, S2 = (idx(v, i, j) for v in range(nv))
                Z = zi(i)
                c[P] += dt
                c[S1] += self.rho_s
                c[S2] += 5.0 * self.rho_s               # convex (tiered) SLO penalty
                c[Bn] += 1e-4                           # mild preference to serve early
                if j == N - 1:
                    c[Bn] += 2.0 * cmax * th.max()      # terminal: backlog costs later
                ub[S1] = s1max
                ub[P] = m.tdp - m.ps
                ub[D] = m.d_max
                prevB = [] if j == 0 else [(idx(3, i, j - 1), -1.0)]
                rhs = B0[i] if j == 0 else 0.0
                add([(Bn, 1.0), (A, -dt), (D, dt)] + prevB, rhs, rhs)
                # power envelopes (scaled by adaptive gain theta)
                add([(P, 1.0), (Z, -th[i] * m.a / (self.T * self.kappa[i])), (D, -th[i] * m.b)], 0, np.inf)
                add([(P, 1.0), (D, -th[i] * m.c)], 0, np.inf)
                add([(P, 1.0), (Z, -m.floor)], 0, np.inf)
                add([(P, 1.0), (Z, -(m.tdp - m.ps))], -np.inf, 0)
                add([(D, 1.0), (Z, -m.d_max)], -np.inf, 0)
                add([(A, 1.0), (Z, -amax)], -np.inf, 0)
                # SLO via Little's law on the waiting queue
                add([(Bn, 1.0), (D, -Wt), (S1, -1.0), (S2, -1.0)], -np.inf, 0)
            add([(idx(0, i, j), 1.0) for i in range(G)], dem[j], dem[j])
            # budget; a GPU switched off while holding running work keeps drawing up
            # to its floor cap until it drains (k_drain steps) -- modelled here
            dr = [i for i in range(G) if kdr[i] > j]
            c[bsl(j)] += 1e3                  # budget is hard in spirit: 1 kJ per W over
            add([(idx(2, i, j), 1.0) for i in range(G)] + [(bsl(j), -1.0)]
                + [(zi(i), -(M[i].cap_min - M[i].ps)) for i in dr],
                -np.inf, bud[j] - ps - sum(M[i].cap_min - M[i].ps for i in dr))
        Aeq = coo_matrix((vv, (ri, ci)), shape=(len(lo_), n)).tocsr()
        res = milp(c, constraints=LinearConstraint(Aeq, lo_, hi_), bounds=Bounds(lb, ub),
                   integrality=integ,
                   options=dict(time_limit=self.time_limit, mip_rel_gap=1e-3, disp=False))
        self.last_status = res.status
        if res.x is None or not np.all(np.isfinite(res.x)):
            # infeasible (budget below floors) or timeout: budget-proportional fallback
            self.n_fallback = getattr(self, "n_fallback", 0) + 1
            caps = project_budget([m.tdp for m in M], M, bud[0])
            return self._out(caps, route=np.ones(G) / G)
        x = res.x
        P0 = np.array([x[idx(2, i, 0)] for i in range(G)])
        A0 = np.maximum(np.array([x[idx(0, i, 0)] for i in range(G)]), 0)
        Z0 = np.array([x[zi(i)] for i in range(G)]) > 0.5
        caps = np.array([m.ps + max(p, m.floor) for m, p in zip(M, P0)])
        # project the serving GPUs onto budget - P_static(idle GPUs), using the
        # *true* nameplate limits (matters in blind mode, whose model is averaged)
        drain_extra = sum(M[i].cap_min - M[i].ps for i in range(G)
                          if (not Z0[i]) and nrun[i] > 0.5)
        caps, _ = shed_and_project(caps, self.fleet.models(self.T),
                                   bud[0] - drain_extra + (0 if self._idle_w is None else
                                             float(np.sum(np.array([m.ps for m in M]) - self._idle_w))),
                                   active=Z0)
        if A0.sum() > 1e-9:
            route = A0 / A0.sum()
        elif Z0.any():
            route = Z0 / Z0.sum()
        else:
            route = np.ones(G) / G
        self._last = dict(P=P0, A=A0, Z=Z0)
        self.z_last = Z0.copy()
        self.p_last = np.where(Z0, P0, 0.0)
        return self._out(caps, route=route)


# ----------------------------------------------------------------------------
# slow outer loop: GPU parking (idle-power-aware consolidation)
# ----------------------------------------------------------------------------
class ParkingLoop(Controller):
    """Wraps any fast controller with a slow (period ~30 s) park/wake decision.
    aware=True : enumerate awake subsets with the per-GPU model; minimise
                 sum_awake(P_static + pi(D_i)) + sum_parked(P_park) s.t. capacity
                 >= margin * recent-peak demand (so it knows B200 idles at 2x H200).
    aware=False: heterogeneity-blind -- wake the fewest GPUs (index order) whose
                 *average* capacity covers the demand.
    Parked GPUs draw p_park_frac*P_static; waking takes t_wake s at P_static."""

    def __init__(self, inner, aware=True, period=30, margin=1.6, p_park_frac=0.3,
                 look=120, name=None):
        self.inner, self.aware, self.period, self.margin = inner, aware, period, margin
        self.ppf, self.look = p_park_frac, look
        self.name = name or f"{inner.name}+{'park' if aware else 'blindpark'}"

    def reset(self, fleet):
        super().reset(fleet)
        self.inner.reset(fleet)
        self.want = np.zeros(fleet.G, bool)
        self.solve_times = []

    @property
    def N(self):
        return getattr(self.inner, "N", 10)

    def _choose(self, D, budget):
        import itertools
        M, G = self.M, self.fleet.G
        cap_i = np.array([min(m.d_max, (m.tdp - m.ps) / m.c) for m in M])
        if not self.aware:
            k = 1
            while k < G and k * cap_i.mean() < self.margin * D:
                k += 1
            awake = np.zeros(G, bool)
            awake[:k] = True
            return awake
        best, best_cost = np.ones(G, bool), np.inf
        for r in range(1, G + 1):
            for S in itertools.combinations(range(G), r):
                S = list(S)
                if cap_i[S].sum() < self.margin * D:
                    continue
                # greedy split: cheapest marginal energy first, each needs >= floor
                cost = sum(M[i].ps for i in S) + sum(self.ppf * M[i].ps
                                                     for i in range(G) if i not in S)
                rem = D
                for i in sorted(S, key=lambda i: M[i].c):
                    d = min(rem, cap_i[i])
                    cost += M[i].pi_needed(d) if d > 0 else 0.0
                    rem -= d
                floors = sum(M[i].cap_min for i in S)
                if floors + sum(self.ppf * M[i].ps for i in range(G) if i not in S) > budget:
                    continue
                if cost < best_cost - 1e-6:
                    best, best_cost = np.zeros(G, bool), cost
                    best[S] = True
        return best

    def act(self, obs):
        t0 = time.perf_counter()
        k = obs["k"]
        if k % self.period == 0:
            hist = obs["arr_hist"][-self.look:] or [0.0]
            D = (max(hist) / obs["dt"]) * self.fleet.gl
            q = sum(x[0] for x in obs["x"]) * self.fleet.gl / (self.period * obs["dt"])
            self.want = ~self._choose(D + q, obs["budget"])
        mode = obs.get("mode", np.zeros(self.fleet.G, int))
        avail = (mode == 0) & ~self.want
        idle_w = np.array([m.ps * (self.ppf if md == 1 else 1.0) for m, md in zip(self.M, mode)])
        obs = dict(obs, avail=avail, idle_w=idle_w)
        out = self.inner.act(obs)
        out["route"] = masked(out["route"], avail) if avail.any() else out["route"]
        out["park"] = self.want.copy()
        self.solve_times.append(time.perf_counter() - t0)
        return out
