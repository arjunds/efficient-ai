"""
controls/plant.py — discrete-time plant model of one GPU serving one LLM under
vLLM (V1) continuous batching, identified from our iter_log / power logs.

    x_{k+1}, y_k = Plant.step(x_k, u_k, d_k)          (pure numpy, deterministic)

State (fixed dimension, see `state_names`):
    q_wait   requests queued, not yet admitted (gateway queue + vLLM waiting)
    n_run    admitted requests (prefilling + decoding)  == n_pf + sum(n_dec_j)
    kv_res   resident KV tokens as vLLM reports them (incl. ~half-block rounding)
    q_tok    prompt tokens held by the queued requests
    pf_tok   prompt tokens of admitted requests whose prefill is not done
    n_pf     admitted requests still in prefill
    n_think  closed-loop clients between a completion and their next request
             (client/front-end resubmit latency `think_s`, identified ~30 ms)
    n_dec_1..n_dec_m, kv_1..kv_m
             decode requests / their KV tokens in m Erlang "progress" stages
             (stage length = gen_len/m tokens -> gen-length CV^2 = 1/m). This
             age structure is what makes completions and KV drain lag the
             admissions realistically (a step in load -> no completions for
             ~gen_len iterations), instead of the memoryless m=1 fluid.

Input u (dict):  'max_running' admission/concurrency cap (actuatable, any time,
                 by a gateway in front of vLLM); 'power_cap_w' (nvidia-smi -pl,
                 actuatable at ~s timescale). Both optional.
Disturbance d:   'arrivals' requests arriving this interval (open loop) and/or
                 'clients' (closed-loop population: queue topped up to this many
                 requests in system, which is what our load generator does);
                 'prompt_len' mean prompt tokens of arrivals; 'gen_len' mean
                 generated tokens per request (population); 'cached_frac'
                 fraction of prompt tokens served from the prefix cache (not
                 computed); 'kv_shared_frac' fraction of prompt tokens whose KV
                 blocks are shared with already-resident requests (not added to
                 kv_res; ~cached_frac for identical synthetic prompts, ~0 for
                 real prompt pools).

Service model (identified, see SYSID.md):  per engine iteration
    t_iter = t_ser + tau*n_run + pnorm_p( t_cpu,
                 max( W/(BW*u_w) + KV_bytes/(BW*u_kv),  F_gemm/(peak*u_f) ) )
i.e. a roofline max on the GPU side, overlapped (p-norm; p->inf = max, p=1 =
sum) with the CPU launch path of eager-mode vLLM, plus serial per-iteration and
per-sequence costs. Iterations carry decode tokens for every decoding request
plus chunked prefill up to the token budget (2048). Energy per iteration is the
validated 3-term model  E = e_wbyte*W + e_kvbyte*KV_bytes + e_gemm*F,  power =
P_static + E_dyn/dt. If that exceeds the power cap the GPU is throttled: busy
time stretches to E_dyn/(P_cap - P_static) (first-order; NOT validated on data);
y['t_iter_s'] / y['tpot_s'] then report the STRETCHED per-iteration time
(y['t_iter_unthrottled_s'], y['throttle_factor'] give the unthrottled one and
the ratio). The requested cap is clipped to [p_cap_min_w, p_cap].
"""
import json
import math
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS_JSON = os.path.join(_HERE, "plant_params.json")

_DEFAULT_HW = {
    "H200": dict(e_wbyte=1.076e-10, e_kvbyte=3.376e-10, e_gemm=0.680e-12,
                 p_static=116.4, p_cap=700.0, bw=4.8e12, peak_flops=9.9e14),
    "B200": dict(e_wbyte=1.252e-10, e_kvbyte=3.043e-10, e_gemm=0.539e-12,
                 p_static=237.8, p_cap=1000.0, bw=8.0e12, peak_flops=2.25e15),
}


_CAP_MIN = {"H200": 200.0, "B200": 300.0, "A5000": 100.0}   # see __init__ note


def _load_params():
    try:
        with open(PARAMS_JSON) as f:
            return json.load(f)
    except Exception:
        return {}


def _pnorm(x, y, p):
    m = max(x, y, 1e-12)
    if p >= 50:
        return m
    return m * ((x / m) ** p + (y / m) ** p) ** (1.0 / p)


class Plant:
    BASE_STATES = ["q_wait", "n_run", "kv_res", "q_tok", "pf_tok", "n_pf", "n_think"]

    def __init__(self, gpu: str, model: str, dt: float = 1.0, **overrides):
        """gpu: 'H200' | 'B200' (or any name with bw/peak_flops/... overrides).
        model: HF short name, e.g. 'Qwen2-7B-Instruct' (see plant_params.json
        model_consts); unknown models need weight_bytes, kv_bytes_per_token,
        gemm_flops_per_token, n_layers overrides and use the GPU-level law.
        overrides: any key of self.gpu_params (e.g. p_static=120, u_f=0.5,
        kv_capacity=5e5, max_num_seqs=128, token_budget=2048, n_stages=4,
        substeps=20, gen_len=200, prompt_len=100, cached_frac=0.0)."""
        P = _load_params()
        self.gpu, self.model, self.dt = gpu, model, float(dt)
        hw = dict(_DEFAULT_HW.get(gpu, {}))
        hw.update({k: v for k, v in (P.get("gpu_hw", {}).get(gpu) or {}).items()
                   if isinstance(v, (int, float))})
        consts = dict(P.get("model_consts", {}).get(model) or {})
        key = f"{gpu}|{model}"
        tm = (P.get("time_model", {}) or {}).get(key)
        law = (P.get("time_law", {}) or {}).get(gpu)
        g = {}
        g.update(hw)
        for k in ("weight_bytes", "kv_bytes_per_token", "gemm_flops_per_token",
                  "n_layers", "moe"):
            g[k] = consts.get(k)
        if tm is not None:
            th = tm["forms"]["phys_p"]["theta"]
            g.update(t_ser=th["t_ser"], t_cpu=th["t_cpu"], tau_seq=th["tau_seq"],
                     u_kv=th["u_kv"], u_f=th["u_f"], u_w=tm["u_w_pinned"],
                     p_overlap=tm["p_pinned"], time_source=f"fit {key}")
        elif law is not None:
            th = law["theta"]
            g.update(t_ser=th["a"], t_cpu=th["b_per_layer"] * float(g.get("n_layers") or 32),
                     tau_seq=th["tau_seq"], u_kv=th["u_kv"], u_f=th["u_f"],
                     u_w=th["u_w"], p_overlap=law["p"], time_source=f"GPU law {gpu}")
        else:   # no identification for this GPU: generic eager-vLLM defaults
            g.update(t_ser=2.0e-3, t_cpu=3.0e-3, tau_seq=1.2e-5, u_kv=0.4, u_f=0.55,
                     u_w=0.75, p_overlap=12.0, time_source="generic defaults")
        wl = (P.get("workload", {}) or {}).get(model, {})
        g.update(kv_capacity=float((P.get("kv_capacity", {}) or {}).get(key) or 1e6),
                 max_num_seqs=128, token_budget=int(P.get("token_budget", 2048)),
                 block_size=int(P.get("block_size", 16)),
                 n_stages=int(wl.get("n_stages", 4)), substeps=20,
                 gen_len=float(wl.get("gen_len", 200.0)),
                 prompt_len=float(wl.get("prompt_len", 100.0)), cached_frac=0.0,
                 kv_shared_frac=0.0,
                 think_s=float((P.get("think_s", {}) or {}).get(gpu, 0.03)))
        # settable power-limit floor. ASSUMPTIONS unless measured: H200 200 W
        # (as H100 SXM), B200 300 W (both as in plant_fp.py); A5000 100 W
        # (measured enforced limit on node-d1). Override with p_cap_min_w=...
        g["p_cap_min_w"] = float(_CAP_MIN.get(gpu, 0.0))
        g.update(overrides)
        if "p_cap_min" in overrides and "p_cap_min_w" not in overrides:
            g["p_cap_min_w"] = float(overrides["p_cap_min"])
        g["p_cap_min"] = g["p_cap_min_w"]          # alias used by controllers.py
        g["mbu"], g["mfu"], g["overhead"] = g["u_w"], g["u_f"], g["t_ser"]
        missing = [k for k in ("weight_bytes", "kv_bytes_per_token",
                               "gemm_flops_per_token", "bw", "peak_flops",
                               "p_static", "p_cap") if g.get(k) is None]
        if missing:
            raise ValueError(f"Plant({gpu},{model}): missing params {missing}; "
                             "pass them as overrides")
        self.gpu_params = g
        m = int(g["n_stages"])
        self.m = m
        self.state_names = (self.BASE_STATES + [f"n_dec_{j+1}" for j in range(m)]
                            + [f"kv_{j+1}" for j in range(m)])
        self.nx = len(self.state_names)
        self._i = {n: i for i, n in enumerate(self.state_names)}

    # ------------------------------------------------------------- physics
    def weight_bytes(self, tok):
        g = self.gpu_params
        moe = g.get("moe")
        if not moe:
            return g["weight_bytes"]
        E, k = moe["n_experts"], moe["top_k"]
        frac = 1.0 - (1.0 - k / E) ** max(tok, 1.0)
        return g["weight_bytes"] + max(0.0, frac * E - k) * moe["expert_bytes_all_layers"]

    def t_iter(self, n_run, kv_tokens, tok):
        """Seconds for one engine iteration processing `tok` tokens with
        n_run sequences and kv_tokens resident."""
        g = self.gpu_params
        W = self.weight_bytes(tok)
        mem = W / (g["bw"] * g["u_w"]) + kv_tokens * g["kv_bytes_per_token"] / (g["bw"] * g["u_kv"])
        cmp_ = g["gemm_flops_per_token"] * tok / (g["peak_flops"] * g["u_f"])
        return (g["t_ser"] + g["tau_seq"] * max(n_run, 1.0)
                + _pnorm(g["t_cpu"], max(mem, cmp_), g["p_overlap"]))

    def e_iter(self, kv_tokens, tok):
        g = self.gpu_params
        return (g["e_wbyte"] * self.weight_bytes(tok)
                + g["e_kvbyte"] * kv_tokens * g["kv_bytes_per_token"]
                + g["e_gemm"] * g["gemm_flops_per_token"] * tok)

    # ------------------------------------------------------------- state io
    def reset(self, x0=None) -> np.ndarray:
        x = np.zeros(self.nx)
        if x0 is None:
            return x
        if isinstance(x0, dict):
            return self.state_from_obs(**x0)
        x0 = np.asarray(x0, float)
        if x0.size == self.nx:
            return x0.copy()
        return self.state_from_obs(*x0[:3])

    def state_from_obs(self, q_wait=0.0, n_run=0.0, kv_res=0.0, prompt_len=None,
                       gen_len=None, clients=None, **_):
        """Full state from the three observables (q_wait, n_run, kv_res),
        assuming the stationary age distribution: requests spread evenly over
        the m decode stages, stage j holding prompt + (j-0.5)/m*gen tokens,
        rescaled to match the observed kv_res. No prefill in flight."""
        g = self.gpu_params
        p = float(prompt_len if prompt_len is not None else g["prompt_len"])
        gl = float(gen_len if gen_len is not None else g["gen_len"])
        m = self.m
        x = np.zeros(self.nx)
        i = self._i
        x[i["q_wait"]] = max(q_wait, 0.0)
        x[i["q_tok"]] = max(q_wait, 0.0) * p
        n = max(n_run, 0.0)
        shared = g["kv_shared_frac"] * p if n > 0.5 else 0.0
        kv_net = max(kv_res - 0.5 * g["block_size"] * n - shared, 0.0)
        shape = np.array([p + (j + 0.5) / m * gl for j in range(m)])
        nj = np.full(m, n / m)
        kvj = nj * shape
        if kvj.sum() > 0:
            kvj *= kv_net / kvj.sum()
        x[i["n_dec_1"]:i["n_dec_1"] + m] = nj
        x[i["kv_1"]:i["kv_1"] + m] = kvj
        x[i["n_run"]] = n
        x[i["kv_res"]] = kv_net + 0.5 * g["block_size"] * n + shared
        if clients is not None:        # closed loop: the rest are in transit
            x[i["n_think"]] = max(float(clients) - n - max(q_wait, 0.0), 0.0)
        return x

    # ------------------------------------------------------------- dynamics
    @staticmethod
    def _erlang_advance(nj, kvj, tau, rate):
        """Exact fluid transport through the m-stage Erlang chain when every
        decoding request generates `tau` tokens (stage exit rate `rate` per
        token): P(advance j stages) = Poisson(j; rate*tau). Returns new
        (nj, kvj) and the completed (n, kv)."""
        m = len(nj)
        lam = rate * tau
        pm = np.empty(m)
        pm[0] = math.exp(-lam)
        for j in range(1, m):
            pm[j] = pm[j - 1] * lam / j
        kv_src = kvj + nj * tau                     # every request grew by tau
        n_new = np.zeros(m)
        kv_new = np.zeros(m)
        done_n = done_kv = 0.0
        for i in range(m):
            if nj[i] <= 0.0 and kv_src[i] <= 0.0:
                continue
            k = m - i                                # stages ahead incl. own
            n_new[i:] += nj[i] * pm[:k]
            kv_new[i:] += kv_src[i] * pm[:k]
            stay = pm[:k].sum()
            done_n += nj[i] * (1.0 - stay)
            done_kv += kv_src[i] * (1.0 - stay)
        return n_new, kv_new, max(done_n, 0.0), max(done_kv, 0.0)

    def step(self, x, u: dict = None, d: dict = None):
        """Advance one control interval dt. Returns (x_next, y).

        Per substep h = dt/substeps:  (1) arrivals / closed-loop returns and
        admission, (2) service: prefill-carrying + decode iterations sized by
        the identified t_iter (and throttled by the power cap), exact Erlang
        transport of decode progress, completions, (3) same-substep client
        returns and a second admission pass (so boundary states match vLLM's
        post-schedule n_running)."""
        u, d = (u or {}), (d or {})
        g = self.gpu_params
        m, i = self.m, self._i
        x = np.array(x, float)
        S = dict(q=x[i["q_wait"]], qtok=x[i["q_tok"]], pf_tok=x[i["pf_tok"]],
                 n_pf=x[i["n_pf"]], n_think=x[i["n_think"]])
        nj = x[i["n_dec_1"]:i["n_dec_1"] + m].copy()
        kvj = x[i["kv_1"]:i["kv_1"] + m].copy()

        cap_run = min(float(u.get("max_running", g["max_num_seqs"])), float(g["max_num_seqs"]))
        p_cap = float(u.get("power_cap_w", g["p_cap"]))
        p_cap_req = p_cap
        # settable range of the power limit: [p_cap_min_w, p_cap] (nvidia-smi -pl)
        p_cap = min(max(p_cap, g["p_cap_min_w"], g["p_static"] + 1.0), g["p_cap"])
        arr = float(d.get("arrivals", 0.0))
        clients = d.get("clients", None)
        p_len = float(d.get("prompt_len", g["prompt_len"]))
        gen = max(float(d.get("gen_len", g["gen_len"])), 1.0)
        cf = min(max(float(d.get("cached_frac", g["cached_frac"])), 0.0), 1.0)
        shf = min(max(float(d.get("kv_shared_frac", g["kv_shared_frac"])), 0.0), 1.0)
        B = float(g["token_budget"])
        half_blk = 0.5 * g["block_size"]
        kv_cap = float(g["kv_capacity"])
        M = int(g["substeps"])
        h = self.dt / M
        Z = max(float(g["think_s"]), 1e-6)
        z_back = 1.0 - math.exp(-h / Z)
        z_same = 1.0 - (Z / h) * z_back            # returned within the same substep
        acc = dict(E=0.0, tok=0.0, comp=0.0, iters=0.0, busy=0.0, adm=0.0,
                   arr=0.0, done=0.0, throttled=False, tc_max=0.0, busy_u=0.0)

        def kv_resident(n_run):
            return kvj.sum() + half_blk * n_run

        def add_arrivals(a_new, back):
            """open-loop arrivals + returning closed-loop clients (population
            capped at `clients`; surplus returners leave, deficit = new)."""
            n_run = S["n_pf"] + nj.sum()
            if clients is not None:
                c_ = float(clients)
                room = max(0.0, c_ - (S["q"] + a_new + n_run + S["n_think"]))
                a_new += min(back, room)
                room = max(0.0, c_ - (S["q"] + a_new + n_run + S["n_think"]))
                a_new += room
            S["q"] += a_new
            S["qtok"] += a_new * p_len
            acc["arr"] += a_new

        def admit():
            q = S["q"]
            if q <= 1e-9:
                S["q"], S["qtok"] = 0.0, 0.0
                return
            n_dec = nj.sum()
            n_run = S["n_pf"] + n_dec
            p_q = S["qtok"] / q
            kv_now = kv_resident(n_run)
            t_d = self.t_iter(max(n_run, 1.0), kv_now, max(n_dec, 1.0))
            budget_tok = (h / t_d) * max(B - n_dec, 1.0)
            pend = S["pf_tok"] * (1.0 - cf)
            a_budget = max(0.0, budget_tok - pend) / max(p_q * (1.0 - cf), 1.0)
            a_kv = max(0.0, 0.98 * kv_cap - kv_now - S["pf_tok"]) / max(p_q + half_blk, 1.0)
            a = max(0.0, min(q, cap_run - n_run, a_budget, a_kv))
            if a > 0:
                S["q"] -= a
                S["qtok"] = max(S["qtok"] - a * p_q, 0.0)
                S["pf_tok"] += a * p_q
                S["n_pf"] += a
                acc["adm"] += a
            if S["q"] < 1e-9:
                S["q"], S["qtok"] = 0.0, 0.0

        for _ in range(M):
            # (1) arrivals, returns, admission
            back = 0.0
            if clients is not None:
                back = S["n_think"] * z_back
                S["n_think"] -= back
            else:
                S["n_think"] = 0.0
            add_arrivals(arr * h / self.dt, back)
            admit()
            n_dec = nj.sum()
            n_pf = S["n_pf"]
            n_run = n_pf + n_dec
            if n_run <= 1e-9:
                continue                           # idle substep: P_static only
            # (2) service. Fluid n_run < 1 means the GPU is busy only a fraction
            # occ = min(1, n_run) of the time (with batch n_run/occ); request
            # progress (tokens per request) is unaffected, iterations/energy scale.
            occ = min(1.0, n_run)
            kv_b = kv_resident(n_run) / occ
            nb, nd_b = n_run / occ, n_dec / occ
            X = S["pf_tok"] * (1.0 - cf) / occ     # computed prefill tokens pending
            t_d = self.t_iter(nb, kv_b, nd_b)
            if X > 0:
                room = max(B - nd_b, 1.0)
                k_c = max(X / room, min(n_pf / occ, h / t_d))   # >=1 iteration per prefill
                tok_c = nd_b + X / max(k_c, 1e-9)
                t_c = self.t_iter(nb, kv_b, tok_c)
                if k_c * t_c <= h:
                    phi, n_c, n_d = 1.0, k_c, (h - k_c * t_c) / t_d
                else:
                    phi = h / (k_c * t_c)
                    n_c, n_d = k_c * phi, 0.0
                acc["tc_max"] = max(acc["tc_max"], t_c)
            else:
                phi, n_c, n_d, t_c, tok_c = 0.0, 0.0, h / t_d, 0.0, 0.0
            E = n_d * self.e_iter(kv_b, nd_b)
            if n_c:
                E += n_c * self.e_iter(kv_b, tok_c)
            E *= occ
            busy = occ * (n_d * t_d + n_c * t_c)
            s = 1.0                                # power-cap throttle
            if E > 0 and g["p_static"] + E / h > p_cap:
                s = (p_cap - g["p_static"]) * h / E
                acc["throttled"] = True
            tau = s * (n_c + n_d)                  # iterations seen by each request
            phi *= s
            acc["E"] += s * E
            acc["iters"] += occ * tau
            acc["busy"] += busy                    # throttled: same busy time, fewer iters
            acc["busy_u"] += s * busy              # unthrottled time of the iterations done
            if n_c and s < 1.0:
                acc["tc_max"] = max(acc["tc_max"], t_c / s)
            # prefill completions -> decode stage 1 (first token emitted)
            done_pf = n_pf * phi
            tok_done = S["pf_tok"] * phi
            acc["comp"] += tok_done * (1.0 - cf)
            S["n_pf"] -= done_pf
            S["pf_tok"] -= tok_done
            if S["n_pf"] < 1e-9:
                S["n_pf"], S["pf_tok"] = 0.0, 0.0
            # decode progress (exact Erlang transport), completions
            acc["tok"] += n_dec * tau + done_pf
            nj_new, kv_new, dn, _dkv = self._erlang_advance(nj, kvj, tau, m / gen)
            nj[:] = nj_new
            kvj[:] = kv_new
            nj[0] += done_pf
            kvj[0] += tok_done * (1.0 - shf) + done_pf
            acc["done"] += dn
            # (3) same-substep client returns + second admission pass
            if clients is not None:
                S["n_think"] += dn * (1.0 - z_same)
                add_arrivals(0.0, dn * z_same)
            admit()

        n_pf, pf_tok = S["n_pf"], S["pf_tok"]
        n_dec = nj.sum()
        n_run = n_pf + n_dec
        q = S["q"]
        xn = np.zeros(self.nx)
        xn[i["q_wait"]], xn[i["q_tok"]] = q, S["qtok"]
        xn[i["pf_tok"]], xn[i["n_pf"]] = pf_tok, n_pf
        xn[i["n_think"]] = S["n_think"] if clients is not None else 0.0
        xn[i["n_dec_1"]:i["n_dec_1"] + m] = nj
        xn[i["kv_1"]:i["kv_1"] + m] = kvj
        xn[i["n_run"]] = n_run
        # shared prompt prefix (identical prompts) is resident once
        xn[i["kv_res"]] = kvj.sum() + half_blk * n_run + (shf * p_len if n_run > 0.5 else 0.0)

        done = acc["done"]
        # Little's law with the queue's departure (admission) rate: expected
        # wait of a request joining now; falls back to the arrival rate.
        lam = (acc["adm"] if acc["adm"] > 1e-9 else max(acc["arr"], done)) / self.dt
        thr = done / self.dt
        # per-iteration time actually experienced (stretched when throttled)
        t_mean = acc["busy"] / acc["iters"] if acc["iters"] > 0 else \
            self.t_iter(max(n_run, 1.0), 0.0, 1.0)
        wq = q / lam if (q > 1e-9 and lam > 0) else 0.0
        y = {
            "power_w": g["p_static"] + acc["E"] / self.dt,
            "energy_j": g["p_static"] * self.dt + acc["E"],
            "tokens_out": acc["tok"],
            "completions": done,
            "admitted": acc["adm"],
            "arrivals": acc["arr"],
            "iterations": acc["iters"],
            "busy_frac": min(acc["busy"] / self.dt, 1.0),
            "t_iter_s": t_mean,
            "prefill_tokens_computed": acc["comp"],
            "queue_delay_s": wq,
            "e2e_latency_s": ((q + n_run) / thr) if thr > 1e-9 else
                             (float("inf") if (q + n_run) > 0 else 0.0),
            "ttft_s": wq + max(acc["tc_max"], t_mean),
            "tpot_s": t_mean,
            "t_iter_unthrottled_s": (acc["busy_u"] / acc["iters"]) if acc["iters"] > 0 else t_mean,
            "throttle_factor": (acc["busy_u"] / acc["busy"]) if acc["busy"] > 0 else 1.0,
            "power_cap_applied_w": p_cap,
            "power_cap_clipped": bool(abs(p_cap - p_cap_req) > 1e-9),
            "throttled": bool(acc["throttled"]),
            "n_waiting": q, "n_running": n_run, "kv_tokens": xn[i["kv_res"]],
        }
        return xn, y

    # ------------------------------------------------------------- helpers
    def rollout(self, x0, us, ds):
        """Simulate len(ds) steps; returns (X [k+1, nx], list of y)."""
        x = self.reset(x0)
        X, Y = [x], []
        for k, d in enumerate(ds):
            u = us[k] if isinstance(us, (list, tuple)) else (us or {})
            x, y = self.step(x, u, d)
            X.append(x)
            Y.append(y)
        return np.array(X), Y
