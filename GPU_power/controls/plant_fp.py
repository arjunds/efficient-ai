"""
controls/plant_fp.py -- first-principles stand-in plant (fluid, per GPU).

Same interface as controls/plant.py (built/validated by another agent):

    p = Plant(gpu="H200", model="Qwen2-7B-Instruct", dt=1.0, **overrides)
    x = p.reset()
    x, y = p.step(x, u={'max_running': 256, 'power_cap_w': 400.0},
                     d={'arrivals': 8.0, 'prompt_len': 50, 'gen_len': 153})

Physics (all from our validated energy + roofline model):
  per vLLM iteration with n_dec decode requests + a chunked-prefill slice:
     W   = active_params * 2 B          (weights re-read every iteration)
     KV  = resident_kv_tokens * kv_bytes/token
     F   = 2 * active_params * tokens_in_iteration
     E_dyn  = e_wbyte*W + e_kvbyte*KV + e_gemm*F            (3-term model)
     t_roof = max((W+KV)/(BW*MBU(n)), F/(peak*MFU))           (roofline, realized MBU)
     t      = max(t_roof, E_dyn/(P_cap - P_static))           (cap throttle = saturation)
     P      = P_static + E_dyn/t
  MBU(n) = mbu1*(1 - kappa*ln n), fitted to realized_utilization.csv (c=1..64).

State (fluid, integrated with nsub sub-steps per dt):
  q_wait  requests waiting for admission
  n_run   running requests (derived: n_pf + sum(hist))
  kv_res  resident KV tokens (derived from the histogram, see _kv())
  n_pf    admitted requests still in prefill
  pf_tok  prompt tokens not yet prefilled
  h_0..h_{R-1}  decode requests binned by *remaining* output tokens (width w)
The remaining-token histogram makes batch completions realistic (a batch of
requests does not drain exponentially), which matters for admission gating.

Known simplifications (the controller does NOT see these -> built-in mismatch):
  * output-length distribution = min(Exp(mu), gen_cap), mu solved for the mean;
  * fractional batches n<1 are interpreted as "busy a fraction n of the time";
  * the cap response assumes energy per unit work is cap-invariant (no DVFS
    efficiency gain) -- the least-validated part of the model (A100-PCIe capped
    data broke the linear model).
"""
import json
import math
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_COEF_JSON = os.path.join(_HERE, "..", "gpu_coefficients.json")

# measured active/total params and fp16 KV bytes/token (canonical ~/models.py)
MODEL_TABLE = {
    "Qwen2-7B-Instruct":    dict(active=7.070e9, total=7.615e9, kvb_tok=57344),
    "Qwen2.5-32B-Instruct": dict(active=31.98e9, total=32.76e9, kvb_tok=262144),
    "Qwen2-72B-Instruct":   dict(active=71.46e9, total=72.70e9, kvb_tok=327680),
}
MODEL_ALIASES = {"7B": "Qwen2-7B-Instruct", "32B": "Qwen2.5-32B-Instruct",
                 "72B": "Qwen2-72B-Instruct"}

# realized memory-bandwidth utilization at c=1 (realized_utilization.csv)
MBU1 = {("H200", "Qwen2-7B-Instruct"): 0.494, ("H200", "Qwen2.5-32B-Instruct"): 0.670,
        ("B200", "Qwen2-7B-Instruct"): 0.325, ("B200", "Qwen2.5-32B-Instruct"): 0.505,
        ("B200", "Qwen2-72B-Instruct"): 0.615}

# fallbacks if gpu_coefficients.json is missing (brief's numbers)
_FALLBACK = {
    "H200": dict(e_wbyte=1.077e-10, e_kvbyte=3.38e-10, e_gemm=0.673e-12, p_static=118.0,
                 p_cap=700.0, bw=4.8e12, peak_flops=9.9e14),
    "B200": dict(e_wbyte=1.189e-10, e_kvbyte=3.511e-10, e_gemm=0.485e-12, p_static=237.0,
                 p_cap=1000.0, bw=8.0e12, peak_flops=2.25e15),
}
# not in the json: memory, minimum settable power limit (ASSUMPTION for B200:
# H100/H200 SXM expose a 200 W floor; B200's is assumed 300 W), realized MFU.
_EXTRA = {
    "H200": dict(mem_bytes=150.7e9, p_cap_min=200.0, mfu=0.6),
    "B200": dict(mem_bytes=192.3e9, p_cap_min=300.0, mfu=0.6),
}


def load_gpu_coeffs(gpu):
    try:
        with open(_COEF_JSON) as f:
            js = json.load(f)
        g = js[gpu]
        base = {k: float(g[k]) for k in ("e_wbyte", "e_kvbyte", "e_gemm", "p_static",
                                         "p_cap", "bw", "peak_flops")}
        base["source"] = "gpu_coefficients.json"
    except Exception:
        base = dict(_FALLBACK[gpu])
        base["source"] = "fallback(brief)"
    base.update(_EXTRA[gpu])
    return base


def gen_len_dist(mean_g, cap=256, w=16):
    """Histogram (over remaining-token bins of width w) of G ~ min(Exp(mu), cap)
    with mu solved so that E[G] = mean_g. Returns (p_bins, edges, E[G|G>=x] per bin)."""
    mean_g = float(min(max(mean_g, 2.0), cap - 1.0))
    lo, hi = 1.0, 1e5
    for _ in range(80):   # bisection on mu: mu*(1-exp(-cap/mu)) = mean
        mu = 0.5 * (lo + hi)
        m = mu * (1 - math.exp(-cap / mu))
        lo, hi = (mu, hi) if m < mean_g else (lo, mu)
    R = int(math.ceil(cap / w))
    edges = np.arange(R + 1) * w
    g = np.arange(1, cap + 1, dtype=float)
    pg = np.exp(-(g - 1) / mu) - np.exp(-g / mu)
    pg[-1] += math.exp(-cap / mu)          # truncation mass at the cap
    pg /= pg.sum()
    p_bins = np.array([pg[(g > edges[r]) & (g <= edges[r + 1])].sum() for r in range(R)])
    # E[generated so far | remaining in bin r] = E[G | G >= x_r] - x_r  (x_r = bin mid)
    mids = 0.5 * (edges[:-1] + edges[1:])
    egen = np.array([(pg[g >= x] * g[g >= x]).sum() / max(pg[g >= x].sum(), 1e-12) - x
                     for x in mids])
    return p_bins, edges, np.maximum(egen, 0.0)


class Plant:
    state_names_base = ["q_wait", "n_run", "kv_res", "n_pf", "pf_tok"]

    def __init__(self, gpu: str, model: str, dt: float = 1.0, nsub: int = 20,
                 prompt_len: float = 50.0, gen_len: float = 153.0, gen_cap: int = 256,
                 bin_w: int = 16, max_batched_tokens: int = 8192, max_num_seqs: int = 256,
                 mbu_kappa: float = 0.037, **overrides):
        model = MODEL_ALIASES.get(model, model)
        self.gpu, self.model, self.dt, self.nsub = gpu, model, float(dt), int(nsub)
        gp = load_gpu_coeffs(gpu)
        mt = MODEL_TABLE[model]
        gp.update(w_bytes=2.0 * mt["active"], p_active=mt["active"], kvb_tok=mt["kvb_tok"],
                  weights_total_bytes=2.0 * mt["total"], mbu1=MBU1.get((gpu, model), 0.45),
                  mbu_kappa=mbu_kappa, max_batched_tokens=max_batched_tokens,
                  max_num_seqs=max_num_seqs, prompt_len=prompt_len, gen_len=gen_len,
                  gen_cap=gen_cap)
        gp.update(overrides)                       # e.g. mismatch: e_wbyte=1.2*nominal
        kv_bytes_avail = 0.9 * gp["mem_bytes"] - gp["weights_total_bytes"] - 2e9
        if kv_bytes_avail <= 0:
            raise ValueError(f"{model} does not fit on {gpu}")
        gp.setdefault("kv_max_tokens", kv_bytes_avail / gp["kvb_tok"])
        self.gpu_params = gp
        self.w = bin_w
        self._set_gen(gen_len)
        self.state_names = self.state_names_base + [f"h{r}" for r in range(self.R)]

    # ---------------- helpers ----------------
    def _set_gen(self, gen_len):
        self._gen_mean = float(gen_len)
        self.p_bins, self.edges, self.egen = gen_len_dist(gen_len, self.gpu_params["gen_cap"],
                                                          self.w)
        self.R = len(self.p_bins)

    def mbu(self, n):
        g = self.gpu_params
        return g["mbu1"] * max(0.6, 1.0 - g["mbu_kappa"] * math.log(max(n, 1.0)))

    def _kv(self, n_pf, pf_tok, hist, lp):
        return max(n_pf * lp - pf_tok, 0.0) + float(hist @ (lp + self.egen))

    def reset(self, x0=None):
        if x0 is not None:
            return np.asarray(x0, float).copy()
        return np.zeros(len(self.state_names_base) + self.R)

    def iteration(self, n_dec, chunk, kv_tok, cap_w):
        """One vLLM iteration (per-iteration quantities, batch >= 1 request).
        Returns t, E_dyn, throttled."""
        g = self.gpu_params
        wb = g["w_bytes"]
        kvb = kv_tok * g["kvb_tok"]
        flops = 2.0 * g["p_active"] * (n_dec + chunk)
        e_dyn = g["e_wbyte"] * wb + g["e_kvbyte"] * kvb + g["e_gemm"] * flops
        t_roof = max((wb + kvb) / (g["bw"] * self.mbu(n_dec + 1e-9)),
                     flops / (g["peak_flops"] * g["mfu"]))
        p_dyn_max = max(cap_w - g["p_static"], 1.0)
        t_cap = e_dyn / p_dyn_max
        return max(t_roof, t_cap), e_dyn, t_cap > t_roof

    # ---------------- dynamics ----------------
    def step(self, x, u: dict, d: dict):
        g = self.gpu_params
        x = np.asarray(x, float)
        q, _, _, n_pf, pf = x[:5]
        hist = x[5:5 + self.R].copy()
        lp = float(d.get("prompt_len", g["prompt_len"]))
        gl = float(d.get("gen_len", self._gen_mean))
        if abs(gl - self._gen_mean) > 1e-6:
            self._set_gen(gl)
        cap = float(np.clip(u.get("power_cap_w", g["p_cap"]), g["p_cap_min"], g["p_cap"]))
        max_run = float(min(u.get("max_running", g["max_num_seqs"]), g["max_num_seqs"]))
        arr = float(d.get("arrivals", 0.0))
        h = self.dt / self.nsub
        edges = self.edges.astype(float)
        E = tok = comp = adm_tot = 0.0
        busy_t = t_sum = thr_t = 0.0
        for _ in range(self.nsub):
            q += arr / self.nsub
            n_run = n_pf + hist.sum()
            kv = self._kv(n_pf, pf, hist, lp)
            kv_room = max(g["kv_max_tokens"] * 0.95 - kv, 0.0)
            adm = min(q, max(max_run - n_run, 0.0), kv_room / (lp + gl))
            q -= adm
            adm_tot += adm
            if d.get("prefilled", False):
                # disaggregated decode pool: prompt KV arrives from a prefill GPU,
                # the request starts decoding at once (KV of lp is resident)
                hist = hist + adm * self.p_bins
            else:
                n_pf += adm
                pf += adm * lp
            n_dec = hist.sum()
            n_eff = n_dec + n_pf
            if n_eff < 1e-9:
                E += g["p_static"] * h
                continue
            chunk = min(pf, max(g["max_batched_tokens"] - n_dec, 0.0))
            beta = min(1.0, n_eff)                      # busy fraction for n < 1
            t, e_dyn, thr = self.iteration(n_dec / beta, chunk / beta, kv / beta, cap)
            n_iter = beta * h / t
            E += g["p_static"] * h + e_dyn * n_iter
            busy_t += beta * h
            t_sum += beta * h * t
            thr_t += beta * h * thr
            s = h / t                                   # tokens advanced per request
            # prefill progress -> requests enter decode with a fresh G draw
            done_pf_tok = min(pf, chunk * s)
            pf -= done_pf_tok
            moved = min(n_pf, done_pf_tok / max(lp, 1.0)) if pf > 1e-9 else n_pf
            n_pf -= moved
            # decode: shift remaining-token histogram down by s tokens
            if n_dec > 0:
                C = np.concatenate([[0.0], np.cumsum(hist)])
                Cs = np.interp(edges + s, edges, C, right=C[-1])
                comp_k = Cs[0]
                hist = np.diff(Cs)
                comp += comp_k
                tok += n_dec * s
            hist = hist + moved * self.p_bins
            tok += moved                                # first token from prefill
        n_run = n_pf + hist.sum()
        kv = self._kv(n_pf, pf, hist, lp)
        lam = arr / self.dt
        y = dict(power_w=E / self.dt, energy_j=E, tokens_out=tok, completions=comp,
                 t_iter_s=(t_sum / busy_t) if busy_t > 0 else 0.0,
                 queue_delay_s=(q / lam) if lam > 1e-9 else (0.0 if q < 1e-9 else self.dt),
                 throttled=(thr_t / busy_t) if busy_t > 0 else 0.0,
                 admitted=adm_tot, busy_frac=busy_t / self.dt, cap_applied_w=cap,
                 kv_frac=kv / g["kv_max_tokens"])
        xn = np.concatenate([[q, n_run, kv, n_pf, pf], hist])
        return xn, y
