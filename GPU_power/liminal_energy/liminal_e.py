#!/usr/bin/env python3
"""
liminal_e.py -- LIMINAL-E: LIMINAL's analytic decode roofline extended with a
MEASURED energy model and a power roofline.  Formalism: LIMINAL_E.md.

Reuses recommend_gpu.py (GPU table + priors via load_gpus(), model resolution,
per-iteration serving work, the realized-utilization time model iter_time()).
What is new here:
  * per-step ENERGY decomposition  E = P_s*t + e_w*W + e_kv*K + e_g*F_gemm (+ e_link*A)
  * a POWER ROOFLINE  P_s + P_0 + e_b*beta + e_f*phi <= P_cap  and a capped regime
    with measured lower J/B and J/FLOP (DVFS) instead of "stretch time at full-clock J"
  * bound classification  bandwidth | compute | overhead | power
  * a LIMINAL-as-published baseline (eta=1, t0=0, 114 us/layer; power = TDP)
  * a CUDA-graphs time variant (per-layer floor reduced; fitted on B200 7B, see
    validate.py) and coefficient profiles that bracket the undetermined KV/compute split
  * provisioning: GPU type x TP degree x replicas minimizing J/token under a TPOT SLO
v2 (2026-10-21): A100-80-PCIe and RTX A6000 added as MEASURED GPUs from
gpus_measured.json (build_measured.py: microbench-only "zeroshot" block; validate_v2.py:
serving-"calibrated" block); a memory-activity power term P_0; an AT-CAP ROOFLINE fitted on
capped LLM-shaped microbench kernels; B200 KV settled (UPDATE 6 review); PCIe TP sync.

Importable without a GPU.  Needs ~/models.py (python >= 3.7: run in the container).
No numpy.

    import liminal_e as le
    r = le.step("B200", "Qwen/Qwen2-7B-Instruct", batch=64, context=1024)
    r["t"], r["power"], r["j_per_token"], r["bound"], r["capped"]
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
for _d in (PARENT, os.path.expanduser("~"), "/workspace"):
    if _d not in sys.path:
        sys.path.append(_d)

import recommend_gpu as rg  # noqa: E402

DTYPE_B = rg.DTYPE_B

# ---------------------------------------------------------------------------
# 0. Constants and their provenance
# ---------------------------------------------------------------------------
# LIMINAL v2 (arXiv 2507.14397 v2, Sec. 5; LIT_LIMINAL.md): memory slope fixed at peak
# bandwidth, exposed per-layer overhead fitted on 8xH100 vLLM, mean 114 us (95-138).
LIMINAL_T_EXPOSED = 114e-6

# CUDA-graphs time variant: the per-layer floor tau is reduced by GRAPHS_DTAU.  Fitted by
# validate.py on the B200_prov graphs 7B runs at c<=64 against the eager-fitted model,
# then tested on 32B and 72B (held out).  See validation.json["graphs_fit"].
GRAPHS_DTAU = {"B200": 34.5e-6}         # s/layer, validate.py fit (validation.json graphs_fit)
GRAPHS_DTAU_PRIOR = 34.5e-6             # other GPUs: B200 value, UNMEASURED (H200 w/ graphs open)

# Tensor-parallel synchronization.  recommend_gpu draws an all-reduce latency tau_ar per
# layer (prior 10-60 us, central 25) and a time-inflation factor (central 1.15).  B200 is
# now MEASURED (logs/B200_prov/tp2, CUDA graphs): tau_ar fitted on 72B TP=2 with no
# inflation factor, tested on 32B TP=2 (validate.py; validation.json tp_fit).
TP_SYNC = {"B200": dict(tau_ar=15.5e-6, pen=1.0)}  # validate.py fit (72B), 32B held-out 1.3%
# A100 PCIe (no NVLink): per-layer latency + all-reduce bytes over an effective link rate;
# fitted by validate_v2.py on 14B TP=2, tested on 32B TP=2 -> read from gpus_measured.json
TP_SYNC_PRIOR = None                            # None -> rg.central_draw (25 us, x1.15)

# Coefficient profiles (J/B, J/B, J/FLOP).  'canonical' = gpu_coefficients.json via
# rg.load_gpus() (same numbers as the recommender).  The others bracket the
# KV-vs-compute split, which is UNDETERMINED (FINDINGS_B200.md "UPDATE 5 review"):
# per-bin KV bytes and GEMM FLOPs correlate 0.74-0.76 on the 7B data.
PROFILES = {
    "B200": {
        # relog, energy counter, free 3-term fit (UPDATE 5; provisioning_summary.json refit.relog)
        "kv_high": dict(e_wbyte=1.2603e-10, e_kvbyte=3.981e-10, e_gemm=0.7130e-12,
                        src="UPDATE 5 relog free fit (smear-free bins)"),
        # relog with e_gemm pinned at the steady uncapped value (UPDATE 5 review)
        "kv_low": dict(e_wbyte=1.250e-10, e_kvbyte=1.22e-10, e_gemm=0.943e-12,
                       src="UPDATE 5 review: relog, e_gemm pinned at steady 0.943 pJ"),
        # whole-run fit, 14 runs (balance_analysis.json runlevel B200/fit_root)
        "runlevel": dict(e_wbyte=1.2262e-10, e_kvbyte=1.0252e-10, e_gemm=0.9272e-12,
                         src="balance_analysis.json runlevel B200/fit_root"),
    },
    "H200": {
        # whole-run fit, 32 runs (balance_analysis.json runlevel H200/fit_root)
        "runlevel": dict(e_wbyte=1.0293e-10, e_kvbyte=2.4232e-10, e_gemm=1.1043e-12,
                         src="balance_analysis.json runlevel H200/fit_root"),
        # NOTE: no kv_low profile for H200 -- no steady calibration exists (access revoked).
    },
}

# Power-roofline coefficients (full clock).  B200: steady microbench, uncapped points only
# (microbench/steady_calib_B200.json: grid marginal 1.251e-10 J/B + 38.7 W activity
# intercept; llmgemm e_flop 0.943 pJ).  H200: NO steady calibration -- serving e_wbyte and
# the smear-free whole-run e_gemm with KV held (BALANCE.md: 0.956 pJ) -> a PREDICTION.
# Capped regime (DVFS throttled): B200 measured -- capped DRAM streams median 1.122e-10 J/B
# (steady_calib_B200.json dram_stream), capped LLM GEMMs 0.57-0.67 pJ/FLOP (UPDATE 4 review)
# -> 0.62 pJ.  Others: B200 capped/uncapped ratios applied as a prior.
POWER_ROOF = {
    "B200": dict(e_b=1.2509e-10, p0=38.73, e_f=0.9425e-12, e_b_cap=1.122e-10, e_f_cap=0.62e-12,
                 src="microbench/steady_calib_B200.json (measured)"),
    "H200": dict(e_b=1.0657e-10, p0=0.0, e_f=0.956e-12, e_b_cap=None, e_f_cap=None,
                 src="serving e_wbyte + run-level e_gemm (BALANCE.md); no steady calib (PREDICTION)"),
}
CAP_SCALE_PRIOR = dict(w=1.122e-10 / 1.2509e-10, g=0.62e-12 / 0.9425e-12)   # 0.897, 0.658

_GPUS = None
_MODELS = {}
MEASURED_JSON = os.path.join(HERE, "gpus_measured.json")
DATASHEET_KEYS = ("gpu_name", "mem_tech", "bw", "peak_flops", "p_cap", "mem", "interconnect")
COEF_KEYS = ("e_wbyte", "e_kvbyte", "e_gemm", "p_static", "p0")


def _apply_measured(table, path=MEASURED_JSON):
    """Overlay gpus_measured.json on the recommender table.  Each GPU may carry
       zeroshot   : microbench + datasheet + idle only (no serving data)
       calibrated : serving-fitted (A100/A6000: at-cap and time scales; B200: settled
                    long-context fit; H200: e_gemm refit with the KV prior)
       at_cap     : at-cap roofline (tau_b, tau_f, k) from capped LLM-shaped kernels
    Canonical coefficients = calibrated if present, else zeroshot.  The September per-bin
    coefficients of H200/B200 are kept as profile 'sept_perbin'."""
    if not os.path.exists(path):
        return table
    js = json.load(open(path))
    for name, e in js.items():
        if name.startswith("_"):
            continue
        g = dict(table.get(name, {}), name=name)
        for k in DATASHEET_KEYS:
            if e.get(k) is not None:
                g[k] = e[k]
        if g.get("measured") and "sept_perbin" not in g:
            g["sept_perbin"] = {k: g.get(k) for k in COEF_KEYS}
        for blk in ("zeroshot", "calibrated", "at_cap", "idle_vllm", "idle_clk", "idle_deep", "duty",
                    "grid", "tp_sync"):
            if e.get(blk) is not None:
                g[blk] = e[blk]
        canon = e.get("calibrated") or e.get("zeroshot")
        if canon:
            for k in COEF_KEYS:
                if canon.get(k) is not None:
                    g[k] = canon[k]
            g["energy_src"] = "gpus_measured.json:%s (%s)" % (
                "calibrated" if e.get("calibrated") else "zeroshot", canon.get("src", ""))
        if not table.get(name, {}).get("measured"):
            g["time_key"] = "_prior"
            g["confidence"] = "measured energy+cap / prior time"
        g["measured"] = True
        g.setdefault("p0", 0.0)
        for r in ("r_wbyte", "r_kvbyte", "r_gemm"):          # keep recommender MC ranges sane
            k = {"r_wbyte": "e_wbyte", "r_kvbyte": "e_kvbyte", "r_gemm": "e_gemm"}[r]
            g[r] = (g[k], 0.9 * g[k], 1.1 * g[k])
        g["r_static"] = (g["p_static"], 0.95 * g["p_static"], 1.05 * g["p_static"])
        if e.get("tp_sync"):
            TP_SYNC[name] = e["tp_sync"]
        table[name] = g
    return table


def gpus():
    """GPU table: recommend_gpu.load_gpus() (gpu_coefficients.json + priors), overlaid with
    liminal_energy/gpus_measured.json (A100-80-PCIe, A6000 measured; B200/H200 updates)."""
    global _GPUS
    if _GPUS is None:
        _GPUS = _apply_measured(rg.load_gpus())
    return _GPUS


def gpu(name_or_dict):
    return name_or_dict if isinstance(name_or_dict, dict) else gpus()[name_or_dict]


def model(mid):
    if not isinstance(mid, str):
        return mid
    if mid not in _MODELS:
        _MODELS[mid] = rg.resolve_model(mid)
    return _MODELS[mid]


def profile_params(g, profile="canonical"):
    """dict(e_w, e_kv, e_g, p_s, p0, ovh_scale, cap_scale, cap_mode, src) for a GPU / profile.
      canonical    : calibrated if available, else zeroshot / recommender values
      zeroshot     : microbench + datasheet + idle only (A100, A6000, B200)
      zeroshot_gemv: zeroshot with e_w = batch-1 GEMV duty-cycle J/B, P_s = vLLM idle, P_0 = 0
      sept_perbin  : the September per-bin serving fit (H200, B200; gpu_coefficients.json)
      kv_as_weight : canonical with e_kv := e_w (sensitivity)
      kv_high/kv_low/runlevel : historical B200/H200 brackets (PROFILES)
    Unknown profile for a GPU -> canonical (noted in src)."""
    g = gpu(g)
    cal = g.get("calibrated") or {}
    base = dict(e_w=g["e_wbyte"], e_kv=g["e_kvbyte"], e_g=g["e_gemm"], p_s=g["p_static"],
                p0=g.get("p0", 0.0), ovh_scale=cal.get("ovh_scale", 1.0), cap_scale=cal.get("cap_scale", 1.0),
                cap_mode=cal.get("cap_mode", "auto"), src=g["energy_src"])
    if profile == "canonical":
        return base
    if profile == "kv_as_weight":
        return dict(base, e_kv=base["e_w"], src=base["src"] + " [sensitivity: e_kvbyte := e_wbyte]")
    if profile in ("zeroshot", "zeroshot_gemv") and g.get("zeroshot"):
        z = g["zeroshot"] if profile == "zeroshot" else g["zeroshot"]["alt_gemv"]
        return dict(e_w=z["e_wbyte"], e_kv=z["e_kvbyte"], e_g=z["e_gemm"], p_s=z["p_static"],
                    p0=z.get("p0", 0.0), ovh_scale=1.0, cap_scale=1.0, cap_mode="auto",
                    src="gpus_measured.json:" + profile)
    if profile == "sept_perbin" and g.get("sept_perbin"):
        z = g["sept_perbin"]
        return dict(base, e_w=z["e_wbyte"], e_kv=z["e_kvbyte"], e_g=z["e_gemm"], p_s=z["p_static"],
                    p0=z.get("p0") or 0.0, src="gpu_coefficients.json (Sept per-bin)")
    ov = PROFILES.get(g["name"], {}).get(profile)
    if ov is None:
        return dict(base, src=base["src"] + " [profile %r n/a -> canonical]" % profile)
    return dict(base, e_w=ov["e_wbyte"], e_kv=ov["e_kvbyte"], e_g=ov["e_gemm"], p0=0.0, src=ov["src"])


def coefficients(g, profile="canonical"):
    """(e_wbyte, e_kvbyte, e_gemm, p_static, source) -- back-compatible view of profile_params."""
    p = profile_params(g, profile)
    return p["e_w"], p["e_kv"], p["e_g"], p["p_s"], p["src"]


def capped_coefficients(g, e_w, e_kv, e_g):
    """J/B and J/FLOP in the power-capped (clock-throttled) regime.  Measured absolute
    values where available (B200), never above the uncapped profile value."""
    pr = POWER_ROOF.get(g["name"], {})
    sw, sg = CAP_SCALE_PRIOR["w"], CAP_SCALE_PRIOR["g"]
    ew_c = min(e_w, pr["e_b_cap"]) if pr.get("e_b_cap") else sw * e_w
    eg_c = min(e_g, pr["e_f_cap"]) if pr.get("e_f_cap") else sg * e_g
    return ew_c, e_kv * (ew_c / e_w), eg_c


# ---------------------------------------------------------------------------
# 1. Workload (GPU-independent)
# ---------------------------------------------------------------------------
def allreduce_bytes(m, tokens, tp):
    """Per-GPU bytes moved by TP all-reduces in one step: 2 all-reduces per layer
    (attention out, MLP out) of tokens*d fp16 values, ring algorithm 2(n-1)/n."""
    if tp <= 1:
        return 0.0
    return 2.0 * m.L * tokens * m.d * DTYPE_B * 2.0 * (tp - 1) / tp


def work(m, batch, context, phase="decode", prefill_tokens=0.0, tp=1):
    """Work of ONE step, summed over the TP group.
      decode : `batch` sequences each produce 1 token at mean resident `context`;
               `prefill_tokens` (optional) are chunked-prefill tokens mixed into the step.
      prefill: `batch` = tokens in the chunk, `context` = prompt length (causal: ~ctx/2).
    W weight bytes (canonical ~/models.py: no input-embedding gather), K KV bytes read,
    Fg dense GEMM FLOPs, Fa attention FLOPs, A all-reduce bytes (all GPUs)."""
    m = model(m)
    kvpt = m.kv_bytes_per_token(DTYPE_B)
    if phase == "prefill":
        T = float(batch)
        W = rg.weight_params(m, T) * DTYPE_B
        K = T * kvpt                                   # KV written once
        Fa = m.attn_flops_per_token(1) * T * context / 2.0
        Fa_pf = Fa                                     # FLOP-bound: charged at e_g
        out = T
    else:
        T = float(batch) + float(prefill_tokens)
        W = rg.weight_params(m, T) * DTYPE_B
        K = float(batch) * context * kvpt
        Fa = m.attn_flops_per_token(1) * float(batch) * context
        Fa_pf = 0.0                                    # decode attention: in e_kv*K
        out = float(batch)
    Fg = 2.0 * m.active_params() * T
    A = allreduce_bytes(m, T, tp) * tp
    return dict(Wb=W, KVb=K, Fg=Fg, Fa=Fa, Fa_pf=Fa_pf, F=Fg + Fa, T=T, A=A, tokens_out=out)


def intensity(w):
    """Arithmetic intensity, FLOP per DRAM byte (weights + KV)."""
    return w["F"] / (w["Wb"] + w["KVb"])


# ---------------------------------------------------------------------------
# 2. Time
# ---------------------------------------------------------------------------
def time_params(g, time_model="eager", ovh_scale=1.0):
    """Time-model parameters.  ovh_scale multiplies the host/launch overhead terms (t0, tau)
    -- the node-specific CPU launch path (A100 node: fitted on eager runs, validate_v2.py).
    Applied in eager mode only (CUDA graphs remove most of the launch path); tau_moe (the
    fused-MoE layer floor, a kernel time) is not scaled."""
    tp = dict(rg.TIME_CENTRAL[g["time_key"]])
    if ovh_scale != 1.0 and time_model == "eager":
        for k in ("t0", "tau"):
            if tp.get(k):
                tp[k] *= ovh_scale
    if time_model == "graphs":
        d = GRAPHS_DTAU.get(g["name"], GRAPHS_DTAU_PRIOR)
        tp["tau"] = max(0.0, tp["tau"] - d)
        if tp.get("tau_moe"):
            tp["tau_moe"] = max(0.0, tp["tau_moe"] - d)
    return tp


def step_time(g, m, w, tp_deg=1, time_model="eager", draw=None, ovh_scale=1.0):
    """Uncapped step time and its decomposition {overhead, bandwidth, compute}.
    time_model: 'eager' (recommender fit, eager-mode vLLM), 'graphs' (CUDA graphs),
    'liminal' (LIMINAL v2 as published: peak BW, no t0, 114 us/layer exposed)."""
    L = m.L
    n = tp_deg
    if time_model == "liminal":
        t_mem = (w["Wb"] + w["KVb"]) / (g["bw"] * n)
        t_cmp = w["F"] / (g["peak_flops"] * n)
        ovh = LIMINAL_T_EXPOSED * L
        return max(t_mem, t_cmp) + ovh, dict(overhead=ovh, bandwidth=t_mem, compute=t_cmp)
    d = draw or dict(rg.central_draw(g))
    sync = TP_SYNC.get(g["name"]) if draw is None else None
    if sync and sync.get("tau_ar") is not None:
        d["tau_ar"], d["tp_pen"] = sync["tau_ar"], sync["pen"]
    tp = time_params(g, time_model, ovh_scale) if draw is None else d["tp"]
    moe = rg.is_moe(m)
    tau_ar = d["tau_ar"] if n > 1 else 0.0
    t, _ = rg.iter_time(g, tp, w, L, moe, d["mfu"], n, tau_ar)
    pen = d["tp_pen"] if n > 1 else 1.0
    t_link = 0.0
    if n > 1 and sync and sync.get("link_bw"):                    # PCIe: all-reduce bytes / link rate
        t_link = (w["A"] / n) / sync["link_bw"]
        t += t_link
    # decomposition (mirrors rg.iter_time)
    bw, pk = g["bw"] * n, g["peak_flops"] * n
    tau = (tp["tau_moe"] if (moe and tp.get("tau_moe")) else tp["tau"]) + tau_ar
    t_w = w["Wb"] / (bw * tp["eta"])
    per_layer = rg.softmax_k(tau, t_w / L, tp["k"])
    t_kv = w["KVb"] / (bw * tp["eta_kv"])
    t_tok = w["F"] / (pk * tp["mu"])
    t_cmp = tp["t0"] + w["F"] / (pk * d["mfu"])
    if t_cmp > tp["t0"] + L * per_layer + t_kv + t_tok:          # compute roofline binds
        parts = dict(overhead=tp["t0"], bandwidth=0.0, compute=t_cmp - tp["t0"])
    else:
        parts = dict(overhead=tp["t0"] + L * per_layer - t_w, bandwidth=t_w + t_kv, compute=t_tok)
    parts["overhead"] += (pen - 1.0) * t + t_link                # TP sync / overlap penalty
    return t * pen, parts


# ---------------------------------------------------------------------------
# 3. Energy, power, power cap
# ---------------------------------------------------------------------------
def at_cap_time(g, w, n, cap_scale=1.0):
    """Step time when the GPU sits at its power cap, from the AT-CAP ROOFLINE fitted on
    capped LLM-shaped microbench kernels (gpus_measured.json at_cap):
        seconds per DRAM byte = softmax_k(cap_scale*tau_b, I*tau_f),  I = FLOP/byte of the step.
    KV and all-reduce bytes are streamed at the weight rate (prior).  cap_scale (serving-
    calibrated, 1 for zero-shot) stretches only the streaming term: in decode the non-GEMM
    kernels of a step share the capped power; compute-bound prefill is left as measured."""
    ac = g["at_cap"]
    byts = (w["Wb"] + w["KVb"]) / n
    F_e = (w["Fg"] + w.get("Fa_pf", 0.0)) / n
    I = F_e / byts if byts > 0 else 0.0
    return byts * rg.softmax_k(cap_scale * ac["tau_b"], I * ac["tau_f"], ac["k"])


def step_work(g, m, w, tp_deg=1, profile="canonical", time_model="eager",
              cap_regime=False, cap_mode="auto", power_model="measured", draw=None, coef=None):
    """Predict one step from a precomputed work dict.  Returns time, power, energy split,
    bound and per-token metrics.  power_model='tdp' gives LIMINAL's STPS/W convention
    (power = n*TDP, no activity dependence).
    Energy:  E = n*(P_s + P_0)*t + e_w*W + e_kv*K + e_g*F + e_link*A   (P_0: memory/activity
    intercept, measured on A100 26 W, A6000 81 W; 0 for the serving-fitted H200/B200 profiles).
    The step is CAPPED when n*(P_s+P_0) + E_dyn/t_uncapped > n*P_cap.  Capped handling
    (cap_mode): 'roof_ovh' (default when the GPU has an at-cap roofline): t = max(t_uncapped,
    at_cap_time + overhead part of the time model) and power = P_cap; 'roof': without the
    overhead term; 'stretch': t = E_dyn/(n*(P_cap-P_s-P_0)) at full-clock J (recommender
    convention, default otherwise); 'dvfs' (or cap_regime=True): stretch with the capped-regime
    coefficients (capped_coefficients())."""
    g, m = gpu(g), model(m)
    n = tp_deg
    if coef is None:
        pp = profile_params(g, profile)
    else:
        pp = dict(e_w=coef[0], e_kv=coef[1], e_g=coef[2], p_s=coef[3], p0=coef[4] if len(coef) > 4 else 0.0,
                  ovh_scale=1.0, cap_scale=1.0, cap_mode="auto", src="explicit")
    e_w, e_kv, e_g, p_s, p0 = pp["e_w"], pp["e_kv"], pp["e_g"], pp["p_s"], pp["p0"]
    src = pp["src"]
    e_link = e_w                                              # PRIOR: an all-reduce byte ~ a DRAM byte
    t_unc, parts = step_time(g, m, w, n, time_model, draw, pp["ovh_scale"])
    # e_g multiplies dense GEMM FLOPs (the fit's convention); decode attention is
    # KV-byte-bound and absorbed in e_kv.  Prefill attention (FLOP-bound, ~ctx^2) is
    # charged at e_g.
    F_e = w["Fg"] + w.get("Fa_pf", 0.0)
    E_full = e_w * w["Wb"] + e_kv * w["KVb"] + e_g * F_e + e_link * w["A"]
    split = dict(weights=e_w * w["Wb"], kv=e_kv * w["KVb"], compute=e_g * F_e, link=e_link * w["A"])
    budget = n * (g["p_cap"] - p_s - p0)                      # dynamic budget above idle + activity
    mode = "dvfs" if cap_regime else (cap_mode if cap_mode != "auto" else pp.get("cap_mode", "auto"))
    if mode == "auto":
        mode = "roof_ovh" if g.get("at_cap") else "stretch"
    t, capped = t_unc, False
    if E_full / t_unc > budget:                               # demanded power exceeds the cap
        capped = True
        if mode in ("roof", "roof_ovh"):
            t_c = at_cap_time(g, w, n, pp["cap_scale"])
            if mode == "roof_ovh":                            # host/launch overhead is not throttled
                t_c += parts["overhead"]
            t = max(t_unc, t_c)
            E_dyn_total = n * (g["p_cap"] - p_s) * t          # power pinned at the cap
            act = n * p0 * t
            s_ = max(0.0, E_dyn_total - act) / E_full
            split = {k: v * s_ for k, v in split.items()}
        else:
            if mode == "dvfs":
                cw, ckv, cg = capped_coefficients(g, e_w, e_kv, e_g)
            else:
                cw, ckv, cg = e_w, e_kv, e_g
            E_cap = cw * w["Wb"] + ckv * w["KVb"] + cg * F_e + cw * w["A"]
            t = max(t_unc, E_cap / budget)                    # time stretches only if needed
            s_ = min(E_full, budget * t) / E_full
            split = {k: v * s_ for k, v in split.items()}
    split["activity"] = n * p0 * t
    E_dyn = sum(split.values())
    if power_model == "tdp":
        power = n * g["p_cap"]
        E_static, E_dyn = power * t, 0.0
        split = {k: 0.0 for k in split}
    else:
        E_static = n * p_s * t
        power = E_static / t + E_dyn / t
    bound = "power" if capped else max(parts, key=parts.get)
    E = E_static + E_dyn
    out = w["tokens_out"]
    return dict(t=t, t_uncapped=t_unc, time_parts=parts, power=power, energy=E,
                e_static=E_static, e_dyn=E_dyn, split=split, bound=bound, capped=capped,
                j_per_token=E / out, tok_s=out / t, n_gpu=n,
                mbu=(w["Wb"] + w["KVb"]) / (g["bw"] * n * t),
                mfu=w["F"] / (g["peak_flops"] * n * t), ai=intensity(w),
                coef=(e_w, e_kv, e_g, p_s, p0), coef_src=src, work=w)


def step(g, m, batch, context, phase="decode", tp=None, prefill_tokens=0.0, **kw):
    """One decode (or prefill) step.  tp=None -> smallest TP degree that fits weights +
    KV (rg.n_gpus_needed).  kw: profile, time_model, cap_regime, power_model."""
    g, m = gpu(g), model(m)
    if tp is None:
        kv_total = float(batch) * context * m.kv_bytes_per_token(DTYPE_B) if phase == "decode" else 0.0
        tp = rg.n_gpus_needed(g, m, kv_total)
        if tp is None:
            return None
    w = work(m, batch, context, phase, prefill_tokens, tp)
    return step_work(g, m, w, tp, **kw)


def serving_work(m, n, P, G, cached_frac=0.0, prefill_attn=True):
    """rg.serving_iter_work plus the causal attention FLOPs of the prefill chunks
    (n*(1-cf)*P/G new tokens per step, each attending ~P/2 keys).  rg omits them; they
    are negligible for short prompts (<0.3% of GEMM FLOPs at P<=200) but ~12% at P=8192."""
    w = rg.serving_iter_work(m, n, P, G, cached_frac)
    w["Fa"] = w["F"] - w["Fg"]
    if prefill_attn:
        fa_pf = m.attn_flops_per_token(1) * n * (1.0 - cached_frac) * P / max(G, 1.0) * P / 2.0
        w["Fa_pf"] = fa_pf
        w["Fa"] += fa_pf
        w["F"] += fa_pf
    return w


def serving(g, m, P, G, conc=None, rate=None, cached_frac=0.0, tp=1, prefill_attn=True, **kw):
    """Steady-state continuous-batching prediction (same workload as
    rg.predict_serving: n running sequences + their share of prefill per step).
    Closed loop (conc) or open loop (rate, Little's law)."""
    g, m = gpu(g), model(m)

    def at(n):
        w = serving_work(m, n, P, G, cached_frac, prefill_attn)
        w["A"] = allreduce_bytes(m, w["T"], tp) * tp
        w["tokens_out"] = n
        return step_work(g, m, w, tp, **kw)

    if conc is not None:
        n = float(conc)
        r = at(n)
        busy = 1.0
    else:
        n = 1.0
        for _ in range(80):
            r = at(n)
            n_new = max(1.0, rate * G * r["t"])
            if abs(n_new - n) < 1e-4 * n:
                break
            n = 0.5 * (n + n_new)
        r = at(n)
        busy = min(1.0, rate * G * r["t"] / n)
    tok_s = n / r["t"] if conc is not None else min(rate * G, n / r["t"])
    p_s = r["coef"][3] * tp
    power = busy * (r["power"] - p_s) + p_s          # idle gaps: back to P_s (no activity term)
    return dict(r, n=n, tok_s=tok_s, power=power, j_per_token=power / tok_s, busy=busy)


# ---------------------------------------------------------------------------
# 4. Rooflines: performance, power, energy
# ---------------------------------------------------------------------------
def power_roof_params(g, regime="full", profile="canonical"):
    """(e_b, e_f, p0, p_s) for the full-clock power roof.  Microbench GPUs (A100, A6000,
    B200): grid slope e_b + intercept P_0 over clocks-up idle, duty-cycle GEMM e_f (B200: the
    uncapped steady LLM-GEMM 0.943 pJ, POWER_ROOF).  H200: serving coefficients (PREDICTION)."""
    g = gpu(g)
    pr = POWER_ROOF.get(g["name"])
    if pr is not None:
        return pr["e_b"], pr["e_f"], pr["p0"], g.get("idle_clk") or g["p_static"]
    if g.get("grid") and g.get("duty"):
        return g["grid"]["e_b"], g["duty"]["gemm_fp16_4096"], g["grid"]["p0"], g["idle_clk"]
    e_w, _, e_g, p_s, _ = coefficients(g, profile)
    return e_w, e_g, 0.0, p_s


def power_limited_rates(g, regime="full", profile="canonical"):
    """Full clock: BW_P = (P_cap - P_s - P_0)/e_b, PHI_P = (P_cap - P_s - P_0)/e_f.
    regime='cap': the at-cap roofline's asymptotes 1/tau_b, 1/tau_f (measured, throttled)."""
    g = gpu(g)
    if regime == "cap" and g.get("at_cap"):
        ac = g["at_cap"]
        return dict(bw_p=1 / ac["tau_b"], phi_p=1 / ac["tau_f"], bw_frac=1 / ac["tau_b"] / g["bw"],
                    flops_frac=1 / ac["tau_f"] / g["peak_flops"], roof_knee=ac["tau_b"] / ac["tau_f"],
                    perf_ridge=g["peak_flops"] / g["bw"], budget=None)
    eb, ef, p0, p_s = power_roof_params(g, regime, profile)
    budget = g["p_cap"] - p_s - p0
    return dict(bw_p=budget / eb, phi_p=budget / ef, budget=budget,
                bw_frac=budget / eb / g["bw"], flops_frac=budget / ef / g["peak_flops"],
                roof_knee=eb / ef, perf_ridge=g["peak_flops"] / g["bw"])


def roofline(g, I, regime="full", profile="canonical"):
    """Attainable FLOP/s at arithmetic intensity I (FLOP/B):
      bandwidth roof I*BW, compute roof PEAK, and a power roof:
      full clock  phi_P(I) = (P_cap-P_s-P_0) / (e_b/I + e_f)   (harmonic; knee at e_b/e_f)
      at cap      phi(I) = I / softmax_k(tau_b, I*tau_f)       (fitted on capped kernels)."""
    g = gpu(g)
    bw_roof = I * g["bw"]
    pk = g["peak_flops"]
    if regime == "cap":
        if not g.get("at_cap"):
            return dict(I=I, bw_roof=bw_roof, compute_roof=pk, power_roof=None, attainable=min(bw_roof, pk),
                        bound=None)
        ac = g["at_cap"]
        pw = I / rg.softmax_k(ac["tau_b"], I * ac["tau_f"], ac["k"])
    else:
        eb, ef, p0, p_s = power_roof_params(g, regime, profile)
        pw = (g["p_cap"] - p_s - p0) / (eb / I + ef)
    att = min(bw_roof, pk, pw)
    bound = "power" if att == pw and pw < min(bw_roof, pk) else ("bandwidth" if bw_roof <= pk else "compute")
    return dict(I=I, bw_roof=bw_roof, compute_roof=pk, power_roof=pw, attainable=att, bound=bound)


def roofline_curves(g, Is, regime="full"):
    return [roofline(g, I, regime) for I in Is]


def energy_per_token_curve(g, m, context, batches, prefill_per_seq=0.0, **kw):
    """Energy 'roofline': J/token vs batch, split into static, weight, KV, compute.
    Weight J/token ~ e_w*W/B (slope -1 on log-log), compute ~ 2*e_g*P_act (flat),
    KV ~ e_kv*ctx*kv_bytes_per_token (flat); they cross at the energy ridge B = e_w/e_g."""
    rows = []
    for b in batches:
        r = step(g, m, b, context, prefill_tokens=prefill_per_seq * b, **kw)
        if r is None:
            rows.append(None)
            continue
        out = r["work"]["tokens_out"]
        rows.append(dict(batch=b, j_per_token=r["j_per_token"], static=r["e_static"] / out,
                         weights=r["split"]["weights"] / out, kv=r["split"]["kv"] / out,
                         compute=r["split"]["compute"] / out, link=r["split"]["link"] / out,
                         activity=r["split"]["activity"] / out,
                         capped=r["capped"], bound=r["bound"], t=r["t"], power=r["power"],
                         n_gpu=r["n_gpu"]))
    return rows


def ridges(g, profile="canonical"):
    """Performance ridge peak/BW, energy ridge e_w/e_g (serving coefficients), and the
    static-equivalent bandwidth P_s/e_w (the stream rate whose dynamic power equals idle)."""
    g = gpu(g)
    e_w, _, e_g, p_s, _ = coefficients(g, profile)
    return dict(perf_ridge=g["peak_flops"] / g["bw"], energy_ridge=e_w / e_g,
                static_equiv_bw=p_s / e_w, static_equiv_frac=p_s / e_w / g["bw"])


# ---------------------------------------------------------------------------
# 5. Provisioning
# ---------------------------------------------------------------------------
def kv_capacity_seqs(g, m, tp, ctx_max):
    """Sequences of length ctx_max whose KV fits next to the weights (vLLM mem util 0.9)."""
    free = tp * rg.MEM_UTIL * g["mem"] * 0.97 - m.total_params() * DTYPE_B
    if free <= 0:
        return 0
    return int(free // (ctx_max * m.kv_bytes_per_token(DTYPE_B)))


def provision_conc(m, conc, P, G, slo_tpot=0.05, gpu_names=None, tps=(1, 2, 4, 8),
                   max_seqs=512, **kw):
    """Closed-loop provisioning: `conc` concurrent requests (prompt P, gen G).  For each
    GPU type and TP degree, use the fewest replicas R such that per-replica batch
    b = conc/R fits in KV memory and max_seqs and meets TPOT <= slo_tpot.  Returns the
    candidates sorted by J/token (energy of all GPUs incl. idle, per generated token)."""
    m = model(m)
    names = gpu_names or list(gpus())
    out = []
    for name in names:
        g = gpu(name)
        for tp in tps:
            cap = kv_capacity_seqs(g, m, tp, P + G)
            if cap < 1:
                continue
            R = max(1, int(math.ceil(conc / float(min(cap, max_seqs)))))
            best = None
            for _ in range(64):
                b = conc / float(R)
                r = serving(g, m, P, G, conc=b, tp=tp, **kw)
                if r["t"] <= slo_tpot:
                    best = r
                    break
                R += 1
                if R > conc:
                    break
            if best is None:
                continue
            out.append(dict(gpu=name, tp=tp, replicas=R, n_gpus=R * tp, batch=conc / float(R),
                            j_per_token=best["j_per_token"], tpot=best["t"],
                            power_per_replica=best["power"], capped=best["capped"],
                            bound=best["bound"], measured=g.get("measured", False),
                            confidence=g.get("confidence")))
    return sorted(out, key=lambda d: d["j_per_token"])


def provision_rate(m, rate, P, G, slo_tpot=0.05, gpu_names=None, tps=(1, 2, 4, 8),
                   max_seqs=512, max_replicas=64, **kw):
    """Open-loop provisioning: Poisson arrivals at `rate` req/s.  Each replica receives
    rate/R; its batch follows Little's law n = (rate/R)*G*t(n).  Idle power of every
    provisioned GPU is charged; this is the amortization term that makes few, busy
    GPUs beat many, idle ones.  Returns candidates sorted by J/token."""
    m = model(m)
    names = gpu_names or list(gpus())
    out = []
    for name in names:
        g = gpu(name)
        for tp in tps:
            cap = min(kv_capacity_seqs(g, m, tp, P + G), max_seqs)
            if cap < 1:
                continue
            for R in range(1, max_replicas + 1):
                r = serving(g, m, P, G, rate=rate / float(R), tp=tp, **kw)
                if r["n"] <= cap and r["t"] <= slo_tpot and r["tok_s"] >= 0.999 * rate * G / R:
                    jt = R * r["power"] / (rate * G)
                    out.append(dict(gpu=name, tp=tp, replicas=R, n_gpus=R * tp, batch=r["n"],
                                    j_per_token=jt, tpot=r["t"], busy=r["busy"],
                                    power_total=R * r["power"], capped=r["capped"],
                                    bound=r["bound"], measured=g.get("measured", False)))
                    break
    return sorted(out, key=lambda d: d["j_per_token"])


if __name__ == "__main__":
    for gname in ("A6000", "A100-80-PCIe", "H200", "B200"):
        g = gpu(gname)
        pl = power_limited_rates(g)
        print("%s  perf ridge %.0f  power-roof knee %.0f  BW_P %.2f TB/s (%.0f%% of peak)  "
              "PHI_P %.0f TF (%.0f%%)" % (gname, pl["perf_ridge"], pl["roof_knee"], pl["bw_p"] / 1e12,
                                         100 * pl["bw_frac"], pl["phi_p"] / 1e12, 100 * pl["flops_frac"]))
        for b in (1, 16, 64, 256):
            r = step(g, "Qwen/Qwen2-7B-Instruct", b, 1024)
            print("   7B b=%4d ctx=1024: t=%.2f ms P=%.0f W  %.1f mJ/tok  bound=%s capped=%s" % (
                b, r["t"] * 1e3, r["power"], r["j_per_token"] * 1e3, r["bound"], r["capped"]))
