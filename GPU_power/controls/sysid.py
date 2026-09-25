#!/usr/bin/env python3
"""
controls/sysid.py — system identification + validation of the serving plant.

Re-runnable, in-container (numpy + scipy; transformers only for the one-off
prompt tokenization step). From GPU_power/:

    python3 controls/sysid.py tokenize     # one-off: prompt-pool token lengths
    python3 controls/sysid.py fit          # iteration-time model  -> plant_params.json
    python3 controls/sysid.py validate     # k-step plant validation -> sysid_results.json
    python3 controls/sysid.py all

What it identifies (see SYSID.md for the write-up):

  1. Iteration-time model (per GPU x model), fitted on 1 s bins of iter_log:
         t_iter = t_base + tau_seq * n_running
                  + max( KV_bytes / (BW * u_kv),  F_gemm / (peak * u_f) - t_w )^+
     i.e. a roofline max between the memory side (weights, folded into t_base,
     plus KV reads) and the compute side (GEMM FLOPs of the tokens processed),
     plus a per-iteration fixed cost and a per-sequence cost. t_base itself is
     decomposed across models on the same GPU as
         t_base = t_oh(GPU, n_layers) + W / (BW * u_w)
     (enforce_eager launch overhead ~ layers, plus weight streaming).
     The stat-logger t_start/t_end are async-jittery, so we never trust a single
     row: target = measured busy time per 1 s bin, prediction = sum over the
     rows in that bin of the per-row model time.

  2. Validation of controls.plant.Plant k-step-ahead (k=1..10, dt=1 s) on the
     measured n_waiting, n_running, kv_tokens_resident and power, against a
     persistence baseline.

Data subtleties this script handles (all verified against the logs):
  * vLLM's IterationStats.num_prompt_tokens (our `prefill_tokens`) is the FULL
    prompt length, logged once at the iteration that emits the first token,
    irrespective of chunking and of prefix-cache hits. The workload generator
    draws prompts with random.Random(1234).choice(pool) *with replacement*, and
    prefix caching is ON (vLLM V1 default), so at c=64 up to ~50-75% of prompt
    tokens are cache hits and are NOT computed. We replay the RNG, tokenize the
    pool, match prefill events to draws, and compute the *computed* prefill
    tokens per row. Synthetic (task=None) runs use one identical prompt, so all
    but the last 16-token block are cached.
  * Poisson arrivals are random.Random(0).expovariate(rate) — replayed exactly.
  * Two B200 runs crashed mid-run (engine error dump) and are excluded.
"""
import argparse
import bisect
import csv
import glob
import json
import math
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PARAMS_JSON = os.path.join(HERE, "plant_params.json")
RESULTS_JSON = os.path.join(HERE, "sysid_results.json")
TOKLEN_JSON = os.path.join(HERE, "prompt_token_lens.json")
COEF_JSON = os.path.join(ROOT, "gpu_coefficients.json")

GPU_KEY = {"NVIDIA H200": "H200", "NVIDIA B200": "B200",
           "NVIDIA A100 80GB PCIe": "A100", "NVIDIA RTX A5000": "A5000"}
RUN_GLOBS = ["logs/ragged/*/*", "logs/ragged_ladder/*/*", "logs/ragged_moe/*/*",
             "logs/B200/*/*", "logs/B200_32B/*/*", "logs/B200_large/*/*",
             "logs/poisson/*/*", "logs/wave", "logs/load_sweep/*/*"]
# crashed runs (EngineCore error dump in run.log; <15% of the window has iters)
BAD_RUNS = {
    "logs/B200/Qwen2-7B-Instruct/sharegpt_c16": "engine crashed ~4 s into window",
    "logs/B200/Mistral-7B-v0.1/sharegpt_c4": "engine crashed early in window",
}
BLOCK = 16                 # vLLM KV block size (prefix-cache granularity)
TOKEN_BUDGET = 2048        # max_num_batched_tokens seen in every run.log
BIN_S = 1.0                # fit / control interval
PRINT = print


# ============================================================ small utils
def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def rel(p):
    return os.path.relpath(p, ROOT)


def load_json(p, default=None):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return default


def gpu_coefficients():
    """Per-GPU energy/hardware constants (lead's single source of truth)."""
    defaults = {
        "H200": dict(e_wbyte=1.077e-10, e_kvbyte=3.38e-10, e_gemm=0.673e-12,
                     p_static=118.0, p_cap=700.0, bw=4.8e12, peak_flops=9.9e14),
        "B200": dict(e_wbyte=1.189e-10, e_kvbyte=3.511e-10, e_gemm=0.485e-12,
                     p_static=237.0, p_cap=1000.0, bw=8.0e12, peak_flops=2.25e15),
    }
    js = load_json(COEF_JSON, {}) or {}
    out = {}
    for g, dflt in defaults.items():
        d = dict(dflt)
        for k in dflt:
            if isinstance(js.get(g), dict) and js[g].get(k) is not None:
                d[k] = float(js[g][k])
        d["source"] = "gpu_coefficients.json" if g in js else "brief defaults"
        out[g] = d
    return out


# ============================================================ model constants
def model_constants(meta):
    """Analytic per-model constants, same accounting as energy_model.py."""
    from energy_model import (model_byte_constants, model_flop_constants,
                              resolve_model_obj, DTYPE_BYTES)
    W, kvbpt, _ = model_byte_constants(meta)
    mm, attn_res, act = model_flop_constants(meta)
    m, _ = resolve_model_obj(meta)
    dtb = DTYPE_BYTES.get(meta.get("dtype", "float16"), 2)
    c = dict(weight_bytes=W, kv_bytes_per_token=kvbpt, gemm_flops_per_token=mm,
             attn_flops_per_resident_token=attn_res, active_params=act,
             n_layers=int(m.L), dtype_bytes=dtb, moe=None)
    if getattr(m, "ffn", "mlp") == "moe" and m.n_routed_experts:
        c["moe"] = dict(n_experts=int(m.n_routed_experts), top_k=int(m.n_active_experts),
                        expert_bytes_all_layers=float(m._mlp_params(m.d_ff_expert)
                                                      * m._n_moe_layers() * dtb))
    return c


def weight_bytes_for_tokens(c, tok):
    """Vectorised MoE-occupancy weight bytes (dense: constant)."""
    tok = np.asarray(tok, float)
    if not c.get("moe"):
        return np.full_like(tok, c["weight_bytes"])
    E, k = c["moe"]["n_experts"], c["moe"]["top_k"]
    frac = 1.0 - (1.0 - k / E) ** np.maximum(tok, 1.0)
    extra = np.maximum(0.0, frac * E - k)
    return c["weight_bytes"] + extra * c["moe"]["expert_bytes_all_layers"]


# ============================================================ prompt replay
def tokenize_pools():
    """One-off: token length of every prompt in each pool under every model's
    tokenizer (vLLM default add_special_tokens=True). Lengths only (no text)."""
    from transformers import AutoTokenizer
    out = load_json(TOKLEN_JSON, {}) or {}
    ids = set()
    for rd in discover_runs():
        m = load_json(os.path.join(ROOT, rd, "run_meta.json"), {})
        if m.get("task") and m.get("model"):
            ids.add(m["model"])
    for hf in sorted(ids):
        if hf in out and all(t in out[hf] for t in ("alpaca", "sharegpt")):
            continue
        tok, src = None, hf
        cands = [hf] + (["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2-7B-Instruct"]
                        if hf.startswith("Qwen/") else [])
        for cand in cands:
            try:
                tok = AutoTokenizer.from_pretrained(cand, use_fast=True,
                                                    trust_remote_code=True)
                src = cand
                break
            except Exception as e:
                PRINT(f"[tokenize] {cand}: tokenizer unavailable ({type(e).__name__})")
        if tok is None:
            continue
        out[hf] = {"_tokenizer": src}
        for task in ("alpaca", "sharegpt"):
            pool = json.load(open(os.path.join(ROOT, f"prompts_{task}.json")))
            out[hf][task] = [len(tok(p)["input_ids"]) for p in pool]
        PRINT(f"[tokenize] {hf}: alpaca mean {np.mean(out[hf]['alpaca']):.1f}, "
              f"sharegpt mean {np.mean(out[hf]['sharegpt']):.1f}")
    with open(TOKLEN_JSON, "w") as f:
        json.dump(out, f)


def replay_draws(n, pool_size):
    """Pool indices drawn by energy_profile_load.make_prompt_fn (Random(1234))."""
    rng = random.Random(1234)
    rng_choice = rng.choice
    seq = range(pool_size)
    return [rng_choice(seq) for _ in range(n)]


def replay_poisson(rate, duration, n_max=100000):
    """Arrival offsets (s after window start) of _run_open_loop_async."""
    rng = random.Random(0)
    t, out = 0.0, []
    while t < duration and len(out) < n_max:
        out.append(t)
        t += rng.expovariate(rate)
    return np.array(out)


# ============================================================ run loading
def discover_runs():
    out = []
    for g in RUN_GLOBS:
        for rd in sorted(glob.glob(os.path.join(ROOT, g))):
            if (os.path.isdir(rd) and os.path.basename(rd) != "idle"
                    and os.path.exists(os.path.join(rd, "iter_log.csv"))
                    and os.path.exists(os.path.join(rd, "run_meta.json"))):
                out.append(rel(rd))
    return out


def model_name(meta):
    return (meta.get("model") or "").split("/")[-1]


def load_run(rd, toklens=None, with_power=True):
    """Parse one run into numpy arrays + derived per-row quantities."""
    ad = os.path.join(ROOT, rd)
    meta = load_json(os.path.join(ad, "run_meta.json"), {})
    res = load_json(os.path.join(ad, "results.json"), {}) or {}
    gpu = GPU_KEY.get(meta.get("gpu_name"), meta.get("gpu_name"))
    with open(os.path.join(ad, "iter_log.csv")) as f:
        rows = list(csv.DictReader(f))
    A = {k: np.array([_f(r.get(k)) for r in rows]) for k in
         ("t_start", "t_end", "n_running", "n_waiting", "kv_tokens_resident",
          "prefill_tokens", "decode_tokens")}
    for k in ("n_running", "n_waiting", "kv_tokens_resident", "prefill_tokens",
              "decode_tokens"):
        A[k] = np.nan_to_num(A[k])
    order = np.argsort(A["t_end"], kind="stable")
    A = {k: v[order] for k, v in A.items()}
    R = dict(rd=rd, meta=meta, res=res, gpu=gpu, model=model_name(meta),
             task=meta.get("task"), A=A)
    R["consts"] = model_constants(meta)
    R["idle_w"] = meta.get("idle_power_w")
    w0, w1 = meta.get("window_wall_t0"), meta.get("window_wall_t1")
    R["w0"], R["w1"] = w0, w1
    rate = float(meta.get("arrival_rate") or 0.0)
    sched = meta.get("concurrency_schedule")
    R["mode"] = "poisson" if rate > 0 else ("schedule" if sched else "closed")
    R["rate"] = rate
    segs = meta.get("segments") or []
    R["segments"] = segs
    R["c0"] = int(segs[0].get("concurrency", 1)) if segs and R["mode"] != "poisson" else 2
    R["label"] = rd.replace("logs/", "")
    R["prefill_comp"], R["n_first"], R["match_rate"], R["cached_frac"] = \
        computed_prefill(R, toklens)
    if with_power:
        from energy_model import load_power_samples
        R["pw_t"], R["pw_p"] = map(np.asarray, load_power_samples(
            os.path.join(ad, "power_trace.csv")))
    return R


def computed_prefill(R, toklens):
    """Per-row computed (non-prefix-cached) prefill tokens and #first-token
    events, by matching logged prefill_tokens to the replayed prompt draws.
    Returns (pf_comp[rows], n_first[rows], match_rate, cached_frac)."""
    A = R["A"]
    pf = A["prefill_tokens"]
    n_rows = len(pf)
    maxlen = int(R["meta"].get("max_model_len") or 0) or None
    if not R["task"]:
        # synthetic: one identical prompt (build_prompt) for every request;
        # only the first request pays the full prefill, the rest ~one block.
        L = int(R["meta"].get("input_len") or 512)
        n_first = np.round(np.where(pf > 0, pf / L, 0.0))
        per_req = L - BLOCK * ((L - 1) // BLOCK)
        pf_comp = n_first * per_req
        first = np.argmax(pf > 0) if (pf > 0).any() else None
        if first is not None:
            pf_comp[first] += L - per_req
        cf = 1.0 - pf_comp.sum() / max(pf.sum(), 1.0)
        return pf_comp, n_first, 1.0, cf
    lens_pool = (toklens or {}).get(R["meta"].get("model"), {}).get(R["task"])
    if lens_pool is None:
        return pf.copy(), np.where(pf > 0, 1.0, 0.0), float("nan"), 0.0
    n_draw = int(R["res"].get("completed_requests", 0)) * 3 + 5000
    draws = replay_draws(n_draw, len(lens_pool))
    seen = set()
    pf_comp = np.zeros(n_rows)
    n_first = np.zeros(n_rows)
    j, ok, tot = 0, 0, 0
    for i in np.nonzero(pf > 0)[0]:
        target = pf[i]
        acc, cnt, comp = 0.0, 0, 0.0
        j_start = j
        while acc < target and j < len(draws):
            idx = draws[j]
            L = lens_pool[idx]
            j += 1
            if maxlen and L >= maxlen:      # rejected by vLLM: consumed, never runs
                continue
            acc += L
            cnt += 1
            cached = BLOCK * ((L - 1) // BLOCK) if idx in seen else 0
            seen.add(idx)
            comp += L - cached
        tot += 1
        if acc == target:
            ok += 1
            pf_comp[i] = comp
            n_first[i] = cnt
        else:
            # desync (reordering / rejected prompt): fall back to the logged
            # value scaled by the cached fraction of the MATCHED events so far,
            # re-anchor the pointer.
            j = j_start + max(1, int(round(target / max(np.mean(lens_pool), 1))))
            m_ok = n_first[:i] > 0
            cf_so_far = (1.0 - pf_comp[:i][m_ok].sum() / pf[:i][m_ok].sum()
                         if pf[:i][m_ok].sum() > 0 else 0.0)
            pf_comp[i] = target * (1.0 - cf_so_far)
            n_first[i] = -max(1, round(target / max(np.mean(lens_pool), 1)))  # <0: unmatched
    n_first = np.abs(n_first)
    mr = ok / tot if tot else float("nan")
    if tot and mr < 0.5:
        # prompts do not follow the replayed sequence (older harness run):
        # no cache information -> assume nothing cached.
        return pf.copy(), np.where(pf > 0, np.maximum(1.0, np.round(pf / np.mean(lens_pool))), 0.0), mr, 0.0
    cf = 1.0 - pf_comp.sum() / max(pf.sum(), 1.0)
    return pf_comp, n_first, mr, cf


# ============================================================ binning
def row_masks(R):
    """valid rows: interval does not contain an engine-idle gap or a stall."""
    A = R["A"]
    dt = A["t_end"] - A["t_start"]
    busy_prev = np.r_[True, (A["n_running"][:-1] + A["n_waiting"][:-1]) > 0]
    ok = busy_prev & (dt >= 0) & (dt < 1.0) & np.isfinite(dt)
    return ok, dt


def bin_edges(R, dt=BIN_S, whole=False):
    A = R["A"]
    if whole or R["w0"] is None:
        t0, t1 = A["t_start"][0], A["t_end"][-1]
    else:
        t0, t1 = R["w0"], R["w1"]
    n = int(math.floor((t1 - t0) / dt))
    return t0 + dt * np.arange(n + 1)


def mean_power_bins(R, edges):
    from energy_model import integrate_window
    ts, ps = list(R["pw_t"]), list(R["pw_p"])
    out = np.full(len(edges) - 1, np.nan)
    for b in range(len(edges) - 1):
        e = integrate_window(ts, ps, edges[b], edges[b + 1])
        if e is not None:
            out[b] = e / (edges[b + 1] - edges[b])
    return out


# ============================================================ iteration-time model
def row_features(R):
    """Per-row analytic work for the time model."""
    A, c = R["A"], R["consts"]
    tok = A["decode_tokens"] + R["prefill_comp"]
    W = weight_bytes_for_tokens(c, tok)
    n = len(tok)
    return dict(
        tok=tok, n=np.maximum(A["n_running"], 1.0),
        Wb=W, KVb=A["kv_tokens_resident"] * c["kv_bytes_per_token"],
        F=c["gemm_flops_per_token"] * tok,
        L=np.full(n, float(c["n_layers"])))


# ---------------------------------------------------------------------------
# Iteration-time model forms. All predict the time of ONE engine iteration from
# the work it does; utilisations u_* are fractions of datasheet BW / peak.
#   roofline      t = max((Wb+KVb)/(bw u_m), F/(peak u_f))            [LIMINAL]
#   roofline+oh   t = t_oh + max((Wb+KVb)/(bw u_m), F/(peak u_f))
#   additive      t = t_oh + tau n + Wb/(bw u_w) + KVb/(bw u_kv) + F/(peak u_f)
#   phys          t = t_oh + tau n + max(Wb/(bw u_w) + KVb/(bw u_kv), F/(peak u_f))
#                 (u_w pinned per GPU from the cross-model law; t_oh per model)
#   law (GPU-level, all dense models jointly; used for leave-one-MODEL-out):
#                 t = a + b L + tau n + max(Wb/(bw u_w) + KVb/(bw u_kv), F/(peak u_f))
# t_oh / a+bL is the per-iteration fixed cost (enforce_eager => python + kernel
# launches ~ per layer); tau n is per-sequence scheduling/sampling/bookkeeping.
# ---------------------------------------------------------------------------
TIME_FORMS = {
    "roofline": ["u_m", "u_f"],
    "roofline+oh": ["t_oh", "u_m", "u_f"],
    "additive": ["t_oh", "tau_seq", "u_kv", "u_f"],
    "phys": ["t_oh", "tau_seq", "u_kv", "u_f"],
    "phys_p": ["t_ser", "t_cpu", "tau_seq", "u_kv", "u_f"],
    "law_sum": ["a", "b_per_layer", "tau_seq", "u_w", "u_kv", "u_f"],
    "law_max": ["a", "b_per_layer", "tau_seq", "u_w", "u_kv", "u_f"],
    "law_p": ["a", "b_per_layer", "tau_seq", "u_w", "u_kv", "u_f", "p"],
}
X0 = {"roofline": [0.5, 0.5], "roofline+oh": [0.003, 0.8, 0.5],
      "additive": [0.004, 1e-5, 0.5, 0.5], "phys": [0.004, 1e-5, 0.5, 0.5],
      "phys_p": [0.002, 0.003, 1e-5, 0.5, 0.5],
      "law_sum": [0.002, 1e-4, 1e-5, 0.8, 0.5, 0.5],
      "law_max": [0.002, 1e-4, 1e-5, 0.8, 0.5, 0.5],
      "law_p": [0.002, 1e-4, 1e-5, 0.8, 0.5, 0.5, 2.0]}
UB = 1.0      # utilisations are physically <= 1 of datasheet
_LAWB = ([0.0, 0.0, 0.0, 0.05, 0.02, 0.02], [0.05, 2e-3, 1e-3, UB, UB, UB])
BOUNDS = {"roofline": ([0.02, 0.02], [UB, UB]),
          "roofline+oh": ([0.0, 0.02, 0.02], [0.2, UB, UB]),
          "additive": ([0.0, 0.0, 0.02, 0.02], [0.2, 1e-3, UB, UB]),
          "phys": ([0.0, 0.0, 0.02, 0.02], [0.2, 1e-3, UB, UB]),
          "phys_p": ([0.0, 0.0, 0.0, 0.02, 0.02], [0.2, 0.2, 1e-3, UB, UB]),
          "law_sum": _LAWB, "law_max": _LAWB,
          "law_p": (_LAWB[0] + [1.0], _LAWB[1] + [12.0])}
LAW_FORMS = ("law_sum", "law_max", "law_p")


def pnorm(x, y, p):
    """(x^p + y^p)^(1/p): p=1 serial (sum), p->inf perfect overlap (max)."""
    m = np.maximum(np.maximum(x, y), 1e-12)
    return m * ((x / m) ** p + (y / m) ** p) ** (1.0 / p)


def time_rows(form, th, X, hw, u_w=None, p=None):
    bw, pk = hw["bw"], hw["peak_flops"]
    if form == "roofline":
        u_m, u_f = th
        return np.maximum((X["Wb"] + X["KVb"]) / (bw * u_m), X["F"] / (pk * u_f))
    if form == "roofline+oh":
        t_oh, u_m, u_f = th
        return t_oh + np.maximum((X["Wb"] + X["KVb"]) / (bw * u_m), X["F"] / (pk * u_f))
    if form == "additive":
        t_oh, tau, u_kv, u_f = th
        return (t_oh + tau * X["n"] + X["Wb"] / (bw * u_w) + X["KVb"] / (bw * u_kv)
                + X["F"] / (pk * u_f))
    if form == "phys":
        t_oh, tau, u_kv, u_f = th
        return t_oh + tau * X["n"] + np.maximum(
            X["Wb"] / (bw * u_w) + X["KVb"] / (bw * u_kv), X["F"] / (pk * u_f))
    if form == "phys_p":
        t_ser, t_cpu, tau, u_kv, u_f = th
        gpu = np.maximum(X["Wb"] / (bw * u_w) + X["KVb"] / (bw * u_kv), X["F"] / (pk * u_f))
        return t_ser + tau * X["n"] + pnorm(t_cpu, gpu, p)
    if form in LAW_FORMS:
        a, b, tau, uw, u_kv, u_f = th[:6]
        gpu = np.maximum(X["Wb"] / (bw * uw) + X["KVb"] / (bw * u_kv), X["F"] / (pk * u_f))
        cpu = b * X["L"]
        if form == "law_sum":
            comb = cpu + gpu
        elif form == "law_max":
            comb = np.maximum(cpu, gpu)
        else:
            comb = pnorm(cpu, gpu, th[6])
        return a + tau * X["n"] + comb
    raise ValueError(form)


def time_bins(R, dt=BIN_S):
    """Bin a run: measured busy seconds per bin + the valid rows in it."""
    ok, rdt = row_masks(R)
    tend = R["A"]["t_end"]
    edges = bin_edges(R, dt)
    idx = np.searchsorted(edges, tend, side="right") - 1
    bins = []
    for b in range(len(edges) - 1):
        sel = np.nonzero((idx == b) & ok)[0]
        if len(sel) < 3:
            continue
        busy = rdt[sel].sum()
        if busy < 0.5 * dt:
            continue
        bins.append((sel, busy))
    return bins


def stack_bins(runs, dt=BIN_S):
    """Concatenate rows of all bins of all runs, with a bin id per row."""
    feats, bin_id, busy, run_id = [], [], [], []
    nb = 0
    for ri, R in enumerate(runs):
        X = row_features(R)
        for sel, bsy in time_bins(R, dt):
            feats.append({k: v[sel] for k, v in X.items()})
            bin_id.append(np.full(len(sel), nb))
            busy.append(bsy)
            run_id.append(ri)
            nb += 1
    if not feats:
        return None
    Xs = {k: np.concatenate([f[k] for f in feats]) for k in feats[0]}
    return dict(X=Xs, bin=np.concatenate(bin_id), busy=np.array(busy),
                run=np.array(run_id), nb=nb)


def predict_bins(form, th, S, hw, u_w=None, p=None):
    t = time_rows(form, th, S["X"], hw, u_w, p)
    return np.bincount(S["bin"], weights=t, minlength=S["nb"])


def fit_time(form, S, hw, x0=None, u_w=None, p=None, quick=False):
    from scipy.optimize import least_squares
    if S is None or S["nb"] < 3:
        return None

    def resid(th):
        return (predict_bins(form, th, S, hw, u_w, p) - S["busy"]) / S["busy"]
    lo, hi = BOUNDS[form]
    starts = [x0 or X0[form], X0[form]]
    if quick and x0 is not None:      # held-out refits: warm start only
        starts = [x0]
    if form in ("phys", "additive"):
        starts += [[0.002, 2e-5, 0.9, 0.3], [0.006, 0.0, 0.1, 0.9]]
    if form == "phys_p":
        starts += [[0.003, 0.001, 1e-5, 0.5, 0.5], [0.001, 0.02, 1e-5, 0.5, 0.5]]
    if form in LAW_FORMS:
        ext = [2.0] if form == "law_p" else []
        starts += [[0.001, 5e-5, 1e-5, 0.9, 0.9, 0.6] + ext,
                   [0.003, 5e-5, 1e-5, 0.8, 0.3, 0.3] + ext,
                   [0.0005, 1.5e-4, 1e-5, 0.7, 0.5, 0.5] + ([6.0] if ext else [])]
    best = None
    for s in starts:
        s = np.clip(np.asarray(s, float), np.asarray(lo) + 1e-12, np.asarray(hi) - 1e-12)
        r = least_squares(resid, s, bounds=(lo, hi), loss="soft_l1", f_scale=0.05,
                          x_scale="jac")
        if best is None or r.cost < best.cost:
            best = r
    return best.x


def err_stats(pred, meas):
    e = (pred - meas) / meas
    return dict(mape=float(np.mean(np.abs(e)) * 100), bias=float(np.mean(e) * 100),
                rmse_rel=float(np.sqrt(np.mean(e ** 2)) * 100), n=int(len(e)))


def time_groups(runs, hwc):
    groups = {}
    for R in runs:
        if R["rd"] in BAD_RUNS or not R["task"] or R["gpu"] not in hwc:
            continue          # time fit on real-prompt runs of H200/B200 only
        groups.setdefault((R["gpu"], R["model"]), []).append(R)
    return groups


def fit_gpu_law(groups, hwc):
    """Joint GPU-level fit over all dense models, for each law variant;
    leave-one-MODEL-out (LOMO). The variant with the best LOMO is selected."""
    law = {}
    for gpu in sorted({g for g, _ in groups}):
        dense = {m: rs for (g, m), rs in groups.items()
                 if g == gpu and not rs[0]["consts"].get("moe")}
        hw = hwc[gpu]
        S = stack_bins([r for rs in dense.values() for r in rs])
        Sm = {m: stack_bins(rs) for m, rs in dense.items()}
        ent = {"models": sorted(dense), "variants": {}}
        for form in LAW_FORMS:
            th = fit_time(form, S, hw)
            ins = err_stats(predict_bins(form, th, S, hw), S["busy"])
            lomo = {}
            for m in dense:
                tr = [r for mm, rs in dense.items() if mm != m for r in rs]
                thm = fit_time(form, stack_bins(tr), hw, x0=list(th), quick=True)
                lomo[m] = dict(err_stats(predict_bins(form, thm, Sm[m], hw), Sm[m]["busy"]),
                               theta=dict(zip(TIME_FORMS[form], map(float, thm))))
            ent["variants"][form] = dict(
                theta=dict(zip(TIME_FORMS[form], map(float, th))), in_sample=ins,
                lomo=lomo, lomo_mape_mean=float(np.mean([v["mape"] for v in lomo.values()])))
        best = min(ent["variants"], key=lambda f: ent["variants"][f]["lomo_mape_mean"])
        ent["best"] = best
        ent.update({k: ent["variants"][best][k] for k in ("theta", "in_sample", "lomo",
                                                          "lomo_mape_mean")})
        th = ent["theta"]
        ent["u_w"] = th["u_w"]
        ent["p"] = th.get("p", 1.0 if best == "law_sum" else 12.0)
        law[gpu] = ent
    return law


def fit_time_models(runs, hwc, law):
    """Per GPU x model fits of every form; leave-one-run-out (LORO) held-out."""
    groups = time_groups(runs, hwc)
    out = {}
    for (gpu, model), rs in sorted(groups.items()):
        hw = hwc[gpu]
        u_w, p = law[gpu]["u_w"], law[gpu]["p"]
        c = rs[0]["consts"]
        S_all = stack_bins(rs)
        ent = {"n_runs": len(rs), "n_bins": int(S_all["nb"]), "forms": {},
               "u_w_pinned": u_w, "p_pinned": p}
        splits = []
        for k in range(len(rs)):
            S_te = stack_bins([rs[k]])
            if S_te is not None:
                splits.append((stack_bins([r for i, r in enumerate(rs) if i != k]),
                               S_te, rs[k]["label"]))
        lth = law[gpu]["theta"]
        x0p = [lth["a"], lth["b_per_layer"] * c["n_layers"], lth["tau_seq"],
               lth["u_kv"], lth["u_f"]]
        for form in ("roofline", "roofline+oh", "additive", "phys", "phys_p"):
            x0 = x0p if form == "phys_p" else None
            th = fit_time(form, S_all, hw, x0=x0, u_w=u_w, p=p)
            ins = err_stats(predict_bins(form, th, S_all, hw, u_w, p), S_all["busy"])
            pe, me, per_run = [], [], {}
            for S_tr, S_te, lab in splits:
                thk = fit_time(form, S_tr, hw, x0=list(th), u_w=u_w, p=p, quick=True)
                pr = predict_bins(form, thk, S_te, hw, u_w, p)
                pe.append(pr); me.append(S_te["busy"])
                per_run[lab] = err_stats(pr, S_te["busy"])["bias"]
            ent["forms"][form] = dict(
                theta=dict(zip(TIME_FORMS[form], map(float, th))), in_sample=ins,
                loro=err_stats(np.concatenate(pe), np.concatenate(me)),
                loro_bias_per_run=per_run)
        if model in law[gpu]["lomo"]:
            ent["law_lomo"] = {k: v for k, v in law[gpu]["lomo"][model].items()
                               if k != "theta"}
        # constant-t baseline (LORO): mean seconds per row from other runs
        pe, me = [], []
        for S_tr, S_te, _ in splits:
            tbar = S_tr["busy"].sum() / len(S_tr["bin"])
            pe.append(tbar * np.bincount(S_te["bin"], minlength=S_te["nb"]))
            me.append(S_te["busy"])
        ent["const_baseline_loro"] = err_stats(np.concatenate(pe), np.concatenate(me))
        ent["n_layers"] = c["n_layers"]
        ent["weight_bytes"] = c["weight_bytes"]
        ent["mean_iter_s"] = float(S_all["busy"].sum() / len(S_all["bin"]))
        out[f"{gpu}|{model}"] = ent
    return out


# ============================================================ plant validation
KS = list(range(1, 11))
VARS = ["n_waiting", "n_running", "kv_tokens_resident", "power_w", "tokens_s"]


def kv_capacity_from_log(rd):
    import re
    p = os.path.join(ROOT, rd, "run.log")
    if not os.path.exists(p):
        return None
    with open(p, errors="ignore") as f:
        for line in f:
            mm = re.search(r"GPU KV cache size: ([0-9,]+) tokens", line)
            if mm:
                return float(mm.group(1).replace(",", ""))
    return None


def draw_stats(R, toklens):
    """Replayed prompt draws of a run: full lengths, computed lengths, cached."""
    pool = (toklens or {}).get(R["meta"].get("model"), {}).get(R["task"])
    if not R["task"] or pool is None:
        L = int(R["meta"].get("input_len") or 512)
        return None, float(L)
    n = int(R["res"].get("completed_requests", 0)) + 2 * R["c0"] + 50
    draws = replay_draws(n, len(pool))
    maxlen = int(R["meta"].get("max_model_len") or 0) or None
    lens = np.array([pool[j] for j in draws], float)
    if maxlen:
        lens = lens[lens < maxlen]
    return (draws, lens), float(lens.mean())


def workload_stats(R, toklens):
    """Per-run workload: mean prompt, mean gen, gen-length CV^2 (from KV)."""
    res = R["res"]
    done = max(int(res.get("completed_requests", 0)), 1)
    gen = res.get("generated_tokens_window", 0) / done
    if not R["task"]:
        gen = float(R["meta"].get("output_len") or gen)
    _, p_mean = draw_stats(R, toklens)
    A = R["A"]
    cv2 = None
    if R["mode"] == "closed" and R["w0"]:
        sel = (A["t_end"] > R["w0"] + 10) & (A["t_end"] < R["w1"]) & (A["n_running"] > 0)
        if sel.sum() > 100:
            kv_per = A["kv_tokens_resident"][sel].sum() / A["n_running"][sel].sum()
            cv2 = 2.0 * (kv_per - 0.5 * BLOCK - p_mean) / gen - 1.0
    # closed loop: last admission (first-token event) ~ generator deadline; the
    # window then drains. Think time Z = (c - n_run - n_wait) / completion rate.
    deadline, think = None, None
    if R["mode"] in ("closed", "schedule") and R["w0"]:
        ev = A["t_end"][R["n_first"] > 0]
        ev = ev[(ev > R["w0"]) & (ev <= R["w1"])]
        if len(ev):
            deadline = float(ev[-1])
    if R["mode"] == "closed" and deadline:
        c = R["c0"]
        t_a, t_b = R["w0"] + 5.0, deadline - 1.0
        sel = (A["t_end"] > t_a) & (A["t_end"] < t_b)
        nf = R["n_first"][sel].sum()
        if t_b - t_a > 5 and nf > 5:
            lam = nf / (t_b - t_a)
            think = max(0.0, (c - A["n_running"][sel].mean() - A["n_waiting"][sel].mean()) / lam)
    return dict(gen_len=gen, prompt_len=p_mean, cv2=cv2, cached_frac=R["cached_frac"],
                kv_shared_frac=0.0 if R["task"] else R["cached_frac"],
                deadline=deadline, think_s=think)


def observe_grid(R, dt=BIN_S):
    """Observed (q_wait, n_run, kv) at grid boundaries, and per-interval
    measured power and decode-token throughput."""
    A = R["A"]
    edges = bin_edges(R, dt)
    te = A["t_end"]
    obs = np.full((len(edges), 3), np.nan)
    for k, T in enumerate(edges):
        lo = np.searchsorted(te, T - 0.1, side="right")
        hi = np.searchsorted(te, T, side="right")
        if hi > lo:
            sl = slice(lo, hi)
        elif hi > 0:
            sl = slice(hi - 1, hi)
        else:
            obs[k] = 0.0          # before the first iteration: empty system
            continue
        obs[k] = [A["n_waiting"][sl].mean(), A["n_running"][sl].mean(),
                  A["kv_tokens_resident"][sl].mean()]
    pw = mean_power_bins(R, edges)
    idx = np.searchsorted(edges, te, side="right") - 1
    tok = np.bincount(idx[(idx >= 0) & (idx < len(edges) - 1)],
                      weights=(A["decode_tokens"])[(idx >= 0) & (idx < len(edges) - 1)],
                      minlength=len(edges) - 1) / dt
    return edges, obs, pw, tok


def disturbances(R, edges, wl, toklens):
    """Per-interval disturbance dicts for the plant (see module docstring)."""
    n = len(edges) - 1
    ds = []
    if R["mode"] == "poisson":
        # every request completes (drained), incl. 2 warmups -> #arrivals exact
        n_arr = max(int(R["res"].get("completed_requests", 0)) - 2, 0)
        arr_t = replay_poisson(R["rate"], 1e9, n_max=n_arr) \
            if n_arr else np.array([])
        ds_draw, _ = draw_stats(R, toklens)
        k_of = np.floor(arr_t / (edges[1] - edges[0])).astype(int)
        for k in range(n):
            js = np.nonzero(k_of == k)[0]
            d = dict(arrivals=float(len(js)), gen_len=wl["gen_len"],
                     cached_frac=wl["cached_frac"], prompt_len=wl["prompt_len"],
                     kv_shared_frac=wl["kv_shared_frac"])
            if ds_draw is not None and len(js):
                draws, _lens = ds_draw
                pool = toklens[R["meta"]["model"]][R["task"]]
                seen = set(draws[:2 + js[0]])
                full = comp = 0.0
                for a in js:
                    idx = draws[2 + a]
                    L = pool[idx]
                    full += L
                    comp += L - (BLOCK * ((L - 1) // BLOCK) if idx in seen else 0)
                    seen.add(idx)
                d["prompt_len"] = full / len(js)
                d["cached_frac"] = 1.0 - comp / max(full, 1.0)
            ds.append(d)
        return ds
    segs = R["segments"]
    for k in range(n):
        tm = 0.5 * (edges[k] + edges[k + 1])
        c = segs[-1]["concurrency"] if segs else R["c0"]
        for sg in segs:
            if sg["t0"] <= tm < sg["t1"]:
                c = sg["concurrency"]
                break
        if R["mode"] == "closed" and wl.get("deadline") and edges[k] >= wl["deadline"]:
            c = 0                     # generator stopped submitting: drain
        ds.append(dict(clients=float(c), prompt_len=wl["prompt_len"], gen_len=wl["gen_len"],
                       cached_frac=wl["cached_frac"], kv_shared_frac=wl["kv_shared_frac"]))
    return ds


def kstep_run(plant, R, edges, obs, pw, tok, ds, kmax=10):
    """Roll the plant from every observed boundary state; collect k-step
    predictions, persistence baselines and a bias-corrected power variant
    (offset-free-MPC style: add the last interval's power residual)."""
    n = len(edges) - 1
    out = {v: {k: ([], [], []) for k in range(1, kmax + 1)} for v in VARS + ["power_bc"]}
    prev_resid = None
    for s in range(1, n):
        if not np.all(np.isfinite(obs[s])):
            prev_resid = None
            continue
        # 1-step prediction of the interval just finished -> power residual
        if np.all(np.isfinite(obs[s - 1])) and np.isfinite(pw[s - 1]):
            x_ = plant.state_from_obs(obs[s - 1, 0], obs[s - 1, 1], obs[s - 1, 2],
                                      prompt_len=ds[s - 1].get("prompt_len"),
                                      gen_len=ds[s - 1].get("gen_len"),
                                      clients=ds[s - 1].get("clients"))
            _, y_ = plant.step(x_, {}, ds[s - 1])
            prev_resid = pw[s - 1] - y_["power_w"]
        d0 = ds[s]
        x = plant.state_from_obs(obs[s, 0], obs[s, 1], obs[s, 2],
                                 prompt_len=d0.get("prompt_len"), gen_len=d0.get("gen_len"),
                                 clients=d0.get("clients"))
        for k in range(1, kmax + 1):
            if s + k > n:
                break
            x, y = plant.step(x, {}, ds[s + k - 1])
            m_obs = obs[s + k]
            pm = pw[s + k - 1]
            tm = tok[s + k - 1]
            if not (np.all(np.isfinite(m_obs)) and np.isfinite(pm)):
                continue
            p_prev = pw[s - 1] if np.isfinite(pw[s - 1]) else pm
            vals = [(y["n_waiting"], m_obs[0], obs[s, 0]),
                    (y["n_running"], m_obs[1], obs[s, 1]),
                    (y["kv_tokens"], m_obs[2], obs[s, 2]),
                    (y["power_w"], pm, p_prev),
                    (y["tokens_out"] / plant.dt, tm, tok[s - 1]),
                    (y["power_w"] + (prev_resid if prev_resid is not None else 0.0), pm, p_prev)]
            for v, (a, b, c) in zip(VARS + ["power_bc"], vals):
                out[v][k][0].append(a); out[v][k][1].append(b); out[v][k][2].append(c)
    return out


def metric(pred, meas):
    pred, meas = np.asarray(pred, float), np.asarray(meas, float)
    if pred.size == 0:
        return dict(rmse=float("nan"), mape=float("nan"), nrmse=float("nan"), bias=float("nan"),
                    mean=float("nan"), n=0)
    e = pred - meas
    nz = np.abs(meas) > 1e-6
    mape = float(np.mean(np.abs(e[nz] / meas[nz])) * 100) if nz.sum() > 0.5 * len(meas) else float("nan")
    rm = float(np.sqrt(np.mean(e ** 2)))
    mu = float(np.mean(np.abs(meas)))
    return dict(rmse=rm, mape=mape, nrmse=rm / mu * 100 if mu > 0 else float("nan"),
                bias=float(np.mean(e)), mean=mu, n=int(len(e)))


def run_category(R):
    syn = "" if R["task"] else "-synthetic"
    lvl = ""
    if R["mode"] == "closed":
        lvl = "-c<=4" if R["c0"] <= 4 else "-c>=8"
    return f"{R['gpu']}:{R['mode']}{lvl}{syn}"


def cmd_validate(args):
    from controls.plant import Plant
    P = load_json(PARAMS_JSON, {})
    hwc = gpu_coefficients()
    toklens = load_json(TOKLEN_JSON, {})
    runs = [R for R in load_all(with_power=True) if R["gpu"] in hwc and R["rd"] not in BAD_RUNS]
    PRINT(f"[validate] {len(runs)} runs")
    # ---- workload priors (per model x task, from runs) + KV capacities
    wls = {R["label"]: workload_stats(R, toklens) for R in runs}
    kvcap = {}
    for R in runs:
        c = kv_capacity_from_log(R["rd"])
        if c:
            kvcap.setdefault(f"{R['gpu']}|{R['model']}", c)
    by_mt = {}
    for R in runs:
        if R["task"]:
            by_mt.setdefault((R["model"], R["task"]), []).append(R["label"])
    PRINT("\n=== workload: gen-length CV^2 implied by closed-loop KV (kv/n = p + g(1+cv2)/2) ===")
    cv2_task = {}
    for (mdl, task), labs in sorted(by_mt.items()):
        cvs = [wls[l]["cv2"] for l in labs if wls[l]["cv2"] is not None]
        gens = [wls[l]["gen_len"] for l in labs]
        cv2_task.setdefault(task, []).extend(cvs)
        PRINT(f"{mdl:28s}{task:9s} gen={np.mean(gens):6.1f} prompt={wls[labs[0]]['prompt_len']:6.1f} "
              f"cv2 per run={np.round(cvs, 2).tolist()}")
    cv2_med = {t: float(np.median(v)) for t, v in cv2_task.items()}
    PRINT(f"median cv2 per task: {cv2_med}")

    def stages_for(cv2):
        return int(np.clip(round(1.0 / max(cv2, 1e-3)), 1, 16))

    # store workload priors for the Plant defaults
    workload = {}
    for (mdl, task), labs in by_mt.items():
        workload.setdefault(mdl, {})[task] = dict(
            gen_len=float(np.mean([wls[l]["gen_len"] for l in labs])),
            prompt_len=float(np.mean([wls[l]["prompt_len"] for l in labs])),
            n_stages=stages_for(cv2_med.get(task, 0.25)))
    for mdl, tt in workload.items():
        tt.update({k: float(np.mean([v[k] for v in tt.values() if isinstance(v, dict)]))
                   for k in ("gen_len", "prompt_len")})
        tt["n_stages"] = int(round(np.mean([v["n_stages"] for k, v in tt.items()
                                            if isinstance(v, dict)])))
    think = {}
    for R in runs:
        z = wls[R["label"]].get("think_s")
        if z is not None:
            think.setdefault(R["gpu"], []).append(z)
    PRINT("closed-loop client think time Z (s), median per GPU: " +
          str({g: round(float(np.median(v)), 4) for g, v in think.items()}) +
          "  IQR: " + str({g: np.round(np.percentile(v, [25, 75]), 4).tolist()
                           for g, v in think.items()}))
    P["think_s"] = {g: float(np.median(v)) for g, v in think.items()}
    P["workload"] = workload
    P["kv_capacity"] = kvcap
    P["gen_cv2_by_task"] = cv2_med
    with open(PARAMS_JSON, "w") as f:
        json.dump(P, f, indent=1)

    # ---- LORO time params per run (refit phys_p without the run)
    groups = time_groups(runs, hwc)
    loro_theta = {}
    for (gpu, model), rs in groups.items():
        tm = P["time_model"][f"{gpu}|{model}"]
        th0 = list(tm["forms"]["phys_p"]["theta"].values())
        for k, R in enumerate(rs):
            S_tr = stack_bins([r for j, r in enumerate(rs) if j != k])
            th = fit_time("phys_p", S_tr, hwc[gpu], x0=th0, u_w=tm["u_w_pinned"],
                          p=tm["p_pinned"], quick=True)
            loro_theta[R["label"]] = dict(zip(TIME_FORMS["phys_p"], map(float, th)))

    # ---- k-step validation
    agg = {}
    per_run = {}
    for R in runs:
        key = f"{R['gpu']}|{R['model']}"
        if key not in P["time_model"]:
            continue
        wl = dict(wls[R["label"]])
        if R["task"]:   # gen length is NOT known in advance: use other runs' mean
            others = [wls[l]["gen_len"] for l in by_mt[(R["model"], R["task"])] if l != R["label"]]
            wl["gen_len"] = float(np.mean(others)) if others else wl["gen_len"]
            m = stages_for(cv2_med.get(R["task"], 0.25))
        else:
            m = 16      # forced exact output length: ~deterministic
        zs = [wls[r2["label"]]["think_s"] for r2 in runs
              if r2["gpu"] == R["gpu"] and r2["label"] != R["label"]
              and wls[r2["label"]].get("think_s") is not None]
        ov = dict(n_stages=m, kv_capacity=kvcap.get(key, 1e6), gen_len=wl["gen_len"],
                  prompt_len=wl["prompt_len"], cached_frac=wl["cached_frac"],
                  kv_shared_frac=wl["kv_shared_frac"],
                  think_s=float(np.median(zs)) if zs else 0.03)
        if R["idle_w"]:
            ov["p_static"] = float(R["idle_w"])
        if R["label"] in loro_theta:
            th = loro_theta[R["label"]]
            ov.update(t_ser=th["t_ser"], t_cpu=th["t_cpu"], tau_seq=th["tau_seq"],
                      u_kv=th["u_kv"], u_f=th["u_f"])
        plant = Plant(R["gpu"], R["model"], dt=BIN_S, **ov)
        edges, obs, pw, tok = observe_grid(R)
        ds = disturbances(R, edges, wl, toklens)
        res = kstep_run(plant, R, edges, obs, pw, tok, ds)
        cat = run_category(R)
        VV = VARS + ["power_bc"]
        per_run[R["label"]] = {v: {k: metric(res[v][k][0], res[v][k][1]) for k in (1, 5, 10)}
                               for v in VARS + ["power_bc"]}
        for v in VV:
            for k in KS:
                a = agg.setdefault(cat, {}).setdefault(v, {}).setdefault(k, ([], [], []))
                for j in range(3):
                    a[j].extend(res[v][k][j])
        # also pool per GPU
        for v in VV:
            for k in KS:
                a = agg.setdefault(f"{R['gpu']}:ALL", {}).setdefault(v, {}).setdefault(k, ([], [], []))
                for j in range(3):
                    a[j].extend(res[v][k][j])
    table = {}
    for cat, dv in agg.items():
        table[cat] = {v: {k: dict(model=metric(p, m_), persist=metric(c, m_))
                          for k, (p, m_, c) in dk.items()} for v, dk in dv.items()}
    out = dict(kstep=table, per_run=per_run, workload=workload, cv2_by_task=cv2_med,
               kv_capacity=kvcap)
    prev = load_json(RESULTS_JSON, {}) or {}
    if "smoke" in prev:
        out["smoke"] = prev["smoke"]
    with open(RESULTS_JSON, "w") as f:
        json.dump(out, f, indent=1)
    report_kstep(table)


def report_kstep(table):
    PRINT("\n=== k-step-ahead validation (dt=1 s): model vs persistence ===")
    PRINT("cells: RMSE / NRMSE%=RMSE/mean|meas| (MAPE% for power)  [persistence same]")
    for cat in sorted(table):
        PRINT(f"\n--- {cat}  (n={table[cat]['n_running'][1]['model']['n']} starts at k=1)")
        for v in VARS + ["power_bc"]:
            cells = []
            for k in (1, 2, 5, 10):
                c = table[cat][v].get(k) or table[cat][v].get(str(k))
                mo, pe = c["model"], c["persist"]
                ex = lambda z: (f"{z['rmse']:.3g}/{z['nrmse']:.0f}%" +
                                (f"/{z['mape']:.1f}%" if v.startswith("power") else ""))
                cells.append(f"k={k}: {ex(mo)} [{ex(pe)}]")
            PRINT(f"  {v:18s} " + " | ".join(cells))


# ============================================================ entry points
def load_all(with_power=True, only=None):
    toklens = load_json(TOKLEN_JSON, {})
    runs = []
    for rd in discover_runs():
        if only and not only(rd):
            continue
        try:
            runs.append(load_run(rd, toklens, with_power=with_power))
        except Exception as e:
            PRINT(f"[skip] {rd}: {e!r}")
    return runs


def cmd_fit(args):
    hwc = gpu_coefficients()
    runs = load_all(with_power=False)
    PRINT(f"[fit] {len(runs)} runs loaded")
    PRINT(f"{'run':52s}{'match':>7s}{'cached%':>8s}")
    for R in runs:
        if R["task"]:
            PRINT(f"{R['label']:52s}{R['match_rate']:7.2f}{100*R['cached_frac']:8.1f}")
    groups = time_groups(runs, hwc)
    law = fit_gpu_law(groups, hwc)
    tfits = fit_time_models(runs, hwc, law)
    ood = ood_synthetic(runs, hwc, tfits)
    prev = load_json(PARAMS_JSON, {}) or {}
    prev["time_ood_synthetic"] = ood
    prev["time_model"] = tfits
    prev["time_law"] = law
    prev["prefill_cache"] = {R["label"]: dict(match_rate=R["match_rate"],
                                              cached_frac=R["cached_frac"])
                             for R in runs if R["task"]}
    prev["model_consts"] = {R["model"]: R["consts"] for R in runs}
    prev["gpu_hw"] = hwc
    prev["token_budget"] = TOKEN_BUDGET
    prev["block_size"] = BLOCK
    with open(PARAMS_JSON, "w") as f:
        json.dump(prev, f, indent=1)
    report_time(tfits, law)


def ood_synthetic(runs, hwc, tfits):
    """Out-of-distribution test of the per-model time fits: synthetic-prompt
    runs (never used in fitting; identical prompts -> ~97% prefix-cache hits),
    predicted with cache-aware computed prefill vs the naive logged count."""
    out = {}
    for R in runs:
        key = f"{R['gpu']}|{R['model']}"
        if R["task"] or key not in tfits or R["rd"] in BAD_RUNS:
            continue
        tm = tfits[key]
        th = list(tm["forms"]["phys_p"]["theta"].values())
        S = stack_bins([R])
        if S is None:
            continue
        aware = err_stats(predict_bins("phys_p", th, S, hwc[R["gpu"]], tm["u_w_pinned"],
                                       tm["p_pinned"]), S["busy"])
        saved = R["prefill_comp"]
        R["prefill_comp"] = R["A"]["prefill_tokens"].copy()
        Sn = stack_bins([R])
        R["prefill_comp"] = saved
        naive = err_stats(predict_bins("phys_p", th, Sn, hwc[R["gpu"]], tm["u_w_pinned"],
                                       tm["p_pinned"]), Sn["busy"])
        out[R["label"]] = dict(cache_aware=aware, naive=naive)
    if out:
        PRINT("\n=== OOD time test on synthetic runs (MAPE / bias %) ===")
        for k, v in out.items():
            PRINT(f"{k:48s} cache-aware {v['cache_aware']['mape']:6.2f}/{v['cache_aware']['bias']:+6.2f}"
                  f"   naive {v['naive']['mape']:7.2f}/{v['naive']['bias']:+7.2f}")
        a = np.array([v["cache_aware"]["mape"] for v in out.values()])
        b = np.array([v["naive"]["mape"] for v in out.values()])
        PRINT(f"mean over {len(a)} runs: cache-aware {a.mean():.2f}%  naive {b.mean():.2f}%")
    return out


def report_time(tfits, law):
    PRINT("\n=== iteration-time model: 1 s bin busy-time error, MAPE % "
          "(in-sample / leave-one-run-out) ===")
    forms = ["roofline", "roofline+oh", "additive", "phys", "phys_p"]
    PRINT(f"{'GPU|model':30s}{'bins':>5s}{'const':>6s}" +
          "".join(f"{f:>13s}" for f in forms) + f"{'lawLOMO':>8s}")
    for k, e in tfits.items():
        lomo = e.get("law_lomo", {}).get("mape")
        PRINT(f"{k:30s}{e['n_bins']:5d}{e['const_baseline_loro']['mape']:6.1f}" +
              "".join(f"{e['forms'][f]['in_sample']['mape']:6.2f}/{e['forms'][f]['loro']['mape']:<6.2f}"
                      for f in forms) + (f"{lomo:8.2f}" if lomo is not None else f"{'-':>8s}"))
    PRINT("\n=== GPU-level laws (joint dense fit), LOMO = leave-one-model-out ===")
    for g, e in law.items():
        for f, v in e["variants"].items():
            th = v["theta"]
            PRINT(f"{g} {f:8s} a={th['a']*1e3:.3f}ms b={th['b_per_layer']*1e6:.1f}us/layer "
                  f"tau={th['tau_seq']*1e6:.2f}us/seq u_w={th['u_w']:.3f} u_kv={th['u_kv']:.3f} "
                  f"u_f={th['u_f']:.3f} p={th.get('p', float('nan')):.2f} | in {v['in_sample']['mape']:.2f}% "
                  f"LOMO {v['lomo_mape_mean']:.2f}%  " +
                  " ".join(f"{m.split('-Instruct')[0]}:{x['bias']:+.1f}" for m, x in v["lomo"].items()))
        PRINT(f"   -> selected {e['best']}")
    PRINT("\n=== per GPU x model 'phys_p' params (u_w, p pinned from GPU law) ===")
    for k, e in tfits.items():
        th = e["forms"]["phys_p"]["theta"]
        wr = max(e["forms"]["phys_p"]["loro_bias_per_run"].items(), key=lambda kv: abs(kv[1]))
        PRINT(f"{k:30s} t_ser={th['t_ser']*1e3:6.3f}ms t_cpu={th['t_cpu']*1e3:6.3f}ms "
              f"tau={th['tau_seq']*1e6:6.2f}us u_kv={th['u_kv']:.3f} u_f={th['u_f']:.3f} "
              f"(u_w={e['u_w_pinned']:.3f},p={e['p_pinned']:.1f}) mean_iter={e['mean_iter_s']*1e3:.2f}ms "
              f"worst-run {wr[0].split('/')[-1]} {wr[1]:+.1f}%")


A5000_HW = dict(bw=768e9, peak_flops=111e12, p_cap=230.0,
                # no A5000 energy calibration exists: H200 coefficients as a prior
                e_wbyte=1.076e-10, e_kvbyte=3.376e-10, e_gemm=0.680e-12)


def cmd_smoke(args):
    """Check the plant's Little's-law latency proxies against the per-request
    latencies of a smoke run with requests.csv (closed-loop c)."""
    from controls.plant import Plant
    rd = args.run_dir
    toklens = load_json(TOKLEN_JSON, {})
    R = load_run(rd, toklens)
    rq = list(csv.DictReader(open(os.path.join(ROOT, rd, "requests.csv"))))
    t0 = np.array([float(r["t_submit"]) for r in rq])
    tf = np.array([_f(r["t_first_token"]) for r in rq])
    t1 = np.array([float(r["t_finish"]) for r in rq])
    gen = np.array([float(r["gen_tokens"]) for r in rq])
    ttft, e2e = tf - t0, t1 - t0
    tpot = (t1 - tf) / np.maximum(gen - 1, 1)
    c = R["c0"]
    A = R["A"]
    wl = workload_stats(R, toklens)
    ts, te = R["w0"] + 3.0, (wl["deadline"] or R["w1"]) - 1.0
    sel = (A["t_end"] > ts) & (A["t_end"] < te)
    n_mean = A["n_running"][sel].mean()
    done_in = ((t1 > ts) & (t1 < te)).sum() / (te - ts)
    little = n_mean / done_in
    PRINT(f"[smoke] {rd}: {len(rq)} requests, c={c}, gen mean {gen.mean():.1f}")
    PRINT(f"measured per-request: TTFT mean {ttft.mean()*1e3:.1f} ms (p50 {np.median(ttft)*1e3:.1f}, "
          f"p90 {np.percentile(ttft, 90)*1e3:.1f}); e2e mean {e2e.mean():.3f} s (p50 {np.median(e2e):.3f}); "
          f"TPOT mean {tpot.mean()*1e3:.2f} ms")
    PRINT(f"Little's law on measured state: L=n_run={n_mean:.2f}, lambda={done_in:.2f}/s -> "
          f"W={little:.3f} s  vs measured mean e2e {e2e[(t1 > ts) & (t1 < te)].mean():.3f} s "
          f"(think Z={wl['think_s']})")
    # measured iteration time in steady bins
    S = stack_bins([R])
    t_meas = S["busy"].sum() / len(S["bin"])
    gpu = R["gpu"]
    hw = dict(A5000_HW)
    hw["p_cap"] = float(R["meta"].get("power_limit_w") or hw["p_cap"])
    hw["p_static"] = float(R["idle_w"] or 58.0)
    P = load_json(PARAMS_JSON, {})
    variants = {}
    # (a) zero-shot: H200 GPU-law structure with A5000 bw/peak
    law = P["time_law"]["H200"]["theta"]
    L = R["consts"]["n_layers"]
    variants["zero-shot (H200 law, A5000 datasheet)"] = dict(
        t_ser=law["a"], t_cpu=law["b_per_layer"] * L, tau_seq=law["tau_seq"],
        u_w=law["u_w"], u_kv=law["u_kv"], u_f=law["u_f"],
        p_overlap=P["time_law"]["H200"]["p"])
    # (b) calibrated on this run (in-sample): phys_p with u_w, p from zero-shot
    th = fit_time("phys_p", S, hw, x0=[law["a"], law["b_per_layer"] * L, law["tau_seq"],
                                         law["u_kv"], law["u_f"]],
                  u_w=law["u_w"], p=P["time_law"]["H200"]["p"])
    variants["calibrated t_iter (this run)"] = dict(
        zip(["t_ser", "t_cpu", "tau_seq", "u_kv", "u_f"], map(float, th)),
        u_w=law["u_w"], p_overlap=P["time_law"]["H200"]["p"])
    out = {"measured": dict(ttft_mean=float(ttft.mean()), ttft_p50=float(np.median(ttft)),
                            e2e_mean=float(e2e.mean()), tpot_mean=float(tpot.mean()),
                            little_W=float(little), n_run=float(n_mean),
                            thr_req_s=float(done_in), t_iter=float(t_meas),
                            power_w=float(R["res"].get("avg_power_window_w") or 0))}
    for name, tp in variants.items():
        pl = Plant(gpu, R["model"], dt=1.0, **hw, **tp, weight_bytes=R["consts"]["weight_bytes"],
                   kv_bytes_per_token=R["consts"]["kv_bytes_per_token"],
                   gemm_flops_per_token=R["consts"]["gemm_flops_per_token"],
                   n_layers=L, gen_len=wl["gen_len"], prompt_len=wl["prompt_len"],
                   cached_frac=wl["cached_frac"], n_stages=7,
                   think_s=wl["think_s"] or 0.015, kv_capacity=615328)
        x = pl.reset()
        ys = []
        for k in range(int(te - R["w0"])):
            x, y = pl.step(x, {}, dict(clients=c))
            if k >= 3:
                ys.append(y)
        mean = lambda key: float(np.mean([y[key] for y in ys]))
        res = dict(t_iter=mean("t_iter_s"), e2e_little=mean("e2e_latency_s"),
                   ttft=mean("ttft_s"), tpot=mean("tpot_s"), n_run=mean("n_running"),
                   thr_req_s=mean("completions"), power_w=mean("power_w"),
                   tokens_s=mean("tokens_out"))
        out[name] = res
        PRINT(f"plant [{name}]: t_iter {res['t_iter']*1e3:.2f} ms (meas {t_meas*1e3:.2f}); "
              f"e2e(Little) {res['e2e_little']:.3f} s (meas mean {e2e.mean():.3f}); "
              f"TTFT proxy {res['ttft']*1e3:.1f} ms (meas mean {ttft.mean()*1e3:.1f}); "
              f"TPOT {res['tpot']*1e3:.2f} ms (meas {tpot.mean()*1e3:.2f}); "
              f"req/s {res['thr_req_s']:.2f} (meas {done_in:.2f}); "
              f"power {res['power_w']:.0f} W (meas {out['measured']['power_w']:.0f})")
    js = load_json(RESULTS_JSON, {}) or {}
    js["smoke"] = {rd: out}
    with open(RESULTS_JSON, "w") as f:
        json.dump(js, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tokenize", "fit", "validate", "all", "smoke", "report"])
    ap.add_argument("--run_dir", default="logs/A5000_smoke/Qwen2.5-1.5B-Instruct/alpaca_c8")
    args = ap.parse_args()
    if args.cmd == "smoke":
        cmd_smoke(args)
        return
    if args.cmd == "report":          # re-print saved tables, no refitting
        P = load_json(PARAMS_JSON, {})
        report_time(P["time_model"], P["time_law"])
        rj = load_json(RESULTS_JSON, {})
        if rj.get("kstep"):
            report_kstep({c: {v: {int(k): x for k, x in dk.items()} for v, dk in dv.items()}
                          for c, dv in rj["kstep"].items()})
        return
    if args.cmd in ("tokenize",):
        tokenize_pools()
    if args.cmd in ("fit", "all"):
        cmd_fit(args)
    if args.cmd in ("validate", "all"):
        cmd_validate(args)




if __name__ == "__main__":
    main()
