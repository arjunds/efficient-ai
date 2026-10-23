#!/usr/bin/env python3
"""
energy_model.py

Calibrate the LIMINAL energy extension from one measured run:

    E_iter = e_bit · (weight_bytes + Σ_seq KV_bytes(ctx_seq)) + P_static · t_iter

P_static is anchored by the idle baseline (run_meta.idle_power_w). e_bit is fit by
least-squares.

IMPORTANT — we fit over TIME BINS, not per scheduler-iteration. The V1 stat logger
timestamps each record() callback, but that cadence is async/jittery (~ms) and
finer than the power sampling, so per-iteration energy = ∫power over a ~6 ms
window is dominated by sampling noise, and instantaneous power = E_iter/t_iter
explodes when t_iter is tiny. Binning to ~200 ms averages the jitter: within a bin
the model becomes

    E_bin = e_bit · bytes_bin + P_static · Δ,   bytes_bin = Σ_{iters in bin} bytes_iter

which is exactly the roofline/energy relation (power ∝ HBM-bytes/s). This is the
methodologically sound version and gives stable coefficients.

We restrict to decode-phase iterations (the LIMINAL decode model); prefill iters
move similar bytes but do far more compute, so their energy needs separate
treatment.

Writes calibration.json. Helpers (bin_run, load_power_samples) are reused by
validate_waveform.py.

PREFILL SPREADING (2026-10-23, TIMEMODEL_HIGHC.md §5): when a run has requests.csv and all
phases are binned, each request's prompt tokens are moved from the step that reported them
back over its [t_submit, t_first_token] window before binning (spread_prefill_rows; run totals
unchanged). Runs without requests.csv bin exactly as before. Disable with --no_spread_prefill
or ENERGY_SPREAD_PREFILL=0.

Usage: python energy_model.py --run_dir logs/<run> [--bin_s 0.2] [--phase decode]
"""

import argparse
import bisect
import csv
import json
import os
from typing import List, Optional, Tuple

DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "fp8": 1, "float32": 4, "auto": 2}
DEFAULT_BIN_S = 0.2


# ---------------- trace loading / integration ----------------
def load_power_samples(power_csv: str) -> Tuple[List[float], List[float]]:
    ts, ps = [], []
    with open(power_csv) as f:
        for row in csv.DictReader(f):
            try:
                t = float(row["t_wall"]); p = float(row["power_w"])
            except (ValueError, KeyError, TypeError):
                continue
            ts.append(t); ps.append(p)
    order = sorted(range(len(ts)), key=lambda i: ts[i])
    return [ts[i] for i in order], [ps[i] for i in order]


POWER_SOURCES = ("energy", "inst", "avg")


def window_energy_fn(power_csv: str, source: str = "auto"):
    """Return (energy(t0, t1) -> J or None, source_used).

    Sources, best first (auto picks the first that is present on >=80% of rows):
      energy  cumulative NVML energy counter (energy_mj): E(t1) - E(t0) from the
              counter's change points, linearly interpolated. True energy, no
              smoothing; resolution ~ the counter update period (~100 ms on B200).
      inst    POWER_INSTANT (power_inst_w), trapezoid-integrated (0.10 s window).
      avg     GetPowerUsage (power_w), trapezoid-integrated. A 0.85 s moving
              average on B200 / 1.0 s on A5000: smears 200 ms bins. Every run
              logged before 2026-10-16 only has this, so old runs bin unchanged.
    Override with env ENERGY_POWER_SOURCE or --power_source."""
    source = os.environ.get("ENERGY_POWER_SOURCE", source) or "auto"
    rows = []
    with open(power_csv) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    n = len(rows)

    def frac(col):
        return sum(1 for r in rows if (r.get(col) or "").strip() not in ("",)) / n if n else 0.0

    order = list(POWER_SOURCES) if source == "auto" else [source]
    for src in order:
        if src == "energy" and frac("energy_mj") >= 0.8:
            pts = []
            for r in rows:
                try:
                    t = float(r["t_wall"]); e = float(r["energy_mj"])
                except (ValueError, KeyError, TypeError):
                    continue
                pts.append((t, e))
            pts.sort()
            chg = [pts[0]] if pts else []
            for t, e in pts[1:]:
                if e != chg[-1][1]:
                    chg.append((t, e))
            if len(chg) >= 5:
                ct = [c[0] for c in chg]; ce = [c[1] for c in chg]
                # Validate against POWER_INSTANT over the whole trace: the counter
                # is exact on B200 (counter/instant 0.9998, steady microbench) but
                # reads ~0.30x on the A5000 (GA102). Reject it if off by >5%.
                if frac("power_inst_w") >= 0.8:
                    ti, pi = [], []
                    for r in rows:
                        try:
                            ti.append(float(r["t_wall"])); pi.append(float(r["power_inst_w"]))
                        except (ValueError, KeyError, TypeError):
                            pass
                    o = sorted(range(len(ti)), key=lambda i: ti[i])
                    ti = [ti[i] for i in o]; pi = [pi[i] for i in o]
                    e_inst = integrate_window(ti, pi, ct[0], ct[-1])
                    ratio = ((ce[-1] - ce[0]) * 1e-3 / e_inst) if e_inst else None
                    if ratio is None or abs(ratio - 1.0) > 0.05:
                        print(f"[energy_model] energy counter rejected for {power_csv}: "
                              f"counter/POWER_INSTANT = {ratio}; using POWER_INSTANT", flush=True)
                        continue

                def energy(t0, t1, ct=ct, ce=ce, tl=pts[-1][0]):
                    # Up to 0.5 s past the first/last counter update (and within
                    # 0.1 s of the last sample: the logger stops just after the window) extrapolate with the edge segment's power; a
                    # window ending between updates would otherwise return None.
                    if t1 <= t0 or t0 < ct[0] - 0.5 or t1 > min(ct[-1] + 0.5, tl + 0.1):
                        return None

                    def E(t):
                        if t > ct[-1]:
                            return ce[-1] + (ce[-1] - ce[-2]) / (ct[-1] - ct[-2]) * (t - ct[-1])
                        if t < ct[0]:
                            return ce[0] - (ce[1] - ce[0]) / (ct[1] - ct[0]) * (ct[0] - t)
                        return _interp(ct, ce, t)
                    return (E(t1) - E(t0)) * 1e-3
                return energy, "energy"
        if src in ("inst", "avg"):
            col = "power_inst_w" if src == "inst" else "power_w"
            if src == "inst" and frac(col) < 0.8:
                continue
            ts, ps = [], []
            for r in rows:
                try:
                    t = float(r["t_wall"]); p = float(r[col])
                except (ValueError, KeyError, TypeError):
                    continue
                ts.append(t); ps.append(p)
            o = sorted(range(len(ts)), key=lambda i: ts[i])
            ts = [ts[i] for i in o]; ps = [ps[i] for i in o]
            if len(ts) >= 2:
                return (lambda t0, t1, ts=ts, ps=ps: integrate_window(ts, ps, t0, t1)), src
    return (lambda t0, t1: None), None


def _interp(ts, ps, t):
    i = bisect.bisect_left(ts, t)
    if i <= 0:
        return ps[0]
    if i >= len(ts):
        return ps[-1]
    t0, t1 = ts[i - 1], ts[i]
    if t1 == t0:
        return ps[i]
    return ps[i - 1] + (ps[i] - ps[i - 1]) * (t - t0) / (t1 - t0)


def integrate_window(ts, ps, t0, t1) -> Optional[float]:
    """Trapezoidal ∫ power dt over [t0, t1] with edge interpolation."""
    if len(ts) < 2 or t1 <= t0 or t1 < ts[0] or t0 > ts[-1]:
        return None
    lo = bisect.bisect_left(ts, t0)
    hi = bisect.bisect_right(ts, t1)
    pts = [(t0, _interp(ts, ps, t0))]
    for i in range(lo, hi):
        if t0 < ts[i] < t1:
            pts.append((ts[i], ps[i]))
    pts.append((t1, _interp(ts, ps, t1)))
    if len(pts) < 2:
        return None
    e = 0.0
    for i in range(len(pts) - 1):
        (ta, pa), (tb, pb) = pts[i], pts[i + 1]
        e += 0.5 * (pa + pb) * (tb - ta)
    return e


# ---------------- model resolution ----------------
def resolve_model_obj(meta):
    """Return (Model, source_key). Try the hand-specified MODELS table first;
    fall back to auto-ingesting a dense model from its HF config.json. This lets
    the size-ladder and any new dense model be analyzed with zero hand entry."""
    from gate_dram import import_models, hf_to_model_key
    models = import_models()
    key = meta.get("model_key") or hf_to_model_key(meta.get("model", ""), models)
    if key is not None and key in models:
        return models[key], f"models.py:{key}"
    hf_id = meta.get("model")
    if hf_id:
        from models import model_from_hf_id
        return model_from_hf_id(hf_id), f"hf_config:{hf_id}"
    raise ValueError(f"model not resolvable: {hf_id} -> {key}")


# ---------------- model byte accounting ----------------
def model_byte_constants(meta: dict) -> Tuple[float, float, str]:
    if meta.get("weight_bytes") and meta.get("kv_bytes_per_token"):
        return float(meta["weight_bytes"]), float(meta["kv_bytes_per_token"]), "run_meta"
    m, src = resolve_model_obj(meta)
    dtype_b = DTYPE_BYTES.get(meta.get("dtype", "float16"), 2)
    kv_b = meta.get("kv_cache_dtype")
    kv_b = DTYPE_BYTES.get(kv_b, dtype_b) if kv_b not in (None, "auto") else dtype_b
    return float(m.active_params() * dtype_b), float(m.kv_bytes_per_token(kv_b)), src


def model_flop_constants(meta: dict):
    """Return (matmul_flops_per_token, attn_flops_per_resident_token, active_params).

    Per-iteration FLOPs (from the fields iter_log records) are modeled as:
      flops_iter = matmul_per_tok · (prefill_tokens + decode_tokens)
                   + attn_per_resident · kv_tokens_resident
    where matmul_per_tok = 2·active_params (MAC=2 flops; includes lm_head) and
    attn_per_resident = attn_flops_per_token(1) (the per-context-token attention
    cost; Σ_seq attn_flops_per_token(ctx_seq) = attn_per_resident·Σctx, exact for
    non-chunked GQA/MLA since attn_flops_per_token is linear in T). Prefill self-
    attention (quadratic) is under-counted, but the matmul term dominates prefill
    and drives the arithmetic-intensity spread; the analysis side can refine.
    """
    m, _src = resolve_model_obj(meta)
    matmul_per_tok = 2.0 * m.active_params()
    attn_per_resident = float(m.attn_flops_per_token(1))
    return matmul_per_tok, attn_per_resident, float(m.active_params())


def get_model_obj(meta):
    return resolve_model_obj(meta)


def weight_params_for_tokens(m, t):
    """Params whose weights are read from HBM for an iteration processing t tokens.
    Dense: active_params (constant). MoE: active_params (top-k) PLUS the extra
    routed experts that get touched once t tokens route across them — expected
    distinct experts = E·(1-(1-k/E)^t) under ~uniform routing. Reduces to
    active_params at t=1 and rises toward total_params at large t (why an MoE's
    per-batch weight traffic is NOT active_params·tokens)."""
    base = m.active_params()
    if getattr(m, "ffn", "mlp") != "moe" or t <= 1:
        return base
    E, k = m.n_routed_experts, m.n_active_experts
    if not E or not k:
        return base
    frac_hit = 1.0 - (1.0 - k / E) ** t          # P(expert touched by >=1 of t tokens)
    extra_experts = max(0.0, frac_hit * E - k)   # beyond the top-k already in base
    per_expert = m._mlp_params(m.d_ff_expert)
    return base + extra_experts * per_expert * m._n_moe_layers()


def iter_rows(iter_csv: str) -> List[dict]:
    with open(iter_csv) as f:
        return list(csv.DictReader(f))


# ---------------- prefill spreading (hidden chunked-prefill steps) ----------------
# vLLM reports a request's prompt tokens (IterationStats.num_prompt_tokens) -- and, in our logs,
# its prefix-cache queries/hits -- only on the step where its prefill FINISHES.  With chunked
# prefill a long prompt's earlier chunk steps report prefill_tokens = 0, so their GEMM FLOPs land
# in the bin of the last chunk while their energy is spent earlier (FINDINGS_B200 UPDATE 6
# review).  spread_prefill_rows() moves each request's tokens back over the steps of its
# [t_submit, t_first_token] window BEFORE binning; run totals are unchanged.
SPREAD_MODES = ("budget", "duration")


def _requests_windows(run_dir):
    """[(t_submit, t_first_token, prompt_tokens)] from requests.csv, or None if the file or the
    columns are missing (-> spreading stays off: old runs bin exactly as before)."""
    p = os.path.join(run_dir, "requests.csv")
    if not os.path.exists(p):
        return None
    out = []
    with open(p) as f:
        rd = csv.DictReader(f)
        if not rd.fieldnames or not {"t_submit", "t_first_token", "prompt_tokens"} <= set(rd.fieldnames):
            return None
        for r in rd:
            s, ft, pt = _f(r.get("t_submit")), _f(r.get("t_first_token")), _f(r.get("prompt_tokens"))
            if s is not None and ft is not None and pt:
                out.append((s, ft, pt))
    return out


def spread_prefill_rows(t_start, t_end, pref, hits, dec, reqs, budget=2048, mode="budget"):
    """Redistribute each request's prompt tokens over the logged steps of its prefill window.

    Inputs are per-row lists in TIME ORDER (t_end ascending).  A request is attributed to the
    reporting row = last row with t_end <= t_first_token (this reproduces the logged
    prefill_tokens on >99% of rows of the B200/A100/A6000 provisioning runs), only if that row
    still has that many unattributed prompt tokens.  Its computed tokens C = P * (1 - hits/pref of
    the reporting row) are placed over rows (t_end > t_submit) .. reporting row:
      mode 'budget'  : backward fill from the reporting row, each row taking at most its free
                       chunked-prefill budget (max_num_batched_tokens - decode tokens - what is
                       already placed there); requests are placed in finish order (vLLM schedules
                       prefills FCFS, so an earlier request owns the steps just before its finish).
                       Tokens that do not fit stay on the reporting row.
      mode 'duration': proportional to step duration over the window.
    Raw prompt tokens and hits move with the same per-row shares, so per-row computed tokens
    (pref - hits) stay >= 0 and the run totals of prompt tokens, hits and computed tokens are
    unchanged.  Returns (pref', hits', info)."""
    n = len(t_end)
    pref2, hits2 = list(pref), list(hits)
    avail = list(pref)
    jobs = []
    for s, ft, P in reqs:
        r = bisect.bisect_right(t_end, ft) - 1
        if r < 0 or avail[r] + 1e-9 < P:
            continue
        avail[r] -= P
        lo = bisect.bisect_right(t_end, s)
        C = P * (1.0 - hits[r] / pref[r]) if pref[r] > 0 else P
        jobs.append((r, C, P, max(0, min(lo, r)), s))
    info = {"spread_mode": mode, "requests_total": len(reqs), "requests_attributed": len(jobs),
            "tokens_attributed": sum(j[2] for j in jobs), "tokens_moved_off_report_row": 0.0,
            "computed_overflow": 0.0, "budget": budget}
    free = [max(0.0, budget - d) for d in dec]
    jobs.sort(key=lambda j: (j[0], j[1], j[4]))         # finish order; short ones claim the row first
    for r, C, P, lo, _s in jobs:
        alloc = {}
        if C <= 0 or lo >= r:
            alloc[r] = C
            if lo >= r:
                free[r] -= C
        elif mode == "duration":
            w = [max(0.0, t_end[k] - t_start[k]) for k in range(lo, r + 1)]
            tot = sum(w)
            for k, wk in zip(range(lo, r + 1), w):
                alloc[k] = C * (wk / tot) if tot > 0 else (C if k == r else 0.0)
        else:
            rem = C
            k = r
            while rem > 1e-9 and k >= lo:
                a = min(rem, free[k])
                if a > 0:
                    alloc[k] = alloc.get(k, 0.0) + a
                    free[k] -= a
                    rem -= a
                k -= 1
            if rem > 1e-9:
                alloc[r] = alloc.get(r, 0.0) + rem
                info["computed_overflow"] += rem
        if C <= 0:
            continue
        h = P - C
        pref2[r] -= P
        hits2[r] -= h
        for k, a in alloc.items():
            f = a / C
            pref2[k] += P * f
            hits2[k] += h * f
            if k != r:
                info["tokens_moved_off_report_row"] += P * f
    for k in range(n):                                   # float dust from the subtraction
        if abs(pref2[k]) < 1e-6:
            pref2[k] = 0.0
        if abs(hits2[k]) < 1e-6:
            hits2[k] = 0.0
    return pref2, hits2, info


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def spread_enabled(run_dir, phase=None, spread_prefill=None):
    """Prefill spreading is ON by default when the run has requests.csv with t_submit /
    t_first_token / prompt_tokens and all phases are binned (phase=None); OFF for runs without
    it (old runs bin exactly as before), for phase-filtered binning (the logged phase tags
    would no longer match), with spread_prefill=False, or with env ENERGY_SPREAD_PREFILL=0."""
    if spread_prefill is False or phase:
        return None
    if os.environ.get("ENERGY_SPREAD_PREFILL", "").strip() in ("0", "false", "off", "no"):
        return None
    return _requests_windows(run_dir)


def bin_run(run_dir: str, bin_s: float = DEFAULT_BIN_S, phase=None,
            include_empty: bool = False, power_source: str = "auto",
            spread_prefill=None, spread_mode: str = "budget"):
    """Bin a run into fixed-width time bins. phase=None includes ALL phases
    (needed for the two-term fit: prefill bins are the high-arithmetic-intensity
    points that pin e_flop). Each bin: {b0,b1,dt,E,bytes,flops,n,prefill,decode}.
    spread_prefill: None = auto (on iff requests.csv exists and phase is None; see
    spread_enabled), False = off, True = on when possible.  spread_mode: 'budget' | 'duration'
    (spread_prefill_rows)."""
    meta = json.load(open(os.path.join(run_dir, "run_meta.json")))
    ts, ps = load_power_samples(os.path.join(run_dir, "power_trace.csv"))
    efn, psrc = window_energy_fn(os.path.join(run_dir, "power_trace.csv"),
                                 power_source)
    rows = iter_rows(os.path.join(run_dir, "iter_log.csv"))
    have_hits = bool(rows) and "prefix_hits" in rows[0]
    weight_bytes, kv_bpt, byte_src = model_byte_constants(meta)  # t=1 reference
    matmul_per_tok, attn_per_resident, active_params = model_flop_constants(meta)
    m_obj, _key = get_model_obj(meta)
    dtype_b = DTYPE_BYTES.get(meta.get("dtype", "float16"), 2)

    # optional: move each request's prompt tokens (and hits) over its prefill window
    spread = {}
    spread_info = None
    reqs = spread_enabled(run_dir, phase, spread_prefill)
    if reqs:
        if spread_mode not in SPREAD_MODES:
            raise ValueError("spread_mode must be one of %s" % (SPREAD_MODES,))
        idx = []
        for i, r in enumerate(rows):
            a, b = _f(r.get("t_start")), _f(r.get("t_end"))
            if a is not None and b is not None:
                idx.append((b, i, a))
        idx.sort()
        p0 = [_f(rows[i].get("prefill_tokens")) or 0.0 for _, i, _ in idx]
        h0 = [((_f(rows[i].get("prefix_hits")) or 0.0) if have_hits else 0.0) for _, i, _ in idx]
        d0 = [_f(rows[i].get("decode_tokens")) or 0.0 for _, i, _ in idx]
        budget = meta.get("max_num_batched_tokens_resolved") or meta.get("max_num_batched_tokens") or 2048
        p1, h1, spread_info = spread_prefill_rows([a for _, _, a in idx], [b for b, _, _ in idx],
                                                  p0, h0, d0, reqs, float(budget), spread_mode)
        spread = {i: (p1[k], h1[k]) for k, (_, i, _) in enumerate(idx)}

    tms, byts, flps, flps1, prefs, decs, hits = [], [], [], [], [], [], []
    wbyt_l, kvbyt_l, gemm_l, attn_l = [], [], [], []   # physical channels
    for ri, r in enumerate(rows):
        if phase and (r.get("phase") or "").strip() != phase:
            continue
        t0 = _f(r.get("t_start")); t1 = _f(r.get("t_end"))
        if t0 is None or t1 is None:
            continue
        kv = _f(r.get("kv_tokens_resident")) or 0.0
        pref = _f(r.get("prefill_tokens")) or 0.0
        if spread:
            pref = spread[ri][0]
        dec = _f(r.get("decode_tokens")) or 0.0
        nr = _f(r.get("n_running")) or 1.0
        tms.append(0.5 * (t0 + t1))
        # per-iteration weight bytes: MoE occupancy-aware (dense = active_params).
        wbytes = weight_params_for_tokens(m_obj, pref + dec) * dtype_b
        kvbytes = kv * kv_bpt
        byts.append(wbytes + kvbytes)
        # v1 (original): attention ~ resident tokens (correct for decode, but
        # under-counts prefill self-attention). v2 (#2 improvement): each token
        # processed this iter attends to the mean per-seq resident context
        # (pref+dec)*kv/n_running -> reduces to kv for pure decode, captures the
        # ~quadratic prefill attention. Aggregate approx (no per-seq ctx available).
        matmul = matmul_per_tok * (pref + dec)
        attn = attn_per_resident * kv
        flps1.append(matmul + attn)
        flps.append(matmul + attn_per_resident * (pref + dec) * kv / max(nr, 1.0))
        prefs.append(pref); decs.append(dec)
        if spread:
            hits.append(spread[ri][1])
        else:
            hits.append((_f(r.get("prefix_hits")) or 0.0) if have_hits else 0.0)
        wbyt_l.append(wbytes); kvbyt_l.append(kvbytes)
        gemm_l.append(matmul); attn_l.append(attn)

    info = {"weight_bytes": weight_bytes, "kv_bytes_per_token": kv_bpt,
            "matmul_flops_per_token": matmul_per_tok,
            "attn_flops_per_resident_token": attn_per_resident,
            "active_params": active_params, "byte_source": byte_src,
            "idle_power_w": meta.get("idle_power_w"), "meta": meta,
            "n_iters_total": len(tms), "bin_s": bin_s, "phase": phase,
            "power_source": psrc, "prefix_hits_logged": have_hits}
    if spread_info is not None:
        info["prefill_spread"] = spread_info
    if not ts or not tms:
        return [], info

    order = sorted(range(len(tms)), key=lambda i: tms[i])
    tms = [tms[i] for i in order]; byts = [byts[i] for i in order]
    flps = [flps[i] for i in order]; flps1 = [flps1[i] for i in order]
    prefs = [prefs[i] for i in order]; decs = [decs[i] for i in order]
    hits = [hits[i] for i in order]
    wbyt_l = [wbyt_l[i] for i in order]; kvbyt_l = [kvbyt_l[i] for i in order]
    gemm_l = [gemm_l[i] for i in order]; attn_l = [attn_l[i] for i in order]

    w0 = meta.get("window_wall_t0") or tms[0]
    w1 = meta.get("window_wall_t1") or tms[-1]
    w0 = max(w0, ts[0]); w1 = min(w1, ts[-1])

    bins = []
    b0 = w0
    while b0 < w1:
        b1 = min(b0 + bin_s, w1)
        E = efn(b0, b1)
        lo = bisect.bisect_left(tms, b0); hi = bisect.bisect_left(tms, b1)
        n = hi - lo
        if E is not None and (n > 0 or include_empty):
            # prefix-cache hits are never computed: per-bin computed prefill
            pref_b = sum(prefs[lo:hi]); pref_c = max(0.0, pref_b - sum(hits[lo:hi]))
            bins.append({"b0": b0, "b1": b1, "dt": b1 - b0, "E": E, "n": n,
                         "bytes": sum(byts[lo:hi]), "flops": sum(flps[lo:hi]),
                         "flops_v1": sum(flps1[lo:hi]),
                         "weight_bytes": sum(wbyt_l[lo:hi]), "kv_bytes": sum(kvbyt_l[lo:hi]),
                         "gemm_flops": sum(gemm_l[lo:hi]), "attn_flops": sum(attn_l[lo:hi]),
                         "prefill": pref_b, "decode": sum(decs[lo:hi]),
                         "prefill_computed": pref_c if have_hits else None,
                         "gemm_flops_computed": (matmul_per_tok * (pref_c + sum(decs[lo:hi]))
                                                 if have_hits else None)})
        b0 = b1
    return bins, info


# ---------------- fitting ----------------
def _ls_through_origin(xs, ys):
    sxx = sum(x * x for x in xs)
    return (sum(x * y for x, y in zip(xs, ys)) / sxx) if sxx else None


def _ls_two_feature(x1, x2, y):
    s11 = sum(a * a for a in x1); s22 = sum(a * a for a in x2)
    s12 = sum(a * b for a, b in zip(x1, x2))
    s1y = sum(a * c for a, c in zip(x1, y)); s2y = sum(a * c for a, c in zip(x2, y))
    det = s11 * s22 - s12 * s12
    if abs(det) < 1e-30:
        return None, None
    return (s1y * s22 - s2y * s12) / det, (s11 * s2y - s12 * s1y) / det


def _r2(y, yhat):
    n = len(y)
    if n == 0:
        return None
    ybar = sum(y) / n
    ss_tot = sum((v - ybar) ** 2 for v in y)
    ss_res = sum((v - h) ** 2 for v, h in zip(y, yhat))
    return 1 - ss_res / ss_tot if ss_tot else None


def calibrate(run_dir: str, phase: str = "decode", bin_s: float = DEFAULT_BIN_S) -> dict:
    bins, info = bin_run(run_dir, bin_s, phase)
    result = {"run_dir": run_dir, "phase": phase, "bin_s": bin_s,
              "n_bins": len(bins), "n_iters_total": info["n_iters_total"],
              "weight_bytes": info["weight_bytes"],
              "kv_bytes_per_token": info["kv_bytes_per_token"],
              "byte_source": info["byte_source"], "idle_power_w": info["idle_power_w"]}
    if len(bins) < 3:
        result["error"] = f"too few {phase} bins ({len(bins)}); try smaller --bin_s"
        return result

    xs = [b["bytes"] for b in bins]
    dts = [b["dt"] for b in bins]
    Es = [b["E"] for b in bins]

    e_bit_joint, p_static_joint = _ls_two_feature(xs, dts, Es)

    p_static = info["idle_power_w"]
    if p_static is None:
        p_static = p_static_joint
        result["p_static_source"] = "joint_fit (no idle baseline!)"
    else:
        result["p_static_source"] = "idle_baseline"

    dyn = [E - p_static * dt for E, dt in zip(Es, dts)]
    e_bit = _ls_through_origin(xs, dyn)
    e_bit_agg = sum(dyn) / sum(xs) if sum(xs) else None
    yhat = [e_bit * x + p_static * dt for x, dt in zip(xs, dts)]

    # per-bin power view (more interpretable than energy for flat signals)
    meas_w = [E / dt for E, dt in zip(Es, dts)]
    pred_w = [y / dt for y, dt in zip(yhat, dts)]

    result.update({
        "e_bit_j_per_byte": e_bit,
        "e_bit_aggregate": e_bit_agg,
        "p_static_w": p_static,
        "e_bit_joint": e_bit_joint,
        "p_static_joint": p_static_joint,
        "r2_energy": _r2(Es, yhat),
        "r2_power": _r2(meas_w, pred_w),
        "mean_power_w": sum(meas_w) / len(meas_w),
        "mean_bytes_per_bin": sum(xs) / len(xs),
        "mean_dynamic_power_w": sum(dyn) / sum(dts) if sum(dts) else None,
    })
    return result


def _pearson(x, y):
    n = len(x)
    if n < 2:
        return None
    mx = sum(x) / n; my = sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x); syy = sum((b - my) ** 2 for b in y)
    d = (sxx * syy) ** 0.5
    return (sxy / d) if d else None


def calibrate_two_term(run_dir: str, bin_s: float = DEFAULT_BIN_S, spread_prefill=None,
                       spread_mode: str = "budget", binned_name: str = "binned_table.csv") -> dict:
    """Fit E_bin = e_bit·bytes + e_flop·flops + P_static·dt over ALL-phase bins.
    P_static anchored from idle; (e_bit, e_flop) by 2-feature LS on the dynamic
    energy. Also exports binned_table.csv (or binned_name) for independent re-fitting.
    Prefill spreading: see bin_run / spread_enabled (auto-on when requests.csv exists)."""
    bins, info = bin_run(run_dir, bin_s, phase=None, spread_prefill=spread_prefill,
                         spread_mode=spread_mode)
    result = {"run_dir": run_dir, "model": "two_term", "bin_s": bin_s,
              "n_bins": len(bins), "n_iters_total": info["n_iters_total"],
              "weight_bytes": info["weight_bytes"],
              "kv_bytes_per_token": info["kv_bytes_per_token"],
              "matmul_flops_per_token": info["matmul_flops_per_token"],
              "attn_flops_per_resident_token": info["attn_flops_per_resident_token"],
              "active_params": info["active_params"],
              "idle_power_w": info["idle_power_w"],
              "task": info["meta"].get("task"),
              "concurrency_or_rate": info["meta"].get("concurrency_or_rate"),
              "power_source": info.get("power_source")}
    if info.get("prefill_spread") is not None:
        result["prefill_spread"] = info["prefill_spread"]

    # always export the binned table (analysis side re-fits independently)
    _export_binned_table(run_dir, bins, info.get("power_source"),
                         spread=(info.get("prefill_spread") or {}).get("spread_mode"), name=binned_name)

    if len(bins) < 4:
        result["error"] = f"too few bins ({len(bins)})"
        return result

    bytes_ = [b["bytes"] for b in bins]
    flops = [b["flops"] for b in bins]
    dts = [b["dt"] for b in bins]
    Es = [b["E"] for b in bins]

    p_static = info["idle_power_w"]
    result["p_static_source"] = "idle_baseline" if p_static is not None else "none"
    if p_static is None:
        result["error"] = "no idle baseline; cannot anchor P_static"
        return result

    dyn = [E - p_static * dt for E, dt in zip(Es, dts)]
    e_bit, e_flop = _ls_two_feature(bytes_, flops, dyn)

    # identifiability: bytes and flops must not be collinear across bins
    r_bf = _pearson(bytes_, flops)
    ai = [f / by for f, by in zip(flops, bytes_) if by]   # arithmetic intensity
    yhat = [(e_bit or 0) * by + (e_flop or 0) * fl + p_static * dt
            for by, fl, dt in zip(bytes_, flops, dts)]
    meas_w = [E / dt for E, dt in zip(Es, dts)]
    pred_w = [y / dt for y, dt in zip(yhat, dts)]

    result.update({
        "e_bit_j_per_byte": e_bit,
        "e_flop_j_per_flop": e_flop,
        "p_static_w": p_static,
        "r2_energy": _r2(Es, yhat),
        "r2_power": _r2(meas_w, pred_w),
        "bytes_flops_pearson": r_bf,
        "identifiable": (r_bf is not None and abs(r_bf) < 0.97),
        "arithmetic_intensity_min": min(ai) if ai else None,
        "arithmetic_intensity_max": max(ai) if ai else None,
        "binned_table": os.path.join(run_dir, "binned_table.csv"),
    })
    if e_flop is not None:
        result["e_flop_pJ_per_flop"] = e_flop * 1e12
    if not result["identifiable"]:
        result["warning"] = (f"bytes~flops collinear (r={r_bf:.3f}); e_bit/e_flop "
                             "not separable — need more arithmetic-intensity spread "
                             "(heavier prefill / longer prompts).")
    return result


def _export_binned_table(run_dir, bins, power_source=None, spread=None, name="binned_table.csv"):
    """Columns are additive: prefill_tokens_computed / gemm_flops_bin_computed only
    when prefix hits were logged (same names controls/prefix_cache_refit.py adds to
    old runs), and power_source records which NVML reading produced energy_bin_j.
    With prefill spreading (spread = mode) prefill_tokens is written with one decimal (spread
    tokens are fractional) and a trailing prefill_spread column names the mode."""
    have_c = bool(bins) and bins[0].get("prefill_computed") is not None
    with open(os.path.join(run_dir, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_mid", "dt_s", "energy_bin_j", "bytes_bin", "flops_bin",
                    "flops_bin_v1", "weight_bytes_bin", "kv_bytes_bin",
                    "gemm_flops_bin", "attn_flops_bin",
                    "prefill_tokens", "decode_tokens", "n_iters"]
                   + (["prefill_tokens_computed", "gemm_flops_bin_computed"] if have_c else [])
                   + (["power_source"] if power_source else [])
                   + (["prefill_spread"] if spread else []))
        for b in bins:
            w.writerow([f"{0.5*(b['b0']+b['b1']):.6f}", f"{b['dt']:.4f}",
                        f"{b['E']:.4f}", f"{b['bytes']:.6e}", f"{b['flops']:.6e}",
                        f"{b.get('flops_v1', b['flops']):.6e}",
                        f"{b.get('weight_bytes', 0):.6e}", f"{b.get('kv_bytes', 0):.6e}",
                        f"{b.get('gemm_flops', 0):.6e}", f"{b.get('attn_flops', 0):.6e}",
                        (f"{b['prefill']:.1f}" if spread else int(b["prefill"])), int(b["decode"]), b["n"]]
                       + ([f"{b['prefill_computed']:.1f}", f"{b['gemm_flops_computed']:.6e}"]
                          if have_c else [])
                       + ([power_source] if power_source else [])
                       + ([spread] if spread else []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--phase", default="decode", choices=["decode", "prefill", "mixed"])
    ap.add_argument("--bin_s", type=float, default=DEFAULT_BIN_S)
    ap.add_argument("--power_source", default="auto", choices=("auto",) + POWER_SOURCES,
                    help="NVML reading used for bin energy (auto: energy counter > "
                         "POWER_INSTANT > GetPowerUsage, whichever was logged)")
    ap.add_argument("--two_term", action="store_true",
                    help="fit E = e_bit·bytes + e_flop·flops + P_static·dt (all phases)")
    ap.add_argument("--no_spread_prefill", action="store_true",
                    help="do NOT redistribute prompt tokens over their prefill windows (default: on "
                         "when the run has requests.csv; env ENERGY_SPREAD_PREFILL=0 also disables)")
    ap.add_argument("--spread_mode", default="budget", choices=SPREAD_MODES)
    ap.add_argument("--binned_out", default="binned_table.csv",
                    help="file name (in run_dir) for the exported binned table")
    args = ap.parse_args()
    os.environ.setdefault("ENERGY_POWER_SOURCE", args.power_source)
    if args.two_term:
        cal = calibrate_two_term(args.run_dir, args.bin_s,
                                 spread_prefill=False if args.no_spread_prefill else None,
                                 spread_mode=args.spread_mode, binned_name=args.binned_out)
        out = os.path.join(args.run_dir, "calibration_two_term.json")
    else:
        cal = calibrate(args.run_dir, args.phase, args.bin_s)
        out = os.path.join(args.run_dir, "calibration.json")
    with open(out, "w") as f:
        json.dump(cal, f, indent=2)
    print(json.dumps(cal, indent=2))


if __name__ == "__main__":
    main()
