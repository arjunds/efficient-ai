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


# ---------------- model byte accounting ----------------
def model_byte_constants(meta: dict) -> Tuple[float, float, str]:
    if meta.get("weight_bytes") and meta.get("kv_bytes_per_token"):
        return float(meta["weight_bytes"]), float(meta["kv_bytes_per_token"]), "run_meta"
    from gate_dram import import_models, hf_to_model_key
    models = import_models()
    key = meta.get("model_key") or hf_to_model_key(meta.get("model", ""), models)
    if key is None or key not in models:
        raise ValueError(f"model_key not resolvable/absent from models.py: "
                         f"{meta.get('model')} -> {key}")
    m = models[key]
    dtype_b = DTYPE_BYTES.get(meta.get("dtype", "float16"), 2)
    kv_b = meta.get("kv_cache_dtype")
    kv_b = DTYPE_BYTES.get(kv_b, dtype_b) if kv_b not in (None, "auto") else dtype_b
    return float(m.active_params() * dtype_b), float(m.kv_bytes_per_token(kv_b)), f"models.py:{key}"


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
    from gate_dram import import_models, hf_to_model_key
    models = import_models()
    key = meta.get("model_key") or hf_to_model_key(meta.get("model", ""), models)
    if key is None or key not in models:
        raise ValueError(f"model_key absent from models.py: {meta.get('model')}")
    m = models[key]
    matmul_per_tok = 2.0 * m.active_params()
    attn_per_resident = float(m.attn_flops_per_token(1))
    return matmul_per_tok, attn_per_resident, float(m.active_params())


def iter_rows(iter_csv: str) -> List[dict]:
    with open(iter_csv) as f:
        return list(csv.DictReader(f))


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def bin_run(run_dir: str, bin_s: float = DEFAULT_BIN_S, phase=None,
            include_empty: bool = False):
    """Bin a run into fixed-width time bins. phase=None includes ALL phases
    (needed for the two-term fit: prefill bins are the high-arithmetic-intensity
    points that pin e_flop). Each bin: {b0,b1,dt,E,bytes,flops,n,prefill,decode}."""
    meta = json.load(open(os.path.join(run_dir, "run_meta.json")))
    ts, ps = load_power_samples(os.path.join(run_dir, "power_trace.csv"))
    rows = iter_rows(os.path.join(run_dir, "iter_log.csv"))
    weight_bytes, kv_bpt, byte_src = model_byte_constants(meta)
    matmul_per_tok, attn_per_resident, active_params = model_flop_constants(meta)

    tms, byts, flps, prefs, decs = [], [], [], [], []
    for r in rows:
        if phase and (r.get("phase") or "").strip() != phase:
            continue
        t0 = _f(r.get("t_start")); t1 = _f(r.get("t_end"))
        if t0 is None or t1 is None:
            continue
        kv = _f(r.get("kv_tokens_resident")) or 0.0
        pref = _f(r.get("prefill_tokens")) or 0.0
        dec = _f(r.get("decode_tokens")) or 0.0
        tms.append(0.5 * (t0 + t1))
        byts.append(weight_bytes + kv * kv_bpt)
        flps.append(matmul_per_tok * (pref + dec) + attn_per_resident * kv)
        prefs.append(pref); decs.append(dec)

    info = {"weight_bytes": weight_bytes, "kv_bytes_per_token": kv_bpt,
            "matmul_flops_per_token": matmul_per_tok,
            "attn_flops_per_resident_token": attn_per_resident,
            "active_params": active_params, "byte_source": byte_src,
            "idle_power_w": meta.get("idle_power_w"), "meta": meta,
            "n_iters_total": len(tms), "bin_s": bin_s, "phase": phase}
    if not ts or not tms:
        return [], info

    order = sorted(range(len(tms)), key=lambda i: tms[i])
    tms = [tms[i] for i in order]; byts = [byts[i] for i in order]
    flps = [flps[i] for i in order]; prefs = [prefs[i] for i in order]
    decs = [decs[i] for i in order]

    w0 = meta.get("window_wall_t0") or tms[0]
    w1 = meta.get("window_wall_t1") or tms[-1]
    w0 = max(w0, ts[0]); w1 = min(w1, ts[-1])

    bins = []
    b0 = w0
    while b0 < w1:
        b1 = min(b0 + bin_s, w1)
        E = integrate_window(ts, ps, b0, b1)
        lo = bisect.bisect_left(tms, b0); hi = bisect.bisect_left(tms, b1)
        n = hi - lo
        if E is not None and (n > 0 or include_empty):
            bins.append({"b0": b0, "b1": b1, "dt": b1 - b0, "E": E, "n": n,
                         "bytes": sum(byts[lo:hi]), "flops": sum(flps[lo:hi]),
                         "prefill": sum(prefs[lo:hi]), "decode": sum(decs[lo:hi])})
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


def calibrate_two_term(run_dir: str, bin_s: float = DEFAULT_BIN_S) -> dict:
    """Fit E_bin = e_bit·bytes + e_flop·flops + P_static·dt over ALL-phase bins.
    P_static anchored from idle; (e_bit, e_flop) by 2-feature LS on the dynamic
    energy. Also exports binned_table.csv for independent re-fitting."""
    bins, info = bin_run(run_dir, bin_s, phase=None)
    result = {"run_dir": run_dir, "model": "two_term", "bin_s": bin_s,
              "n_bins": len(bins), "n_iters_total": info["n_iters_total"],
              "weight_bytes": info["weight_bytes"],
              "kv_bytes_per_token": info["kv_bytes_per_token"],
              "matmul_flops_per_token": info["matmul_flops_per_token"],
              "attn_flops_per_resident_token": info["attn_flops_per_resident_token"],
              "active_params": info["active_params"],
              "idle_power_w": info["idle_power_w"],
              "task": info["meta"].get("task"),
              "concurrency_or_rate": info["meta"].get("concurrency_or_rate")}

    # always export the binned table (analysis side re-fits independently)
    _export_binned_table(run_dir, bins)

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


def _export_binned_table(run_dir, bins):
    with open(os.path.join(run_dir, "binned_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_mid", "dt_s", "energy_bin_j", "bytes_bin", "flops_bin",
                    "prefill_tokens", "decode_tokens", "n_iters"])
        for b in bins:
            w.writerow([f"{0.5*(b['b0']+b['b1']):.6f}", f"{b['dt']:.4f}",
                        f"{b['E']:.4f}", f"{b['bytes']:.6e}", f"{b['flops']:.6e}",
                        int(b["prefill"]), int(b["decode"]), b["n"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--phase", default="decode", choices=["decode", "prefill", "mixed"])
    ap.add_argument("--bin_s", type=float, default=DEFAULT_BIN_S)
    ap.add_argument("--two_term", action="store_true",
                    help="fit E = e_bit·bytes + e_flop·flops + P_static·dt (all phases)")
    args = ap.parse_args()
    if args.two_term:
        cal = calibrate_two_term(args.run_dir, args.bin_s)
        out = os.path.join(args.run_dir, "calibration_two_term.json")
    else:
        cal = calibrate(args.run_dir, args.phase, args.bin_s)
        out = os.path.join(args.run_dir, "calibration.json")
    with open(out, "w") as f:
        json.dump(cal, f, indent=2)
    print(json.dumps(cal, indent=2))


if __name__ == "__main__":
    main()
