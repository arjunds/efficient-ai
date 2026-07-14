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


def iter_rows(iter_csv: str) -> List[dict]:
    with open(iter_csv) as f:
        return list(csv.DictReader(f))


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def bin_run(run_dir: str, bin_s: float = DEFAULT_BIN_S, phase: str = "decode",
            include_empty: bool = False):
    """Bin a run into fixed-width time bins. Returns (bins, info) where each bin is
    {b0,b1,dt,E,bytes,n} with E=measured energy (J), bytes=Σ iter bytes whose
    midpoint lands in the bin, n=#iters."""
    meta = json.load(open(os.path.join(run_dir, "run_meta.json")))
    ts, ps = load_power_samples(os.path.join(run_dir, "power_trace.csv"))
    rows = iter_rows(os.path.join(run_dir, "iter_log.csv"))
    weight_bytes, kv_bpt, byte_src = model_byte_constants(meta)

    tms, byts = [], []
    for r in rows:
        if phase and (r.get("phase") or "").strip() != phase:
            continue
        t0 = _f(r.get("t_start")); t1 = _f(r.get("t_end"))
        if t0 is None or t1 is None:
            continue
        kv = _f(r.get("kv_tokens_resident")) or 0.0
        tms.append(0.5 * (t0 + t1))
        byts.append(weight_bytes + kv * kv_bpt)

    info = {"weight_bytes": weight_bytes, "kv_bytes_per_token": kv_bpt,
            "byte_source": byte_src, "idle_power_w": meta.get("idle_power_w"),
            "meta": meta, "n_iters_total": len(tms), "bin_s": bin_s, "phase": phase}
    if not ts or not tms:
        return [], info

    order = sorted(range(len(tms)), key=lambda i: tms[i])
    tms = [tms[i] for i in order]; byts = [byts[i] for i in order]

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
            bins.append({"b0": b0, "b1": b1, "dt": b1 - b0,
                         "E": E, "bytes": sum(byts[lo:hi]), "n": n})
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--phase", default="decode", choices=["decode", "prefill", "mixed"])
    ap.add_argument("--bin_s", type=float, default=DEFAULT_BIN_S)
    args = ap.parse_args()
    cal = calibrate(args.run_dir, args.phase, args.bin_s)
    with open(os.path.join(args.run_dir, "calibration.json"), "w") as f:
        json.dump(cal, f, indent=2)
    print(json.dumps(cal, indent=2))


if __name__ == "__main__":
    main()
