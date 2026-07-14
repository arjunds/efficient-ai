#!/usr/bin/env python3
"""
energy_model.py

Calibrate the LIMINAL energy extension from one measured run:

    E_iter = e_bit · (weight_bytes + Σ_seq KV_bytes(ctx_seq)) + P_static · t_iter

- P_static is anchored by the idle baseline (run_meta.idle_power_w), which fixes
  the static/dynamic split independently of the fit (handoff requirement).
- e_bit is then fit by least-squares through the origin against the per-iteration
  *dynamic* energy (measured iteration energy minus P_static·t_iter).
- We also report a joint 2-parameter fit (e_bit, P_static) as a cross-check.

bytes/iter uses ~/models.py: weight_bytes = active_params·dtype_bytes,
Σ KV_bytes = kv_tokens_resident · kv_bytes_per_token. (For non-chunked GQA/MLA,
Σ_seq KV_bytes(ctx_seq) is exactly linear in Σ ctx = kv_tokens_resident.)

Writes calibration.json. Also exposes helpers reused by validate_waveform.py.

Usage:
  python energy_model.py --run_dir logs/<run>
"""

import argparse
import bisect
import csv
import json
import os
from typing import List, Optional, Tuple

DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "fp8": 1, "float32": 4, "auto": 2}


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
    # ensure sorted by time
    order = sorted(range(len(ts)), key=lambda i: ts[i])
    return [ts[i] for i in order], [ps[i] for i in order]


def integrate_window(ts: List[float], ps: List[float], t0: float, t1: float
                     ) -> Optional[float]:
    """Trapezoidal ∫ power dt over [t0, t1], interpolating at the edges."""
    if len(ts) < 2 or t1 <= t0:
        return None
    lo = bisect.bisect_left(ts, t0)
    hi = bisect.bisect_right(ts, t1)
    pts = []
    if lo > 0:  # left edge interpolation
        pts.append((t0, _interp(ts, ps, t0)))
    for i in range(lo, hi):
        pts.append((ts[i], ps[i]))
    if hi < len(ts):
        pts.append((t1, _interp(ts, ps, t1)))
    pts = [(t, p) for (t, p) in pts if t0 <= t <= t1]
    if len(pts) < 2:
        return None
    e = 0.0
    for i in range(len(pts) - 1):
        (ta, pa), (tb, pb) = pts[i], pts[i + 1]
        e += 0.5 * (pa + pb) * (tb - ta)
    return e


def _interp(ts, ps, t):
    i = bisect.bisect_left(ts, t)
    if i <= 0:
        return ps[0]
    if i >= len(ts):
        return ps[-1]
    t0, t1 = ts[i - 1], ts[i]
    p0, p1 = ps[i - 1], ps[i]
    if t1 == t0:
        return p0
    return p0 + (p1 - p0) * (t - t0) / (t1 - t0)


# ---------------- model byte accounting ----------------
def model_byte_constants(meta: dict) -> Tuple[float, float, str]:
    """Return (weight_bytes, kv_bytes_per_token, source)."""
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
    weight_bytes = m.active_params() * dtype_b
    kv_bytes_per_token = m.kv_bytes_per_token(kv_b)
    return float(weight_bytes), float(kv_bytes_per_token), f"models.py:{key}"


def iter_rows(iter_csv: str) -> List[dict]:
    out = []
    with open(iter_csv) as f:
        for row in csv.DictReader(f):
            out.append(row)
    return out


def iter_bytes(row: dict, weight_bytes: float, kv_bpt: float) -> Optional[float]:
    kv = row.get("kv_tokens_resident")
    try:
        kv_tokens = float(kv) if kv not in (None, "") else 0.0
    except ValueError:
        kv_tokens = 0.0
    return weight_bytes + kv_tokens * kv_bpt


# ---------------- fitting ----------------
def _ls_through_origin(xs, ys):
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    return (sxy / sxx) if sxx else None


def _ls_two_feature(x1, x2, y):
    """Least squares y = a·x1 + b·x2 (no intercept). Returns (a, b)."""
    s11 = sum(a * a for a in x1)
    s22 = sum(a * a for a in x2)
    s12 = sum(a * b for a, b in zip(x1, x2))
    s1y = sum(a * c for a, c in zip(x1, y))
    s2y = sum(a * c for a, c in zip(x2, y))
    det = s11 * s22 - s12 * s12
    if abs(det) < 1e-30:
        return None, None
    a = (s1y * s22 - s2y * s12) / det
    b = (s11 * s2y - s12 * s1y) / det
    return a, b


def _r2(y, yhat):
    n = len(y)
    if n == 0:
        return None
    ybar = sum(y) / n
    ss_tot = sum((v - ybar) ** 2 for v in y)
    ss_res = sum((v - h) ** 2 for v, h in zip(y, yhat))
    return 1 - ss_res / ss_tot if ss_tot else None


def calibrate(run_dir: str, phase: str = "decode") -> dict:
    meta = json.load(open(os.path.join(run_dir, "run_meta.json")))
    ts, ps = load_power_samples(os.path.join(run_dir, "power_trace.csv"))
    rows = iter_rows(os.path.join(run_dir, "iter_log.csv"))
    weight_bytes, kv_bpt, byte_src = model_byte_constants(meta)

    p_static = meta.get("idle_power_w")

    bytes_list, t_list, e_list = [], [], []
    for r in rows:
        if (r.get("phase") or "").strip() != phase:
            continue
        try:
            t0 = float(r["t_start"]); t1 = float(r["t_end"])
        except (ValueError, KeyError):
            continue
        t_iter = t1 - t0
        if t_iter <= 0:
            continue
        e_meas = integrate_window(ts, ps, t0, t1)
        if e_meas is None:
            continue
        b = iter_bytes(r, weight_bytes, kv_bpt)
        bytes_list.append(b); t_list.append(t_iter); e_list.append(e_meas)

    n = len(e_list)
    result = {
        "run_dir": run_dir, "phase": phase, "n_iters_fit": n,
        "weight_bytes": weight_bytes, "kv_bytes_per_token": kv_bpt,
        "byte_source": byte_src, "idle_power_w": p_static,
    }
    if n < 3:
        result["error"] = f"too few {phase} iters with clean energy windows ({n})"
        return result

    # Joint 2-parameter fit (cross-check).
    e_bit_joint, p_static_joint = _ls_two_feature(bytes_list, t_list, e_list)

    # Primary: anchor P_static from idle, fit e_bit through the origin on dynamic E.
    if p_static is None:
        p_static = p_static_joint  # fall back to fitted static if no idle baseline
        result["p_static_source"] = "joint_fit (no idle baseline!)"
    else:
        result["p_static_source"] = "idle_baseline"
    dyn_e = [e - p_static * t for e, t in zip(e_list, t_list)]
    e_bit = _ls_through_origin(bytes_list, dyn_e)

    yhat = [e_bit * b + p_static * t for b, t in zip(bytes_list, t_list)]
    result.update({
        "e_bit_j_per_byte": e_bit,
        "p_static_w": p_static,
        "e_bit_joint": e_bit_joint,
        "p_static_joint": p_static_joint,
        "r2": _r2(e_list, yhat),
        "mean_bytes_per_iter": sum(bytes_list) / n,
        "mean_energy_per_iter_j": sum(e_list) / n,
        "mean_t_iter_s": sum(t_list) / n,
    })
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--phase", default="decode",
                    choices=["decode", "prefill", "mixed"])
    args = ap.parse_args()
    cal = calibrate(args.run_dir, args.phase)
    out = os.path.join(args.run_dir, "calibration.json")
    with open(out, "w") as f:
        json.dump(cal, f, indent=2)
    print(json.dumps(cal, indent=2))
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
