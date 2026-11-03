#!/usr/bin/env python3
"""Real-trace decode-only replays (TRACES.md): per GPU x model x trace x load level,
  - measured: throughput, average power, J/token, idle share, TPOT / TTFT percentiles,
    prefix-cache hit fraction (decode purity);
  - roofline placement: arithmetic intensity I = FLOP / DRAM byte (weights + KV) of the
    steps actually run, achieved FLOP/s and B/s vs the bandwidth, compute and power roofs
    (liminal_e.roofline);
  - energy breakdown with the GPU's LIMINAL-E coefficients on the MEASURED timeline:
    static (P_s x T), activity (P_0 x busy time), weights (e_w x W), KV (e_kv x K),
    GEMM (e_g x F), with each power-capped step's dynamic terms scaled down to P_cap;
    residual = measured - sum (host/launch overhead not in the terms, model error);
    LIMINAL-E's predicted window energy is the same sum (capT% = time in capped steps);
  - capacity at latency SLOs: the highest measured level whose TPOT p99 and TTFT p99 meet
    the SLO, its request rate and J/token.

Steady window per level: [t0 + ramp, last arrival], ramp = median request latency of the
level (capped at 40% of the arrival span), so the empty-start ramp and the drain tail are
excluded. Idle time = window time with no request in the system (requests.csv).

  python3 trace_analysis.py --root logs/A100_prov --root logs/A6000_prov [--out trace_summary.json]
"""
import argparse
import bisect
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "liminal_energy"))
from energy_model import window_energy_fn  # noqa: E402

SLOS = {"tight": dict(tpot=0.050, ttft=1.0), "relaxed": dict(tpot=0.100, ttft=2.0)}


def gpu_key(name):
    n = (name or "").upper()
    for k, pat in (("B200", "B200"), ("H200", "H200"), ("A100-80-PCIe", "A100"), ("A6000", "A6000")):
        if pat in n:
            return k
    return None


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else None


def union_len(intervals, lo, hi):
    """Length of the union of [a, b] intervals clipped to [lo, hi]."""
    iv = sorted((max(a, lo), min(b, hi)) for a, b in intervals if b > lo and a < hi)
    tot, cur_a, cur_b = 0.0, None, None
    for a, b in iv:
        if cur_b is None or a > cur_b:
            if cur_b is not None:
                tot += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    if cur_b is not None:
        tot += cur_b - cur_a
    return tot


def find_runs(roots):
    for root in roots:
        for dp, _, fns in os.walk(root):
            if "results.json" in fns and "run_meta.json" in fns:
                try:
                    meta = json.load(open(os.path.join(dp, "run_meta.json")))
                    res = json.load(open(os.path.join(dp, "results.json")))
                except Exception:
                    continue
                if meta.get("trace") and res.get("levels"):
                    yield dp, meta, res


def load_iters(rd):
    rows = []
    with open(os.path.join(rd, "iter_log.csv")) as f:
        for r in csv.DictReader(f):
            t0, t1 = fnum(r.get("t_start")), fnum(r.get("t_end"))
            if t0 is None or t1 is None:
                continue
            pf, hits = fnum(r.get("prefill_tokens")) or 0.0, fnum(r.get("prefix_hits")) or 0.0
            rows.append(dict(t0=t0, t1=t1, n=fnum(r.get("decode_tokens")) or 0.0,
                             nrun=fnum(r.get("n_running")) or 0.0,
                             kv=fnum(r.get("kv_tokens_resident")) or 0.0,
                             pf=max(0.0, pf - hits)))           # computed prefill tokens
    rows.sort(key=lambda r: r["t1"])
    return rows


def load_requests(rd):
    out = []
    with open(os.path.join(rd, "requests.csv")) as f:
        for r in csv.DictReader(f):
            out.append(dict(sub=fnum(r["t_submit"]), first=fnum(r.get("t_first_token")),
                            fin=fnum(r["t_finish"]), P=fnum(r["prompt_tokens"]) or 0.0,
                            G=fnum(r["gen_tokens"]) or 0.0, cached=fnum(r.get("cached_tokens")),
                            x=fnum(r.get("trace_speedup"))))
    return out


def analyze_level(rd, meta, lv, iters, reqs, Efn, lim, g, m, pp):
    x = lv["trace_speedup"]
    R = [r for r in reqs if r["x"] == x] or [r for r in reqs if lv["t0"] - 1 <= r["sub"] <= lv["t1"]]
    span = lv["t_last_arrival"] - lv["t0"]
    ramp = min(0.4 * span, pct([r["fin"] - r["sub"] for r in R], 0.5) or 0.0)
    ws, we = lv["t0"] + ramp, lv["t_last_arrival"]
    if we - ws < 5.0:                                    # too short: use the whole level
        ws, we = lv["t0"], lv["t1"]
    S = [r for r in iters if r["t0"] >= ws and r["t1"] <= we]
    if not S:
        return None
    ws, we = S[0]["t0"], S[-1]["t1"]
    T = we - ws
    E = Efn(ws, we)
    tokens = sum(r["n"] for r in S)
    T_busy = union_len([(r["sub"], r["fin"]) for r in R], ws, we)
    T_idle = max(0.0, T - T_busy)
    dts = [r["t1"] - r["t0"] for r in S]
    nbar = sum(r["nrun"] * d for r, d in zip(S, dts)) / max(T, 1e-9)
    ctx = sum(r["kv"] for r in S) / max(1.0, sum(r["nrun"] for r in S))
    Rs = [r for r in R if ws <= r["sub"] <= we]          # arrived in the steady window
    tpot = [(r["fin"] - r["first"]) / (r["G"] - 1) for r in Rs if r["first"] and r["G"] > 1]
    ttft = [r["first"] - r["sub"] for r in Rs if r["first"]]
    hit_den = sum(r["P"] for r in R)
    out = dict(trace_speedup=x, req_per_s=len(Rs) / T if T else None, window_s=T, ramp_s=ramp,
               tokens=tokens, tok_per_s=tokens / T, energy_j=E,
               avg_power_w=E / T if E else None, j_per_token=E / tokens if E and tokens else None,
               idle_frac_time=T_idle / T, mean_running=nbar, mean_context=ctx,
               computed_prefill_tokens=sum(r["pf"] for r in S),
               prefix_hit_frac=(sum(r["cached"] or 0 for r in R) / hit_den) if hit_den else None,
               tpot_p50_ms=1e3 * pct(tpot, .5) if tpot else None, tpot_p99_ms=1e3 * pct(tpot, .99) if tpot else None,
               ttft_p50_ms=1e3 * pct(ttft, .5) if ttft else None, ttft_p99_ms=1e3 * pct(ttft, .99) if ttft else None,
               aborted=lv.get("aborted", 0), requests=len(Rs))
    if lim is None or g is None or m is None or not E:
        return out
    # work of the steps actually run (LIMINAL-E work(): canonical byte/FLOP counts)
    # Per step: full-clock dynamic energy e_w*W + e_kv*K + e_g*F; a step whose demanded
    # power exceeds the cap is clipped to P_cap (its dynamic terms scaled down together,
    # as LIMINAL-E's roof does), so the residual is model error, not the cap.
    W = K = F = Fg = 0.0
    dyn = dict(weights=0.0, kv=0.0, gemm=0.0)
    E_pred = 0.0
    p_cap = g["p_cap"]
    capped_t = 0.0
    for r, d in zip(S, dts):
        if r["n"] <= 0 and r["pf"] <= 0:
            continue
        nb = max(r["n"], 1.0)
        w = lim.work(m, nb, r["kv"] / max(r["nrun"], 1.0), prefill_tokens=r["pf"])
        fg = w["Fg"] + w.get("Fa_pf", 0.0)
        W += w["Wb"]; K += w["KVb"]; F += w["F"]; Fg += fg
        parts_s = dict(weights=pp["e_w"] * w["Wb"], kv=pp["e_kv"] * w["KVb"], gemm=pp["e_g"] * fg)
        e_dyn = sum(parts_s.values())
        room = max(0.0, (p_cap - pp["p_s"] - pp["p0"]) * d)
        sc = min(1.0, room / e_dyn) if e_dyn > 0 else 1.0
        capped_t += d if sc < 1.0 else 0.0
        for k in dyn:
            dyn[k] += sc * parts_s[k]
        E_pred += (pp["p_s"] + pp["p0"]) * d + sc * e_dyn
    E_pred -= pp["p0"] * T_idle                           # no activity power with nothing running
    parts = dict(static=pp["p_s"] * T, activity=pp["p0"] * T_busy, **dyn)
    parts["residual"] = E - sum(parts.values())
    I = F / (W + K) if W + K else None
    roof = lim.roofline(g, I) if I else {}
    out.update(intensity=I, flops_per_s=F / T, bytes_per_s=(W + K) / T,
               bw_util=(W + K) / T / g["bw"], power_frac_cap=(E / T) / p_cap,
               roof_attainable=roof.get("attainable"), roof_bound=roof.get("bound"),
               roof_frac=(F / T) / roof["attainable"] if roof.get("attainable") else None,
               kv_byte_share=K / (W + K) if W + K else None,
               energy_parts_j=parts, energy_shares={k: v / E for k, v in parts.items()},
               liminal_pred_j=E_pred, liminal_err=(E_pred - E) / E, coef_src=pp["src"],
               capped_time_frac=capped_t / T,
               unclipped_dynamic_j=pp["e_w"] * W + pp["e_kv"] * K + pp["e_g"] * Fg)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--gpu_key", default=None, help="override GPU coefficient key (testing)")
    a = ap.parse_args()
    try:
        import liminal_e as lim
    except Exception as e:
        print("[warn] liminal_e unavailable (%r): measured columns only" % (e,))
        lim = None
    rows = []
    for rd, meta, res in sorted(find_runs(a.root)):
        gk = a.gpu_key or gpu_key(meta.get("gpu_name"))
        g = m = pp = None
        if lim is not None and gk:
            try:
                g, m = lim.gpu(gk), lim.model(meta["model"])
                pp = lim.profile_params(g)
            except Exception as e:
                print("[warn] %s: no LIMINAL-E coefficients (%r)" % (rd, e))
                g = m = pp = None
        Efn, src = window_energy_fn(os.path.join(rd, "power_trace.csv"), "auto")
        iters, reqs = load_iters(rd), load_requests(rd)
        tr = meta["trace"]
        for lv in res["levels"]:
            r = analyze_level(rd, meta, lv, iters, reqs, Efn, lim, g, m, pp)
            if r is None:
                continue
            r.update(run_dir=rd, gpu=gk or meta.get("gpu_name"), model=meta["model"].split("/")[-1],
                     trace=os.path.basename(tr["trace_file"]).split(".")[0], power_source=src,
                     idle_power_w=meta.get("idle_power_w"), p_cap=meta.get("power_limit_total_w"))
            rows.append(r)
    if not rows:
        print("no trace runs under", a.root)
        return
    # capacity at SLO: highest load level meeting TPOT p99 and TTFT p99
    cap = {}
    for r in rows:
        for s, lim_ in SLOS.items():
            ok = (r["tpot_p99_ms"] is not None and r["tpot_p99_ms"] <= 1e3 * lim_["tpot"]
                  and r["ttft_p99_ms"] is not None and r["ttft_p99_ms"] <= 1e3 * lim_["ttft"])
            k = (r["gpu"], r["model"], r["trace"], s)
            if ok and (k not in cap or r["req_per_s"] > cap[k]["req_per_s"]):
                cap[k] = dict(req_per_s=r["req_per_s"], tok_per_s=r["tok_per_s"], j_per_token=r["j_per_token"],
                              trace_speedup=r["trace_speedup"], avg_power_w=r["avg_power_w"])
    hdr = ("gpu", "model", "trace", "x", "req/s", "tok/s", "W", "%cap", "J/tok", "idle%", "nrun", "ctx",
           "I", "BW%", "TPOT99", "TTFT99", "hit", "wts", "kv", "gemm", "stat", "act", "resid", "LimErr", "capT%")
    print(" ".join("%8s" % h for h in hdr))
    for r in sorted(rows, key=lambda r: (r["gpu"], r["model"], r["trace"], r["trace_speedup"])):
        sh = r.get("energy_shares") or {}
        f = lambda v, fmt: (fmt % v) if v is not None else "-"
        print(" ".join("%8s" % c for c in (
            r["gpu"][:8], r["model"][:8], r["trace"][:8], "%g" % r["trace_speedup"], f(r["req_per_s"], "%.2f"),
            f(r["tok_per_s"], "%.0f"), f(r["avg_power_w"], "%.0f"), f(r.get("power_frac_cap") and 100 * r["power_frac_cap"], "%.0f"),
            f(r["j_per_token"], "%.4f"), f(100 * r["idle_frac_time"], "%.0f"), f(r["mean_running"], "%.1f"),
            f(r["mean_context"], "%.0f"), f(r.get("intensity"), "%.1f"), f(r.get("bw_util") and 100 * r["bw_util"], "%.0f"),
            f(r["tpot_p99_ms"], "%.0f"), f(r["ttft_p99_ms"], "%.0f"), f(r["prefix_hit_frac"], "%.3f"),
            *[f(sh.get(k) and 100 * sh[k], "%.0f") for k in ("weights", "kv", "gemm", "static", "activity", "residual")],
            f(r.get("liminal_err") and 100 * r["liminal_err"], "%+.1f"),
            f(r.get("capped_time_frac") is not None and 100 * r["capped_time_frac"], "%.0f"))))
    print("\ncapacity at SLO (TPOT p99 / TTFT p99: tight 50 ms / 1 s, relaxed 100 ms / 2 s):")
    for k in sorted(cap):
        c = cap[k]
        print("  %-14s %-22s %-16s %-7s %7.2f req/s  %7.0f tok/s  %.4f J/tok  (x%g, %.0f W)" % (
            k[0], k[1], k[2], k[3], c["req_per_s"], c["tok_per_s"], c["j_per_token"] or 0, c["trace_speedup"],
            c["avg_power_w"] or 0))
    out = a.out or os.path.join(a.root[0], "trace_summary.json")
    json.dump({"levels": rows, "capacity": [dict(zip(("gpu", "model", "trace", "slo"), k), **v) for k, v in cap.items()]},
              open(out, "w"), indent=1, default=str)
    print("wrote", out)


if __name__ == "__main__":
    main()
