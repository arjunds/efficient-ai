#!/usr/bin/env python3
"""
thermal_leakage.py: is "idle" power a function of chip temperature (leakage), and
does that bias P_static, the activity intercept P_0 and the power roof?

Motivation (PI, 2026-11-09): idle power is a by-product of the preceding dynamic power,
not a fixed number. Idle samples inside our trace replays rise with temperature
(A100 97 W at 35-45 C -> 119 W at 60-65 C; B200 241 -> 268 W at 30-55 C), but those
are confounded with NVML's power averaging right after a busy step. This separates the
two: averaging decays in ~1 s, the die cools over tens of seconds. No root, no clock
or power-limit change; one CUDA context open throughout (the serving idle state).

Phases (all at 10 ms NVML: POWER_INSTANT, averaged power, energy counter, SM/mem
clock, temperature, throttle reasons):
  baseline   idle with the context open, until temperature is steady
  heat:<op>  run <op> back-to-back until temperature plateaus
  cool:<op>  stop; idle until back within 2 C of the baseline (or the time cap)
             -> P_idle(T): if idle power tracks temperature well after the ~1 s
                averaging tail, it is leakage. Heating with two different ops (DRAM
                stream, fp16 GEMM) tests that idle depends on T alone, not on what
                heated the chip.
  ramp       DRAM stream at increasing then decreasing #programs (the occupancy
             grid), each level held until temperature is steady.
             -> up vs down at the same bandwidth differ only in temperature:
                hysteresis dP/dT cross-checks the cool-down slope, and
                P = a + e*BW + k*(T - T_ref) gives a leakage-corrected e_b and P_0.

  python3 microbench/thermal_leakage.py --out microbench/thermal_A5000          # run
  python3 microbench/thermal_leakage.py --analyze microbench/thermal_A5000      # analyse
Writes <out>/samples.csv, <out>/phases.json, and on --analyze <out>/summary.json and
<out>/thermal.png.
"""
import argparse
import csv
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

COLS = ("t", "energy_mj", "p_inst_w", "p_avg_w", "sm_mhz", "mem_mhz", "temp_c", "reasons")


# ---------------------------------------------------------------- run
def run(a):
    import pynvml
    import torch
    import energy_microbench as mb

    os.makedirs(a.out, exist_ok=True)
    pynvml.nvmlInit()
    torch.cuda.init()
    torch.zeros(1, device="cuda")                       # open the context first
    h = mb.Bench._match_handle(None)                    # UUID match, as energy_microbench (uses no state)
    smp = mb.Sampler(h, a.sample_ms / 1000.0).start()
    name = pynvml.nvmlDeviceGetName(h)
    name = name.decode() if isinstance(name, bytes) else name
    info = dict(gpu=name, sm_count=torch.cuda.get_device_properties(0).multi_processor_count,
                power_limit_w=pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1e3,
                host=os.uname().nodename, started=time.strftime("%Y-%m-%d %H:%M:%S"))
    print("[thermal]", info, flush=True)
    phases = []

    def temp_now():
        return pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)

    def steady(t_start, window_s, tol_c):
        """temperature range over the last window_s is within tol_c"""
        rows = smp.window(time.time() - window_s, time.time())
        ts = [r[6] for r in rows if r[6] is not None]
        return (time.time() - t_start >= window_s and len(ts) > 10 and max(ts) - min(ts) <= tol_c)

    def idle_until(label, min_s, max_s, done):
        torch.cuda.synchronize()
        t0 = time.time()
        while True:
            time.sleep(1.0)
            el = time.time() - t0
            if el >= max_s or (el >= min_s and done(t0)):
                break
        phases.append(dict(phase=label, t0=t0, t1=time.time(), temp0=None, temp1=temp_now()))
        print(f"[thermal] {label}: {time.time() - t0:.0f} s, T={temp_now()} C", flush=True)

    def busy_until(label, op, min_s, max_s, done, bw=None):
        op.tune(0.03)
        torch.cuda.synchronize()
        t0 = time.time()
        calls = 0
        last_chk = t0
        while True:
            op(); calls += 1
            torch.cuda.synchronize()        # ~30 ms calls: a sync per call keeps the queue empty
            now = time.time()               # (an unbounded queue overran max_s by ~30 s in the smoke)
            if now - last_chk >= 0.5:
                last_chk = now
                el = now - t0
                if el >= max_s or (el >= min_s and done(t0)):
                    break
        torch.cuda.synchronize()
        t1 = time.time()
        rec = dict(phase=label, t0=t0, t1=t1, temp1=temp_now(), calls=calls)
        b = getattr(op, "bytes_per_call", 0.0) * calls
        f = getattr(op, "flops_per_call", 0.0) * calls
        rec.update(bytes_per_s=b / (t1 - t0), flops_per_s=f / (t1 - t0))
        phases.append(rec)
        print(f"[thermal] {label}: {t1 - t0:.0f} s, T={temp_now()} C, "
              f"{rec['bytes_per_s'] / 1e9:.0f} GB/s {rec['flops_per_s'] / 1e12:.1f} TF/s", flush=True)
        return rec

    ws = mb.dram_ws()
    n = 8192 if a.gemm_n is None else a.gemm_n
    A = torch.randn(n, n, device="cuda", dtype=torch.float16)
    B = torch.randn(n, n, device="cuda", dtype=torch.float16)
    C = torch.empty(n, n, device="cuda", dtype=torch.float16)
    heaters = {"stream": lambda: mb.StreamOp(ws),
               "gemm": lambda: mb.TorchOp(lambda: torch.matmul(A, B, out=C), 0.0, 2.0 * n ** 3)}
    T_base = None
    try:
        if "baseline" in a.phases:
            idle_until("baseline", a.baseline_min_s, a.baseline_max_s,
                       lambda t0: steady(t0, 60, 1))
            T_base = temp_now()
        T_base = T_base if T_base is not None else temp_now()
        for op_name in [p.split(":", 1)[1] for p in a.phases if p.startswith("heat:")]:
            busy_until(f"heat:{op_name}", heaters[op_name](), a.heat_min_s, a.heat_max_s,
                       lambda t0: steady(t0, 60, 1))
            idle_until(f"cool:{op_name}", a.cool_min_s, a.cool_max_s,
                       lambda t0: temp_now() <= T_base + 2)
        if "ramp" in a.phases:
            sm = info["sm_count"]
            fr = [0.125, 0.25, 0.5, 1, 2, 4, 8]
            for direction, fs in (("up", fr), ("down", fr[::-1])):
                for f in fs:
                    nprog = max(1, int(round(sm * f)))
                    r = busy_until(f"ramp_{direction}:{nprog}", mb.StreamOp(ws, nprog=nprog),
                                   a.level_min_s, a.level_max_s, lambda t0: steady(t0, 20, 0.5))
                    r.update(direction=direction, nprog=nprog)
            idle_until("ramp_cool", a.cool_min_s, a.cool_max_s, lambda t0: temp_now() <= T_base + 2)
    finally:
        smp.stop()
        with open(os.path.join(a.out, "samples.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(COLS)
            w.writerows(smp.rows)
        json.dump(dict(info=info, t_base_c=T_base, phases=phases, args=vars(a)),
                  open(os.path.join(a.out, "phases.json"), "w"), indent=1)
        print(f"[thermal] wrote {len(smp.rows)} samples to {a.out}", flush=True)


# ---------------------------------------------------------------- analysis
def _fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _lstsq(X, y):
    """ordinary least squares via normal equations (no numpy dependency)"""
    k = len(X[0])
    XtX = [[sum(r[i] * r[j] for r in X) for j in range(k)] for i in range(k)]
    Xty = [sum(r[i] * yy for r, yy in zip(X, y)) for i in range(k)]
    M = [row[:] + [v] for row, v in zip(XtX, Xty)]
    for c in range(k):
        p = max(range(c, k), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(k):
            if r != c and M[c][c]:
                fct = M[r][c] / M[c][c]
                M[r] = [x - fct * z for x, z in zip(M[r], M[c])]
    beta = [M[i][k] / M[i][i] for i in range(k)]
    res = [yy - sum(b * x for b, x in zip(beta, r)) for r, yy in zip(X, y)]
    ss = sum(e * e for e in res)
    my = sum(y) / len(y)
    r2 = 1 - ss / max(1e-12, sum((yy - my) ** 2 for yy in y))
    return beta, r2, math.sqrt(ss / max(1, len(y) - k))


def analyze(d):
    S = []
    for r in csv.DictReader(open(os.path.join(d, "samples.csv"))):
        S.append({k: _fnum(r[k]) for k in COLS})
    meta = json.load(open(os.path.join(d, "phases.json")))
    P = meta["phases"]
    pw = "p_inst_w" if sum(1 for s in S if s["p_inst_w"]) > 0.5 * len(S) else "p_avg_w"
    out = dict(info=meta["info"], power_field=pw, t_base_c=meta["t_base_c"])

    def win(t0, t1):
        return [s for s in S if t0 <= s["t"] <= t1 and s[pw] is not None and s["temp_c"] is not None]

    # baseline
    b = next((p for p in P if p["phase"] == "baseline"), None)
    if b:
        w = win(b["t1"] - 30, b["t1"])
        out["baseline"] = dict(p_w=sum(s[pw] for s in w) / len(w), temp_c=sum(s["temp_c"] for s in w) / len(w),
                               sm_mhz=sorted(s["sm_mhz"] for s in w)[len(w) // 2])
    # cool-downs: P_idle vs T, excluding the first `skip` s after the stop (averaging tail)
    cools = {}
    for p in P:
        if not p["phase"].startswith("cool:") and p["phase"] != "ramp_cool":
            continue
        w = win(p["t0"], p["t1"])
        if len(w) < 50:
            continue
        # decay of the first seconds: time for power to fall within 5% of its value at +10 s
        p10 = [s[pw] for s in w if 9.5 <= s["t"] - p["t0"] <= 10.5]
        ref = sum(p10) / len(p10) if p10 else None
        settle = next((s["t"] - p["t0"] for s in w if ref and s[pw] <= 1.05 * ref), None)
        fits = {}
        for skip in (2.0, 5.0, 10.0):
            ww = [s for s in w if s["t"] - p["t0"] >= skip]
            # 1 s bins so 10 ms noise does not dominate
            bins = {}
            for s in ww:
                bins.setdefault(int(s["t"]), []).append(s)
            pts = [(sum(x["temp_c"] for x in v) / len(v), sum(x[pw] for x in v) / len(v),
                    sorted(x["sm_mhz"] for x in v)[len(v) // 2]) for v in bins.values() if len(v) >= 5]
            if len(pts) < 10:
                continue
            clk = max(set(c for _, _, c in pts), key=lambda c: sum(1 for *_, cc in pts if cc == c))
            pts_c = [(t, pw_) for t, pw_, c in pts if c == clk]        # one clock state only
            if len(pts_c) < 10 or max(t for t, _ in pts_c) - min(t for t, _ in pts_c) < 3:
                continue
            (a0, k), r2, rmse = _lstsq([[1.0, t] for t, _ in pts_c], [pw_ for _, pw_ in pts_c])
            fits[f"skip{int(skip)}s"] = dict(w_per_c=k, p_at_tbase=a0 + k * (meta["t_base_c"] or 0), r2=r2,
                                            rmse_w=rmse, n_bins=len(pts_c), sm_mhz=clk,
                                            t_range=[min(t for t, _ in pts_c), max(t for t, _ in pts_c)])
        cools[p["phase"]] = dict(t_start_c=w[0]["temp_c"], t_end_c=w[-1]["temp_c"], p_start_w=w[0][pw],
                                 p_end_w=w[-1][pw], p_at_10s_w=ref, s_to_within_5pct_of_10s=settle, fits=fits)
    out["cooldowns"] = cools
    # ramp: per level, last 10 s
    lv = []
    for p in P:
        if not p["phase"].startswith("ramp_up") and not p["phase"].startswith("ramp_down"):
            continue
        w = win(p["t1"] - 10, p["t1"])
        if not w:
            continue
        lv.append(dict(direction=p["direction"], nprog=p["nprog"], bw_gbs=p["bytes_per_s"] / 1e9,
                       p_w=sum(s[pw] for s in w) / len(w), temp_c=sum(s["temp_c"] for s in w) / len(w),
                       sm_mhz=sorted(s["sm_mhz"] for s in w)[len(w) // 2],
                       capped_frac=sum(1 for s in w if s["reasons"] and int(s["reasons"]) & 0x4) / len(w),
                       hold_s=p["t1"] - p["t0"]))
    out["ramp_levels"] = lv
    unc = [x for x in lv if x["capped_frac"] < 0.05]
    if len(unc) >= 4:
        (p0, e), r2, rmse = _lstsq([[1.0, x["bw_gbs"] * 1e9] for x in unc], [x["p_w"] for x in unc])
        tr = meta["t_base_c"] or min(x["temp_c"] for x in unc)
        (p0t, et, kt), r2t, rmset = _lstsq([[1.0, x["bw_gbs"] * 1e9, x["temp_c"] - tr] for x in unc],
                                           [x["p_w"] for x in unc])
        out["ramp_fit"] = dict(
            no_temp=dict(p0_w=p0, e_j_per_b=e, r2=r2, rmse_w=rmse),
            with_temp=dict(p0_at_tbase_w=p0t, e_j_per_b=et, w_per_c=kt, r2=r2t, rmse_w=rmset, t_ref_c=tr),
            n=len(unc))
        hy = []
        for u in [x for x in unc if x["direction"] == "up"]:
            dn = next((x for x in unc if x["direction"] == "down" and x["nprog"] == u["nprog"]), None)
            if dn and abs(dn["temp_c"] - u["temp_c"]) >= 1:
                hy.append(dict(nprog=u["nprog"], dT=dn["temp_c"] - u["temp_c"], dP=dn["p_w"] - u["p_w"],
                               dbw_pct=100 * (dn["bw_gbs"] / u["bw_gbs"] - 1),
                               w_per_c=(dn["p_w"] - u["p_w"]) / (dn["temp_c"] - u["temp_c"])))
        out["hysteresis"] = hy
    json.dump(out, open(os.path.join(d, "summary.json"), "w"), indent=1)
    _plot(d, S, P, pw, out)
    return out


def _plot(d, S, P, pw, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print("[analyze] no matplotlib:", e)
        return
    t0 = S[0]["t"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    a = ax[0]
    a.plot([s["t"] - t0 for s in S], [s[pw] for s in S], lw=0.4, color="#2a78d6", label=pw)
    a2 = a.twinx()
    a2.plot([s["t"] - t0 for s in S], [s["temp_c"] for s in S], lw=1, color="#d03b3b", label="temp")
    a.set_xlabel("s"); a.set_ylabel("W"); a2.set_ylabel("C", color="#d03b3b")
    for p in P:
        a.axvspan(p["t0"] - t0, p["t1"] - t0, color="#eee" if p["phase"].startswith(("cool", "base", "ramp_cool")) else "#fde",
                  lw=0, zorder=-1)
    a.set_title("timeline (pink = load, grey = idle)", loc="left", fontsize=9)
    a = ax[1]
    for p in P:
        if not (p["phase"].startswith("cool") or p["phase"] == "ramp_cool"):
            continue
        w = [s for s in S if p["t0"] + 2 <= s["t"] <= p["t1"] and s[pw] and s["temp_c"]]
        a.scatter([s["temp_c"] for s in w], [s[pw] for s in w], s=2, alpha=0.3, label=p["phase"])
    b = out.get("baseline")
    if b:
        a.plot(b["temp_c"], b["p_w"], "k*", ms=12, label="baseline")
    a.set_xlabel("temperature (C)"); a.set_ylabel("idle power (W)")
    a.set_title("idle power vs temperature, >2 s after load stops", loc="left", fontsize=9); a.legend(fontsize=7, markerscale=4)
    a = ax[2]
    for dr, m in (("up", "o"), ("down", "s")):
        L = [x for x in out.get("ramp_levels", []) if x["direction"] == dr]
        sc = a.scatter([x["bw_gbs"] for x in L], [x["p_w"] for x in L], c=[x["temp_c"] for x in L],
                       cmap="coolwarm", marker=m, label=dr)
    if out.get("ramp_levels"):
        fig.colorbar(sc, ax=a, label="C")
    a.set_xlabel("DRAM GB/s"); a.set_ylabel("W"); a.legend(fontsize=7)
    a.set_title("occupancy ramp up (o) / down (s)", loc="left", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(d, "thermal.png"), dpi=140)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="microbench/thermal")
    ap.add_argument("--analyze", default=None, help="analyse an existing output directory and exit")
    ap.add_argument("--phases", default="baseline,heat:stream,heat:gemm,ramp")
    ap.add_argument("--sample_ms", type=float, default=10.0)
    ap.add_argument("--baseline_min_s", type=float, default=90)
    ap.add_argument("--baseline_max_s", type=float, default=300)
    ap.add_argument("--heat_min_s", type=float, default=120)
    ap.add_argument("--heat_max_s", type=float, default=600)
    ap.add_argument("--cool_min_s", type=float, default=120)
    ap.add_argument("--cool_max_s", type=float, default=600)
    ap.add_argument("--level_min_s", type=float, default=30)
    ap.add_argument("--level_max_s", type=float, default=150)
    ap.add_argument("--gemm_n", type=int, default=None)
    a = ap.parse_args()
    if a.analyze:
        s = analyze(a.analyze)
        print(json.dumps({k: v for k, v in s.items() if k != "ramp_levels"}, indent=1))
        return
    a.phases = a.phases.split(",")
    run(a)
    analyze(a.out)


if __name__ == "__main__":
    main()
