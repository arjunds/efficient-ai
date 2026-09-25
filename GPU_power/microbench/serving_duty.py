#!/usr/bin/env python3
"""
serving_duty.py -- duty-cycled vLLM serving: J/byte at full clocks on a
power-capped GPU.

Why: node-d1's A5000s are admin-capped at 100 W (TDP 230 W) and idle at 61-79 W,
so ANY sustained decode pins power at the cap and the firmware drops the SM
clock to 210 MHz (decode step 95 ms instead of ~22 ms). Coefficients from such
runs describe the cap controller, not the hardware. Here the offered load is a
train of short requests (8-token synthetic prompt, exactly 8 output tokens:
1 prefill + 7 decode = 8 full weight sweeps) separated by idle gaps; the gap is
varied block by block. Mean power per block vs weight-byte rate is fit as
    P_block = a + e_serv * byte_rate      (a = floor at the operating clocks)
exactly like the microbenchmark duty-cycle method, so e_serv is directly
comparable to the GEMV / DRAM-stream slopes and is immune to the idle-state
ambiguity (61 W @1695 MHz vs 79 W @1905 MHz on this part).

  run:      python3 microbench/serving_duty.py run --model Qwen/Qwen2-7B-Instruct
  analyze:  python3 microbench/serving_duty.py analyze logs/A5000_duty/Qwen2-7B-Instruct
"""
import argparse
import bisect
import csv
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def build_schedule(gaps, block_s, on_s, idle_s):
    """Returns (schedule string, blocks). Block = repeated (1:on_s, 0:gap)."""
    segs, blocks = [], []
    segs.append(f"0:{idle_s}"); blocks.append({"gap_s": None, "kind": "idle", "n_seg": 1})
    for g in gaps:
        n = max(3, int(round(block_s / (g + 0.25))))
        for _ in range(n):
            segs.append(f"1:{on_s}"); segs.append(f"0:{g}")
        blocks.append({"gap_s": g, "kind": "duty", "n_seg": 2 * n})
    segs.append(f"0:{idle_s}"); blocks.append({"gap_s": None, "kind": "idle", "n_seg": 1})
    return ",".join(segs), blocks


def do_run(a):
    short = a.model.split("/")[-1]
    rd = os.path.join(a.log_root, short, a.tag)
    os.makedirs(rd, exist_ok=True)
    gaps = [float(x) for x in a.gaps.split(",")]
    sched, blocks = build_schedule(gaps, a.block_s, a.on_s, a.idle_s)
    json.dump({"gaps": gaps, "block_s": a.block_s, "on_s": a.on_s, "idle_s": a.idle_s,
               "blocks": blocks, "input_len": a.input_len, "output_len": a.output_len},
              open(os.path.join(rd, "duty_schedule.json"), "w"), indent=1)
    # idle baseline dir (the harness looks for baselines.json in parent dir)
    cmd = [sys.executable, "-u", "energy_profile_load.py", "--mode", "load",
           "--model", a.model, "--dtype", "float16", "--run_dir", rd,
           "--max_model_len", str(a.max_model_len), "--gpu_memory_utilization",
           str(a.gpu_mem), "--input_len", str(a.input_len), "--output_len",
           str(a.output_len), "--concurrency_schedule", sched]
    print(" ".join(cmd[:14]), "... (schedule with", sched.count(",") + 1, "segments)", flush=True)
    with open(os.path.join(rd, "run.log"), "w") as lf:
        subprocess.run(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT)
    do_analyze(argparse.Namespace(run_dir=rd, e_gemm=a.e_gemm))


def _linfit(xs, ys):
    import numpy as np
    X = np.c_[np.ones(len(xs)), xs]
    c, *_ = np.linalg.lstsq(X, ys, rcond=None)
    res = ys - X @ c
    s2 = (res ** 2).sum() / max(1, len(xs) - 2)
    cov = s2 * np.linalg.inv(X.T @ X)
    ss = ((ys - ys.mean()) ** 2).sum()
    return c[0], c[1], cov[0, 0] ** 0.5, cov[1, 1] ** 0.5, (1 - (res ** 2).sum() / ss) if ss else None


def do_analyze(a):
    import numpy as np
    from energy_model import model_byte_constants, model_flop_constants
    rd = a.run_dir
    meta = json.load(open(os.path.join(rd, "run_meta.json")))
    sch = json.load(open(os.path.join(rd, "duty_schedule.json")))
    segs = meta["segments"]
    wbytes, kv_bpt, src = model_byte_constants(meta)
    mm_per_tok, attn_per_res, _ = model_flop_constants(meta)

    P = list(csv.DictReader(open(os.path.join(rd, "power_trace.csv"))))
    pt = np.array([float(r["t_wall"]) for r in P])

    def col(k):
        return np.array([float(r[k]) if r.get(k) not in (None, "") else np.nan for r in P])
    p_inst, p_avg, sm, mem, temp = col("power_inst_w"), col("power_w"), col("sm_mhz"), col("mem_mhz"), col("temp_c")
    pw = p_inst if np.isfinite(p_inst).mean() > 0.9 else p_avg

    it = list(csv.DictReader(open(os.path.join(rd, "iter_log.csv"))))
    its = np.array([float(r["t_start"]) for r in it])
    ite = np.array([float(r["t_end"]) for r in it])
    ntok = np.array([float(r["prefill_tokens"] or 0) + float(r["decode_tokens"] or 0) for r in it])
    kv = np.array([float(r["kv_tokens_resident"] or 0) for r in it])
    ph = [r["phase"] for r in it]

    out = []
    k = 0
    for b in sch["blocks"]:
        s0 = segs[k]["t0"]; s1 = segs[k + b["n_seg"] - 1]["t1"]; k += b["n_seg"]
        t0, t1 = s0 + 1.5, s1 - 0.5            # trim edges (1 s-avg lag, settle)
        m = (pt >= t0) & (pt <= t1)
        mi = (its >= t0) & (its < t1)
        dur = t1 - t0
        n_it = int(mi.sum())
        wb = n_it * wbytes; kb = float((kv[mi] * kv_bpt).sum())
        fl = float((ntok[mi] * mm_per_tok).sum())
        dec = [(ite[i] - its[i]) for i in np.where(mi)[0] if ph[i] == "decode" and ite[i] > its[i]]
        # SM clock while an iteration is running (burst clocks)
        busy = np.zeros(len(pt), bool)
        for i in np.where(mi)[0]:
            lo = bisect.bisect_left(pt, its[i]); hi = bisect.bisect_right(pt, ite[i])
            busy[lo:hi] = True
        out.append({**b, "t0": t0, "t1": t1, "dur": dur, "n_iters": n_it,
                    "p_mean": float(np.nanmean(pw[m])), "p_avg_mean": float(np.nanmean(p_avg[m])),
                    "sm_mean": float(np.nanmean(sm[m])), "sm_busy_mean": float(np.nanmean(sm[m & busy])) if (m & busy).any() else None,
                    "sm_busy_min": float(np.nanmin(sm[m & busy])) if (m & busy).any() else None,
                    "mem_mean": float(np.nanmean(mem[m])), "temp": float(np.nanmean(temp[m])),
                    "p_max": float(np.nanmax(pw[m])),
                    "weight_byte_rate": wb / dur, "kv_byte_rate": kb / dur, "flop_rate": fl / dur,
                    "decode_step_ms_median": float(np.median(dec) * 1e3) if dec else None,
                    "burst_bw_from_decode_step": (wbytes / np.median(dec)) if dec else None})
    xs = np.array([o["weight_byte_rate"] + o["kv_byte_rate"] for o in out])
    ys = np.array([o["p_mean"] for o in out])
    fr = np.array([o["flop_rate"] for o in out])
    res = {"run_dir": rd, "weight_bytes_per_iter": wbytes, "byte_source": src,
           "idle_power_w_vllm_baseline": meta.get("idle_power_w"), "power_limit_w": meta.get("power_limit_w"),
           "blocks": out}
    if len(out) >= 3:
        a0, e, sa, se, r2 = _linfit(xs, ys)
        res["fit_raw"] = {"a_w": a0, "e_j_per_byte": e, "se": se, "se_a": sa, "r2": r2}
        if a.e_gemm:
            a1, e1, sa1, se1, r21 = _linfit(xs, ys - a.e_gemm * fr)
            res["fit_compute_corrected"] = {"e_gemm_used": a.e_gemm, "a_w": a1, "e_j_per_byte": e1,
                                            "se": se1, "r2": r21}
    json.dump(res, open(os.path.join(rd, "duty_analysis.json"), "w"), indent=1)
    print(f"{'block':<10}{'dur':>6}{'iters':>7}{'P W':>8}{'Pavg':>8}{'GB/s':>8}{'TF/s':>7}"
          f"{'sm':>6}{'sm_busy':>8}{'min':>6}{'step ms':>9}{'Pmax':>7}")
    for o in out:
        print(f"{str(o['gap_s']):<10}{o['dur']:>6.1f}{o['n_iters']:>7}{o['p_mean']:>8.2f}{o['p_avg_mean']:>8.2f}"
              f"{(o['weight_byte_rate']+o['kv_byte_rate'])/1e9:>8.2f}{o['flop_rate']/1e12:>7.3f}"
              f"{o['sm_mean']:>6.0f}{(o['sm_busy_mean'] or 0):>8.0f}{(o['sm_busy_min'] or 0):>6.0f}"
              f"{(o['decode_step_ms_median'] or 0):>9.2f}{o['p_max']:>7.1f}")
    for k2 in ("fit_raw", "fit_compute_corrected"):
        if k2 in res:
            f = res[k2]
            print(f"{k2}: e = {f['e_j_per_byte']:.4e} ± {f['se']:.1e} J/byte, floor a = {f['a_w']:.2f} W, R2 = {f['r2']:.4f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--log_root", default="logs/A5000_duty")
    r.add_argument("--tag", default="duty8x8")
    r.add_argument("--gaps", default="5,3,2,1.5")
    r.add_argument("--block_s", type=float, default=45.0)
    r.add_argument("--on_s", type=float, default=0.02)
    r.add_argument("--idle_s", type=float, default=25.0)
    r.add_argument("--input_len", type=int, default=8)
    r.add_argument("--output_len", type=int, default=8)
    r.add_argument("--max_model_len", type=int, default=2048)
    r.add_argument("--gpu_mem", type=float, default=0.90)
    r.add_argument("--e_gemm", type=float, default=None, help="J/flop for compute correction")
    z = sp.add_parser("analyze")
    z.add_argument("run_dir")
    z.add_argument("--e_gemm", type=float, default=None)
    a = ap.parse_args()
    if a.cmd == "run":
        do_run(a)
    else:
        do_analyze(a)


if __name__ == "__main__":
    main()
