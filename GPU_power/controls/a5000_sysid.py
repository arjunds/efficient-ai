#!/usr/bin/env python3
"""
controls/a5000_sysid.py -- identify the (100-W-capped) A5000 serving plant for
the live admission-control experiment. Does NOT modify the shared harness or
controls/sysid.py; it drives / reuses them.

  python3 -m controls.a5000_sysid staircase_cmd --model M --run_dir D
        prints the energy_profile_load.py command for a concurrency staircase
        (closed loop; 25 s per level, an idle step and a down-staircase so the
        power waveform moves; reuses --concurrency_schedule).
  python3 -m controls.a5000_sysid table RUN_DIR [RUN_DIR ...] --out T.json
        throughput/power table n_running -> tok/s, P_busy, P_idle, gen_mean from
        closed-loop segments (staircase and/or fixed-c sweep runs). This is the
        live MPC's internal model (controls/live_controllers.py).
  python3 -m controls.a5000_sysid fit --glob 'logs/A5000/*/alpaca_c*,logs/A5000_live/staircase*' \
        --out controls/plant_params_a5000.json
        runs controls/sysid.py's iteration-time identification on the A5000 runs
        (RUN_GLOBS / PARAMS_JSON / gpu table monkeypatched -- nothing shared is
        overwritten), so controls/plant.py can be instantiated for the A5000 with
        PARAMS_JSON pointed at the output.
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

STAIRCASE = "1:25,2:25,4:25,8:25,16:25,32:25,64:25,0:20,48:25,24:25,12:25,6:25,3:25"


def staircase_cmd(a):
    return (f"python3 -u energy_profile_load.py --mode load --model {a.model} --task {a.task} "
            f"--output_len 256 --concurrency_schedule {STAIRCASE} --run_dir {a.run_dir} "
            f"--max_model_len 4096")


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _power(rd):
    ts, ps = [], []
    with open(os.path.join(rd, "power_trace.csv")) as f:
        for r in csv.DictReader(f):
            p = _f(r.get("power_inst_w")) if r.get("power_inst_w") else _f(r.get("power_w"))
            t = _f(r.get("t_wall"))
            if np.isfinite(p) and np.isfinite(t):
                ts.append(t)
                ps.append(p)
    return np.array(ts), np.array(ps)


def _iters(rd):
    with open(os.path.join(rd, "iter_log.csv")) as f:
        rows = list(csv.DictReader(f))
    A = {k: np.array([_f(r.get(k)) for r in rows]) for k in
         ("t_start", "t_end", "n_running", "decode_tokens", "prefill_tokens")}
    return {k: np.nan_to_num(v) for k, v in A.items()}


def segments_table(run_dirs, trim_s=5.0):
    rows = []
    idle = []
    gens = []
    for rd in run_dirs:
        meta = json.load(open(os.path.join(rd, "run_meta.json")))
        segs = meta.get("segments") or []
        if meta.get("idle_power_w"):
            idle.append(float(meta["idle_power_w"]))
        try:
            res = json.load(open(os.path.join(rd, "results.json")))
            if res.get("completed_requests"):
                gens.append(res["generated_tokens_window"] / res["completed_requests"])
        except Exception:
            pass
        ts, ps = _power(rd)
        A = _iters(rd)
        for s in segs:
            c = s.get("concurrency")
            t0, t1 = s["t0"] + trim_s, s["t1"]
            if t1 - t0 < 5:
                continue
            pm = ps[(ts >= t0) & (ts <= t1)]
            if c == 0:
                if len(pm):
                    idle.append(float(np.median(pm)))
                continue
            m = (A["t_end"] >= t0) & (A["t_end"] <= t1)
            if m.sum() < 5:
                continue
            dur = np.maximum(A["t_end"][m] - A["t_start"][m], 0)
            n = float(np.average(A["n_running"][m], weights=np.maximum(dur, 1e-6)))
            tok = float(A["decode_tokens"][m].sum()) / (t1 - t0)
            rows.append(dict(run=os.path.relpath(rd, ROOT), c=c, n_mean=n, tok_s=tok,
                             p_mean=float(pm.mean()) if len(pm) else float("nan"),
                             p_p95=float(np.percentile(pm, 95)) if len(pm) else float("nan")))
    return rows, idle, gens


def cmd_table(a):
    rows, idle, gens = segments_table(a.run_dirs)
    if not rows:
        print("no usable segments")
        return
    # table keyed by (rounded) mean running batch; average duplicates
    tab = {}
    for r in rows:
        k = max(1, int(round(r["n_mean"])))
        tab.setdefault(k, []).append(r["tok_s"])
    table = {k: float(np.mean(v)) for k, v in sorted(tab.items())}
    # enforce monotone throughput (noise at small n)
    ks = sorted(table)
    for i in range(1, len(ks)):
        table[ks[i]] = max(table[ks[i]], table[ks[i - 1]])
    busy = [r["p_mean"] for r in rows if np.isfinite(r["p_mean"])]
    out = dict(table=table, p_busy=float(np.median(busy)) if busy else None,
               p_idle=float(np.median(idle)) if idle else None,
               gen_mean=float(np.mean(gens)) if gens else None,
               rows=rows, sources=[os.path.relpath(d, ROOT) for d in a.run_dirs])
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    for r in rows:
        print(f"  {r['run']:55s} c={r['c']:>3} n={r['n_mean']:6.1f} tok/s={r['tok_s']:8.1f} "
              f"P={r['p_mean']:6.1f} (p95 {r['p_p95']:.1f})")


def cmd_fit(a):
    from controls import sysid
    # the GPU-level law needs >= 2 models: include the physics agent's A5000 sweep
    sysid.RUN_GLOBS = [g.strip() for g in a.glob.split(",") if g.strip()]
    sysid.PARAMS_JSON = a.out
    sysid.RESULTS_JSON = a.out.replace(".json", "_results.json")
    base = sysid.gpu_coefficients

    def gpu_coeffs_with_a5000():
        out = base()
        js = sysid.load_json(sysid.COEF_JSON, {}) or {}
        g = dict(js.get("A5000") or {})
        spec = sysid.load_json(os.path.join(ROOT, "logs", "A5000", "specs.json"), {}) or {}
        out["A5000"] = dict(e_wbyte=float(g.get("e_wbyte", 2.07e-10)),
                            e_kvbyte=float(g.get("e_kvbyte", 5.85e-10)),
                            e_gemm=float(g.get("e_gemm", 2.5e-13)),
                            p_static=float(g.get("p_static", 61.1)),
                            p_cap=float(spec.get("p_cap", 100.0)),       # ENFORCED limit
                            bw=float(g.get("bw") or 7.68e11),
                            peak_flops=float(g.get("peak_flops") or spec.get("peak_flops", 1.111e14)),
                            source="gpu_coefficients.json + logs/A5000/specs.json")
        return out
    sysid.gpu_coefficients = gpu_coeffs_with_a5000
    print(f"[a5000_sysid] fitting runs matching {a.glob} -> {a.out}")
    sysid.cmd_fit(argparse.Namespace())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("staircase_cmd")
    s.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    s.add_argument("--task", default="alpaca")
    s.add_argument("--run_dir", default="logs/A5000_live/staircase")
    t = sub.add_parser("table")
    t.add_argument("run_dirs", nargs="+")
    t.add_argument("--out", default="logs/A5000_live/a5000_table.json")
    f = sub.add_parser("fit")
    f.add_argument("--glob", default="logs/A5000/*/alpaca_c*,logs/A5000/*/sharegpt_c*,logs/A5000_live/staircase*")
    f.add_argument("--out", default=os.path.join(HERE, "plant_params_a5000.json"))
    a = ap.parse_args()
    if a.cmd == "staircase_cmd":
        print(staircase_cmd(a))
    elif a.cmd == "table":
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        cmd_table(a)
    elif a.cmd == "fit":
        cmd_fit(a)


if __name__ == "__main__":
    main()
