#!/usr/bin/env python3
"""
provisioning_analysis.py — summarize run_provisioning_sweep.py output
(HANDOFF_B200_PROVISIONING.md). CPU-only; numpy + stdlib (+ recommend_gpu for the
H200/B200 predictions, skipped if it cannot import).

  python3 provisioning_analysis.py [--root logs/B200_prov] [--out <root>/provisioning_summary.json]

Sections
  1. runs      per run: tok/s, req/s, J/token (energy counter), avg W, MBU, median
               step, % of samples at the SW power cap, peak KV %, % of steps with
               requests waiting, TTFT / TPOT p50 & p99, prefix-cache hit rate.
  2. graphs    eager (relog) vs CUDA graphs on matched 7B runs: step time,
               throughput, J/token. How much of the per-step overhead was eager mode?
  3. refit     3-term fit on the new smear-free bins (relog, graphs) vs the per-bin
               coefficients in gpu_coefficients.json and the steady 0.943 pJ/FLOP.
  4. highconc  measured B200 at c=128..512 vs the recommender's B200 and H200
               predictions at the same load (H200 has no measurement: model only).
  5. prefill   J per prompt token, % at cap, and effective J/FLOP of prefill.
  6. tp        TP=2 vs TP=1 J/token and throughput at the same load.
"""
import argparse
import csv
import glob
import json
import os

import numpy as np

CAP_BITS = 0x4 | 0x80          # SW power cap | HW power brake


def jload(p):
    try:
        return json.load(open(p))
    except Exception:
        return {}


def rows(p):
    try:
        return list(csv.DictReader(open(p)))
    except Exception:
        return []


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def pct(a, q):
    a = [x for x in a if x is not None]
    return float(np.percentile(a, q)) if a else None


def run_summary(rd):
    res, meta = jload(os.path.join(rd, "results.json")), jload(os.path.join(rd, "run_meta.json"))
    if res.get("aggregate_tokens_per_sec") is None:
        return None
    t0, t1 = meta.get("window_wall_t0"), meta.get("window_wall_t1")
    win = (t1 - t0) if (t0 and t1) else res.get("window_duration_s")
    gen = res.get("generated_tokens_window") or 0
    E = res.get("energy_window_j")
    p_avg = (E / win) if (E and win) else res.get("avg_power_window_w")
    tr = [r for r in rows(os.path.join(rd, "power_trace.csv"))
          if t0 is None or t0 <= (fnum(r["t_wall"]) or 0) <= t1]
    capn = [r for r in tr if (r.get("clk_reasons") or "").strip()]
    cap_frac = (sum(1 for r in capn if int(r["clk_reasons"]) & CAP_BITS) / len(capn)) if capn else None
    it = [r for r in rows(os.path.join(rd, "iter_log.csv"))
          if t0 is None or (t0 <= (fnum(r["t_start"]) or 0) and (fnum(r["t_end"]) or 0) <= t1)]
    steps = [fnum(r["t_end"]) - fnum(r["t_start"]) for r in it
             if fnum(r["t_end"]) and fnum(r["t_start"]) and fnum(r["t_end"]) > fnum(r["t_start"])]
    kvmax = max([fnum(r.get("kv_cache_pct")) or 0 for r in it], default=None)
    waitf = (sum(1 for r in it if (fnum(r.get("n_waiting")) or 0) > 0) / len(it)) if it else None
    q = sum(fnum(r.get("prefix_queries")) or 0 for r in it); h = sum(fnum(r.get("prefix_hits")) or 0 for r in it)
    rq = rows(os.path.join(rd, "requests.csv"))
    ttft = [fnum(r.get("ttft_s")) for r in rq]
    tpot = []
    ptok, gtok = [], []
    for r in rq:
        e2e, tf, g = fnum(r.get("e2e_s")), fnum(r.get("ttft_s")), fnum(r.get("gen_tokens"))
        if e2e and tf is not None and g and g > 1:
            tpot.append((e2e - tf) / (g - 1))
        if fnum(r.get("prompt_tokens")):
            ptok.append(fnum(r["prompt_tokens"]))
        if g:
            gtok.append(g)
    bt = rows(os.path.join(rd, "binned_table.csv"))
    T = sum(fnum(b["dt_s"]) or 0 for b in bt)
    B = sum((fnum(b["weight_bytes_bin"]) or 0) + (fnum(b["kv_bytes_bin"]) or 0) for b in bt)
    return dict(run_dir=rd, gpu=meta.get("gpu_name"), model=(meta.get("model") or "").split("/")[-1],
                tp=meta.get("tp_degree"), eager=meta.get("enforce_eager"),
                tok_s=res["aggregate_tokens_per_sec"], req_s=(res.get("completed_requests") or 0) / win if win else None,
                j_tok=(p_avg * win / gen) if (p_avg and gen) else None, p_avg=p_avg,
                energy_source=res.get("energy_window_source") or "GetPowerUsage",
                idle=meta.get("idle_power_w"), mbu=(B / T / 8e12 / (meta.get("tp_degree") or 1)) if T else None,
                step_ms=pct(steps, 50) * 1e3 if steps else None, cap_frac=cap_frac,
                kv_max=kvmax, wait_frac=waitf, hit_rate=(h / q) if q else None,
                ttft_p50=pct(ttft, 50), ttft_p99=pct(ttft, 99), tpot_p50=pct(tpot, 50), tpot_p99=pct(tpot, 99),
                mean_prompt=float(np.mean(ptok)) if ptok else None, mean_gen=float(np.mean(gtok)) if gtok else None,
                n_req=len(rq), power_source=(bt[0].get("power_source") if bt else None))


def fit3(rds, use_computed=True):
    X, y = [], []
    for rd in rds:
        idle = jload(os.path.join(rd, "run_meta.json")).get("idle_power_w")
        if idle is None:
            continue
        for b in rows(os.path.join(rd, "binned_table.csv")):
            g = b.get("gemm_flops_bin_computed") if use_computed else None
            g = fnum(g) if g not in (None, "") else fnum(b["gemm_flops_bin"])
            X.append([fnum(b["weight_bytes_bin"]), fnum(b["kv_bytes_bin"]), g])
            y.append(fnum(b["energy_bin_j"]) - idle * fnum(b["dt_s"]))
    if len(y) < 20:
        return None
    X, y = np.array(X), np.array(y)
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - ((y - X @ c) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    rng = np.random.default_rng(0); bs = []
    for _ in range(200):
        i = rng.integers(0, len(y), len(y)); b, *_ = np.linalg.lstsq(X[i], y[i], rcond=None); bs.append(b)
    lo, hi = np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)
    return dict(e_wbyte=c[0], e_kvbyte=c[1], e_gemm=c[2], ci_lo=lo.tolist(), ci_hi=hi.tolist(),
                r2=float(r2), n_bins=len(y))


def fmt(v, f="{:.2f}", none="-"):
    return none if v is None else f.format(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="logs/B200_prov")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = {}
    runs = {}
    for rd in sorted(glob.glob(os.path.join(a.root, "*", "*", "*"))):
        if os.path.basename(rd) == "idle" or not os.path.isdir(rd):
            continue
        s = run_summary(rd)
        if s:
            arm = rd.split(os.sep)[-3]
            runs[(arm, s["model"], os.path.basename(rd))] = s
    out["runs"] = {"/".join(k): v for k, v in runs.items()}

    print("== 1. runs ==")
    print(f"{'arm/model/tag':<44}{'tok/s':>8}{'req/s':>7}{'mJ/tok':>8}{'W':>6}{'MBU':>6}{'step':>6}"
          f"{'cap%':>6}{'KV%':>6}{'wait%':>6}{'TTFT50/99 s':>13}{'TPOT50/99 ms':>14}{'hit%':>6}")
    for k, s in runs.items():
        print(f"{'/'.join(k)[:43]:<44}{s['tok_s']:>8.0f}{fmt(s['req_s'], '{:.1f}'):>7}{fmt(s['j_tok'] and s['j_tok']*1e3, '{:.0f}'):>8}"
              f"{fmt(s['p_avg'], '{:.0f}'):>6}{fmt(s['mbu']):>6}{fmt(s['step_ms'], '{:.1f}'):>6}"
              f"{fmt(s['cap_frac'] and s['cap_frac']*100, '{:.0f}', '0' if s['cap_frac'] == 0 else '-'):>6}"
              f"{fmt(s['kv_max'] and s['kv_max']*100, '{:.0f}'):>6}{fmt(s['wait_frac'] and s['wait_frac']*100, '{:.0f}', '0'):>6}"
              f"{fmt(s['ttft_p50'], '{:.2f}')+'/'+fmt(s['ttft_p99'], '{:.2f}'):>13}"
              f"{fmt(s['tpot_p50'] and s['tpot_p50']*1e3, '{:.1f}')+'/'+fmt(s['tpot_p99'] and s['tpot_p99']*1e3, '{:.1f}'):>14}"
              f"{fmt(s['hit_rate'] and s['hit_rate']*100, '{:.0f}'):>6}")
    srcs = sorted({s["power_source"] for s in runs.values() if s["power_source"]})
    print(f"   bin energy source(s): {srcs}  ('energy' = NVML energy counter, validated against "
          f"POWER_INSTANT; 'inst' = POWER_INSTANT; 'avg' = smoothed GetPowerUsage)")

    # 2. eager vs graphs
    pairs = [(k, runs[("graphs",) + k[1:]]) for k in runs if k[0] == "relog" and ("graphs",) + k[1:] in runs]
    if pairs:
        print("\n== 2. eager vs CUDA graphs (same model, task, concurrency) ==")
        out["graphs_vs_eager"] = []
        for k, g in pairs:
            e = runs[k]
            row = dict(tag=k[2], step_ratio=(g["step_ms"] / e["step_ms"]) if (g["step_ms"] and e["step_ms"]) else None,
                       tok_ratio=g["tok_s"] / e["tok_s"],
                       jtok_ratio=(g["j_tok"] / e["j_tok"]) if (g["j_tok"] and e["j_tok"]) else None,
                       step_eager_ms=e["step_ms"], step_graphs_ms=g["step_ms"], mbu_eager=e["mbu"], mbu_graphs=g["mbu"])
            out["graphs_vs_eager"].append(row)
            print(f"  {k[1]}/{k[2]:<14} step {fmt(e['step_ms'], '{:.1f}')} -> {fmt(g['step_ms'], '{:.1f}')} ms  "
                  f"tok/s x{row['tok_ratio']:.2f}  J/tok x{fmt(row['jtok_ratio'])}  MBU {fmt(e['mbu'])} -> {fmt(g['mbu'])}")

    # 3. refit
    coef = jload("gpu_coefficients.json").get("B200", {})
    steady = jload("microbench/steady_calib_B200.json").get("llmgemm_fit", {}).get("e_flop")
    print("\n== 3. smear-free per-bin refit (Qwen2-7B) vs earlier estimates ==")
    print(f"  per-bin (GetPowerUsage, Sept): e_w {coef.get('e_wbyte', 0):.3e}  e_kv {coef.get('e_kvbyte', 0):.3e}  "
          f"e_g {coef.get('e_gemm', 0)*1e12:.3f} pJ;  steady microbench e_g {fmt(steady and steady*1e12, '{:.3f}')} pJ")
    out["refit"] = {}
    for arm in ("relog", "graphs"):
        rds = [s["run_dir"] for k, s in runs.items() if k[0] == arm and k[1] == "Qwen2-7B-Instruct"]
        f = fit3(rds) if rds else None
        if f:
            out["refit"][arm] = f
            print(f"  {arm:<7} n={f['n_bins']:>5}: e_w {f['e_wbyte']:.3e} [{f['ci_lo'][0]:.3e},{f['ci_hi'][0]:.3e}]  "
                  f"e_kv {f['e_kvbyte']:.3e}  e_g {f['e_gemm']*1e12:.3f} pJ [{f['ci_lo'][2]*1e12:.3f},{f['ci_hi'][2]*1e12:.3f}]  R2 {f['r2']:.3f}")

    # 4. highconc vs model predictions
    hc = {k: s for k, s in runs.items() if k[0] in ("highconc", "graphs")}
    if any(k[0] == "highconc" for k in hc):
        print("\n== 4. B200 measured vs recommender (B200 and H200) at the same load ==")
        try:
            import recommend_gpu as rg
            gpus = rg.load_gpus()
            out["highconc"] = []
            for k, s in sorted(hc.items(), key=lambda kv: (kv[0][1], kv[0][2].split("_c")[0], int(kv[0][2].split("_c")[-1]))):
                c = int(k[2].split("_c")[-1])
                m = rg.resolve_model({"Qwen2-7B-Instruct": "Qwen/Qwen2-7B-Instruct",
                                      "Qwen2.5-32B-Instruct": "Qwen/Qwen2.5-32B-Instruct"}.get(k[1], k[1]))
                P, G = s["mean_prompt"] or 512, s["mean_gen"] or 256
                pb = rg.predict_serving(gpus["B200"], m, P, G, conc=c)
                ph = rg.predict_serving(gpus["H200"], m, P, G, conc=c)
                row = dict(key="/".join(k), conc=c, b200_meas=s["j_tok"], b200_pred=pb["j_tok"], h200_pred=ph["j_tok"],
                           b200_meas_tok_s=s["tok_s"], b200_pred_tok_s=pb["tok_s"], h200_pred_tok_s=ph["tok_s"])
                out["highconc"].append(row)
                print(f"  {'/'.join(k)[:40]:<41} c={c:<4} B200 meas {fmt(s['j_tok'] and s['j_tok']*1e3, '{:.0f}'):>5} mJ/tok "
                      f"(pred {pb['j_tok']*1e3:.0f})  H200 pred {ph['j_tok']*1e3:.0f}  ->  meas B200/pred H200 = "
                      f"{fmt(s['j_tok'] and s['j_tok']/ph['j_tok'])}   tok/s meas {s['tok_s']:.0f} / pred {pb['tok_s']:.0f}")
            print("  (the time model was fitted on EAGER runs at c<=64; compare 'graphs' rows to see its bias)")
        except Exception as e:
            print(f"  [skipped: recommender import failed: {e!r}]")

    # 5. prefill
    pf = {k: s for k, s in runs.items() if k[0] == "prefill"}
    if pf:
        print("\n== 5. prefill-dominated serving ==")
        out["prefill"] = []
        for k, s in pf.items():
            rd = s["run_dir"]; idle = s["idle"] or 0
            bt = rows(os.path.join(rd, "binned_table.csv"))
            E = sum(fnum(b["energy_bin_j"]) for b in bt); T = sum(fnum(b["dt_s"]) for b in bt)
            W = sum(fnum(b["weight_bytes_bin"]) for b in bt); Kb = sum(fnum(b["kv_bytes_bin"]) for b in bt)
            Fg = sum(fnum(b.get("gemm_flops_bin_computed") or b["gemm_flops_bin"]) for b in bt)
            pt = sum(fnum(b.get("prefill_tokens_computed") or b["prefill_tokens"]) for b in bt)
            ew, ek = coef.get("e_wbyte", 1.24e-10), coef.get("e_kvbyte", 2.85e-10)
            eff = (E - idle * T - ew * W - ek * Kb) / Fg if Fg else None
            row = dict(key="/".join(k), j_per_prompt_tok=(E / pt) if pt else None, cap_frac=s["cap_frac"],
                       eff_pj_per_flop=eff and eff * 1e12, p_avg=s["p_avg"])
            out["prefill"].append(row)
            print(f"  {k[2]:<12} J/prompt-token {fmt(row['j_per_prompt_tok'], '{:.4f}')}  P {fmt(s['p_avg'], '{:.0f}')} W  "
                  f"at cap {fmt(s['cap_frac'] and s['cap_frac']*100, '{:.0f}', '0')}%  effective {fmt(row['eff_pj_per_flop'], '{:.3f}')} pJ/FLOP "
                  f"(uncapped steady 0.943; capped GEMM loops 0.57-0.67)")

    # 6. TP
    tp = {k: s for k, s in runs.items() if k[0] in ("tp2", "tp1ref", "graphs")}
    if any(k[0] == "tp2" for k in tp):
        print("\n== 6. TP=2 vs TP=1 (CUDA graphs; TP=2 meters both GPUs) ==")
        out["tp"] = []
        for k, s in tp.items():
            if k[0] != "tp2":
                continue
            ref = tp.get(("tp1ref",) + k[1:]) or tp.get(("graphs",) + k[1:])
            row = dict(key="/".join(k), tp2_jtok=s["j_tok"], tp2_tok_s=s["tok_s"],
                       tp1_jtok=ref and ref["j_tok"], tp1_tok_s=ref and ref["tok_s"])
            out["tp"].append(row)
            print(f"  {k[1]}/{k[2]:<12} TP2 {fmt(s['j_tok'] and s['j_tok']*1e3, '{:.0f}')} mJ/tok, {s['tok_s']:.0f} tok/s"
                  + (f"   TP1 {fmt(ref['j_tok'] and ref['j_tok']*1e3, '{:.0f}')} mJ/tok, {ref['tok_s']:.0f} tok/s"
                     f"   J/tok x{fmt(s['j_tok'] and ref['j_tok'] and s['j_tok']/ref['j_tok'])}" if ref else "   (no TP1 reference)"))

    path = a.out or os.path.join(a.root, "provisioning_summary.json")
    json.dump(out, open(path, "w"), indent=1, default=float)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
