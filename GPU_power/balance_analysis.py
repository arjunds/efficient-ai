#!/usr/bin/env python3
"""Balance / overprovisioning analysis (PI note: "is the system overprovisioned or
balanced, on energy as well as performance?").

Data in hand only (no GPU): gpu_coefficients.json (3-term per-bin fit), the binned
serving runs (logs/...), iter_log.csv step times, results.json request counts, and the
recommender's realized-utilization time model (recommend_gpu.py) for counterfactuals.

Sections
  1. ridges        performance ridge peak/BW vs energy ridge e_byte/e_flop (FLOP/B).
                   In decode, FLOP per weight byte ~= tokens per step, so both ridges
                   read directly as a batch size.
  2. runs          per run: realized BW, MBU, MFU, arithmetic intensity, tok/s,
                   requests/s, J/token, J/request, median step time, and the energy
                   split static | weights | KV | GEMM | residual.
  3. matched       H200 vs B200 on the same (model, task, concurrency): J/token ratio
                   factored into static (idle ratio / speedup) and dynamic parts, and
                   "capability conversion" = realized speedup / BW ratio.
  4. slo           time-model counterfactual: at a TPOT SLO, the largest batch each GPU
                   can run, its J/token and split, and the BREAK-EVEN idle power at
                   which the bigger GPU stops winning. If the bigger GPU's real idle
                   exceeds break-even, it is overprovisioned for that workload.
  5. runlevel      smear-free cross-check: 3-term OLS on whole-run sums (45 s windows
                   integrate out the NVML smoothing / lag), vs the per-bin fit.

  python3 balance_analysis.py [--fig plots_proposal/fig9_balance.png]
  -> balance_analysis.json, balance_runs.csv (+ figure if matplotlib is importable)
"""
import argparse, csv, glob, json, os
import numpy as np

import reconcile_gpus as rc
import recommend_gpu as rg

COEF = json.load(open("gpu_coefficients.json"))
MEASURED = [g for g in ("H200", "B200") if g in COEF]


# --------------------------------------------------------------------------- 1
def ridges(gpus):
    out = {}
    for n, g in gpus.items():
        if not g.get("peak_flops") or not g.get("e_gemm"):
            continue
        perf = g["peak_flops"] / g["bw"]
        energy = g["e_wbyte"] / g["e_gemm"]
        out[n] = dict(measured=bool(g.get("measured")), mem_tech=g.get("mem_tech"),
                      perf_ridge=perf, energy_ridge=energy, ratio=energy / perf,
                      e_wbyte=g["e_wbyte"], e_gemm=g["e_gemm"], p_static=g["p_static"],
                      bw=g["bw"], peak_flops=g["peak_flops"],
                      idle_w_per_tbs=g["p_static"] / (g["bw"] / 1e12),
                      idle_w_per_pflops=g["p_static"] / (g["peak_flops"] / 1e15))
    return out


# --------------------------------------------------------------------------- 2
def step_time(rd, t0, t1):
    p = os.path.join(rd, "iter_log.csv")
    if not os.path.exists(p):
        return None
    dts = []
    for r in csv.DictReader(open(p)):
        try:
            a, b = float(r["t_start"]), float(r["t_end"])
        except (KeyError, ValueError):
            continue
        if t0 <= a and b <= t1 and b > a:
            dts.append(b - a)
    return float(np.median(dts)) if dts else None


def run_rows(include_invalid=False):
    rows = []
    for gpu in MEASURED:
        grp, c = rc.GROUPS[gpu], COEF[gpu]
        ew, ek, eg = c["e_wbyte"], c["e_kvbyte"], c["e_gemm"]
        for root in grp["roots"]:
            for rd in sorted(glob.glob(os.path.join(root, "*", "*"))):
                if not os.path.isdir(rd):
                    continue
                r = rc.run_record(rd)
                if not r or r["conc"] is None:
                    continue
                res = json.load(open(os.path.join(rd, "results.json")))
                meta = json.load(open(os.path.join(rd, "run_meta.json")))
                T = r["T"].sum(); W, K, G, Y = r["W"].sum(), r["K"].sum(), r["G"].sum(), r["Y"].sum()
                E_static = r["idle"] * T
                E_tot = E_static + Y
                win = res.get("window_duration_s") or T
                ntok = res.get("generated_tokens_window") or 0
                nreq = res.get("completed_requests") or 0
                p_avg = res.get("avg_power_window_w")
                e_w, e_k, e_g = ew * W, ek * K, eg * G
                # validity: a stalled load generator leaves few active bins and power
                # at/below idle (2 B200 sharegpt runs: 17 requests, P_avg < idle)
                cover = T / win if win else 0.0
                valid = cover >= 0.5 and (p_avg or 0) > r["idle"]
                rows.append(dict(valid=valid, active_cover=cover,
                    gpu=gpu, root=root, model=r["model"], task=r["task"], conc=r["conc"],
                    idle_w=r["idle"], p_avg_w=p_avg,
                    realized_bw_tbs=(W + K) / T / 1e12, mbu=(W + K) / T / grp["bw"],
                    realized_tflops=G / T / 1e12, mfu=G / T / grp["peak_flops"],
                    ai_flop_per_byte=G / (W + K), ai_flop_per_wbyte=G / W,
                    tok_s=res.get("aggregate_tokens_per_sec"), req_s=nreq / win if win else None,
                    j_tok=(p_avg * win / ntok) if (p_avg and ntok) else None,
                    j_req=(p_avg * win / nreq) if (p_avg and nreq) else None,
                    step_ms=(lambda s: s * 1e3 if s else None)(
                        step_time(rd, meta.get("window_wall_t0", 0), meta.get("window_wall_t1", 9e18))),
                    share_static=E_static / E_tot, share_weights=e_w / E_tot,
                    share_kv=e_k / E_tot, share_gemm=e_g / E_tot,
                    share_resid=(Y - e_w - e_k - e_g) / E_tot,
                    # sums kept for section 5
                    _W=W, _K=K, _G=G, _Y=Y, _T=T, run_dir=rd))
    bad = [r for r in rows if not r["valid"]]
    for r in bad:
        print(f"  [excluded] {r['run_dir']}: active cover {r['active_cover']:.0%}, "
              f"P_avg {r['p_avg_w']:.0f} W vs idle {r['idle_w']:.0f} W, tok/s {r['tok_s']:.0f}")
    return rows if include_invalid else [r for r in rows if r["valid"]]


# --------------------------------------------------------------------------- 3
def matched(rows):
    idx = {(r["gpu"], r["model"], r["task"], r["conc"]): r for r in rows}
    out = []
    for (g, m, t, c), h in sorted(idx.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][3])):
        if g != "H200" or ("B200", m, t, c) not in idx:
            continue
        b = idx[("B200", m, t, c)]
        if not (h["tok_s"] and b["tok_s"] and h["j_tok"] and b["j_tok"]):
            continue
        sp = b["tok_s"] / h["tok_s"]
        st_h, st_b = h["idle_w"] / h["tok_s"], b["idle_w"] / b["tok_s"]
        dy_h, dy_b = h["j_tok"] - st_h, b["j_tok"] - st_b
        out.append(dict(model=m, task=t, conc=c, speedup=sp,
                        bw_ratio=rc.GROUPS["B200"]["bw"] / rc.GROUPS["H200"]["bw"],
                        conversion=sp / (rc.GROUPS["B200"]["bw"] / rc.GROUPS["H200"]["bw"]),
                        idle_ratio=b["idle_w"] / h["idle_w"],
                        jtok_h=h["j_tok"], jtok_b=b["j_tok"], jtok_ratio=b["j_tok"] / h["j_tok"],
                        static_jtok_ratio=st_b / st_h, dyn_jtok_ratio=dy_b / dy_h,
                        static_share_h=st_h / h["j_tok"], static_share_b=st_b / b["j_tok"],
                        step_ms_h=h["step_ms"], step_ms_b=b["step_ms"],
                        req_s_h=h["req_s"], req_s_b=b["req_s"]))
    return out


# --------------------------------------------------------------------------- 4
def decode_at(g, m, batch, prompt, gen):
    return rg.phase_predict(g, m, "decode", prompt, gen, batch, rg.central_draw(g))


def max_batch_under_slo(g, m, slo_s, prompt, gen, bmax=4096):
    """Largest decode batch with per-token latency (= iteration time) <= SLO on ONE GPU
    (iteration time is monotone in batch -> bisection)."""
    def ok(b):
        r = decode_at(g, m, b, prompt, gen)
        return r is not None and r["n_gpu"] == 1 and r["t"] / gen <= slo_s
    if not ok(1):
        return None
    lo, hi = 1, 2
    while hi <= bmax and ok(hi):
        lo, hi = hi, hi * 2
    hi = min(hi, bmax + 1)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo, decode_at(g, m, lo, prompt, gen)


def split(g, m, batch, prompt, gen):
    """J/token split for a decode batch (central coefficients)."""
    r = decode_at(g, m, batch, prompt, gen)
    kvpt = m.kv_bytes_per_token(rg.DTYPE_B)
    Wb = rg.weight_params(m, batch) * rg.DTYPE_B
    KVb = batch * (prompt + gen / 2.0) * kvpt
    Fg = 2.0 * m.active_params() * batch
    t_it = r["t"] / gen
    parts = dict(static=g["p_static"] * t_it, weights=g["e_wbyte"] * Wb,
                 kv=g["e_kvbyte"] * KVb, gemm=g["e_gemm"] * Fg)
    tot = sum(parts.values())
    return {k: v / tot for k, v in parts.items()}, tot / batch, t_it


def slo_study(gpus, models, slos, prompt=1024, gen=256):
    out = []
    for mid in models:
        m = rg.resolve_model(mid)
        for slo in slos:
            row = dict(model=mid, slo_ms=slo * 1e3)
            for n in ("H200", "B200"):
                mb = max_batch_under_slo(gpus[n], m, slo, prompt, gen)
                if mb is None:
                    row[n] = None
                    continue
                b, r = mb
                sh, jt, t_it = split(gpus[n], m, b, prompt, gen)
                row[n] = dict(batch=b, j_tok=r["j_tok"], tok_s=b / t_it, t_iter_ms=t_it * 1e3,
                              mbu=r["mbu"], power=r["power"], bound=r["bound"], shares=sh)
            h, b = row["H200"], row["B200"]
            if h and b:
                # break-even B200 idle: J/tok_B(P_s') = J/tok_H  ->  P_s' = P_s + (J_H - J_B)*tok_s_B
                row["b200_breakeven_idle_w"] = gpus["B200"]["p_static"] + (h["j_tok"] - b["j_tok"]) * b["tok_s"]
                row["b200_actual_idle_w"] = gpus["B200"]["p_static"]
                row["jtok_ratio_b_over_h"] = b["j_tok"] / h["j_tok"]
                # same demand (H200's SLO batch) served on B200: B200 runs faster than needed
                same = decode_at(gpus["B200"], m, h["batch"], prompt, gen)
                row["b200_at_h200_batch_jtok_ratio"] = same["j_tok"] / h["j_tok"]
            out.append(row)
    return out


def what_if(gpus, models, slo, prompt=1024, gen=256):
    """H200 with one capability scaled (idle held): how much J/token falls -> the idle
    power increase that upgrade could afford before it raises J/token."""
    base = gpus["H200"]
    knobs = {"BW x1.5, TDP held 700 W": dict(bw=base["bw"] * 1.5),
             "BW x1.5, TDP 1000 W": dict(bw=base["bw"] * 1.5, p_cap=1000.0),
             "FLOPs x2": dict(peak_flops=base["peak_flops"] * 2),
             "B200 caps (8 TB/s, 2.25 PF, 1000 W)": dict(bw=8e12, peak_flops=2.25e15, p_cap=1000.0)}
    out = []
    for mid in models:
        m = rg.resolve_model(mid)
        mb0 = max_batch_under_slo(base, m, slo, prompt, gen)
        if mb0 is None:
            continue
        b0, r0 = mb0
        for k, mod in knobs.items():
            g = dict(base, **mod)
            mb = max_batch_under_slo(g, m, slo, prompt, gen)
            if mb is None:
                continue
            b, r = mb
            tok_s = b * gen / r["t"]
            out.append(dict(model=mid, knob=k, batch0=b0, batch=b, jtok0=r0["j_tok"], jtok=r["j_tok"],
                            capped=r["capped"], power=r["power"],
                            jtok_change=r["j_tok"] / r0["j_tok"] - 1,
                            affordable_idle_increase_w=(r0["j_tok"] - r["j_tok"]) * tok_s))
    return out


# --------------------------------------------------------------------------- 5
def runlevel_fit(rows, gpu, roots=None, n_boot=500, seed=0):
    rr = [r for r in rows if r["gpu"] == gpu and (roots is None or r["root"] in roots)]
    X = np.array([[r["_W"], r["_K"], r["_G"]] for r in rr]); y = np.array([r["_Y"] for r in rr])
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    yh = X @ c
    r2 = 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    mape = float(np.mean(np.abs((y - yh) / y)) * 100)
    rng = np.random.default_rng(seed); B = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y)); b, *_ = np.linalg.lstsq(X[i], y[i], rcond=None); B.append(b)
    B = np.array(B)
    lo, hi = np.percentile(B, 2.5, 0), np.percentile(B, 97.5, 0)
    # per-bin coefficients predicting run totals
    cb = np.array([COEF[gpu]["e_wbyte"], COEF[gpu]["e_kvbyte"], COEF[gpu]["e_gemm"]])
    pb = X @ cb
    # collinearity check: KV bytes and GEMM FLOPs both grow with concurrency across runs,
    # so also fit with e_kv held at the per-bin value (2 free terms)
    y2 = y - cb[1] * X[:, 1]
    c2, *_ = np.linalg.lstsq(X[:, [0, 2]], y2, rcond=None)
    B2 = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y)); b, *_ = np.linalg.lstsq(X[i][:, [0, 2]], y2[i], rcond=None); B2.append(b)
    B2 = np.array(B2)
    kvfix = dict(e_wbyte=c2[0], e_gemm=c2[1],
                 e_gemm_ci=[float(np.percentile(B2[:, 1], 2.5)), float(np.percentile(B2[:, 1], 97.5))],
                 ratio_e_gemm=c2[1] / cb[2], ratio_e_wbyte=c2[0] / cb[0],
                 corr_kv_gemm=float(np.corrcoef(X[:, 1], X[:, 2])[0, 1]))
    return dict(n_runs=len(rr), e_wbyte=c[0], e_kvbyte=c[1], e_gemm=c[2], kv_fixed=kvfix,
                ci=[[lo[i], hi[i]] for i in range(3)], r2=float(r2), mape=mape,
                perbin_coef_runtotal_mape=float(np.mean(np.abs((y - pb) / y)) * 100),
                perbin_coef_runtotal_bias=float(np.sum(pb) / np.sum(y) - 1),
                ratio_vs_perbin=dict(e_wbyte=c[0] / cb[0], e_kvbyte=c[1] / cb[1], e_gemm=c[2] / cb[2]))


# --------------------------------------------------------------------------- report
def agg(rows, gpu):
    """Mean over tasks per (model, conc)."""
    d = {}
    for r in rows:
        if r["gpu"] == gpu:
            d.setdefault((r["model"], r["conc"]), []).append(r)
    out = []
    for (m, c), rs in sorted(d.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        f = lambda k: float(np.mean([x[k] for x in rs if x[k] is not None])) if any(x[k] is not None for x in rs) else None
        out.append(dict(model=m, conc=c, **{k: f(k) for k in (
            "realized_bw_tbs", "mbu", "mfu", "ai_flop_per_wbyte", "tok_s", "req_s", "j_tok", "j_req",
            "step_ms", "share_static", "share_weights", "share_kv", "share_gemm", "share_resid")}))
    return out


def figure(path, rows, rid, slo_rows, slo_ms_fig):
    import sys
    try:
        import matplotlib
    except ImportError:
        # pydeps only AFTER all transformers use (its hf-hub 1.x breaks transformers);
        # prepended so its pyparsing wins over the container's system one
        sys.path.insert(0, "/shared_data0/adsampat/pydeps")
        import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.0))
    # (a) energy shares vs concurrency, 7B-class mean, both GPUs
    a = ax[0]; parts = ["share_static", "share_weights", "share_kv", "share_gemm", "share_resid"]
    cols = ["#888888", "#1f77b4", "#9467bd", "#d62728", "#dddddd"]
    labels = ["static (idle)", "weight bytes", "KV bytes", "GEMM FLOPs", "residual"]
    xs, ticks = [], []
    k = 0
    for gpu in MEASURED:
        for c in (1, 4, 16, 64):
            rs = [r for r in rows if r["gpu"] == gpu and r["conc"] == c and r["model"] in
                  ("Qwen2-7B-Instruct", "Mistral-7B-v0.1")]
            if not rs:
                continue
            bottom = 0.0
            for p, col, lab in zip(parts, cols, labels):
                v = float(np.mean([r[p] for r in rs]))
                a.bar(k, v, bottom=bottom if v >= 0 else 0, color=col, label=lab if k == 0 else None)
                bottom += max(v, 0)
            ticks.append(f"{gpu}\nc={c}"); xs.append(k); k += 1
        k += 0.6
    a.set_xticks(xs); a.set_xticklabels(ticks, fontsize=8); a.set_ylabel("share of run energy")
    a.set_title("(a) where the energy goes (7B models, measured)"); a.set_ylim(-0.05, 1.22)
    a.legend(fontsize=7, loc="upper center", ncol=5, columnspacing=0.8, handlelength=1.2)
    # (b) operating points vs ridges
    b = ax[1]
    for gpu, mk in (("H200", "o"), ("B200", "s")):
        rs = [r for r in rows if r["gpu"] == gpu]
        sc = b.scatter([r["ai_flop_per_wbyte"] for r in rs], [r["mbu"] for r in rs], marker=mk,
                       c=[np.log2(r["conc"]) for r in rs], cmap="viridis", s=22, label=gpu, alpha=.8)
        b.axvline(rid[gpu]["perf_ridge"], ls="--", color="k" if gpu == "H200" else "r", lw=1)
        b.axvline(rid[gpu]["energy_ridge"], ls=":", color="k" if gpu == "H200" else "r", lw=1.5)
    b.set_xscale("log"); b.set_xlabel("arithmetic intensity, GEMM FLOP / weight byte (~ tokens/step)")
    b.set_ylabel("realized MBU"); b.legend(fontsize=8)
    b.set_title("(b) operating points; -- perf ridge, : energy ridge\n(black H200, red B200; color = log2 conc)", fontsize=9)
    # (c) break-even idle
    c = ax[2]
    sel = [r for r in slo_rows if abs(r["slo_ms"] - slo_ms_fig) < 1e-6 and r.get("b200_breakeven_idle_w")]
    names = [r["model"].split("/")[-1].replace("-Instruct", "") for r in sel]
    c.bar(range(len(sel)), [r["b200_breakeven_idle_w"] for r in sel], color="#2ca02c", label="break-even B200 idle")
    c.axhline(COEF["B200"]["p_static"], color="r", ls="--", label=f"B200 measured idle {COEF['B200']['p_static']:.0f} W")
    c.axhline(COEF["H200"]["p_static"], color="k", ls=":", label=f"H200 measured idle {COEF['H200']['p_static']:.0f} W")
    c.set_xticks(range(len(sel))); c.set_xticklabels(names, rotation=30, fontsize=8)
    c.set_ylabel("W"); c.legend(fontsize=7)
    c.set_title(f"(c) decode at TPOT<={slo_ms_fig:.0f} ms, max batch per GPU:\nB200 wins only if its idle < green", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=130)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", default="plots_proposal/fig9_balance.png")
    ap.add_argument("--out", default="balance_analysis.json")
    a = ap.parse_args()
    gpus = rg.load_gpus()
    R = {}

    rid = ridges(gpus); R["ridges"] = rid
    print("== 1. ridges (FLOP/byte; decode: ~ tokens per step) ==")
    print(f"{'gpu':<14}{'mem':<7}{'perf':>7}{'energy':>8}{'E/P':>6}{'idle W':>8}{'W/(TB/s)':>10}{'W/PFLOPs':>10}")
    for n, r in rid.items():
        print(f"{n:<14}{r['mem_tech']:<7}{r['perf_ridge']:>7.0f}{r['energy_ridge']:>8.0f}{r['ratio']:>6.2f}"
              f"{r['p_static']:>8.0f}{r['idle_w_per_tbs']:>10.1f}{r['idle_w_per_pflops']:>10.1f}"
              + ("" if r["measured"] else "   (prior)"))

    allrows = run_rows(include_invalid=True)
    rows = [r for r in allrows if r["valid"]]
    R["excluded_runs"] = [r["run_dir"] for r in allrows if not r["valid"]]
    keep = [k for k in allrows[0] if not k.startswith("_")]
    with open("balance_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keep); w.writeheader()
        for r in allrows:
            w.writerow({k: r[k] for k in keep})
    print(f"\n== 2. runs ({len(rows)}; per-run table -> balance_runs.csv; mean over tasks below) ==")
    R["runs_by_model_conc"] = {}
    for gpu in MEASURED:
        A = agg(rows, gpu); R["runs_by_model_conc"][gpu] = A
        print(f"-- {gpu}")
        print(f"{'model':<22}{'c':>3}{'TB/s':>6}{'MBU':>5}{'MFU':>6}{'AI':>6}{'tok/s':>7}{'req/s':>6}"
              f"{'mJ/tok':>8}{'J/req':>8}{'step':>6} | {'stat':>5}{'wts':>5}{'kv':>5}{'gemm':>5}{'res':>6}")
        for r in A:
            print(f"{r['model'][:21]:<22}{r['conc']:>3}{r['realized_bw_tbs']:>6.2f}{r['mbu']:>5.2f}{r['mfu']:>6.3f}"
                  f"{r['ai_flop_per_wbyte']:>6.0f}{r['tok_s']:>7.0f}{(r['req_s'] or 0):>6.1f}{r['j_tok']*1e3:>8.0f}"
                  f"{(r['j_req'] or 0):>8.1f}{(r['step_ms'] or 0):>6.1f} | {r['share_static']:>5.2f}"
                  f"{r['share_weights']:>5.2f}{r['share_kv']:>5.2f}{r['share_gemm']:>5.2f}{r['share_resid']:>6.2f}")

    M = matched(rows); R["matched"] = M
    print("\n== 3. matched H200 vs B200 (same model/task/conc) ==")
    print(f"{'model':<22}{'task':<9}{'c':>3}{'speedup':>8}{'conv':>6}{'J/tok B/H':>10}{'static':>8}{'dyn':>6}"
          f"{'stat%H':>7}{'stat%B':>7}")
    for r in M:
        print(f"{r['model'][:21]:<22}{r['task']:<9}{r['conc']:>3}{r['speedup']:>8.2f}{r['conversion']:>6.2f}"
              f"{r['jtok_ratio']:>10.2f}{r['static_jtok_ratio']:>8.2f}{r['dyn_jtok_ratio']:>6.2f}"
              f"{r['static_share_h']*100:>6.0f}%{r['static_share_b']*100:>6.0f}%")
    if M:
        R["matched_summary"] = {c: dict(
            speedup=float(np.mean([r["speedup"] for r in M if r["conc"] == c])),
            jtok_ratio=float(np.mean([r["jtok_ratio"] for r in M if r["conc"] == c])),
            static_jtok_ratio=float(np.mean([r["static_jtok_ratio"] for r in M if r["conc"] == c])),
            dyn_jtok_ratio=float(np.mean([r["dyn_jtok_ratio"] for r in M if r["conc"] == c])))
            for c in sorted({r["conc"] for r in M})}
        print("  mean by conc:", {c: {k: round(v, 2) for k, v in d.items()} for c, d in R["matched_summary"].items()})

    models = ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-32B-Instruct",
              "Qwen/Qwen2-72B-Instruct"]
    slos = [0.020, 0.035, 0.050, 0.100]
    S = slo_study(gpus, models, slos); R["slo"] = S
    print("\n== 4. decode at a TPOT SLO (prompt 1024, gen 256; max batch on 1 GPU; time model) ==")
    print(f"{'model':<24}{'SLO':>5} | {'H b':>5}{'mJ/tok':>7}{'stat%':>6}{'MBU':>5} | {'B b':>5}{'mJ/tok':>7}"
          f"{'stat%':>6}{'MBU':>5} | {'B/H':>5}{'B@Hb':>6}{'BE idle':>8}")
    for r in S:
        h, b = r["H200"], r["B200"]
        if not (h and b):
            print(f"{r['model'].split('/')[-1]:<24}{r['slo_ms']:>5.0f} | infeasible on one GPU"); continue
        print(f"{r['model'].split('/')[-1]:<24}{r['slo_ms']:>5.0f} | {h['batch']:>5}{h['j_tok']*1e3:>7.1f}"
              f"{h['shares']['static']*100:>5.0f}%{h['mbu']:>5.2f} | {b['batch']:>5}{b['j_tok']*1e3:>7.1f}"
              f"{b['shares']['static']*100:>5.0f}%{b['mbu']:>5.2f} | {r['jtok_ratio_b_over_h']:>5.2f}"
              f"{r['b200_at_h200_batch_jtok_ratio']:>6.2f}{r['b200_breakeven_idle_w']:>8.0f}")

    WI = what_if(gpus, models, 0.050); R["what_if_h200_slo50"] = WI
    print("\n== 4b. what-if on H200 at TPOT<=50 ms (idle held): J/token change and affordable idle increase ==")
    for r in WI:
        print(f"{r['model'].split('/')[-1]:<24}{r['knob']:<36} batch {r['batch0']:>4}->{r['batch']:<5}"
              f"J/tok {r['jtok_change']*100:+6.1f}%  P {r['power']:.0f} W{' CAPPED' if r['capped'] else '       '}"
              f"  affordable +{r['affordable_idle_increase_w']:.0f} W idle")

    print("\n== 5. run-level (smear-free) 3-term fit vs per-bin ==")
    R["runlevel"] = {}
    for gpu in MEASURED:
        for lab, roots in (("fit_root", [rc.GROUPS[gpu]["fit_root"]]), ("all_roots", None)):
            f = runlevel_fit(rows, gpu, roots); R["runlevel"][f"{gpu}/{lab}"] = f
            print(f"{gpu:<5}{lab:<10} n={f['n_runs']:>3}  e_w={f['e_wbyte']:.3e} [{f['ci'][0][0]:.3e},{f['ci'][0][1]:.3e}]"
                  f"  e_kv={f['e_kvbyte']:.3e}  e_g={f['e_gemm']*1e12:.3f} pJ [{f['ci'][2][0]*1e12:.3f},{f['ci'][2][1]*1e12:.3f}]"
                  f"  R2={f['r2']:.3f} MAPE={f['mape']:.1f}%  | ratio vs per-bin: w {f['ratio_vs_perbin']['e_wbyte']:.3f}"
                  f" kv {f['ratio_vs_perbin']['e_kvbyte']:.2f} g {f['ratio_vs_perbin']['e_gemm']:.2f}"
                  f" | per-bin coef on run totals: MAPE {f['perbin_coef_runtotal_mape']:.1f}% bias {f['perbin_coef_runtotal_bias']*100:+.1f}%")
            k = f["kv_fixed"]
            print(f"{'':<15}e_kv held at per-bin: e_g={k['e_gemm']*1e12:.3f} pJ [{k['e_gemm_ci'][0]*1e12:.3f},"
                  f"{k['e_gemm_ci'][1]*1e12:.3f}] (x{k['ratio_e_gemm']:.2f})  e_w x{k['ratio_e_wbyte']:.3f}"
                  f"   corr(KV,FLOP) across runs {k['corr_kv_gemm']:.2f}")

    json.dump(R, open(a.out, "w"), indent=1, default=float)
    print("wrote", a.out, "balance_runs.csv")
    try:
        figure(a.fig, rows, rid, S, 50.0)
    except ImportError:
        print("(matplotlib not importable; skipped figure)")


if __name__ == "__main__":
    main()
