#!/usr/bin/env python3
"""
timemodel_highc.py -- why LIMINAL-E over-predicts THROUGHPUT at high concurrency, and the fix.
Report: TIMEMODEL_HIGHC.md.  Writes timemodel_highc.json.  Run in the container (14B/32B need
transformers to resolve the model config).  CPU only.

  1. Per-step evidence (iter_log.csv; a request's prefill is attributed to the step whose t_end is
     the last before its t_first_token -- that reproduces the logged prefill_tokens on >99% of
     rows).  Residual = measured step - LIMINAL-E step (canonical, no highc) at the step's own
     n_running, scheduled tokens and resident KV.  Steps are split into
       pure   : no request admitted, <= capture_max tokens
       mixed  : >= 1 request admitted (prefill in the step), <= capture_max tokens
       eager  : > capture_max (512) scheduled tokens -> vLLM runs it without CUDA graphs
     Steps inside the [t_submit, report] window of a prompt > 512 tokens are dropped (hidden
     chunked prefill, FINDINGS_B200 UPDATE 6 review).
  2. Hypothesis tests: per-sequence host cost vs KV/attention growth, GEMM growth with M, sampling,
     hidden prefill, the CUDA-graph capture limit and FA2's mixed-batch attention.
  3. Fit (B200 Qwen2-7B high-c ONLY, 6 runs): LAD regression of the residual on
     n_running (tau_seq), [admitted >= 1] (t_mix) and L*[tokens > 512] (dtau_eager); leave-one-run-
     out ranges.
  4. Run-level validation (same runs and measured values as validate_v2.py) before / after, with
     the terms switched on one at a time: tok/s, J/token and power MAPE by group.
"""
import bisect
import csv
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "liminal_energy"))
sys.path.insert(1, HERE)
os.chdir(HERE)

import liminal_e as le            # noqa: E402
import validate_v2 as v2          # noqa: E402

CAP = 512                          # vLLM 0.10.2 max_capture_size (run.log compilation_config)
LONG_PROMPT = 512                  # prompts that may be chunked (budget 2048 minus running decodes)
OUT = os.path.join(HERE, "timemodel_highc.json")
E_G_STEADY_B200 = 0.943e-12        # steady uncapped LLM-GEMM J/FLOP (microbench/steady_calib_B200.json)


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def gkey(name):
    n = (name or "").upper()
    if "B200" in n:
        return "B200"
    if "A100" in n:
        return "A100-80-PCIe"
    if "A6000" in n:
        return "A6000"
    if "H200" in n:
        return "H200"
    return name


# ---------------------------------------------------------------------------------------------
# 1. per-step data
# ---------------------------------------------------------------------------------------------
def load_steps(rd):
    meta = json.load(open(os.path.join(rd, "run_meta.json")))
    w0, w1 = meta.get("window_wall_t0"), meta.get("window_wall_t1")
    R = []
    for r in csv.DictReader(open(os.path.join(rd, "iter_log.csv"))):
        a, b = fnum(r.get("t_start")), fnum(r.get("t_end"))
        if a is None or b is None:
            continue
        pf = fnum(r.get("prefill_tokens")) or 0.0
        h = fnum(r.get("prefix_hits")) or 0.0
        R.append([a, b, fnum(r.get("n_running")) or 0.0, fnum(r.get("decode_tokens")) or 0.0, pf,
                  max(0.0, pf - h), fnum(r.get("kv_tokens_resident")) or 0.0])
    R.sort(key=lambda x: x[1])
    A = np.array(R)
    te = list(A[:, 1])
    nnew = np.zeros(len(A))
    inlong = np.zeros(len(A), bool)
    rq = os.path.join(rd, "requests.csv")
    if os.path.exists(rq):
        for r in csv.DictReader(open(rq)):
            s, f, p = fnum(r.get("t_submit")), fnum(r.get("t_first_token")), fnum(r.get("prompt_tokens"))
            if None in (s, f, p):
                continue
            i = bisect.bisect_right(te, f) - 1
            if i >= 0:
                nnew[i] += 1
                if p > LONG_PROMPT:
                    inlong[bisect.bisect_right(te, s):i] = True
    win = (A[:, 0] >= w0) & (A[:, 1] <= w1) if w0 else np.ones(len(A), bool)
    S = dict(dt=A[:, 1] - A[:, 0], n=A[:, 2], dec=A[:, 3], pf=A[:, 4], pfc=A[:, 5], kv=A[:, 6],
             nnew=nnew, inlong=inlong, win=win, meta=meta, rd=rd)
    S["T"] = S["dec"] + S["pfc"]
    S["g"] = le.gpu(gkey(meta.get("gpu_name")))
    S["m"] = le.model(meta["model"])
    S["tm"] = "eager" if meta.get("enforce_eager", True) is not False else "graphs"
    S["tp"] = int(meta.get("tp_degree") or 1)
    return S


_PC = {}


def predict_steps(S, profile="canonical", highc=None):
    """LIMINAL-E time of every logged step at its own operating point."""
    g, m, tm, tp = S["g"], S["m"], S["tm"], S["tp"]
    kvpt = m.kv_bytes_per_token(2)
    Wb = le.rg.weight_params(m, 1) * 2
    ap, af = m.active_params(), m.attn_flops_per_token(1)
    hk = json.dumps(highc, sort_keys=True, default=str) if isinstance(highc, dict) else highc
    out = np.zeros(len(S["dt"]))
    for k in range(len(out)):
        T, kv, n, mix = S["T"][k], S["kv"][k], S["n"][k], S["nnew"][k] > 0
        key = (g["name"], m.L, ap, tm, tp, profile, hk, int(T), int(kv / 16), int(n), bool(mix))
        if key not in _PC:
            w = dict(Wb=Wb, KVb=kv * kvpt, Fg=2 * ap * T, Fa=af * kv, Fa_pf=0.0, T=T,
                     A=le.allreduce_bytes(m, T, tp) * tp, tokens_out=max(S["dec"][k], 1.0),
                     n_seq=n, KV_dec=kv * kvpt, f_mix=1.0 if mix else 0.0)
            w["F"] = w["Fg"] + w["Fa"]
            _PC[key] = le.step_work(g, m, w, tp, profile=profile, time_model=tm, highc=highc)["t"]
        out[k] = _PC[key]
    return out


def lad(X, y, it=60):
    """Least-absolute-deviation regression by IRLS (robust to logger timing jitter)."""
    w = np.ones(len(y))
    c = np.zeros(X.shape[1])
    for _ in range(it):
        c, *_ = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)
        w = 1.0 / np.maximum(np.abs(y - X @ c), 0.05e-3)
    return c


def classify(S):
    ok = S["win"] & ~S["inlong"] & (S["dt"] > 0)
    pure = ok & (S["nnew"] == 0) & (S["T"] <= CAP)
    mix = ok & (S["nnew"] > 0) & (S["T"] <= CAP)
    eag = ok & (S["T"] > CAP)
    return ok, pure, mix, eag


EVIDENCE_PATTERNS = [
    ("B200", "logs/B200_prov/graphs/Qwen2-7B-Instruct/*_c*"),
    ("B200", "logs/B200_prov/highconc/Qwen2-7B-Instruct/*_c*"),
    ("B200", "logs/B200_prov/graphs/Qwen2.5-32B-Instruct/*_c*"),
    ("B200", "logs/B200_prov/highconc/Qwen2.5-32B-Instruct/*_c*"),
    ("A100", "logs/A100_prov/a100_graphs/Qwen2-7B-Instruct/*_c*"),
    ("A100", "logs/A100_prov/a100_highconc/Qwen2-7B-Instruct/*_c*"),
    ("A100", "logs/A100_prov/a100_graphs/Qwen2.5-14B-Instruct/*_c*"),
    ("A6000", "logs/A6000_prov/gpu_graphs/Qwen2-7B-Instruct/*_c*"),
    ("A6000", "logs/A6000_prov/gpu_highconc/Qwen2-7B-Instruct/*_c*"),
    ("A6000", "logs/A6000_prov/gpu_graphs/Qwen2.5-14B-Instruct/*_c*"),
]


def evidence():
    runs = {}
    for _, pat in EVIDENCE_PATTERNS:
        for rd in sorted(glob.glob(pat)):
            if not os.path.exists(os.path.join(rd, "iter_log.csv")):
                continue
            S = load_steps(rd)
            S["pred"] = predict_steps(S)
            S["res"] = S["dt"] - S["pred"]
            runs[os.path.relpath(rd, "logs")] = S
    table = []
    print("\n== 1. per-step residual (measured - LIMINAL-E), medians in ms ==")
    print("%-52s %5s | %-24s | %-24s | %-24s | %s" % ("run", "n", "pure: N  resid", "mixed: N  resid  (n_adm)",
                                                      "eager >512: N  resid", "FA2 extra-KV pred (mixed)"))
    for key, S in runs.items():
        ok, pure, mix, eag = classify(S)
        g, m = S["g"], S["m"]
        rec = dict(run=key, gpu=g["name"], n=float(S["n"][ok].mean()), tm=S["tm"])
        for lab, sel in (("pure", pure), ("mixed", mix), ("eager", eag)):
            rec[lab] = dict(N=int(sel.sum()), resid_ms=float(np.median(S["res"][sel]) * 1e3) if sel.sum() >= 10 else None,
                            n=float(S["n"][sel].mean()) if sel.sum() else None,
                            adm=float(S["nnew"][sel].mean()) if sel.sum() else None,
                            meas_ms=float(np.median(S["dt"][sel]) * 1e3) if sel.sum() >= 10 else None,
                            pred_ms=float(np.median(S["pred"][sel]) * 1e3) if sel.sum() >= 10 else None)
        # mixed-step penalty over pure steps at the same n (linear baseline in n from pure steps)
        rec["mix_penalty_ms"] = None
        if pure.sum() >= 30 and mix.sum() >= 30 and np.ptp(S["n"][pure]) > 0:
            cb = np.polyfit(S["n"][pure], S["res"][pure], 1)
            pen = S["res"][mix] - np.polyval(cb, S["n"][mix])
            rec["mix_penalty_ms"] = float(np.median(pen) * 1e3)
            # FA2 prediction for the penalty: (g-1)*K_decode at the step's streaming rate
            grp = float(m.n_heads) / m.n_kv_heads
            Kx = (grp - 1.0) * S["kv"][mix] * m.kv_bytes_per_token(2)
            if g.get("at_cap"):
                rate = 1.0 / g["at_cap"]["tau_b"]
            else:
                rate = g["bw"] * le.time_params(g, S["tm"])["eta_kv"]
            rec["fa2_pred_ms"] = float(np.median(Kx) / rate * 1e3)
            rec["fa2_rate_TBs"] = rate / 1e12
        table.append(rec)
        f = lambda d: ("%5d %+7.2f" % (d["N"], d["resid_ms"])) if d["resid_ms"] is not None else "%5d     -  " % d["N"]
        print("%-52s %5.0f | %-24s | %-24s | %-24s | pen %s pred %s" % (
            key, rec["n"], f(rec["pure"]), f(rec["mixed"]) + " (%.1f)" % (rec["mixed"]["adm"] or 0), f(rec["eager"]),
            "%+6.2f" % rec["mix_penalty_ms"] if rec["mix_penalty_ms"] is not None else "  -   ",
            "%6.2f" % rec["fa2_pred_ms"] if rec.get("fa2_pred_ms") is not None else "  -"))
    return runs, table


def hypothesis_tests(runs):
    print("\n== 2. hypothesis tests (B200 Qwen2-7B, graphs + high-c) ==")
    out = {}
    sel = [S for k, S in runs.items() if k.startswith("B200_prov/") and "Qwen2-7B" in k]
    # (a) pure-decode residual vs n and vs resident KV: per-sequence vs KV/attention growth
    X, y, task = [], [], []
    for S in sel:
        ok, pure, mix, eag = classify(S)
        X.append(np.c_[np.ones(pure.sum()), S["n"][pure], S["kv"][pure] / 1e3])
        y.append(S["res"][pure])
        task += [S["meta"].get("task")] * int(pure.sum())
    X, y, task = np.vstack(X), np.concatenate(y), np.array(task)
    c = lad(X, y)
    out["pure_vs_n_kv"] = dict(us_per_seq=c[1] * 1e6, us_per_ktok_kv=c[2] * 1e6, const_ms=c[0] * 1e3, N=len(y))
    print("(a) pure-decode residual = %.3f ms + %.2f us x n_running + %.3f us x (resident KV / 1k tok)   [LAD, N=%d]"
          % (c[0] * 1e3, c[1] * 1e6, c[2] * 1e6, len(y)))
    for t in ("alpaca", "sharegpt"):
        s = task == t
        ct = lad(X[s][:, :2], y[s])
        mctx = float(np.sum(X[s][:, 2] * 1e3) / max(1.0, np.sum(X[s][:, 1])))
        out["pure_slope_" + t] = dict(us_per_seq=ct[1] * 1e6, mean_ctx=mctx)
        print("    %-8s alone: %.2f us per running sequence (mean context %.0f tokens)" % (t, ct[1] * 1e6, mctx))
    # (b) GEMM growth with M: does a prefill token cost more than the model's GEMM term?
    X, y = [], []
    for S in sel:
        ok, pure, mix, eag = classify(S)
        s = ok & (S["T"] <= CAP) & (S["n"] >= 100)
        X.append(np.c_[np.ones(s.sum()), S["n"][s], (S["nnew"][s] > 0).astype(float), S["pfc"][s]])
        y.append(S["res"][s])
    X, y = np.vstack(X), np.concatenate(y)
    c = lad(X, y)
    out["prefill_token_extra_us"] = c[3] * 1e6
    print("(b) extra cost per computed prefill token beyond the model's GEMM term: %+.2f us "
          "(model charges %.1f us/token) -> GEMM growth with M is already covered"
          % (c[3] * 1e6, 2 * sel[0]["m"].active_params() / (sel[0]["g"]["peak_flops"] *
                                                            le.time_params(sel[0]["g"], "graphs")["mu"]) * 1e6))
    # (c) capture limit: discontinuity at 512 scheduled tokens (c=512 runs)
    out["capture_jump"] = {}
    for S in sel:
        if "_c512" not in S["rd"]:
            continue
        ok = S["win"] & ~S["inlong"]
        lo = ok & (S["T"] > CAP - 8) & (S["T"] <= CAP)
        hi = ok & (S["T"] > CAP) & (S["T"] <= CAP + 16)
        if lo.sum() > 20 and hi.sum() > 20:
            j = float(np.median(S["res"][hi]) - np.median(S["res"][lo]))
            out["capture_jump"][os.path.basename(S["rd"])] = dict(ms=j * 1e3, us_per_layer=j / S["m"].L * 1e6,
                                                                  n_lo=int(lo.sum()), n_hi=int(hi.sum()))
            print("(c) %s: residual jumps %+.2f ms (%.1f us/layer) between T in (504,512] (N=%d) and (512,528] (N=%d)"
                  % (os.path.basename(S["rd"]), j * 1e3, j / S["m"].L * 1e6, lo.sum(), hi.sum()))
    # (d) hidden chunked prefill: how many high-c steps sit in a long prompt's window
    for S in sel:
        if "highconc" in S["rd"]:
            ok = S["win"]
            out.setdefault("hidden_prefill_frac", {})[os.path.basename(S["rd"])] = float(S["inlong"][ok].mean())
    print("(d) fraction of high-c steps inside a >512-token prompt's prefill window:",
          {k: round(v, 3) for k, v in out["hidden_prefill_frac"].items()})
    # (e) sampling over the vocabulary: greedy (temperature 0) -> fp32 logits + argmax
    m = sel[0]["m"]
    vb = getattr(m, "vocab", None) or getattr(m, "vocab_size", None) or 152064
    by = vb * (2 + 4 + 4)            # bf16 logits read, fp32 write, fp32 argmax read
    out["sampling_us_per_seq"] = by / (sel[0]["g"]["bw"] * 0.5) * 1e6
    print("(e) greedy sampling over %d logits moves ~%.1f MB/sequence -> %.2f us/seq on B200 at 50%% of peak BW "
          "(an upper bound; the measured excess is ~4 us/seq)" % (vb, by / 1e6, out["sampling_us_per_seq"]))
    return out


# ---------------------------------------------------------------------------------------------
# 3. fit on B200 7B high-c
# ---------------------------------------------------------------------------------------------
def fit_b200(runs):
    keys = [k for k in runs if k.startswith("B200_prov/highconc/Qwen2-7B-Instruct/")]

    def design(S):
        ok = S["win"] & ~S["inlong"] & (S["dt"] > 0)
        X = np.c_[np.ones(ok.sum()), S["n"][ok], (S["nnew"][ok] > 0).astype(float),
                  S["m"].L * (S["T"][ok] > CAP).astype(float)]
        return X, S["res"][ok]

    def fit(ks):
        XY = [design(runs[k]) for k in ks]
        X = np.vstack([a for a, _ in XY])
        y = np.concatenate([b for _, b in XY])
        return lad(X, y), len(y)
    c, N = fit(keys)
    loo = [fit([k for k in keys if k != kk])[0] for kk in keys]
    L = np.array(loo)
    res = dict(runs=keys, N=N, const_ms=c[0] * 1e3, tau_seq_us=c[1] * 1e6, t_mix_ms=c[2] * 1e3,
               dtau_eager_us=c[3] * 1e6,
               loo_range=dict(tau_seq_us=[L[:, 1].min() * 1e6, L[:, 1].max() * 1e6],
                              t_mix_ms=[L[:, 2].min() * 1e3, L[:, 2].max() * 1e3],
                              dtau_eager_us=[L[:, 3].min() * 1e6, L[:, 3].max() * 1e6]))
    # tau_seq alone (the task's single-term hypothesis): slope of the residual on n, nothing else
    XY = [design(runs[k]) for k in keys]
    X = np.vstack([a for a, _ in XY])[:, :2]
    y = np.concatenate([b for _, b in XY])
    c1 = lad(X, y)
    res["tau_seq_only_us"] = c1[1] * 1e6
    res["tau_seq_only_const_ms"] = c1[0] * 1e3
    print("\n== 3. fit on B200 Qwen2-7B high-c only (%d runs, %d steps, LAD) ==" % (len(keys), N))
    print("residual = %.3f ms + tau_seq %.2f us x n + t_mix %.3f ms x [admits] + dtau_eager %.1f us x L x [T>512]"
          % (c[0] * 1e3, c[1] * 1e6, c[2] * 1e3, c[3] * 1e6))
    print("  leave-one-run-out: tau_seq %s us, t_mix %s ms, dtau_eager %s us/layer" % (
        ["%.2f" % v for v in res["loo_range"]["tau_seq_us"]], ["%.3f" % v for v in res["loo_range"]["t_mix_ms"]],
        ["%.1f" % v for v in res["loo_range"]["dtau_eager_us"]]))
    print("  single-term (tau_seq only): %.2f us x n  (const %.3f ms)" % (c1[1] * 1e6, c1[0] * 1e3))
    return res


# ---------------------------------------------------------------------------------------------
# 4. run-level validation
# ---------------------------------------------------------------------------------------------
GROUPS = [
    ("B200 7B high-c c128-512 (FIT SET)", lambda x: x["gpu"] == "B200" and x["arm"] == "highconc" and "7B" in x["model"]),
    ("B200 32B high-c c128-512 (held out)", lambda x: x["gpu"] == "B200" and x["arm"] == "highconc" and "32B" in x["model"]),
    ("B200 graphs c<=64 (7B, 32B)", lambda x: x["gpu"] == "B200" and x["arm"] == "graphs"),
    ("B200 longctx 7B/32B (c<=16)", lambda x: x["gpu"] == "B200" and x["arm"] in ("longctx", "longctx32")),
    ("B200 prefill arm", lambda x: x["gpu"] == "B200" and x["arm"] == "prefill"),
    ("B200 eager relog/loggerab (c<=64)", lambda x: x["gpu"] == "B200" and x["arm"] in ("relog", "loggerab")),
    ("B200 TP2 + TP1 ref", lambda x: x["gpu"] == "B200" and x["arm"] in ("tp2", "tp1ref")),
    ("B200 Sept eager (c<=64)", lambda x: x["arm"] == "sept_B200"),
    ("H200 Sept eager (other node)", lambda x: x["arm"] == "sept_H200"),
    ("A100 high-c c128-256", lambda x: x["gpu"] == "A100-80-PCIe" and x["arm"] == "a100_highconc"),
    ("A100 graphs c<=64 (7B, 14B)", lambda x: x["gpu"] == "A100-80-PCIe" and x["arm"] == "a100_graphs"),
    ("A100 other dense TP1 c<=64", lambda x: x["gpu"] == "A100-80-PCIe" and x["tp"] == 1 and not x["moe"]
     and x["arm"] not in ("a100_highconc", "a100_graphs")),
    ("A100 TP2 PCIe", lambda x: x["gpu"] == "A100-80-PCIe" and x["tp"] == 2),
    ("A100 MoE", lambda x: x["gpu"] == "A100-80-PCIe" and x["moe"]),
    ("A6000 high-c c128-256", lambda x: x["gpu"] == "A6000" and x["arm"] == "gpu_highconc"),
    ("A6000 c<=64 (graphs+eager)", lambda x: x["gpu"] == "A6000" and x["arm"] in ("gpu_graphs", "gpu_eager")),
    ("A6000 longctx", lambda x: x["gpu"] == "A6000" and x["arm"] == "gpu_longctx"),
]


def ape(p, m):
    return 100.0 * abs(p - m) / m


def summarize(rows, pk, mk):
    xs = [(x[pk], x[mk]) for x in rows if x.get(pk) is not None and x.get(mk)]
    if not xs:
        return None
    return dict(n=len(xs), mape=sum(ape(p, m) for p, m in xs) / len(xs),
                bias=sum(100.0 * (p / m - 1) for p, m in xs) / len(xs))


def validate(fit):
    keys = set(r["key"] for r in csv.DictReader(open(os.path.join(HERE, "liminal_energy", "validation_v2_runs.csv"))))
    runs = [r for r in v2.load_all() if r["key"] in keys]       # identical population to validate_v2
    host = dict(tau_seq={"_default": fit["tau_seq_us"] * 1e-6}, t_mix={"_default": fit["t_mix_ms"] * 1e-3},
                dtau_eager={"_default": fit["dtau_eager_us"] * 1e-6})
    off = dict(tau_seq={"_default": 0.0}, t_mix={"_default": 0.0}, capture_max=None, attn={})
    full = dict(host, capture_max=CAP, attn=le.HIGHC["attn"])
    variants = [
        ("base", None),
        ("tau_seq only", dict(off, tau_seq={"_default": fit["tau_seq_only_us"] * 1e-6})),
        ("B200 terms (tau_seq+t_mix+capture)", dict(host, attn={})),
        ("FA2 mixed only", dict(off, attn=le.HIGHC["attn"])),
        ("full (B200 terms + FA2)", full),
        ("full + B200 e_g 0.943", full),          # energy sensitivity: steady uncapped GEMM J/FLOP
    ]
    rows = []
    for r in runs:
        row = dict(key=r["key"], gpu=r["gpu"], arm=r["arm"], model=r["model"].split("/")[-1], conc=r["conc"],
                   tp=r["tp"], moe=le.rg.is_moe(r["m"]), tok_meas=r["tok_s_meas"], j_meas=r["j_meas"],
                   p_meas=r["p_meas"])
        profs = ["canonical"] + (["zeroshot"] if r["gpu"] in ("A100-80-PCIe", "A6000") else [])
        for prof in profs:
            for vn, hc in variants:
                kw = {}
                if vn.endswith("e_g 0.943") and r["gpu"] == "B200":
                    pp = le.profile_params(le.gpu("B200"), prof)
                    kw["coef"] = (pp["e_w"], pp["e_kv"], E_G_STEADY_B200, pp["p_s"], pp["p0"])
                    kw["cap_mode"] = pp["cap_mode"]
                p = v2.predict(r, profile=prof, highc=hc, **kw)
                tag = "%s|%s" % (prof, vn)
                row["tok|" + tag], row["j|" + tag], row["p|" + tag] = p["tok_s"], p["j_per_token"], p["power"]
        rows.append(row)
    summ = {}
    print("\n== 4. run-level validation: MAPE (bias) %; tok/s | J/token | power ==")
    for prof in ("canonical", "zeroshot"):
        print("--- profile %s" % prof)
        print("%-40s %3s | " % ("group", "n") + " | ".join("%-27s" % vn[:27] for vn, _ in variants))
        for gname, f in GROUPS:
            sel = [x for x in rows if f(x) and ("tok|%s|base" % prof) in x]
            if not sel:
                continue
            cells = []
            for vn, _ in variants:
                tag = "%s|%s" % (prof, vn)
                st = {k: summarize(sel, "%s|%s" % (k, tag), mk) for k, mk in (("tok", "tok_meas"), ("j", "j_meas"),
                                                                               ("p", "p_meas"))}
                summ.setdefault(prof, {}).setdefault(gname, {})[vn] = st
                cells.append("%4.1f(%+5.1f) %4.1f(%+5.1f) %3.1f" % (st["tok"]["mape"], st["tok"]["bias"], st["j"]["mape"],
                                                                   st["j"]["bias"], st["p"]["mape"]))
            print("%-40s %3d | " % (gname[:40], len(sel)) + " | ".join(cells))
    print("\nper-run, high-c (tok/s meas | base | full;  J/token meas | base | full; canonical, zero-shot for A100/A6000):")
    for x in rows:
        if x["arm"] in ("highconc", "a100_highconc", "gpu_highconc"):
            prof = "zeroshot" if x["gpu"] in ("A100-80-PCIe", "A6000") else "canonical"
            b, fu = "%s|base" % prof, "%s|full (B200 terms + FA2)" % prof
            print("  %-6s %-30s c=%3s  tok %7.0f | %7.0f (%+5.1f) | %7.0f (%+5.1f)   J %.4f | %.4f (%+5.1f) | %.4f (%+5.1f)" % (
                x["gpu"][:6], x["model"] + " " + x["key"].split("/")[-1], x["conc"], x["tok_meas"],
                x["tok|" + b], 100 * (x["tok|" + b] / x["tok_meas"] - 1), x["tok|" + fu], 100 * (x["tok|" + fu] / x["tok_meas"] - 1),
                x["j_meas"], x["j|" + b], 100 * (x["j|" + b] / x["j_meas"] - 1), x["j|" + fu], 100 * (x["j|" + fu] / x["j_meas"] - 1)))
    return rows, summ, [vn for vn, _ in variants]


def step_level_check(runs, fit):
    """Mean measured step vs mean predicted step at the logged operating points (no steady-state
    approximation): base vs the full model with the per-step indicators (admits, T > 512)."""
    hc = dict(tau_seq={"_default": fit["tau_seq_us"] * 1e-6}, t_mix={"_default": fit["t_mix_ms"] * 1e-3},
              dtau_eager={"_default": fit["dtau_eager_us"] * 1e-6}, capture_max=CAP)
    print("\n== 5. step level: mean step time at the logged operating points, ms (meas | base | full) ==")
    out = {}
    for key, S in runs.items():
        ok = S["win"] & (S["dt"] > 0)
        prof = "zeroshot" if S["g"]["name"] in ("A100-80-PCIe", "A6000") else "canonical"
        pb = predict_steps(S, prof) if prof != "canonical" else S["pred"]
        pf = predict_steps(S, prof, highc=hc)
        mm, b, f = S["dt"][ok].mean() * 1e3, pb[ok].mean() * 1e3, pf[ok].mean() * 1e3
        out[key] = dict(meas=mm, base=b, full=f, profile=prof)
        print("  %-52s %7.2f | %7.2f (%+5.1f%%) | %7.2f (%+5.1f%%)" % (key, mm, b, 100 * (b / mm - 1), f, 100 * (f / mm - 1)))
    return out


def main():
    runs, table = evidence()
    tests = hypothesis_tests(runs)
    fit = fit_b200(runs)
    stp = step_level_check(runs, fit)
    rows, summ, vnames = validate(fit)
    json.dump(dict(evidence=table, tests=tests, fit=fit, step_level=stp, validation=summ, variants=vnames,
                   runs=rows, capture_max=CAP), open(OUT, "w"), indent=1, default=float)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
