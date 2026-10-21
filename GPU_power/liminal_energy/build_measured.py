#!/usr/bin/env python3
"""
build_measured.py -- derive LIMINAL-E's MICROBENCH-ONLY (zero-shot) GPU parameters from the
steady microbenchmark files, plus idle states and datasheet values.  No serving data is used.
Writes liminal_energy/gpus_measured.json (the "zeroshot" block of each GPU).  validate_v2.py
later adds a "calibrated" block fitted on serving runs.  Pure python (3.6 ok).

Per GPU:
  idle_vllm  : vLLM loaded, idle (logs/*_prov/idle_states.json)   -> reported, provisioning
  idle_clk   : clocks-up idle (results_*_steady.json idle_p0_pre)  -> P_s while serving
  grid       : uncapped occupancy grid, P - idle_clk = P_0 + e_b*BW   (memory-activity term P_0)
  duty       : duty-cycle full-clock J/B, J/FLOP (dram_stream, gemv_b1, gemm_fp16_4096, L2, L1)
  at_cap     : capped LLM-shaped kernels (llmgemm M=1..8192; + gemv if <4 small-M capped),
               all at P_cap: seconds per byte = softmax_k(tau_b, I*tau_f), I = FLOP/B
               (an at-cap roofline; least squares in log byte rate, k on a grid)
               -> capped streaming rate 1/tau_b, capped FLOP rate 1/tau_f,
                  at-cap effective J/B = (P_cap - idle_clk)*tau_b, J/FLOP = (P_cap - idle_clk)*tau_f
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MB = os.path.join(ROOT, "microbench")

DATASHEET = {
    "A100-80-PCIe": dict(gpu_name="NVIDIA A100 80GB PCIe", mem_tech="HBM2e", bw=1.935e12, peak_flops=3.12e14,
                         p_cap=300.0, mem=80e9, steady="results_A100_steady.json",
                         idle_states="logs/A100_prov/idle_states.json", interconnect="PCIe Gen4, no NVLink"),
    "A6000": dict(gpu_name="NVIDIA RTX A6000", mem_tech="GDDR6", bw=0.768e12, peak_flops=1.55e14,
                  p_cap=300.0, mem=48e9, steady="results_A6000_steady.json",
                  idle_states="logs/A6000_prov/idle_states.json", interconnect="PCIe"),
    "B200": dict(gpu_name="NVIDIA B200", mem_tech="HBM3e", bw=8.0e12, peak_flops=2.25e15, p_cap=1000.0,
                 mem=180e9, steady="results_B200_steady.json", idle_states=None, interconnect="NVLink"),
}
# B200 duty-cycle fits live in the earlier microbench file (microbench/results_B200.json)
B200_DUTY_FILE = "results_B200.json"


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def rep_mean(rec, key):
    return mean(r.get(key) or 0.0 for r in rec["repeats"])


def softk(a, b, k):
    m = max(a, b)
    return m * ((a / m) ** k + (b / m) ** k) ** (1.0 / k)


def fit_cap_roof(rows):
    """At-cap roofline fitted to capped kernels: seconds per byte at intensity I is
    softmax_k(tau_b, I*tau_f).  rows = [(bytes_per_s, flops_per_s)].  Minimizes the squared
    log error of the predicted byte rate; grid over k, coordinate search over log tau."""
    import math
    best = None
    for k in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0):
        lb = -math.log(max(b for b, f in rows))
        lf = -math.log(max(f for b, f in rows))
        step = 0.5

        def loss(lb, lf):
            return sum(math.log(b * softk(math.exp(lb), (f / b) * math.exp(lf), k)) ** 2 for b, f in rows)
        cur = loss(lb, lf)
        while step > 1e-5:
            moved = False
            for db, df in ((step, 0), (-step, 0), (0, step), (0, -step)):
                v = loss(lb + db, lf + df)
                if v < cur:
                    lb, lf, cur, moved = lb + db, lf + df, v, True
            if not moved:
                step /= 2
        if best is None or cur < best[0]:
            best = (cur, k, math.exp(lb), math.exp(lf))
    cur, k, tb, tf = best
    return tb, tf, k, (cur / len(rows)) ** 0.5


def lsq2(rows):
    """min sum (1 - a*x - b*y)^2 -> (a, b)."""
    sxx = sum(x * x for x, y in rows)
    syy = sum(y * y for x, y in rows)
    sxy = sum(x * y for x, y in rows)
    sx = sum(x for x, y in rows)
    sy = sum(y for x, y in rows)
    det = sxx * syy - sxy * sxy
    return (sx * syy - sy * sxy) / det, (sy * sxx - sx * sxy) / det


def build(name, ds):
    st = json.load(open(os.path.join(MB, ds["steady"])))
    idle_clk = st["idle_p0_pre"]["p_inst_w"]
    idle_deep = st["idle_pre"]["p_inst_w"]
    duty_src = st if "duty" in st else json.load(open(os.path.join(MB, B200_DUTY_FILE)))
    duty = {k: v["fit_pooled"]["e"] for k, v in duty_src["duty"].items()}
    duty_r2 = {k: v["fit_pooled"]["r2"] for k, v in duty_src["duty"].items()}
    # occupancy grid, uncapped points, relative to clocks-up idle
    pts = []
    for p in st["grid"]["points"]:
        if rep_mean(p, "frac_sw_power_cap") > 0.05:
            continue
        pts.append((rep_mean(p, "bytes_per_s"), rep_mean(p, "p_w") - idle_clk))
    n = len(pts)
    mx, my = mean(x for x, _ in pts), mean(y for _, y in pts)
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    e_b = sum((x - mx) * (y - my) for x, y in pts) / sxx
    p0 = my - e_b * mx
    # at-cap fit on capped LLM-shaped kernels
    cap_rows, cap_pts = [], []
    capped_llm = [r for r in st.get("llmgemm", []) if rep_mean(r, "frac_sw_power_cap") >= 0.5]
    small_m = [r for r in capped_llm if r.get("M", 0) <= 64]
    recs = capped_llm + (st.get("gemv", []) if len(small_m) < 4 else [])   # B200: few small-M capped
    for rec in recs:
        if rep_mean(rec, "frac_sw_power_cap") < 0.5:
            continue
        b, f = rep_mean(rec, "bytes_per_s"), rep_mean(rec, "flops_per_s")
        cap_rows.append((b, f))
        cap_pts.append(dict(label=rec["label"], bytes_per_s=b, flops_per_s=f, sm_mhz=rep_mean(rec, "sm_mhz"),
                            p_w=rep_mean(rec, "p_w")))
    tau_b, tau_f, k_cap, rmsle = fit_cap_roof(cap_rows)
    budget = ds["p_cap"] - idle_clk
    # uncapped llmgemm points (B200 only) for the figure
    unc_pts = [dict(label=rec["label"], bytes_per_s=rep_mean(rec, "bytes_per_s"),
                    flops_per_s=rep_mean(rec, "flops_per_s"), p_w=rep_mean(rec, "p_w"))
               for rec in st.get("llmgemm", []) if rep_mean(rec, "frac_sw_power_cap") < 0.5]
    idle_vllm = None
    if ds["idle_states"]:
        idle_vllm = json.load(open(os.path.join(ROOT, ds["idle_states"])))["3_vllm_loaded"]["p_inst_w"]
    out = {k: v for k, v in ds.items() if k not in ("steady", "idle_states")}
    out.update(
        source=dict(steady="microbench/" + ds["steady"], idle_states=ds["idle_states"],
                    duty=("microbench/" + B200_DUTY_FILE) if "duty" not in st else "microbench/" + ds["steady"]),
        idle_deep=idle_deep, idle_clk=idle_clk, idle_vllm=idle_vllm,
        duty=duty, duty_r2=duty_r2,
        grid=dict(e_b=e_b, p0=p0, n_uncapped=n, points=pts),
        at_cap=dict(tau_b=tau_b, tau_f=tau_f, k=k_cap, bw_cap=1.0 / tau_b, flops_cap=1.0 / tau_f,
                    eps_b=budget * tau_b, eps_f=budget * tau_f, budget_w=budget, n=len(cap_rows),
                    rmsle=rmsle, points=cap_pts,
                    form="seconds/byte at the cap = softmax_k(tau_b, I*tau_f), I = FLOP/byte"),
        uncapped_llmgemm=unc_pts)
    # ZERO-SHOT serving coefficients (pre-registered rule, see LIMINAL_E.md sec. 10):
    #   P_s = idle_clk, activity intercept P_0 = grid intercept, e_w = grid slope,
    #   e_kv = 1.25 * e_w (B200 settled ratio 1.2-1.3, transferred as a prior), e_g = duty GEMM.
    out["zeroshot"] = dict(p_static=idle_clk, p0=p0, e_wbyte=e_b, e_kvbyte=1.25 * e_b,
                           e_gemm=duty["gemm_fp16_4096"],
                           alt_gemv=dict(p_static=idle_vllm or idle_clk, p0=0.0, e_wbyte=duty["gemv_b1"],
                                         e_kvbyte=1.25 * duty["gemv_b1"], e_gemm=duty["gemm_fp16_4096"]))
    return out


def main():
    res = {"_meta": dict(script="liminal_energy/build_measured.py",
                         note="zeroshot block: microbench + datasheet + idle only. calibrated block: "
                              "validate_v2.py (serving runs). Does NOT modify gpu_coefficients.json.")}
    old = {}
    path = os.path.join(HERE, "gpus_measured.json")
    if os.path.exists(path):
        old = json.load(open(path))
    for name, ds in DATASHEET.items():
        res[name] = build(name, ds)
        if name in old and "calibrated" in old[name]:
            res[name]["calibrated"] = old[name]["calibrated"]
    # B200 canonical serving coefficients: KV SETTLED (FINDINGS_B200.md "UPDATE 6 review";
    # logs/B200_prov/longctx_summary.json, 7B clean decode bins, free 3-term fit, R2 0.90,
    # corr(KV,FLOP) 0.21; e_kv/e_w 1.18 [1.16,1.20]; slope method 1.60-1.61e-10).
    lc = json.load(open(os.path.join(ROOT, "logs/B200_prov/longctx_summary.json")))
    f7 = lc["longctx/Qwen2-7B-Instruct"]["decode_only"]
    res["B200"]["calibrated"] = dict(
        e_wbyte=f7["e_wbyte"], e_kvbyte=f7["e_kvbyte"], e_gemm=f7["e_gemm"], p_static=237.822, p0=0.0,
        cap_mode="stretch",   # validated best on B200 capped runs (validation.json, validation_v2.json)
        src="B200 long-context clean-decode fit (longctx_summary.json; UPDATE 6 review)",
        e_kvbyte_slope=[lc["longctx/Qwen2-7B-Instruct"]["slope"][c]["e_kvbyte"] for c in ("4", "16")] +
        [lc["longctx32/Qwen2.5-32B-Instruct"]["slope"]["4"]["e_kvbyte"]])
    for k in old:
        if k not in res:
            res[k] = old[k]
    json.dump(res, open(path, "w"), indent=1)
    for name in DATASHEET:
        g = res[name]
        z, c = g["zeroshot"], g["at_cap"]
        print("%-13s idle deep/clk/vllm %5.1f/%5.1f/%s W | grid P0 %5.1f W + %.3e J/B (n=%d) | duty gemv %.3e "
              "stream %.3e gemm %.3g pJ | at-cap: %.2f TB/s (%.0f%% BW), %.0f TF (%.0f%%), eps_b %.3e J/B, "
              "eps_f %.3g pJ, k=%.1f (n=%d, rmsle %.3f)" % (
                  name, g["idle_deep"], g["idle_clk"], g["idle_vllm"], g["grid"]["p0"], g["grid"]["e_b"],
                  g["grid"]["n_uncapped"], g["duty"]["gemv_b1"], g["duty"]["dram_stream"],
                  g["duty"]["gemm_fp16_4096"] * 1e12, c["bw_cap"] / 1e12, 100 * c["bw_cap"] / g["bw"],
                  c["flops_cap"] / 1e12, 100 * c["flops_cap"] / g["peak_flops"], c["eps_b"],
                  c["eps_f"] * 1e12, c["k"], c["n"], c["rmsle"]))


if __name__ == "__main__":
    main()
