#!/usr/bin/env python3
"""
ncu_hierarchy.py

Parse an Nsight Compute CSV covering a batch=1 decode and answer two questions
that the NVML-based energy model cannot answer on its own.

Q1 (HANDOFF gotcha #5): is the ANALYTIC byte model right?
    Every coefficient in this project is joules per *modelled* byte --
    `weight_bytes = active_params x dtype`, asserted, never measured. Compare the
    hardware DRAM byte counters against that assertion.

Q2 (the leading explanation for the B200 result): does our lumped `e_wbyte`
    actually measure the WHOLE memory hierarchy rather than DRAM alone?
    A weight byte pulled from HBM must also cross L2, L1 and shared memory to
    reach the tensor cores, and each level costs energy. If so:

        e_wbyte(lumped)  ~=  SUM_level  (bytes_at_level / analytic_bytes) x e_level

    where e_level comes from a hardware-counter decomposition. We use the B200
    coefficients a colleague fit from GEMM microbenchmarks (group project space,
    2026-07-21). If that sum reproduces our measured ~1.19e-10 J/byte, the
    scaling law plausibly governs only the DRAM component (~0.66e-10, which does
    match the datasheet prediction of 0.646e-10) and our lumped coefficient was
    simply measuring something broader than the law describes.

Usage:
    python3 ncu_hierarchy.py --csv ncu_metrics.csv --run_dir <dir> \
        --model Qwen/Qwen2-7B-Instruct --prompt_len 8 --gen_tokens 24
"""
import argparse
import csv
import json
import os

UNIT_SCALE = {"byte": 1, "bytes": 1, "b": 1, "kbyte": 1e3, "mbyte": 1e6,
              "gbyte": 1e9, "tbyte": 1e12, "kib": 1024, "mib": 1024**2,
              "gib": 1024**3}

# NCU metric -> the channel it represents. Names verified against a colleague's
# B200 capture, so they are known to exist on sm_100.
METRICS = {
    "dram__bytes.sum":                              "dram",
    "dram__bytes_read.sum":                         "dram_read",
    "dram__bytes_write.sum":                        "dram_write",
    "lts__t_bytes.sum":                             "l2",
    "l1tex__t_bytes.sum":                           "l1",
    "l1tex__m_xbar2l1tex_read_bytes_pipe_tma.sum":  "tma",
    "sm__sass_data_bytes_mem_shared.sum":           "smem",
}

# B200 per-channel energy, J/byte, from the group's hardware-counter fit
# (results/coefficients/pooled_phys3_ols.json, 200 dense GEMM shapes, R2=0.999).
# NOTE: their workload is isolated GEMMs in bf16/fp32, not fp16 LLM serving, and
# their own WLS variant disagrees with OLS by ~15%. Treat as indicative.
VGAO_OLS = {"dram": 6.616e-11, "l2": 1.714e-11, "l1": 1.277e-11,
            "smem": 1.447e-11, "tma": 0.152e-11}
VGAO_WLS = {"dram": 5.832e-11, "l2": 3.848e-11, "l1": -0.268e-11,
            "smem": 1.723e-11, "tma": -1.365e-11}

# What we measured on B200 with NVML + analytic bytes (logs/B200, 3-term fit).
OUR_E_WBYTE = 1.189e-10
SCALING_LAW_PREDICTION = 0.646e-10   # H200 1.077e-10 x (4.8/8.0)


def _num(s):
    try:
        return float((s or "").strip().replace(",", ""))
    except ValueError:
        return None


def parse_csv(path):
    """Sum each metric across all profiled kernels. NCU 'raw page' CSV is long
    format: one row per kernel x metric."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    hdr_i = next((i for i, r in enumerate(rows)
                  if any("Metric Name" in c for c in r)), None)
    if hdr_i is None:
        raise SystemExit(f"no 'Metric Name' header found in {path} -- "
                         "did ncu fail? check the log for ERR_NVGPUCTRPERM")
    hdr = rows[hdr_i]
    ni = next(j for j, c in enumerate(hdr) if "Metric Name" in c)
    ui = next((j for j, c in enumerate(hdr) if "Metric Unit" in c), None)
    # ncu emits two shapes. Raw (one row per kernel LAUNCH) has "Metric Value".
    # With --print-summary it aggregates into Minimum/Maximum/Average plus an
    # "Invocations" count, and the total is Average x Invocations.
    vi = next((j for j, c in enumerate(hdr) if "Metric Value" in c), None)
    ai = next((j for j, c in enumerate(hdr) if c.strip() == "Average"), None)
    ii = next((j for j, c in enumerate(hdr) if "Invocations" in c), None)
    if vi is None and (ai is None or ii is None):
        raise SystemExit("CSV has neither 'Metric Value' nor "
                         "'Average'+'Invocations' columns: " + ",".join(hdr))
    totals, counts = {}, {}
    for r in rows[hdr_i + 1:]:
        need = vi if vi is not None else max(ai, ii)
        if len(r) <= max(ni, need):
            continue
        chan = METRICS.get(r[ni].strip())
        if chan is None:
            continue
        if vi is not None:
            v = _num(r[vi])
        else:
            avg, inv = _num(r[ai]), _num(r[ii])
            v = avg * inv if (avg is not None and inv is not None) else None
        if v is None:
            continue
        unit = (r[ui].strip().lower() if ui is not None and len(r) > ui else "")
        totals[chan] = totals.get(chan, 0.0) + v * UNIT_SCALE.get(unit, 1)
        counts[chan] = counts.get(chan, 0) + 1
    return totals, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--run_dir", default="logs/ncu_hier")
    ap.add_argument("--model", default="Qwen/Qwen2-7B-Instruct")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--prompt_len", type=int, default=8)
    ap.add_argument("--gen_tokens", type=int, default=24)
    ap.add_argument("--ncu_rc", type=int, default=0,
                    help="ncu's exit code; nonzero means the profile is partial")
    ap.add_argument("--expect_kernels", type=int, default=0,
                    help="approx kernel launches expected; warns if far fewer")
    args = ap.parse_args()

    totals, counts = parse_csv(args.csv)
    if not totals:
        raise SystemExit("no target metrics found in the CSV")

    ntok_p = os.path.join(args.run_dir, "n_tokens.txt")
    gen = int(open(ntok_p).read().strip()) if os.path.exists(ntok_p) else args.gen_tokens

    from models import model_from_hf_id
    m = model_from_hf_id(args.model)
    dtb = 2 if args.dtype in ("float16", "bfloat16") else 4
    ctx = args.prompt_len + gen // 2
    # One forward pass per generated token re-reads all weights; KV grows with ctx.
    # PASS COUNT: vLLM's prefill emits the FIRST output token, then each decode
    # step emits one more. max_tokens=N therefore means N forward passes, not
    # N+1. (An earlier version used gen+1 and got a 17% discrepancy.)
    passes = gen
    # EMBEDDING TABLE: active_params() counts embed_tokens AND lm_head. lm_head
    # really is streamed (a GEMM over the vocab), but embed_tokens during decode
    # is a LOOKUP of one row (~7 KB), not a 1.09 GB read. Counting it inflates
    # the expected weight traffic by ~7% for Qwen2-7B.
    embed_bytes = m.vocab * m.d_model * dtb
    weight_streamed = m.active_params() * dtb - embed_bytes
    analytic_per_tok = weight_streamed + ctx * m.kv_bytes_per_token(dtb)
    analytic_total = analytic_per_tok * passes
    naive_total = (m.active_params() * dtb + ctx * m.kv_bytes_per_token(dtb)) * passes

    print(f"model            : {args.model}")
    print(f"kernels profiled : {max(counts.values())} (per metric)")
    print(f"generated tokens : {gen}  => {passes} forward passes "
          f"(prefill emits token 1)")
    print(f"weights streamed : {weight_streamed/1e9:.3f} GB/pass "
          f"(active_params x{dtb} minus the {embed_bytes/1e9:.2f} GB embed table)")
    print(f"analytic bytes   : {analytic_total/1e9:.1f} GB total "
          f"({naive_total/1e9:.1f} GB if the embed table were counted)\n")

    print("=== measured traffic per level (whole profile) ===")
    print(f"{'channel':<12}{'bytes':>14}{'GB':>10}{'per analytic byte':>20}")
    for ch in ("dram", "dram_read", "dram_write", "l2", "l1", "tma", "smem"):
        if ch not in totals:
            continue
        v = totals[ch]
        print(f"{ch:<12}{v:>14.4g}{v/1e9:>10.1f}{v/analytic_total:>20.3f}")

    dram = totals.get("dram") or (totals.get("dram_read", 0) + totals.get("dram_write", 0))
    ratio = dram / analytic_total

    # ---- completeness guard -------------------------------------------------
    # A truncated profile silently produces a tiny ratio that LOOKS like a real
    # finding ("caches absorb everything"). It is not: it just means ncu never
    # got through the kernels. Refuse to interpret a partial capture.
    n_kernels = max(counts.values())
    partial = []
    if args.ncu_rc != 0:
        partial.append(f"ncu exited {args.ncu_rc} (124 = timeout)")
    if args.expect_kernels and n_kernels < 0.7 * args.expect_kernels:
        partial.append(f"only {n_kernels} kernel launches captured, "
                       f"expected ~{args.expect_kernels}")
    if ratio < 0.5:
        partial.append(f"DRAM/analytic = {ratio:.3f}: a decode CANNOT serve "
                       f"{m.active_params()*dtb/1e9:.0f} GB of weights from a "
                       f"~126 MB L2, so this is physically impossible")
    if partial:
        print("\n" + "=" * 68)
        print("PROFILE INCOMPLETE -- NOT INTERPRETABLE. Reasons:")
        for r in partial:
            print("  * " + r)
        print("Numbers below are from a PARTIAL capture. Do not quote them.")
        print("=" * 68)

    print("\n=== Q1: is the analytic byte model right? (gotcha #5) ===")
    print(f"  measured DRAM / analytic = {ratio:.3f}")
    if 0.8 <= ratio <= 1.25:
        print("  => ANALYTIC MODEL VALIDATED. The denominator under every")
        print("     coefficient in this project is sound.")
    elif ratio > 1.25:
        print(f"  => analytic UNDERCOUNTS by {ratio:.2f}x. Every J/analytic-byte")
        print(f"     figure is inflated by that factor; true J/DRAM-byte is")
        print(f"     {OUR_E_WBYTE/ratio*1e10:.3f}e-10 (vs {SCALING_LAW_PREDICTION*1e10:.3f}e-10 predicted).")
    else:
        print(f"  => analytic OVERCOUNTS ({ratio:.2f}x) -- caches absorbing reuse.")

    print("\n=== Q2: does the lumped e_wbyte span the whole hierarchy? ===")
    for name, coeffs in (("OLS", VGAO_OLS), ("WLS", VGAO_WLS)):
        tot, parts = 0.0, []
        for ch, e in coeffs.items():
            b = totals.get(ch)
            if b is None:
                continue
            contrib = (b / analytic_total) * e
            tot += contrib
            parts.append(f"{ch}={contrib*1e10:.3f}")
        if tot:
            err = 100 * (tot - OUR_E_WBYTE) / OUR_E_WBYTE
            print(f"  [{name}] predicted e_wbyte = {tot*1e10:.3f}e-10   "
                  f"({' '.join(parts)})")
            print(f"        vs our measured {OUR_E_WBYTE*1e10:.3f}e-10  -> {err:+.0f}%")
    dram_only = (dram / analytic_total) * VGAO_OLS["dram"]
    print(f"  DRAM channel alone       = {dram_only*1e10:.3f}e-10  "
          f"vs scaling-law {SCALING_LAW_PREDICTION*1e10:.3f}e-10 "
          f"-> {100*(dram_only-SCALING_LAW_PREDICTION)/SCALING_LAW_PREDICTION:+.0f}%")
    print("\n  If the full-hierarchy sum lands near our measured value while the")
    print("  DRAM-only term lands near the scaling-law prediction, then the law")
    print("  governs DRAM only and our analytic-byte coefficient is broader.")

    out = {"model": args.model, "gen_tokens": gen, "weight_sweeps": passes,
           "analytic_bytes_per_token": analytic_per_tok,
           "analytic_bytes_total": analytic_total,
           "measured": totals, "kernels": counts,
           "dram_over_analytic": ratio,
           "our_e_wbyte": OUR_E_WBYTE,
           "scaling_law_prediction": SCALING_LAW_PREDICTION}
    os.makedirs(args.run_dir, exist_ok=True)
    p = os.path.join(args.run_dir, "ncu_hierarchy.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
