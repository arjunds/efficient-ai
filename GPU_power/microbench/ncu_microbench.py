#!/usr/bin/env python3
"""
ncu_microbench.py -- hardware byte counts for the energy microbenchmarks.

Energy is measured WITHOUT a profiler (energy_microbench.py); this script runs
the *same* ops under Nsight Compute so each op's analytic "bytes" can be split
into what actually crossed DRAM vs L2 vs L1. Joined by op name.

  # 1) permission probe (tiny kernel):
  ncu --metrics dram__bytes_read.sum python3 microbench/ncu_microbench.py --op probe
  #    -> "ERR_NVGPUCTRPERM" in the output means counters are admin-locked.

  # 2) one report per op (profiler range = the measured launches only):
  for op in dram_stream l2_stream_half l1_stream_8KBperprog gemv_b1 gemm_fp16_4096; do
    ncu --profile-from-start off --cache-control none --clock-control none \
        --metrics dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,l1tex__t_bytes.sum \
        --csv --page raw --log-file microbench/ncu/$op.csv \
        python3 microbench/ncu_microbench.py --op $op --launches 3 --meta microbench/ncu/$op.json
  done
  python3 microbench/ncu_microbench.py --parse microbench/ncu    # table + ncu_summary.json

--cache-control none keeps L2 warm between kernels (realistic for back-to-back
launches); --clock-control none leaves clocks as they are (the energy runs are
not clock-locked either). Re-run with --cache-control all for the cold-L2 bound.
"""
import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

UNIT = {"byte": 1, "bytes": 1, "b": 1, "kbyte": 1e3, "mbyte": 1e6, "gbyte": 1e9,
        "tbyte": 1e12, "kib": 1024, "mib": 1024 ** 2, "gib": 1024 ** 3}
METRICS = ["dram__bytes_read.sum", "dram__bytes_write.sum", "lts__t_bytes.sum",
           "l1tex__t_bytes.sum"]


def run_op(args):
    import torch
    import energy_microbench as em
    if args.op == "probe":
        x = torch.ones(1 << 20, device="cuda"); (x * 2).sum().item()
        print("probe kernel ran"); return
    fac, unit = em.DUTY_OPS[args.op]
    op = fac()
    op.tune(0.002)                      # ~2 ms per launch: a few hundred replays max
    op(); torch.cuda.synchronize()      # warm (compile/autotune) OUTSIDE the range
    torch.cuda.profiler.start()
    for _ in range(args.launches):
        op()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    meta = {"op": args.op, "unit": unit, "launches": args.launches,
            "analytic_bytes_total": op.bytes_per_call * args.launches,
            "analytic_flops_total": op.flops_per_call * args.launches,
            "gpu": torch.cuda.get_device_name(0),
            "l2_bytes": getattr(torch.cuda.get_device_properties(0), "L2_cache_size", None)}
    if args.meta:
        os.makedirs(os.path.dirname(args.meta) or ".", exist_ok=True)
        json.dump(meta, open(args.meta, "w"), indent=1)
    print(json.dumps(meta))


def sum_metrics(csv_path):
    """Sum each metric over all profiled kernels in an ncu --csv --page raw file
    (wide format: one row per kernel, metric names in the header row, units in
    the 2nd row)."""
    rows = list(csv.reader(open(csv_path, newline="", errors="replace")))
    hi = next((i for i, r in enumerate(rows) if "Kernel Name" in r or "ID" in r[:3]), None)
    if hi is None:
        return None, 0
    hdr = rows[hi]
    units = rows[hi + 1] if hi + 1 < len(rows) else [""] * len(hdr)
    has_units = any(u.strip().lower() in UNIT for u in units)
    body = rows[hi + 2:] if has_units else rows[hi + 1:]
    out = {m: 0.0 for m in METRICS}
    nk = 0
    for r in body:
        if len(r) < len(hdr):
            continue
        nk += 1
        for m in METRICS:
            if m in hdr:
                j = hdr.index(m)
                try:
                    v = float(r[j].replace(",", ""))
                except ValueError:
                    continue
                u = units[j].strip().lower() if has_units else "byte"
                out[m] += v * UNIT.get(u, 1)
    return out, nk


def parse_dir(d):
    summ = {}
    print(f"{'op':<22}{'kernels':>8}{'analytic GB':>13}{'DRAM rd+wr/an':>15}"
          f"{'L2 (lts)/an':>13}{'L1tex/an':>10}")
    for mp in sorted(glob.glob(os.path.join(d, "*.json"))):
        meta = json.load(open(mp))
        cp = mp[:-5] + ".csv"
        if not os.path.exists(cp):
            continue
        tot, nk = sum_metrics(cp)
        if not tot:
            print(f"{meta['op']:<22} (no metrics -- check {cp} for ERR_NVGPUCTRPERM)"); continue
        a = meta["analytic_bytes_total"]
        dram = tot["dram__bytes_read.sum"] + tot["dram__bytes_write.sum"]
        rec = {**meta, **tot, "kernels": nk, "dram_over_analytic": dram / a,
               "l2_over_analytic": tot["lts__t_bytes.sum"] / a,
               "l1_over_analytic": tot["l1tex__t_bytes.sum"] / a}
        summ[meta["op"]] = rec
        print(f"{meta['op']:<22}{nk:>8}{a/1e9:>13.3f}{rec['dram_over_analytic']:>15.3f}"
              f"{rec['l2_over_analytic']:>13.3f}{rec['l1_over_analytic']:>10.3f}")
    json.dump(summ, open(os.path.join(d, "ncu_summary.json"), "w"), indent=1)
    print("wrote", os.path.join(d, "ncu_summary.json"))
    print("\nEnergy attribution: J per *DRAM* byte = (J/analytic byte from "
          "energy_microbench) / dram_over_analytic.")


def parse_serving(csv_path, model, n_tokens, ctx, dtype="float16"):
    """ncu CSV from ncu_dram_check.py --mode generate (batch-1 decode, n_tokens
    generated + 1 prefill) -> hardware bytes per token vs analytic weight+KV."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from energy_model import model_byte_constants
    wb, kvb, src = model_byte_constants({"model": model, "dtype": dtype})
    tot, nk = sum_metrics(csv_path)
    sweeps = n_tokens            # 1 prefill (-> token 1) + (n-1) decode steps = n weight sweeps
    analytic = sweeps * wb + n_tokens * ctx * kvb
    dram = tot["dram__bytes_read.sum"] + tot["dram__bytes_write.sum"]
    rec = {"model": model, "byte_source": src, "kernels": nk, "n_tokens": n_tokens, "ctx": ctx,
           "analytic_bytes": analytic, **tot, "dram_over_analytic": dram / analytic,
           "dram_write_frac": tot["dram__bytes_write.sum"] / dram if dram else None,
           "l2_over_analytic": tot["lts__t_bytes.sum"] / analytic,
           "l1_over_analytic": tot["l1tex__t_bytes.sum"] / analytic}
    print(json.dumps(rec, indent=1))
    json.dump(rec, open(os.path.splitext(csv_path)[0] + "_serving_summary.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serving_csv")
    ap.add_argument("--model", default="Qwen/Qwen2-7B-Instruct")
    ap.add_argument("--n_tokens", type=int, default=24)
    ap.add_argument("--ctx", type=int, default=20, help="mean context during decode")
    ap.add_argument("--op")
    ap.add_argument("--launches", type=int, default=3)
    ap.add_argument("--meta")
    ap.add_argument("--parse")
    a = ap.parse_args()
    if a.serving_csv:
        parse_serving(a.serving_csv, a.model, a.n_tokens, a.ctx)
    elif a.parse:
        parse_dir(a.parse)
    else:
        run_op(a)


if __name__ == "__main__":
    main()
