#!/usr/bin/env python3
"""
predict_moe.py

The #6 held-out transferability test. Fit (e_bit, e_flop) on the DENSE 4-model
H200 data (logs/ragged), then PREDICT the MoE's per-bin dynamic energy from those
coefficients — the MoE was never in the fit and has a different architecture
(active_params != total, MoE routing). If it predicts well, the coefficients are
hardware constants and the analytic byte/FLOP accounting (which uses active_params)
handles the architecture difference.

  python3 predict_moe.py --dense logs/ragged --moe logs/ragged_moe/Qwen3-30B-A3B
"""
import argparse, csv, glob, json, os
from diagnose_fit import ls2, r2, mape


def load(run_glob):
    by, fl, dyn = [], [], []
    for rd in glob.glob(run_glob):
        tbl = os.path.join(rd, "binned_table.csv"); mp = os.path.join(rd, "run_meta.json")
        if not (os.path.exists(tbl) and os.path.exists(mp)):
            continue
        idle = json.load(open(mp)).get("idle_power_w")
        if idle is None:
            continue
        for r in csv.DictReader(open(tbl)):
            try:
                by.append(float(r["bytes_bin"])); fl.append(float(r["flops_bin"]))
                dyn.append(float(r["energy_bin_j"]) - idle * float(r["dt_s"]))
            except (ValueError, KeyError):
                pass
    return by, fl, dyn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", default="logs/ragged")
    ap.add_argument("--moe", default="logs/ragged_moe/Qwen3-30B-A3B")
    args = ap.parse_args()

    dby, dfl, ddy = load(os.path.join(args.dense, "*", "*"))
    eb, ef = ls2(dby, dfl, ddy)                       # fit on DENSE models only
    print(f"Dense-fit (train) coefficients: e_bit={eb*1e10:.3f}e-10 J/byte, "
          f"e_flop={ef*1e12:.3f} pJ/flop  (n={len(dby)})")

    mby, mfl, mdy = load(os.path.join(args.moe, "*"))
    if not mby:
        print("no MoE bins found"); return
    pred = [eb*b + ef*f for b, f in zip(mby, mfl)]
    print(f"\nHELD-OUT MoE ({os.path.basename(args.moe)}, n={len(mby)} bins):")
    print(f"  predict from dense coeffs:  R2={r2(mdy,pred):.3f}  MAPE={mape(mdy,pred):.1f}%")

    # MoE self-fit for reference (upper bound)
    eb2, ef2 = ls2(mby, mfl, mdy)
    selfpred = [eb2*b + ef2*f for b, f in zip(mby, mfl)]
    print(f"  MoE self-fit (reference):   R2={r2(mdy,selfpred):.3f}  MAPE={mape(mdy,selfpred):.1f}%"
          f"  (e_bit={eb2*1e10:.3f}e-10, e_flop={ef2*1e12:.3f} pJ)")
    # naive baseline: mean dynamic power of the MoE
    kP = sum(mdy)/len(mdy)
    print(f"  naive (mean dyn energy):    MAPE={mape(mdy,[kP]*len(mdy)):.1f}%")


if __name__ == "__main__":
    main()
