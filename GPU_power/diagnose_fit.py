#!/usr/bin/env python3
"""
diagnose_fit.py

Honest interrogation of the two-term fit on the H200 ragged data. Answers:
  1. Does adding the FLOP term actually help? (bytes-only R^2 vs two-term R^2,
     on the SAME bins — the apples-to-apples test I skipped before.)
  2. Coefficient confidence via bootstrap (are e_bit/e_flop tight?).
  3. Held-out transferability: fit on 3 models, predict the 4th's per-bin dynamic
     energy; report MAPE. (The real "hardware constant" test.)
  4. Residual-vs-arithmetic-intensity correlation (structure => missing term).

Reads logs/ragged/*/binned_table.csv + run_meta idle. Pure stdlib.
"""
import csv, glob, json, os, random

ROOT = "logs/ragged"


def load_model_bins(model_dir):
    bytes_, flops, dyn = [], [], []
    for rd in glob.glob(os.path.join(model_dir, "*")):
        tbl = os.path.join(rd, "binned_table.csv")
        meta_p = os.path.join(rd, "run_meta.json")
        if not (os.path.exists(tbl) and os.path.exists(meta_p)):
            continue
        idle = json.load(open(meta_p)).get("idle_power_w")
        if idle is None:
            continue
        for r in csv.DictReader(open(tbl)):
            try:
                by = float(r["bytes_bin"]); fl = float(r["flops_bin"])
                E = float(r["energy_bin_j"]); dt = float(r["dt_s"])
            except (ValueError, KeyError):
                continue
            bytes_.append(by); flops.append(fl); dyn.append(E - idle * dt)
    return bytes_, flops, dyn


def ls1(x, y):                       # y = a*x  (through origin)
    sxx = sum(a * a for a in x)
    return sum(a * b for a, b in zip(x, y)) / sxx if sxx else 0.0


def ls2(x1, x2, y):                  # y = a*x1 + b*x2
    s11 = sum(a * a for a in x1); s22 = sum(a * a for a in x2)
    s12 = sum(a * b for a, b in zip(x1, x2))
    s1y = sum(a * c for a, c in zip(x1, y)); s2y = sum(a * c for a, c in zip(x2, y))
    det = s11 * s22 - s12 * s12
    if abs(det) < 1e-40:
        return 0.0, 0.0
    return (s1y * s22 - s2y * s12) / det, (s11 * s2y - s12 * s1y) / det


def r2(y, yhat):
    n = len(y); yb = sum(y) / n
    sst = sum((v - yb) ** 2 for v in y); ssr = sum((v - h) ** 2 for v, h in zip(y, yhat))
    return 1 - ssr / sst if sst else float("nan")


def mape(y, yhat):
    p = [abs(a - b) / abs(a) for a, b in zip(y, yhat) if a]
    return 100 * sum(p) / len(p) if p else float("nan")


def main():
    models = sorted(os.path.basename(d) for d in glob.glob(f"{ROOT}/*")
                    if os.path.isdir(d) and glob.glob(os.path.join(d, "*", "binned_table.csv")))
    data = {m: load_model_bins(os.path.join(ROOT, m)) for m in models}
    rng = random.Random(0)

    print(f"{'model':<20}{'n':>6}{'R2_bytesonly':>14}{'R2_twoterm':>12}"
          f"{'e_bit(e-10)':>13}{'e_flop(pJ)':>12}{'resid~AI r':>12}")
    for m in models:
        by, fl, dyn = data[m]
        if len(by) < 20:
            continue
        # bytes-only
        a1 = ls1(by, dyn); yh1 = [a1 * x for x in by]
        # two-term
        eb, ef = ls2(by, fl, dyn); yh2 = [eb * x + ef * y for x, y in zip(by, fl)]
        # residual vs AI
        resid = [d - h for d, h in zip(dyn, yh2)]
        ai = [f / b if b else 0 for b, f in zip(by, fl)]
        # pearson(resid, ai)
        n = len(ai); ma = sum(ai)/n; mr = sum(resid)/n
        cov = sum((a-ma)*(r-mr) for a, r in zip(ai, resid))
        va = sum((a-ma)**2 for a in ai) ** .5; vr = sum((r-mr)**2 for r in resid) ** .5
        rr = cov/(va*vr) if va and vr else 0
        print(f"{m:<20}{len(by):>6}{r2(dyn,yh1):>14.3f}{r2(dyn,yh2):>12.3f}"
              f"{eb*1e10:>13.3f}{ef*1e12:>12.3f}{rr:>12.3f}")

    # bootstrap CIs (pool all models) + leave-one-model-out held-out
    allby = sum((data[m][0] for m in models), [])
    allfl = sum((data[m][1] for m in models), [])
    alldy = sum((data[m][2] for m in models), [])
    ebs, efs = [], []
    idx = list(range(len(allby)))
    for _ in range(200):
        s = [rng.choice(idx) for _ in idx]
        eb, ef = ls2([allby[i] for i in s], [allfl[i] for i in s], [alldy[i] for i in s])
        ebs.append(eb*1e10); efs.append(ef*1e12)
    ebs.sort(); efs.sort()
    lo, hi = int(.025*len(ebs)), int(.975*len(ebs))
    print(f"\nBootstrap 95% CI (pooled): e_bit={ebs[len(ebs)//2]:.3f} "
          f"[{ebs[lo]:.3f},{ebs[hi]:.3f}] e-10 J/byte ; "
          f"e_flop={efs[len(efs)//2]:.3f} [{efs[lo]:.3f},{efs[hi]:.3f}] pJ/flop")

    print("\nLeave-one-model-out (fit 3, predict 4th's dynamic energy):")
    for held in models:
        trby = sum((data[m][0] for m in models if m != held), [])
        trfl = sum((data[m][1] for m in models if m != held), [])
        trdy = sum((data[m][2] for m in models if m != held), [])
        eb, ef = ls2(trby, trfl, trdy)
        hby, hfl, hdy = data[held]
        yh = [eb*x + ef*y for x, y in zip(hby, hfl)]
        print(f"  predict {held:<20} R2={r2(hdy,yh):>7.3f}  MAPE={mape(hdy,yh):>6.1f}%")


if __name__ == "__main__":
    main()
