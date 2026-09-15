#!/usr/bin/env python3
"""
roofline_energy.py — tie the fitted energy model back into LIMINAL's roofline.

Core idea: performance is bounded by the BOTTLENECK resource (a max), but energy
is the SUM across resources (memory + compute draw power simultaneously). So:

  LIMINAL (time):   t_iter = max(B / (bw·MBU), F / (peak·MFU))     [roofline]
  Ours   (energy):  E_iter = e_wbyte·Wb + e_kvbyte·KVb + e_gemm·F  [additive]
  Together:         power = E_iter / t_iter,   perf/watt = tok/s / power

This script, per operating point (run), sums the analytic work from binned_table,
computes the roofline time, and reports:
  1. implied utilization (MBU for memory-bound decode, MFU for compute-bound
     prefill) — is it consistent? (LIMINAL's validation, independent of energy)
  2. arithmetic intensity F/B and the roofline regime (vs ridge point peak/bw)
  3. measured power & perf/watt vs the analytic (roofline-time + energy) prediction

Data in hand. Run in-container.
  python3 roofline_energy.py logs/ragged
"""
import glob, os, csv, json, sys
import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else "logs/ragged"
BW, PEAK = 4.8e12, 990e12               # H200: HBM bytes/s, fp16 dense FLOP/s
RIDGE = PEAK / BW                        # arithmetic intensity at the roofline knee
# energy coefficients (3-term channel fit, H200 pooled dense)
E_WBYTE, E_KVBYTE, E_GEMM = 1.077e-10, 3.38e-10, 0.673e-12


def run_totals(rd):
    """Sum analytic work + measured time/energy over a run's bins."""
    t = os.path.join(rd, "binned_table.csv"); m = os.path.join(rd, "run_meta.json")
    rj = os.path.join(rd, "results.json")
    if not all(os.path.exists(p) for p in (t, m, rj)):
        return None
    meta = json.load(open(m)); res = json.load(open(rj))
    idle = meta.get("idle_power_w")
    if idle is None:
        return None
    W = Kv = Gf = Af = dt = Edyn = 0.0
    for r in csv.DictReader(open(t)):
        try:
            W += float(r["weight_bytes_bin"]); Kv += float(r["kv_bytes_bin"])
            Gf += float(r["gemm_flops_bin"]); Af += float(r["attn_flops_bin"])
            dt += float(r["dt_s"]); Edyn += float(r["energy_bin_j"]) - idle*float(r["dt_s"])
        except (KeyError, ValueError):
            pass
    if dt <= 0:
        return None
    B = W + Kv; F = Gf + Af
    return dict(model=os.path.basename(os.path.dirname(rd)),
                tag=os.path.basename(rd), W=W, Kv=Kv, Gf=Gf, Af=Af, B=B, F=F,
                dt=dt, Edyn=Edyn, idle=idle,
                tok_s=res.get("aggregate_tokens_per_sec", 0.0),
                p_meas=res.get("avg_power_window_w", 0.0))


rows = [r for rd in sorted(glob.glob(os.path.join(ROOT, "*", "*")))
        if os.path.isdir(rd) and (r := run_totals(rd))]
if not rows:
    print("no runs"); sys.exit()

print(f"ROOT={ROOT}   ridge point (peak/bw) = {RIDGE:.0f} FLOP/byte")
print(f"{'run':<34}{'AI':>7}{'regime':>7}{'util%':>7}{'t_roof/t_meas':>14}"
      f"{'P_meas':>8}{'P_pred':>8}{'pw_err%':>8}")
util_mem, util_cmp = [], []
for r in rows:
    ai = r["F"] / r["B"] if r["B"] else 0
    mem_bound = ai < RIDGE
    # ideal (100% util) roofline time for the total work
    t_mem = r["B"] / BW; t_cmp = r["F"] / PEAK
    t_roof_ideal = max(t_mem, t_cmp)
    # implied utilization of the bottleneck resource = ideal_time / measured_time
    util = t_roof_ideal / r["dt"]
    (util_mem if mem_bound else util_cmp).append(util)
    # analytic power prediction: energy model / measured time (+ static)
    e_pred = E_WBYTE*r["W"] + E_KVBYTE*r["Kv"] + E_GEMM*r["F"]
    p_pred = e_pred / r["dt"] + r["idle"]
    pw_err = 100*(p_pred - r["p_meas"]) / r["p_meas"] if r["p_meas"] else 0
    name = f"{r['model'].replace('-Instruct','')}/{r['tag']}"
    print(f"{name:<34}{ai:>7.1f}{'mem' if mem_bound else 'cmp':>7}{util*100:>6.1f}%"
          f"{t_roof_ideal/r['dt']:>14.3f}{r['p_meas']:>8.0f}{p_pred:>8.0f}{pw_err:>7.1f}%")

print(f"\nImplied bottleneck utilization (roofline consistency check):")
if util_mem:
    a = np.array(util_mem)*100
    print(f"  memory-bound runs (n={len(a)}): MBU = {a.mean():.1f}% ± {a.std():.1f}%")
if util_cmp:
    a = np.array(util_cmp)*100
    print(f"  compute-bound runs (n={len(a)}): MFU = {a.mean():.1f}% ± {a.std():.1f}%")
print("  (a ~consistent utilization => LIMINAL roofline predicts iteration TIME;"
      "\n   our coefficients give ENERGY; together => perf/watt.)")

# dump CSV + figure
with open("roofline_points.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "tag", "AI", "regime", "MBU", "p_meas", "p_pred",
                "compute_energy_frac"])
    for r in rows:
        ai = r["F"]/r["B"] if r["B"] else 0
        util = max(r["B"]/BW, r["F"]/PEAK)/r["dt"]
        e_pred = E_WBYTE*r["W"] + E_KVBYTE*r["Kv"] + E_GEMM*r["F"]
        cef = E_GEMM*r["F"] / (E_WBYTE*r["W"] + E_KVBYTE*r["Kv"] + E_GEMM*r["F"])
        w.writerow([r["model"], r["tag"], f"{ai:.3f}",
                    "mem" if ai < RIDGE else "cmp", f"{util:.4f}",
                    f"{r['p_meas']:.1f}", f"{e_pred/r['dt']+r['idle']:.1f}",
                    f"{cef:.4f}"])

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ai = np.array([r["F"]/r["B"] if r["B"] else 0 for r in rows])
    mbu = np.array([max(r["B"]/BW, r["F"]/PEAK)/r["dt"] for r in rows]) * 100
    pm = np.array([r["p_meas"] for r in rows])
    pp = np.array([(E_WBYTE*r["W"]+E_KVBYTE*r["Kv"]+E_GEMM*r["F"])/r["dt"]+r["idle"] for r in rows])
    cef = np.array([E_GEMM*r["F"]/(E_WBYTE*r["W"]+E_KVBYTE*r["Kv"]+E_GEMM*r["F"])*100 for r in rows])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3), dpi=150)
    sc = a1.scatter(ai, cef, c=mbu, cmap="viridis", s=42, edgecolor="w", linewidth=.4)
    a1.axvline(RIDGE, ls="--", color="#a33862")
    a1.text(RIDGE*0.92, a1.get_ylim()[1]*0.7, "roofline ridge\n(206 FLOP/byte)",
            ha="right", fontsize=8, color="#a33862")
    a1.set_xscale("log"); a1.set_xlabel("arithmetic intensity (FLOP/byte)")
    a1.set_ylabel("compute share of dynamic energy (%)")
    a1.set_title("Energy roofline: memory-bound → compute\nas AI rises toward the ridge")
    a1.grid(alpha=.3, which="both"); fig.colorbar(sc, ax=a1, label="impl. MBU (%)")
    a2.scatter(pp, pm, s=42, color="#1f5fbf", edgecolor="w", linewidth=.4)
    lim = [min(pm.min(), pp.min())-10, max(pm.max(), pp.max())+10]
    a2.plot(lim, lim, "k--", lw=1, alpha=.6)
    a2.fill_between(lim, [x*0.95 for x in lim], [x*1.05 for x in lim], color="#1f5fbf", alpha=.08)
    a2.set_xlabel("predicted power (W)  [roofline time + energy coeffs]")
    a2.set_ylabel("measured power (W)")
    a2.set_title(f"Analytic power vs measured\n(MBU {mbu.mean():.0f}%±{mbu.std():.0f}%, ±5% band)")
    a2.grid(alpha=.3)
    fig.tight_layout(); fig.savefig("plots_proposal/fig6_roofline_energy.png")
    print("wrote plots_proposal/fig6_roofline_energy.png")
except Exception as e:
    print("plot skipped:", repr(e))
