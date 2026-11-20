#!/usr/bin/env python3
"""How the A100 power roof moves with SM clock, from EnergAIzer's released clock-locked
database (Lee et al., ISPASS 2026; A100-PCIe "yz8", 40 GB, 250 W, 1555 GB/s).
Answers the PI's "does the gap between the roofline and the power roof move with clock"
without changing clocks on our hardware.

Per locked SM clock f (510, 705, 900, 1410 MHz; 'nolock' = the default governor):
  e_b(f) = (P - P_idle(f)) / BW        large (>=1 GB) relu/scalar elementwise kernels
  e_f(f) = (P - P_idle(f) - e_b(f) * B_dram/t) / FLOP/s   large bf16 tensor-core GEMMs
           (B_dram = A + B + C once: a lower bound, so e_f is an upper bound)
  BW(f), F(f) = the achieved rates of those kernels (the clock-dependent roofs)
  P_idle(f) = their idle_power.json at the locked clock
Power roof at the cap:  phi_P(I) = (P_cap - P_idle(f)) / (e_b/I + e_f)
Attainable at clock f:  min(BW(f) * I, F(f), phi_P(I)); the envelope over f is what an
ideal clock governor at the cap could reach.

Coefficients come only from runs below the cap. 510/705/900 MHz are uncapped. At 1410 MHz there is
a locked elementwise table, but it ran at ~1308 MHz at 240 W: e_b and BW are taken from it at its
achieved clock and idle. There is no uncapped GEMM above 900 MHz (the default governor's GEMMs sit
at the cap at ~1035 MHz), so at 1410:
  The whole row sits at the achieved ~1308 MHz (even streaming cannot hold 1410 under the cap):
  F        = the locked clocks' TF/MHz (linear: 0.198 at 510/705/900) x 1308
  e_f      = a band: lo = e_f(900) (flat over 510-900 MHz), hi = e_f(900) * e_b(1410)/e_b(900)
             (the per-byte energy's rise from 900 to ~1308 MHz as a voltage proxy)
The default governor's capped GEMM is plotted as a measured point, not used as a roof.
  python3 liminal_energy/energaizer_clock_roof.py [--db /shared_data0/adsampat/lit/energaizer]
Writes liminal_energy/energaizer_clock_roof.json and figs/fig_clock_roof.png.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
P_CAP = 250.0                     # A100-PCIe 40 GB board limit (the yz8 card)
PEAK_BW, PEAK_F = 1555e9, 312e12
CLOCKS = ["510", "705", "900", "1410", "nolock"]
IS = [1, 3, 10, 30, 100, 300, 1000]


def load(db):
    idle = json.load(open(os.path.join(db, "src/config/dvfs/yz8/idle_power.json")))
    out = {}
    for f in CLOCKS:
        p = os.path.join(db, f"db/yz8_misc_elementwise_freq{f}_lut.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        d = d[d.op.isin(["relu", "scalar_mul", "scalar_add"])].copy()
        d["bytes"] = 2 * d.dim * d.prec.map({"bf16": 2, "fp32": 4})       # read + write
        d["bw"] = d.bytes / (d.time * 1e-3)
        d["P"] = d.energy / (d.time * 1e-3)
        big = d[d.bytes >= 1e9]
        fclk = float(big.avg_freq.median())
        pid = idle[str(int(round(fclk / 15) * 15))] if f != "nolock" else None
        r = dict(clock=f, elem_sm_mhz=fclk, bw=float(big.bw.median()), p_stream=float(big.P.median()),
                 p_idle=pid, n_elem=int(len(big)))
        # no locked-1410 GEMM table, and the default governor's GEMMs are capped (~1035 MHz): 1410's
        # compute roof and e_f are extrapolated from the uncapped locked clocks in main()
        g = os.path.join(db, f"db/yz8_gemmex_bf16bf16_freq{f}_noflush_lut_v2.csv")
        if os.path.exists(g):
            G = pd.read_csv(g)
            G["F"] = 2.0 * G.batch * G.dimM * G.dimN * G.dimK
            G["B"] = 2.0 * G.batch * (G.dimM * G.dimK + G.dimK * G.dimN + G.dimM * G.dimN)
            G["fps"] = G.F / (G.time * 1e-3)
            G["P"] = G.energy / (G.time * 1e-3)
            G = G[(G.time > 1.0) & (G.F / G.B > 500)]                       # long, compute-bound
            top = G[G.fps >= G.fps.quantile(0.9)]
            r.update(gemm_sm_mhz=float(top.avg_freq.median()), f_rate=float(top.fps.median()),
                     p_gemm=float(top.P.median()), b_rate_gemm=float((top.B / (top.time * 1e-3)).median()),
                     n_gemm=int(len(top)))
        out[f] = r
    # the default governor's idle: take the locked clock nearest to what it ran at
    if "nolock" in out:
        f = out["nolock"]
        f["p_idle_elem"] = idle[str(int(round(f["elem_sm_mhz"] / 15) * 15))]
        f["p_idle_gemm"] = idle[str(int(round(f.get("gemm_sm_mhz", 1410) / 15) * 15))]
    return out, idle


def coefficients(r):
    pi_e = r["p_idle"] if r["p_idle"] is not None else r["p_idle_elem"]
    pi_g = r["p_idle"] if r["p_idle"] is not None else r["p_idle_gemm"]
    e_b = (r["p_stream"] - pi_e) / r["bw"]
    e_f = None
    if r.get("f_rate") and r.get("p_gemm") is not None:
        e_f = (r["p_gemm"] - pi_g - e_b * r["b_rate_gemm"]) / r["f_rate"]
    return dict(e_b=e_b, e_f=e_f, p_idle=pi_e)


def roof(r, c, I):
    bw_roof = r["bw"] * I
    f_roof = r.get("f_rate", float("inf"))
    p_roof = (P_CAP - c["p_idle"]) / (c["e_b"] / I + (c["e_f"] or 0.0))
    att = min(bw_roof, f_roof, p_roof)
    bound = "power" if att == p_roof else ("bandwidth" if att == bw_roof else "compute")
    return dict(I=I, attainable_tf=att / 1e12, power_roof_tf=p_roof / 1e12, bound=bound,
                power_over_unconstrained=p_roof / min(bw_roof, f_roof))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/shared_data0/adsampat/lit/energaizer")
    a = ap.parse_args()
    data, idle = load(a.db)
    res = {}
    print(f"{'clock':>7} {'SM MHz':>7} {'idle W':>7} {'BW GB/s':>8} {'pJ/B':>6} {'TF/s':>6} {'pJ/FLOP':>8} "
          f"{'P_gemm':>7} | attainable TF/s (bound) at I = " + " ".join(f"{i:>6}" for i in IS))
    # 1410: compute roof and e_f band from the uncapped locked clocks (see the docstring)
    lk = [f for f in ("510", "705", "900") if data.get(f, {}).get("f_rate")]
    tf_per_mhz = float(np.median([data[f]["f_rate"] / data[f]["gemm_sm_mhz"] for f in lk]))
    c900 = coefficients(data["900"])
    if "1410" in data:
        r = data["1410"]
        c14 = coefficients(r)
        r.update(gemm_sm_mhz=r["elem_sm_mhz"], f_rate=tf_per_mhz * r["elem_sm_mhz"], b_rate_gemm=0.0, p_gemm=None, n_gemm=0,
                 extrapolated=True, e_f_lo=c900["e_f"], e_f_hi=c900["e_f"] * c14["e_b"] / c900["e_b"],
                 tf_per_mhz=tf_per_mhz)
    for f, r in data.items():
        c = coefficients(r)
        if r.get("extrapolated"):
            c["e_f"] = r["e_f_lo"]           # optimistic end of the band (highest power roof)
        if f == "nolock":
            r["capped"] = True             # default governor at the cap: a measured point, not a roof
        rows = [roof(r, c, I) for I in IS]
        res[f] = dict(r, **c, roof=rows)
        print(f"{f:>7} {r['elem_sm_mhz']:7.0f} {c['p_idle']:7.1f} {r['bw'] / 1e9:8.0f} {c['e_b'] * 1e12:6.0f} "
              f"{r.get('f_rate', 0) / 1e12:6.0f} {(c['e_f'] or 0) * 1e12:8.2f} {(r.get('p_gemm') or 0):7.0f} | "
              + " ".join(f"{x['attainable_tf']:5.1f}{x['bound'][0]}" for x in rows))
    locked = [f for f in res if f != "nolock" and res[f].get("e_f") is not None]
    # 1410 with the pessimistic end of the e_f band
    if "1410" in res:
        r = res["1410"]; ch = dict(e_b=r["e_b"], e_f=r["e_f_hi"], p_idle=r["p_idle"])
        r["roof_hi_ef"] = [roof(r, ch, I) for I in IS]
    # what the default governor actually got on its capped GEMMs
    if "nolock" in res:
        n = res["nolock"]
        n["gemm_I"] = n["f_rate"] / n["b_rate_gemm"]
        print(f"\ndefault governor GEMM (capped): {n['f_rate'] / 1e12:.0f} TF/s at {n['gemm_sm_mhz']:.0f} MHz, "
              f"{n['p_gemm']:.0f} W, I = {n['gemm_I']:.0f}")
    if "1410" in res:
        r = res["1410"]
        print(f"1410 lock (ran {r['elem_sm_mhz']:.0f} MHz) extrapolated: F = {r['f_rate'] / 1e12:.0f} TF/s ({r['tf_per_mhz'] / 1e9:.3f} TF/GHz), "
              f"e_f band {r['e_f_lo'] * 1e12:.2f}-{r['e_f_hi'] * 1e12:.2f} pJ/FLOP, idle {r['p_idle']:.1f} W")
    env = []
    for k, I in enumerate(IS):
        best = max(locked, key=lambda f: res[f]["roof"][k]["attainable_tf"])
        env.append(dict(I=I, best_clock=best, attainable_tf=res[best]["roof"][k]["attainable_tf"],
                        frac_peak_bw=res[best]["roof"][k]["attainable_tf"] * 1e12 / I / PEAK_BW,
                        frac_peak_f=res[best]["roof"][k]["attainable_tf"] * 1e12 / PEAK_F))
    print("\nenvelope over locked clocks (an ideal governor at the 250 W cap):")
    for e in env:
        print(f"  I={e['I']:>5}: {e['attainable_tf']:6.1f} TF/s at {e['best_clock']} MHz "
              f"({100 * e['frac_peak_bw']:.0f}% peak BW, {100 * e['frac_peak_f']:.0f}% peak FLOPs)")
    json.dump(dict(p_cap_w=P_CAP, peak_bw=PEAK_BW, peak_flops=PEAK_F, clocks=res, envelope=env,
                   idle_vs_clock=idle), open(os.path.join(HERE, "energaizer_clock_roof.json"), "w"), indent=1)
    plot(res, env)


def plot(res, env):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Is = np.logspace(0, 3, 200)
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw=dict(width_ratios=[1.45, 1]))
    cols = {"510": "#9ecae1", "705": "#4292c6", "900": "#08519c", "1410": "#d03b3b"}
    a = ax[0]
    for f, col in cols.items():
        if f not in res or res[f].get("e_f") is None:
            continue
        r = res[f]
        un = [min(r["bw"] * I, r["f_rate"]) / 1e12 for I in Is]
        a.plot(Is, un, color=col, ls="--", lw=1)
        if r.get("extrapolated"):
            lo = [(P_CAP - r["p_idle"]) / (r["e_b"] / I + r["e_f_hi"]) / 1e12 for I in Is]
            hi = [(P_CAP - r["p_idle"]) / (r["e_b"] / I + r["e_f_lo"]) / 1e12 for I in Is]
            a.fill_between(Is, lo, hi, color=col, alpha=0.25, lw=0)
            a.plot(Is, hi, color=col, lw=1.6,
                   label=f"locked 1410, held {r['elem_sm_mhz']:.0f} MHz: power roof band (e_f {r['e_f_lo'] * 1e12:.2f}–{r['e_f_hi'] * 1e12:.2f} pJ/FLOP, extrapolated)")
        else:
            pr = [(P_CAP - r["p_idle"]) / (r["e_b"] / I + r["e_f"]) / 1e12 for I in Is]
            a.plot(Is, pr, color=col, lw=1.6, label=f"{f} MHz locked: power roof (solid), BW/compute roof (dashed)")
    if "nolock" in res and res["nolock"].get("gemm_I"):
        n = res["nolock"]
        a.plot([min(n["gemm_I"], 1000)], [n["f_rate"] / 1e12], "*", ms=12, color="#d03b3b", mec="k", mew=0.6, zorder=5,
               label=f"default governor GEMM, measured: capped at {n['gemm_sm_mhz']:.0f} MHz")
    a.set_xscale("log"); a.set_yscale("log"); a.set_ylim(1, 400)
    a.set_xlabel("arithmetic intensity (FLOP / DRAM byte)"); a.set_ylabel("TFLOP/s")
    a.set_title("A100-PCIe 40 GB (EnergAIzer data), 250 W cap: roofs vs locked SM clock", loc="left", fontsize=9)
    a.legend(fontsize=7, loc="upper left")
    a = ax[1]
    fs = [f for f in ["510", "705", "900", "1410"] if f in res]
    x = [float(res[f]["elem_sm_mhz"]) for f in fs]
    a.plot(x, [res[f]["e_b"] * 1e12 for f in fs], "o-", color="#2a78d6", label="e_b (pJ/B), at the achieved clock")
    lk = [f for f in fs if not res[f].get("extrapolated")]
    a.plot([float(res[f]["gemm_sm_mhz"]) for f in lk], [res[f]["e_f"] * 1e12 * 100 for f in lk], "s-", color="#eb6834",
           label="e_f (pJ/FLOP ×100), uncapped GEMMs")
    if "1410" in res:
        r = res["1410"]
        a.errorbar([r["elem_sm_mhz"]], [(r["e_f_lo"] + r["e_f_hi"]) / 2 * 1e14], yerr=[[(r["e_f_hi"] - r["e_f_lo"]) / 2 * 1e14]] * 2,
                   fmt="s", mfc="white", color="#eb6834", capsize=4, label="e_f at ~1308: extrapolated band")
    idle = json.load(open(os.path.join(HERE, "energaizer_clock_roof.json")))["idle_vs_clock"]
    ks = sorted(idle, key=float)
    a.plot([float(k) for k in ks], [idle[k] for k in ks], "-", color="#52514e", lw=1.2, label="idle at each locked clock (W)")
    a.set_xlabel("SM clock (MHz)"); a.legend(fontsize=7.5)
    a.set_title("coefficients vs clock: voltage rises above ~1000 MHz", loc="left", fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "figs", "fig_clock_roof.png"), dpi=150)


if __name__ == "__main__":
    main()
