#!/usr/bin/env python3
"""Figures for the proposal/presentation. Recomputes from binned tables; writes
PNGs to plots_proposal/. Run in-container with PYTHONPATH=pydeps (matplotlib)."""
import glob, os, csv, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "plots_proposal"; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3})
C = {"mem": "#2266cc", "compute": "#dd7722", "wbyte": "#2266cc", "kv": "#22aa88",
     "accent": "#aa3366"}


def load(md, depth=1):
    """depth=1: md is a MODEL dir, glob its run dirs. depth=2: md is a ROOT,
    glob model/run dirs (pool across models)."""
    cols = ["weight_bytes_bin", "kv_bytes_bin", "gemm_flops_bin", "bytes_bin",
            "flops_bin", "energy_bin_j", "dt_s", "prefill_tokens", "decode_tokens"]
    o = {c: [] for c in cols}; o["dyn"] = []
    pat = os.path.join(md, "*", "*") if depth == 2 else os.path.join(md, "*")
    for rd in glob.glob(pat):
        t = os.path.join(rd, "binned_table.csv"); m = os.path.join(rd, "run_meta.json")
        if not (os.path.exists(t) and os.path.exists(m)):
            continue
        idle = json.load(open(m)).get("idle_power_w")
        if idle is None:
            continue
        for r in csv.DictReader(open(t)):
            try:
                v = {c: float(r[c]) for c in cols}
            except (KeyError, ValueError):
                continue
            for c in cols:
                o[c].append(v[c])
            o["dyn"].append(v["energy_bin_j"] - idle * v["dt_s"])
    return {k: np.array(v) for k, v in o.items()}


def fit(X, y):
    c, *_ = np.linalg.lstsq(X, y, rcond=None); yh = X @ c
    return c, 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()


SIZES = [("Qwen2.5-0.5B-Instruct", 0.49), ("Qwen2.5-1.5B-Instruct", 1.54),
         ("Qwen2.5-3B-Instruct", 3.09), ("Qwen2.5-7B-Instruct", 7.07),
         ("Qwen2.5-14B-Instruct", 13.99), ("Qwen2.5-32B-Instruct", 31.98)]


def fig_size_independence():
    xs, e_bit, e_wbyte, e_flop, r2_2, r2_3 = [], [], [], [], [], []
    for name, ap in SIZES:
        d = load(os.path.join("logs/ragged_ladder", name))
        if len(d["dyn"]) < 20:
            continue
        (eb, ef), r2 = fit(np.c_[d["bytes_bin"], d["flops_bin"]], d["dyn"])
        (ew, ek, eg), r3 = fit(np.c_[d["weight_bytes_bin"], d["kv_bytes_bin"], d["gemm_flops_bin"]], d["dyn"])
        xs.append(ap); e_bit.append(eb*1e10); e_wbyte.append(ew*1e10)
        e_flop.append(ef*1e12); r2_2.append(r2); r2_3.append(r3)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    ax1.plot(xs, e_wbyte, "o-", color=C["wbyte"], label="e_wbyte (weight bytes)")
    ax1.plot(xs, e_bit, "s--", color="#88aadd", label="e_bit (lumped bytes)")
    ax1.axhspan(1.06, 1.14, color=C["wbyte"], alpha=0.10)
    ax1.set_xscale("log"); ax1.set_xlabel("active params (B)")
    ax1.set_ylabel("memory coeff (×10⁻¹⁰ J/byte)"); ax1.set_ylim(0.9, 1.5)
    ax1.set_title("Memory coefficient is size-independent\n(e_wbyte ±7% over 64× size)")
    ax1.legend(fontsize=9)
    ax2b = ax2.twinx()
    ax2.plot(xs, e_flop, "^-", color=C["compute"], label="e_flop (pJ/flop)")
    ax2b.plot(xs, r2_2, "o:", color="#999", label="R² 2-term")
    ax2b.plot(xs, r2_3, "o-", color="#333", label="R² 3-term (channels)")
    ax2.set_xscale("log"); ax2.set_xlabel("active params (B)")
    ax2.set_ylabel("compute coeff e_flop (pJ/flop)", color=C["compute"])
    ax2b.set_ylabel("fit R²"); ax2.set_title("Compute coeff drifts; channels lift R²\n(honest limitation at extremes)")
    l1, la1 = ax2.get_legend_handles_labels(); l2, la2 = ax2b.get_legend_handles_labels()
    ax2.legend(l1+l2, la1+la2, fontsize=8, loc="center right")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig1_size_independence.png"); plt.close(fig)


def fig_phase():
    d = load("logs/ragged", depth=2)   # dense pooled across models
    (eb, ef), _ = fit(np.c_[d["bytes_bin"], d["flops_bin"]], d["dyn"])
    groups = {"decode-heavy\n(prefill<5%)": [], "mixed\n(5–50%)": [], "prefill-heavy\n(>50%)": []}
    keys = list(groups)
    for i in range(len(d["dyn"])):
        tot = d["prefill_tokens"][i] + d["decode_tokens"][i]
        if tot <= 0:
            continue
        pf = d["prefill_tokens"][i] / tot
        g = (keys[0] if pf < 0.05 else keys[2] if pf > 0.5 else keys[1])
        comp = ef*d["flops_bin"][i]; mem = eb*d["bytes_bin"][i]
        if comp+mem > 0:
            groups[g].append(comp/(comp+mem))
    labels = list(groups); comp = [100*np.mean(groups[g]) for g in labels]
    mem = [100-c for c in comp]
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    ax.bar(labels, mem, color=C["mem"], label="memory (e_bit·bytes)")
    ax.bar(labels, comp, bottom=mem, color=C["compute"], label="compute (e_flop·flops)")
    for i, c in enumerate(comp):
        ax.text(i, mem[i]+c/2, f"{c:.0f}%", ha="center", color="white", fontweight="bold")
    ax.set_ylabel("share of dynamic energy (%)"); ax.set_ylim(0, 100)
    ax.set_title("Energy splits by phase: decode=memory-bound,\nprefill shifts toward compute (6× rise)")
    ax.legend(loc="lower center", fontsize=9); ax.grid(axis="x")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig2_phase_decomposition.png"); plt.close(fig)


def fig_transfer():
    # established scalars from diagnostics
    labels = ["dense\nheld-out", "MoE naive\n(active_params)", "MoE +occupancy\nfix"]
    mape = [8.5, 61.6, 20.6]; r2 = [0.69, -1.35, 0.79]
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    colors = [C["wbyte"], C["accent"], C["compute"]]
    bars = ax.bar(labels, mape, color=colors)
    for b, r in zip(bars, r2):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+2, f"R²={r:.2f}",
                ha="center", fontsize=9)
    ax.axhline(115, ls="--", color="#999"); ax.text(2.5, 117, "naive baseline 115%", ha="right", fontsize=8, color="#666")
    ax.set_ylabel("held-out MAPE (%)"); ax.set_ylim(0, 130)
    ax.set_title("Coefficients transfer to held-out models —\nMoE needs the expert-occupancy byte fix")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_transfer.png"); plt.close(fig)


def fig_recommender():
    import recommend_gpu as R
    from models import model_from_hf_id, MODELS
    m = MODELS.get("Qwen2-7B-Instruct") or model_from_hf_id("Qwen/Qwen2-7B-Instruct")
    prompt, gen, batch = 2048, 256, 32
    phases = {"prefill": R.phase_cost(m, 1, prompt*batch, batch, prompt, True),
              "decode": R.phase_cost(m, gen, batch, batch, prompt+gen//2, False)}
    gpus = [R.H200] + [R.scaled_gpu(n, s) for n, s in R.SPECS.items()]
    names = [g["name"].split(" ")[0] for g in gpus]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, (ph, (wb, kvb, gemm)) in zip(axes, phases.items()):
        e = [R.predict(g, wb, kvb, gemm)["energy"] for g in gpus]
        e = np.array(e)/min(e)
        order = np.argsort(e)
        cols = [C["compute"] if ph == "prefill" else C["mem"]]*len(gpus)
        ax.barh([names[i] for i in order][::-1], [e[i] for i in order][::-1], color=cols)
        ax.set_xlabel("relative energy (lower=better)")
        ax.set_title(f"{ph.upper()} ({'compute-bound' if ph=='prefill' else 'memory-bound'})")
        ax.grid(axis="y")
    fig.suptitle("Energy-optimal GPU differs by phase (L40S↔A100 reversal)\n"
                 "H200 measured; others datasheet-scaled (proposed)", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig4_recommender.png"); plt.close(fig)


def fig_size_transfer():
    # leave-one-size-out: fit 3-term on other sizes, predict held size
    feats = ["weight_bytes_bin", "kv_bytes_bin", "gemm_flops_bin"]
    D = {}
    for name, ap in SIZES:
        d = load(os.path.join("logs/ragged_ladder", name))
        if len(d["dyn"]) < 20:
            continue
        D[name] = (np.c_[[d[c] for c in feats]].T, d["dyn"], ap)
    labels, mapes, aps = [], [], []
    for held in D:
        Xh, yh, ap = D[held]
        Xtr = np.vstack([D[n][0] for n in D if n != held])
        ytr = np.concatenate([D[n][1] for n in D if n != held])
        coef, _ = fit(Xtr, ytr)
        yhat = Xh @ coef
        m = np.nanmean(np.abs((yh - yhat) / np.where(yh == 0, np.nan, yh))) * 100
        labels.append(held.replace("Qwen2.5-", "").replace("-Instruct", ""))
        mapes.append(m); aps.append(ap)
    order = np.argsort(aps)
    labels = [labels[i] for i in order]; mapes = [mapes[i] for i in order]
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    bars = ax.bar(labels, mapes, color=C["mem"])
    for b, m in zip(bars, mapes):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, f"{m:.1f}%",
                ha="center", fontsize=9)
    ax.axhline(7.2, ls="--", color=C["compute"])
    ax.text(len(labels)-0.5, 7.6, "single 7B → all sizes: 7.2%",
            ha="right", fontsize=9, color=C["compute"])
    ax.set_ylabel("held-out MAPE (%)"); ax.set_xlabel("held-out model size")
    ax.set_ylim(0, max(mapes)+3)
    ax.set_title("Calibrate on other sizes, predict a held-out size\n"
                 "(coefficients transfer across 64× model size)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig5_size_transfer.png"); plt.close(fig)


for f in [fig_size_independence, fig_phase, fig_transfer, fig_recommender,
          fig_size_transfer]:
    try:
        f(); print("ok", f.__name__)
    except Exception as e:
        print("FAIL", f.__name__, repr(e))
print("figures in", OUT)
