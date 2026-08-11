#!/usr/bin/env python3
"""
recommend_gpu.py  (#20 — recommender prototype + GPU-spec scaling PoC)

Given a workload (model + prefill/decode token counts + context), predict the
per-phase ENERGY and latency on each candidate GPU and recommend the
energy-optimal GPU — separately for the prefill phase (compute-bound) and the
decode phase (memory-bound), which can pick DIFFERENT GPUs (the disaggregated-
serving payoff).

Energy model (3-term, the channel-split winner):
   E = e_wbyte·weight_bytes + e_kvbyte·kv_bytes + e_gemm·gemm_flops + P_static·t
Latency via roofline:  t = max(bytes / HBM_BW, flops / peak_FLOPS).
Power cap:  instantaneous power is clamped to P_cap (a capped GPU's energy is
governed by P_cap·t, not the work — the A100-PCIe regime we measured).

COEFFICIENT PROVENANCE (honest):
  * H200 coefficients are MEASURED (fit from NVML power on ragged traffic).
  * Other GPUs' coefficients are DATASHEET-SCALED ESTIMATES via the physical
    priors  e_byte ∝ 1/HBM_BW  and  e_flop ∝ 1/peak_FLOPS  anchored on H200.
    These are the recommender's *proposed* generalization, to be validated by
    measuring 2-3 more (uncapped) GPUs — the key next experiment.
"""
import sys
from models import model_from_hf_id, MODELS

DTYPE_B = 2  # fp16

# --- H200 measured coefficients (from fit_channels 3-term, logs/ragged) ---
H200 = dict(name="H200 (measured)", e_wbyte=1.077e-10, e_kvbyte=3.380e-10,
            e_gemm=0.673e-12, p_static=118.0, p_cap=700.0,
            bw=4.8e12, peak_flops=990e12, measured=True)

# --- datasheet specs for scaling other GPUs (BW bytes/s, peak fp16 FLOP/s, TDP) ---
SPECS = {
    "A100-80-PCIe": dict(bw=1.94e12, peak_flops=312e12, tdp=300.0, p_static=74.0, p_cap=300.0),
    "A100-80-SXM":  dict(bw=2.039e12, peak_flops=312e12, tdp=400.0, p_static=80.0, p_cap=400.0),
    "L40S":         dict(bw=0.864e12, peak_flops=362e12, tdp=350.0, p_static=45.0, p_cap=350.0),
    "H100-SXM":     dict(bw=3.35e12, peak_flops=990e12, tdp=700.0, p_static=110.0, p_cap=700.0),
}


def scaled_gpu(name, spec):
    """Datasheet-scaled coefficients anchored on H200 physical priors."""
    return dict(name=f"{name} (scaled)",
                e_wbyte=H200["e_wbyte"] * H200["bw"] / spec["bw"],
                e_kvbyte=H200["e_kvbyte"] * H200["bw"] / spec["bw"],
                e_gemm=H200["e_gemm"] * H200["peak_flops"] / spec["peak_flops"],
                p_static=spec["p_static"], p_cap=spec["p_cap"],
                bw=spec["bw"], peak_flops=spec["peak_flops"], measured=False)


ETA = 0.7   # realized fraction of peak BW / FLOPS (roofline is a lower bound on time)


def phase_cost(m, n_passes, tokens_per_pass, batch, ctx, is_prefill):
    """(weight_bytes, kv_bytes, gemm_flops) totals for a phase.

    KEY: weights are reloaded from HBM on EVERY forward pass. Decode = one pass
    per generated token (n_passes = gen_len), each processing only `batch` tokens
    -> weight traffic is amortized over few FLOPs => memory-bound. Prefill = ~one
    pass over the whole prompt*batch => compute-bound."""
    try:
        from energy_model import weight_params_for_tokens
        wp = weight_params_for_tokens(m, tokens_per_pass)
    except Exception:
        wp = m.active_params()
    total_tokens = tokens_per_pass * n_passes
    weight_bytes = wp * DTYPE_B * n_passes                 # reloaded each pass
    kv_per_tok = m.kv_bytes_per_token(DTYPE_B)
    if is_prefill:
        kv_bytes = 0.5 * batch * ctx * kv_per_tok          # causal build-up (triangular)
    else:
        kv_bytes = batch * ctx * kv_per_tok * n_passes     # read resident KV each step
    gemm = 2.0 * m.active_params() * total_tokens
    return weight_bytes, kv_bytes, gemm


def predict(gpu, wbytes, kvbytes, gemm):
    bytes_tot = wbytes + kvbytes
    energy_dyn = (gpu["e_wbyte"] * wbytes + gpu["e_kvbyte"] * kvbytes + gpu["e_gemm"] * gemm)
    t_roof = max(bytes_tot / (gpu["bw"] * ETA), gemm / (gpu["peak_flops"] * ETA))
    p_dyn = energy_dyn / max(t_roof, 1e-9)
    if p_dyn + gpu["p_static"] <= gpu["p_cap"]:
        t, power, capped = t_roof, p_dyn + gpu["p_static"], False
    else:  # power-capped: GPU throttles -> runs longer, power pinned at cap
        power, capped = gpu["p_cap"], True
        t = energy_dyn / (gpu["p_cap"] - gpu["p_static"])
    energy = energy_dyn + gpu["p_static"] * t
    bound = "compute" if gemm / (gpu["peak_flops"] * ETA) > bytes_tot / (gpu["bw"] * ETA) else "memory"
    return dict(t=t, power=power, energy=energy, bound=bound, capped=capped)


def recommend(model_id, prompt_len=2048, gen_len=256, batch=32):
    try:
        m = MODELS.get(model_id.split("/")[-1]) or model_from_hf_id(model_id)
    except Exception:
        m = model_from_hf_id(model_id)
    gpus = [H200] + [scaled_gpu(n, s) for n, s in SPECS.items()]

    # prefill: 1 pass over prompt_len*batch tokens, ctx ~ prompt_len
    # decode: gen_len passes, each `batch` tokens, ctx ~ prompt_len + gen/2
    phases = {
        "PREFILL (prompt ingest)": phase_cost(m, 1, prompt_len * batch, batch, prompt_len, True),
        "DECODE  (token gen)": phase_cost(m, gen_len, batch, batch, prompt_len + gen_len // 2, False),
    }
    print(f"\n### Workload: {model_id}  prompt={prompt_len} gen={gen_len} batch={batch}")
    print(f"    active_params={m.active_params()/1e9:.2f}B")
    for phase, (wb, kvb, gemm) in phases.items():
        print(f"\n {phase}")
        rows = []
        for g in gpus:
            r = predict(g, wb, kvb, gemm)
            rows.append((r["energy"], g["name"], r))
        rows.sort()
        best = rows[0][0]
        print(f"   {'GPU':<22}{'energy(J)':>12}{'rel':>7}{'bound':>9}{'capped':>8}")
        for e, name, r in rows:
            print(f"   {name:<22}{e:>12.1f}{e/best:>6.2f}x{r['bound']:>9}"
                  f"{'  CAP' if r['capped'] else '':>8}")
        print(f"   -> energy-optimal: {rows[0][1]}")


if __name__ == "__main__":
    mid = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-7B-Instruct"
    recommend(mid)
