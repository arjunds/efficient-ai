#!/usr/bin/env python3
"""
recommend_gpu.py  (v2 — measured coefficients + REALIZED utilization)

Given a workload (model + prompt/gen lengths + batch), predict per-phase
(prefill vs decode) time, power and ENERGY on candidate GPUs and recommend the
energy-optimal GPU for each phase, with uncertainty.

WHAT CHANGED FROM v1 (and why) — see RECOMMENDER.md
  v1 derived every non-H200 GPU with `scaled_gpu()`:  e_byte ∝ 1/HBM_BW,
  e_flop ∝ 1/peak_FLOPS, and timed everything at a flat 70% of datasheet peak.
  The B200 measurement falsified the byte law (J/byte is ~invariant, B200 is
  ~16% *higher* per byte than H200, not 40% lower) and showed that realized
  bandwidth is far below peak and depends on model size (7B: 31-35% of B200
  peak; 72B: 62%). v2 therefore:
    * uses MEASURED coefficients per GPU (gpu_coefficients.json; built-in
      defaults if absent) — no 1/BW scaling anywhere;
    * for unmeasured GPUs uses a MEMORY-TECHNOLOGY prior (HBM3/3e band measured
      on H200+B200; HBM2e from the cap-bound A100 data; GDDR6 wide until the
      A5000 entry appears — it is picked up automatically) and flags LOW
      CONFIDENCE;
    * times iterations with a REALIZED-UTILIZATION model fitted to our own
      vLLM runs (recommender_backtest.py --fit), not datasheet peaks.

ENERGY (per iteration, 3-term channel model; the fit's convention — canonical
~/models.py byte accounting, e_gemm multiplies dense GEMM flops):
    E = e_wbyte·Wb + e_kvbyte·KVb + e_gemm·F_gemm + P_static·t

TIME (per iteration) — the utilization model:
    t_mem = t0 + Σ_layers softmax_k( τ , (Wb/L) / (BW·η) )      # latency-floored weight stream
               + KVb / (BW·η_kv) + F / (peak·μ)                  # KV stream + per-token marginal
    t_cmp = t0 + F / (peak·MFU_prefill)                          # compute roofline (prefill)
    t     = max(t_mem, t_cmp)
  t0   = per-step framework overhead (scheduler, sampling, launch)  [ms]
  τ    = per-layer latency floor: a layer cannot finish faster than this even
         if its weights are tiny (why a 0.5B model runs at 5% MBU)
  η    = asymptotic weight-streaming efficiency (MBU for very large layers)
  softmax_k(a,b) = (a^k + b^k)^(1/k)  (k=3 central; k=1 additive, k→∞ hard max)
  => realized MBU(model, batch, GPU) = (Wb+KVb) / (BW·t)  is a *derived* quantity
     that rises with per-layer weight size and saturates at η.
Power cap: if E_dyn/t + P_static > P_cap the GPU throttles: t = E_dyn/(P_cap-P_static).
Decode = one weight sweep per generated token (MoE: expert occupancy via
energy_model.weight_params_for_tokens). Prefill = chunked passes over the prompt.

UNCERTAINTY: every prediction is a Monte-Carlo over (a) coefficient ranges,
(b) an ensemble of time-model fits (3 functional forms × leave-one-model-out
refits), (c) the prefill-MFU prior. Rankings are reported with P(win).

CLI (compatible with v1):
  python3 recommend_gpu.py <hf_model_id> [--prompt 2048 --gen 256 --batch 32]
         [--gpus H200,B200,...] [--draws 300] [--winmap]
Needs ~/models.py importable (run in the container, or any python>=3.7 with it
on sys.path). No numpy needed.
"""
import argparse
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (HERE, os.path.expanduser("~"), "/workspace"):
    if _d not in sys.path:
        sys.path.append(_d)

DTYPE_B = 2                   # fp16/bf16 weights and KV
PREFILL_CHUNK = 8192          # vLLM V1 max_num_batched_tokens-style chunk
MEM_UTIL = 0.9                # vLLM gpu_memory_utilization
COEFF_JSON = os.path.join(HERE, "gpu_coefficients.json")

# ---------------------------------------------------------------------------
# 1. GPU hardware + energy coefficients
# ---------------------------------------------------------------------------
# Built-in MEASURED defaults (used only if gpu_coefficients.json is absent).
# Canonical ~/models.py accounting (reconcile_gpus.py, 2026-09-24).
BUILTIN_MEASURED = {
    "H200": dict(gpu_name="NVIDIA H200", mem_tech="HBM3e", bw=4.8e12, peak_flops=9.9e14,
                 p_cap=700.0, p_static=116.4, p_static_range=[115.9, 118.4],
                 e_wbyte=1.076e-10, e_kvbyte=3.376e-10, e_gemm=0.680e-12,
                 e_wbyte_ci=[1.073e-10, 1.080e-10], e_kvbyte_ci=[3.285e-10, 3.465e-10],
                 e_gemm_ci=[0.656e-12, 0.702e-12], source="builtin: logs/ragged 3-term fit"),
    "B200": dict(gpu_name="NVIDIA B200", mem_tech="HBM3e", bw=8.0e12, peak_flops=2.25e15,
                 p_cap=1000.0, p_static=237.8, p_static_range=[236.3, 241.7],
                 e_wbyte=1.252e-10, e_kvbyte=3.043e-10, e_gemm=0.539e-12,
                 e_wbyte_ci=[1.247e-10, 1.257e-10], e_kvbyte_ci=[2.729e-10, 3.319e-10],
                 e_gemm_ci=[0.497e-12, 0.588e-12], source="builtin: logs/B200 3-term fit"),
}
# Memory capacity (bytes) — not in the coefficient json.
MEM_BYTES = {"H200": 141e9, "B200": 180e9, "A100-80-PCIe": 80e9, "A100-80-SXM": 80e9,
             "H100-SXM": 80e9, "L40S": 48e9, "A5000": 24e9}

# Unmeasured / partially measured GPUs: datasheet specs + memory-technology prior.
# p_static values here are ESTIMATES (except A100-PCIe: measured idle 73-76 W).
DATASHEET = {
    "A100-80-PCIe": dict(gpu_name="NVIDIA A100 80GB PCIe", mem_tech="HBM2e", bw=1.94e12,
                         peak_flops=3.12e14, p_cap=300.0, p_static=74.0,
                         p_static_range=[73.0, 76.0], note="idle MEASURED; power-capped at every measured point"),
    "A100-80-SXM": dict(gpu_name="NVIDIA A100 80GB SXM", mem_tech="HBM2e", bw=2.039e12,
                        peak_flops=3.12e14, p_cap=400.0, p_static=60.0, p_static_range=[50.0, 90.0]),
    "H100-SXM": dict(gpu_name="NVIDIA H100 SXM", mem_tech="HBM3", bw=3.35e12, peak_flops=9.9e14,
                     p_cap=700.0, p_static=110.0, p_static_range=[70.0, 130.0]),
    "L40S": dict(gpu_name="NVIDIA L40S", mem_tech="GDDR6", bw=0.864e12, peak_flops=3.62e14,
                 p_cap=350.0, p_static=35.0, p_static_range=[25.0, 60.0]),
    "A5000": dict(gpu_name="NVIDIA RTX A5000", mem_tech="GDDR6", bw=0.768e12, peak_flops=1.11e14,
                  p_cap=230.0, p_static=25.0, p_static_range=[15.0, 35.0]),
}

# Memory-technology priors for GPUs with no measured coefficients: (center, lo, hi).
#  HBM3/3e : spans every measured direct J/byte on H200 and B200 (models >=3B).
#  HBM2e   : A100-PCIe direct c=1 dyn-energy/byte = 1.7-1.8e-10 while pinned at its
#            300 W cap — the only HBM2e evidence we have; wide.
#  GDDR6   : UNKNOWN until the A5000 measurement lands — very wide. If
#            gpu_coefficients.json gains an entry with mem_tech GDDR6 (the A5000),
#            the prior is replaced by that measurement (±15%).
#  e_gemm  : between "invariant" (H200/B200 measured values) and "∝ 1/peak"
#            scaled from H200 — the law is only weakly supported for compute.
MEM_PRIOR = {
    "HBM3e": dict(e_wbyte=(1.16e-10, 1.03e-10, 1.27e-10), e_kvbyte=(3.2e-10, 2.7e-10, 3.5e-10)),
    "HBM3":  dict(e_wbyte=(1.16e-10, 1.03e-10, 1.30e-10), e_kvbyte=(3.2e-10, 2.7e-10, 3.6e-10)),
    "HBM2e": dict(e_wbyte=(1.8e-10, 1.4e-10, 2.4e-10), e_kvbyte=(4.5e-10, 3.0e-10, 6.5e-10)),
    "GDDR6": dict(e_wbyte=(1.8e-10, 1.0e-10, 3.5e-10), e_kvbyte=(4.5e-10, 2.5e-10, 9.0e-10)),
}
MFU_PREFILL = (0.65, 0.50, 0.75)   # prior; B200 cuBLAS 8192^3 measured 68% of dense peak

# ---------------------------------------------------------------------------
# 2. Realized-utilization (iteration-time) model — fitted by
#    `recommender_backtest.py --fit` on our vLLM 0.10.2 runs. Keys:
#    t0 [s], tau [s/layer], eta, eta_kv, mu (per-token marginal, units of peak), k,
#    tau_moe [s/layer] (MoE layer latency floor; H200 Qwen3-30B-A3B only).
#    CENTRAL = form k=3 fit on all dense runs of that GPU; ENSEMBLE = forms
#    k∈{1,3,50} × {all, leave-one-model-out}. "_prior" (unmeasured GPUs) = union of
#    the measured GPUs' ensembles (same software stack assumed).
# ---------------------------------------------------------------------------
TIME_CENTRAL = {}    # filled below from TIME_FIT
TIME_ENSEMBLE = {}
TIME_FIT = {"B200": {"central": {"eta": 0.867301796, "eta_kv": 0.369685427, "form": "additive(k=1)", "held_out": None, "k": 1.0, "lomo_mape_t": 2.4806937, "mu": 0.685375689, "t0": 0.001376957, "tau": 8.0076e-05, "tau_moe": 0.000353202, "tau_moe_note": "scaled from H200 (no MoE run on this GPU)"}, "ensemble": [{"eta": 0.867301796, "eta_kv": 0.369685427, "form": "additive(k=1)", "held_out": None, "k": 1.0, "lomo_mape_t": 2.4806937, "mu": 0.685375689, "t0": 0.001376957, "tau": 8.0076e-05, "tau_moe": 0.000353202, "tau_moe_note": "scaled from H200 (no MoE run on this GPU)"}, {"eta": 0.857113606, "eta_kv": 0.404651846, "form": "additive(k=1)", "held_out": "Qwen/Qwen2-72B-Instruct", "k": 1.0, "mu": 0.645180473, "t0": 0.001432324, "tau": 7.7245e-05, "tau_moe": 0.000340716, "tau_moe_note": "scaled from H200 (no MoE run on this GPU)"}, {"eta": 0.879263044, "eta_kv": 0.267912716, "form": "additive(k=1)", "held_out": "Qwen/Qwen2-7B-Instruct", "k": 1.0, "mu": 0.899660075, "t0": 0.001123418, "tau": 8.8101e-05, "tau_moe": 0.000388601, "tau_moe_note": "scaled from H200 (no MoE run on this GPU)"}, {"eta": 0.8866134, "eta_kv": 0.557865605, "form": "additive(k=1)", "held_out": "Qwen/Qwen2.5-32B-Instruct", "k": 1.0, "mu": 0.55224043, "t0": 0.001253641, "tau": 8.5084e-05, "tau_moe": 0.000375295, "tau_moe_note": "scaled from H200 (no MoE run on this GPU)"}, {"eta": 0.865383167, "eta_kv": 0.206980593, "form": "additive(k=1)", "held_out": "mistralai/Mistral-7B-v0.1", "k": 1.0, "mu": 0.924080658, "t0": 0.001393021, "tau": 8.012e-05, "tau_moe": 0.0003534, "tau_moe_note": "scaled from H200 (no MoE run on this GPU)"}, {"eta": 0.657122552, "eta_kv": 1.0, "form": "soft(k=3)", "held_out": None, "k": 3.0, "mu": 0.521580346, "t0": 0.0, "tau": 0.000188217, "tau_moe": 0.000830196, "tau_moe_note": "scaled from H200 (no MoE run on this GPU)"}, {"eta": 0.606144973, "eta_kv": 0.404651786, "form": "soft(k=3)", "held_out": "Qwen/Qwen2-72B-Instruct", "k": 3.0, "mu": 0.645180512, "t0": 0.001868065, "tau": 0.000110496, "tau_moe": 0.000487382, "tau_moe_note": "scaled from H200 (no MoE run on this GPU)"}, {"eta": 0.648154031, "eta_kv": 0.461718732, "form": "soft(k=3)", "held_out": "Qwen/Qwen2-7B-Instruct", "k": 3.0, "mu": 0.620458277, "t0": 0.0, "tau": 0.000183903, "tau_moe": 0.00081117, "tau_moe_note": "scaled from H200 (no MoE run on this GPU)"}, {"eta": 0.691457156, "eta_kv": 0.557865375, "form": "soft(k=3)", "held_out": "Qwen/Qwen2.5-32B-Instruct", "k": 3.0, "mu": 0.552240495, "t0": 0.002028938, "tau": 0.000110946, "tau_moe": 0.000489365, "tau_moe_note": "scaled from H200 (no MoE run on this GPU)"}, {"eta": 0.662507433, "eta_kv": 0.177381837, "form": "soft(k=3)", "held_out": "mistralai/Mistral-7B-v0.1", "k": 3.0, "mu": 1.012636405, "t0": 0.0, "tau": 0.000197003, "tau_moe": 0.000868953, "tau_moe_note": "scaled from H200 (no MoE run on this GPU)"}, {"eta": 0.679179225, "eta_kv": 0.360284087, "form": "max(k=50)", "held_out": None, "k": 50.0, "mu": 0.625088766, "t0": 0.003116846, "tau": 8.8279e-05, "tau_moe": 0.000389384, "tau_moe_note": "scaled from H200 (no MoE run on this GPU)"}, {"eta": 0.618884531, "eta_kv": 0.404651807, "form": "max(k=50)", "held_out": "Qwen/Qwen2-72B-Instruct", "k": 50.0, "mu": 0.645180489, "t0": 0.002784949, "tau": 9.9776e-05, "tau_moe": 0.000440097, "tau_moe_note": "scaled from H200 (no MoE run on this GPU)"}, {"eta": 0.692125214, "eta_kv": 0.415381968, "form": "max(k=50)", "held_out": "Qwen/Qwen2-7B-Instruct", "k": 50.0, "mu": 0.62179391, "t0": 0.003469729, "tau": 3e-05, "tau_moe": 0.000132326, "tau_moe_note": "scaled from H200 (no MoE run on this GPU)"}, {"eta": 0.710493804, "eta_kv": 0.557865553, "form": "max(k=50)", "held_out": "Qwen/Qwen2.5-32B-Instruct", "k": 50.0, "mu": 0.552240445, "t0": 0.003065723, "tau": 9.111e-05, "tau_moe": 0.000401873, "tau_moe_note": "scaled from H200 (no MoE run on this GPU)"}, {"eta": 0.67587757, "eta_kv": 0.120308006, "form": "max(k=50)", "held_out": "mistralai/Mistral-7B-v0.1", "k": 50.0, "mu": 1.446572148, "t0": 0.003115816, "tau": 6.4331e-05, "tau_moe": 0.000283753, "tau_moe_note": "scaled from H200 (no MoE run on this GPU)"}], "lomo": {"Qwen/Qwen2-72B-Instruct": {"eta": 0.857113606, "eta_kv": 0.404651846, "form": "additive(k=1)", "held_out": "Qwen/Qwen2-72B-Instruct", "k": 1.0, "mu": 0.645180473, "t0": 0.001432324, "tau": 7.7245e-05, "tau_moe": 0.000340716, "tau_moe_note": "scaled from H200 (no MoE run on this GPU)"}, "Qwen/Qwen2-7B-Instruct": {"eta": 0.879263044, "eta_kv": 0.267912716, "form": "additive(k=1)", "held_out": "Qwen/Qwen2-7B-Instruct", "k": 1.0, "mu": 0.899660075, "t0": 0.001123418, "tau": 8.8101e-05, "tau_moe": 0.000388601, "tau_moe_note": "scaled from H200 (no MoE run on this GPU)"}, "Qwen/Qwen2.5-32B-Instruct": {"eta": 0.8866134, "eta_kv": 0.557865605, "form": "additive(k=1)", "held_out": "Qwen/Qwen2.5-32B-Instruct", "k": 1.0, "mu": 0.55224043, "t0": 0.001253641, "tau": 8.5084e-05, "tau_moe": 0.000375295, "tau_moe_note": "scaled from H200 (no MoE run on this GPU)"}, "mistralai/Mistral-7B-v0.1": {"eta": 0.865383167, "eta_kv": 0.206980593, "form": "additive(k=1)", "held_out": "mistralai/Mistral-7B-v0.1", "k": 1.0, "mu": 0.924080658, "t0": 0.001393021, "tau": 8.012e-05, "tau_moe": 0.0003534, "tau_moe_note": "scaled from H200 (no MoE run on this GPU)"}}}, "H200": {"central": {"eta": 0.724375312, "eta_kv": 0.222481722, "form": "max(k=50)", "held_out": None, "k": 50.0, "lomo_mape_t": 5.32153283, "mu": 3.210777598, "t0": 0.002148718, "tau": 0.00010979, "tau_moe": 0.000484267}, "ensemble": [{"eta": 0.95, "eta_kv": 0.269553948, "form": "additive(k=1)", "held_out": None, "k": 1.0, "mu": 5.0, "t0": 0.003241441, "tau": 2.4697e-05, "tau_moe": 0.000295277}, {"eta": 0.95, "eta_kv": 0.281693566, "form": "additive(k=1)", "held_out": "Qwen/Qwen2-7B-Instruct", "k": 1.0, "mu": 5.0, "t0": 0.003500406, "tau": 1.9906e-05, "tau_moe": 0.000290262}, {"eta": 0.95, "eta_kv": 0.255567609, "form": "additive(k=1)", "held_out": "Qwen/Qwen2.5-0.5B-Instruct", "k": 1.0, "mu": 5.0, "t0": 0.002068835, "tau": 5.6579e-05, "tau_moe": 0.000319224}, {"eta": 0.95, "eta_kv": 0.259870642, "form": "additive(k=1)", "held_out": "Qwen/Qwen2.5-1.5B-Instruct", "k": 1.0, "mu": 5.0, "t0": 0.002949937, "tau": 3.0205e-05, "tau_moe": 0.000301022}, {"eta": 0.95, "eta_kv": 0.274218838, "form": "additive(k=1)", "held_out": "Qwen/Qwen2.5-14B-Instruct", "k": 1.0, "mu": 5.0, "t0": 0.003072837, "tau": 3.0854e-05, "tau_moe": 0.00029894}, {"eta": 0.95, "eta_kv": 0.276222949, "form": "additive(k=1)", "held_out": "Qwen/Qwen2.5-32B-Instruct", "k": 1.0, "mu": 5.0, "t0": 0.003773105, "tau": 6.401e-06, "tau_moe": 0.000284414}, {"eta": 0.95, "eta_kv": 0.25700819, "form": "additive(k=1)", "held_out": "Qwen/Qwen2.5-3B-Instruct", "k": 1.0, "mu": 3.646666694, "t0": 0.003790196, "tau": 1.214e-06, "tau_moe": 0.000283085}, {"eta": 0.95, "eta_kv": 0.28303588, "form": "additive(k=1)", "held_out": "Qwen/Qwen2.5-7B-Instruct", "k": 1.0, "mu": 4.844761718, "t0": 0.003547173, "tau": 1.8907e-05, "tau_moe": 0.000289299}, {"eta": 0.95, "eta_kv": 0.310244035, "form": "additive(k=1)", "held_out": "google/gemma-7b", "k": 1.0, "mu": 4.255474481, "t0": 0.003113535, "tau": 2.8081e-05, "tau_moe": 0.000298942}, {"eta": 0.95, "eta_kv": 0.274079253, "form": "additive(k=1)", "held_out": "meta-llama/Meta-Llama-3-8B", "k": 1.0, "mu": 5.0, "t0": 0.003184443, "tau": 2.9383e-05, "tau_moe": 0.00029661}, {"eta": 0.95, "eta_kv": 0.258473697, "form": "additive(k=1)", "held_out": "mistralai/Mistral-7B-v0.1", "k": 1.0, "mu": 5.0, "t0": 0.003188256, "tau": 2.9492e-05, "tau_moe": 0.000296007}, {"eta": 0.758854645, "eta_kv": 0.220153436, "form": "soft(k=3)", "held_out": None, "k": 3.0, "mu": 3.550015206, "t0": 0.001582495, "tau": 0.000125538, "tau_moe": 0.000472383}, {"eta": 0.75429443, "eta_kv": 0.216944935, "form": "soft(k=3)", "held_out": "Qwen/Qwen2-7B-Instruct", "k": 3.0, "mu": 4.980991678, "t0": 0.001647608, "tau": 0.000123515, "tau_moe": 0.000470718}, {"eta": 0.754523641, "eta_kv": 0.220965136, "form": "soft(k=3)", "held_out": "Qwen/Qwen2.5-0.5B-Instruct", "k": 3.0, "mu": 3.582995395, "t0": 0.001548359, "tau": 0.000126198, "tau_moe": 0.000472833}, {"eta": 0.755423022, "eta_kv": 0.220669761, "form": "soft(k=3)", "held_out": "Qwen/Qwen2.5-1.5B-Instruct", "k": 3.0, "mu": 3.69087307, "t0": 0.001588476, "tau": 0.000124341, "tau_moe": 0.000472026}, {"eta": 0.750917635, "eta_kv": 0.218386494, "form": "soft(k=3)", "held_out": "Qwen/Qwen2.5-14B-Instruct", "k": 3.0, "mu": 3.510123795, "t0": 0.001516031, "tau": 0.000127934, "tau_moe": 0.000473099}, {"eta": 0.77917241, "eta_kv": 0.217203028, "form": "soft(k=3)", "held_out": "Qwen/Qwen2.5-32B-Instruct", "k": 3.0, "mu": 3.723801814, "t0": 0.001706596, "tau": 0.000121701, "tau_moe": 0.000471156}, {"eta": 0.768182617, "eta_kv": 0.221893472, "form": "soft(k=3)", "held_out": "Qwen/Qwen2.5-3B-Instruct", "k": 3.0, "mu": 3.573532515, "t0": 0.001832387, "tau": 0.000113591, "tau_moe": 0.000467617}, {"eta": 0.756960475, "eta_kv": 0.222657173, "form": "soft(k=3)", "held_out": "Qwen/Qwen2.5-7B-Instruct", "k": 3.0, "mu": 4.351150599, "t0": 0.001743838, "tau": 0.000120222, "tau_moe": 0.000468923}, {"eta": 0.779378679, "eta_kv": 0.519091518, "form": "soft(k=3)", "held_out": "google/gemma-7b", "k": 3.0, "mu": 1.133673909, "t0": 0.001046281, "tau": 0.000145849, "tau_moe": 0.000489584}, {"eta": 0.756121131, "eta_kv": 0.220686527, "form": "soft(k=3)", "held_out": "meta-llama/Meta-Llama-3-8B", "k": 3.0, "mu": 3.593760045, "t0": 0.001620341, "tau": 0.000124677, "tau_moe": 0.000471335}, {"eta": 0.756875254, "eta_kv": 0.211663899, "form": "soft(k=3)", "held_out": "mistralai/Mistral-7B-v0.1", "k": 3.0, "mu": 3.625334374, "t0": 0.001627422, "tau": 0.000124976, "tau_moe": 0.000470799}, {"eta": 0.724375312, "eta_kv": 0.222481722, "form": "max(k=50)", "held_out": None, "k": 50.0, "lomo_mape_t": 5.32153283, "mu": 3.210777598, "t0": 0.002148718, "tau": 0.00010979, "tau_moe": 0.000484267}, {"eta": 0.720797555, "eta_kv": 0.218179842, "form": "max(k=50)", "held_out": "Qwen/Qwen2-7B-Instruct", "k": 50.0, "mu": 4.403819474, "t0": 0.002169355, "tau": 0.000109238, "tau_moe": 0.000484002}, {"eta": 0.734349458, "eta_kv": 0.225233565, "form": "max(k=50)", "held_out": "Qwen/Qwen2.5-0.5B-Instruct", "k": 50.0, "mu": 3.231184721, "t0": 0.002233156, "tau": 0.000110046, "tau_moe": 0.000482647}, {"eta": 0.723745594, "eta_kv": 0.222878581, "form": "max(k=50)", "held_out": "Qwen/Qwen2.5-1.5B-Instruct", "k": 50.0, "mu": 3.338398396, "t0": 0.002150155, "tau": 0.000110478, "tau_moe": 0.000484309}, {"eta": 0.724527812, "eta_kv": 0.221869886, "form": "max(k=50)", "held_out": "Qwen/Qwen2.5-14B-Instruct", "k": 50.0, "mu": 3.067894021, "t0": 0.002145612, "tau": 0.000109852, "tau_moe": 0.000484238}, {"eta": 0.650752804, "eta_kv": 0.237893012, "form": "max(k=50)", "held_out": "Qwen/Qwen2.5-32B-Instruct", "k": 50.0, "mu": 2.493802837, "t0": 0.001659385, "tau": 0.00012672, "tau_moe": 0.000494755}, {"eta": 0.737735972, "eta_kv": 0.223131485, "form": "max(k=50)", "held_out": "Qwen/Qwen2.5-3B-Instruct", "k": 50.0, "mu": 3.364155512, "t0": 0.002260655, "tau": 0.000101667, "tau_moe": 0.00048203}, {"eta": 0.72418673, "eta_kv": 0.22365946, "form": "max(k=50)", "held_out": "Qwen/Qwen2.5-7B-Instruct", "k": 50.0, "mu": 3.94639194, "t0": 0.002223607, "tau": 0.000107357, "tau_moe": 0.000483022}, {"eta": 0.739245346, "eta_kv": 0.243220079, "form": "max(k=50)", "held_out": "google/gemma-7b", "k": 50.0, "mu": 1.994984986, "t0": 0.002085278, "tau": 0.000111739, "tau_moe": 0.000485656}, {"eta": 0.723414756, "eta_kv": 0.221084418, "form": "max(k=50)", "held_out": "meta-llama/Meta-Llama-3-8B", "k": 50.0, "mu": 3.36096876, "t0": 0.002140147, "tau": 0.000110106, "tau_moe": 0.00048444}, {"eta": 0.727338767, "eta_kv": 0.219300386, "form": "max(k=50)", "held_out": "mistralai/Mistral-7B-v0.1", "k": 50.0, "mu": 3.074032473, "t0": 0.002172441, "tau": 0.00010889, "tau_moe": 0.000483557}], "lomo": {"Qwen/Qwen2-7B-Instruct": {"eta": 0.720797555, "eta_kv": 0.218179842, "form": "max(k=50)", "held_out": "Qwen/Qwen2-7B-Instruct", "k": 50.0, "mu": 4.403819474, "t0": 0.002169355, "tau": 0.000109238, "tau_moe": 0.000484002}, "Qwen/Qwen2.5-0.5B-Instruct": {"eta": 0.734349458, "eta_kv": 0.225233565, "form": "max(k=50)", "held_out": "Qwen/Qwen2.5-0.5B-Instruct", "k": 50.0, "mu": 3.231184721, "t0": 0.002233156, "tau": 0.000110046, "tau_moe": 0.000482647}, "Qwen/Qwen2.5-1.5B-Instruct": {"eta": 0.723745594, "eta_kv": 0.222878581, "form": "max(k=50)", "held_out": "Qwen/Qwen2.5-1.5B-Instruct", "k": 50.0, "mu": 3.338398396, "t0": 0.002150155, "tau": 0.000110478, "tau_moe": 0.000484309}, "Qwen/Qwen2.5-14B-Instruct": {"eta": 0.724527812, "eta_kv": 0.221869886, "form": "max(k=50)", "held_out": "Qwen/Qwen2.5-14B-Instruct", "k": 50.0, "mu": 3.067894021, "t0": 0.002145612, "tau": 0.000109852, "tau_moe": 0.000484238}, "Qwen/Qwen2.5-32B-Instruct": {"eta": 0.650752804, "eta_kv": 0.237893012, "form": "max(k=50)", "held_out": "Qwen/Qwen2.5-32B-Instruct", "k": 50.0, "mu": 2.493802837, "t0": 0.001659385, "tau": 0.00012672, "tau_moe": 0.000494755}, "Qwen/Qwen2.5-3B-Instruct": {"eta": 0.737735972, "eta_kv": 0.223131485, "form": "max(k=50)", "held_out": "Qwen/Qwen2.5-3B-Instruct", "k": 50.0, "mu": 3.364155512, "t0": 0.002260655, "tau": 0.000101667, "tau_moe": 0.00048203}, "Qwen/Qwen2.5-7B-Instruct": {"eta": 0.72418673, "eta_kv": 0.22365946, "form": "max(k=50)", "held_out": "Qwen/Qwen2.5-7B-Instruct", "k": 50.0, "mu": 3.94639194, "t0": 0.002223607, "tau": 0.000107357, "tau_moe": 0.000483022}, "google/gemma-7b": {"eta": 0.739245346, "eta_kv": 0.243220079, "form": "max(k=50)", "held_out": "google/gemma-7b", "k": 50.0, "mu": 1.994984986, "t0": 0.002085278, "tau": 0.000111739, "tau_moe": 0.000485656}, "meta-llama/Meta-Llama-3-8B": {"eta": 0.723414756, "eta_kv": 0.221084418, "form": "max(k=50)", "held_out": "meta-llama/Meta-Llama-3-8B", "k": 50.0, "mu": 3.36096876, "t0": 0.002140147, "tau": 0.000110106, "tau_moe": 0.00048444}, "mistralai/Mistral-7B-v0.1": {"eta": 0.727338767, "eta_kv": 0.219300386, "form": "max(k=50)", "held_out": "mistralai/Mistral-7B-v0.1", "k": 50.0, "mu": 3.074032473, "t0": 0.002172441, "tau": 0.00010889, "tau_moe": 0.000483557}}}}  # generated by recommender_backtest.py --fit


def _load_time_fit():
    global TIME_CENTRAL, TIME_ENSEMBLE
    fit = TIME_FIT or {}
    for g, d in fit.items():
        TIME_CENTRAL[g] = d["central"]
        TIME_ENSEMBLE[g] = d["ensemble"]
    if TIME_ENSEMBLE:
        pooled = [p for g, e in TIME_ENSEMBLE.items() for p in e]
        TIME_ENSEMBLE["_prior"] = pooled
        # central prior: parameter-wise median of the measured centrals
        keys = ["t0", "tau", "eta", "eta_kv", "mu", "k", "tau_moe"]
        cents = list(TIME_CENTRAL.values())
        TIME_CENTRAL["_prior"] = {k: sorted(c[k] for c in cents)[len(cents) // 2] for k in keys}


# ---------------------------------------------------------------------------
# 3. GPU table assembly
# ---------------------------------------------------------------------------
def _rng(v, ci, lo_rel=None):
    lo, hi = (ci if ci else (v, v))
    return (v, min(lo, v), max(hi, v))


def load_gpus(path=COEFF_JSON):
    """Return {name: gpu dict}. Measured entries come from gpu_coefficients.json
    (else built-ins); every other GPU gets datasheet specs + memory-tech prior."""
    measured = {}
    src = "builtin"
    if os.path.exists(path):
        try:
            js = json.load(open(path))
            measured = {k: v for k, v in js.items() if not k.startswith("_") and isinstance(v, dict)}
            src = os.path.basename(path)
        except Exception as e:  # malformed json -> builtins
            print("[warn] could not read %s (%r); using built-ins" % (path, e), file=sys.stderr)
    for k, v in BUILTIN_MEASURED.items():
        measured.setdefault(k, v)

    gpus = {}
    for name, e in measured.items():
        if e.get("e_wbyte") is None:
            continue
        spec = dict(DATASHEET.get(name, {}))
        spec.update({k: v for k, v in e.items() if v is not None})
        if not spec.get("peak_flops") or not spec.get("bw"):
            continue
        g = dict(spec, name=name, energy_src="measured (%s)" % (e.get("source") or src),
                 measured=True, mem=MEM_BYTES.get(name, spec.get("mem_bytes", 80e9)))
        g["r_wbyte"] = _rng(e["e_wbyte"], e.get("e_wbyte_ci"))
        # widen the byte coefficient to the cross-model spread of direct c=1 J/byte
        dj = [v for m, v in (e.get("direct_c1_jbyte") or {}).items()
              if not any(s in m for s in ("0.5B", "1.5B"))]   # MBU<0.2 inflates J/byte
        if dj:
            c, lo, hi = g["r_wbyte"]
            g["r_wbyte"] = (c, min(lo, min(dj)), max(hi, max(dj)))
        g["r_kvbyte"] = _rng(e["e_kvbyte"], e.get("e_kvbyte_ci"))
        # e_gemm: bootstrap CIs are far too narrow (the compute coefficient drifts
        # with model size: B200 7B->72B ~0.54->0.37 pJ; H200 ladder ~2x). Use ±25%.
        cg = e["e_gemm"]
        lo_g, hi_g = (e.get("e_gemm_ci") or [cg, cg])
        g["r_gemm"] = (cg, min(lo_g, 0.72 * cg), max(hi_g, 1.12 * cg))
        g["r_static"] = _rng(g["p_static"], e.get("p_static_range"))
        g["time_key"] = name if name in TIME_CENTRAL else "_prior"
        g["confidence"] = "measured" if g["time_key"] == name else "measured-energy / prior-time"
        gpus[name] = g

    # memory-tech prior, possibly replaced by a measured GPU of that technology
    prior = {k: dict(v) for k, v in MEM_PRIOR.items()}
    for name, g in gpus.items():
        tech = g.get("mem_tech")
        if tech and tech not in ("HBM3e",):   # a measured non-HBM3e GPU defines its tech's prior
            c = g["e_wbyte"]
            prior[tech] = dict(e_wbyte=(c, 0.85 * c, 1.15 * c),
                               e_kvbyte=(g["e_kvbyte"], 0.8 * g["e_kvbyte"], 1.25 * g["e_kvbyte"]))
    h = gpus.get("H200")
    for name, spec in DATASHEET.items():
        if name in gpus:
            continue
        pr = prior.get(spec["mem_tech"], prior["GDDR6"])
        e_g_inv = h["e_gemm"] if h else 0.68e-12
        e_g_law = e_g_inv * (h["peak_flops"] if h else 9.9e14) / spec["peak_flops"]
        lo_g, hi_g = sorted([e_g_inv, e_g_law])
        cg = math.sqrt(lo_g * hi_g)
        g = dict(spec, name=name, measured=False, mem=MEM_BYTES.get(name, 80e9),
                 e_wbyte=pr["e_wbyte"][0], e_kvbyte=pr["e_kvbyte"][0], e_gemm=cg,
                 r_wbyte=pr["e_wbyte"], r_kvbyte=pr["e_kvbyte"],
                 r_gemm=(cg, 0.8 * lo_g, 1.2 * hi_g),
                 r_static=(spec["p_static"],) + tuple(spec["p_static_range"]),
                 energy_src="PRIOR (%s band)" % spec["mem_tech"], time_key="_prior",
                 confidence="LOW (prior)")
        gpus[name] = g
    return gpus


# ---------------------------------------------------------------------------
# 4. Model resolution + per-iteration work
# ---------------------------------------------------------------------------
def resolve_model(model_id):
    from models import MODELS, model_from_hf_id
    short = model_id.split("/")[-1]
    try:
        from gate_dram import HF_TO_KEY
        if model_id in HF_TO_KEY and HF_TO_KEY[model_id] in MODELS:
            return MODELS[HF_TO_KEY[model_id]]
    except Exception:
        pass
    if short in MODELS:
        return MODELS[short]
    norm = short.lower().replace("_", "-")
    for k in MODELS:
        if k.lower() == norm:
            return MODELS[k]
    return model_from_hf_id(model_id)


def weight_params(m, tokens):
    try:
        from energy_model import weight_params_for_tokens
        return weight_params_for_tokens(m, tokens)
    except Exception:
        return m.active_params()


def is_moe(m):
    return getattr(m, "ffn", "mlp") == "moe"


def serving_iter_work(m, n, P, G):
    """Mean per-iteration work in steady-state continuous batching: n running
    sequences each generating 1 token, plus prefill chunks for arriving requests
    (n·P/G tokens on average); mean resident context P + G/2."""
    T = n * (1.0 + P / max(G, 1.0))
    ctx = P + G / 2.0
    Wb = weight_params(m, T) * DTYPE_B
    KVb = n * ctx * m.kv_bytes_per_token(DTYPE_B)
    Fg = 2.0 * m.active_params() * T
    Fa = m.attn_flops_per_token(1) * n * ctx
    return dict(Wb=Wb, KVb=KVb, Fg=Fg, F=Fg + Fa, T=T)


# ---------------------------------------------------------------------------
# 5. Iteration prediction
# ---------------------------------------------------------------------------
def softmax_k(a, b, k):
    mx = max(a, b)
    if mx <= 0:
        return 0.0
    return mx * ((a / mx) ** k + (b / mx) ** k) ** (1.0 / k)


def iter_time(g, tp, work, L, moe=False, mfu_pf=None, n_gpu=1, tau_ar=0.0):
    """Uncapped iteration time (s) from the realized-utilization model, and which
    resource dominates it ("memory": weight/KV streaming + per-layer floors;
    "compute": the FLOP terms)."""
    bw = g["bw"] * n_gpu
    pk = g["peak_flops"] * n_gpu
    tau = (tp["tau_moe"] if (moe and tp.get("tau_moe")) else tp["tau"]) + tau_ar
    per_layer = softmax_k(tau, work["Wb"] / L / (bw * tp["eta"]), tp["k"])
    t_stream = tp["t0"] + L * per_layer + work["KVb"] / (bw * tp["eta_kv"])
    t_tok = work["F"] / (pk * tp["mu"])
    mfu = mfu_pf if mfu_pf is not None else MFU_PREFILL[0]
    t_cmp = tp["t0"] + work["F"] / (pk * mfu)
    t = max(t_stream + t_tok, t_cmp)
    bound = "compute" if max(t_tok, t_cmp - tp["t0"]) > t_stream - tp["t0"] else "memory"
    return t, bound


# TP>1 is NOT measured: all-reduce latency per layer (tau_ar) plus a time-inflation
# factor for synchronization / imperfect overlap; both drawn in Monte Carlo.
TP_TIME_PENALTY = (1.15, 1.0, 1.35)


def iter_predict(g, coef, tp, work, L, moe=False, mfu_pf=None, n_gpu=1, tau_ar=0.0, t_scale=1.0):
    """-> dict(t, power, e_dyn, energy, capped, bound). coef = (e_w, e_kv, e_g, p_static)."""
    e_w, e_kv, e_g, p_s = coef
    t, bound = iter_time(g, tp, work, L, moe, mfu_pf, n_gpu, tau_ar)
    t *= t_scale
    e_dyn = e_w * work["Wb"] + e_kv * work["KVb"] + e_g * work["Fg"]
    p_s_tot, cap = p_s * n_gpu, g["p_cap"] * n_gpu
    capped = False
    if e_dyn / t + p_s_tot > cap:          # throttle: power pinned at cap
        t = e_dyn / (cap - p_s_tot)
        capped = True
    power = e_dyn / t + p_s_tot
    return dict(t=t, power=power, e_dyn=e_dyn, energy=e_dyn + p_s_tot * t,
                capped=capped, bound=bound)


def central_draw(g):
    return dict(coef=(g["e_wbyte"], g["e_kvbyte"], g["e_gemm"], g["p_static"]),
                tp=TIME_CENTRAL[g["time_key"]], mfu=MFU_PREFILL[0], tau_ar=25e-6,
                tp_pen=TP_TIME_PENALTY[0])


def random_draw(g, rnd):
    u = lambda r: rnd.uniform(r[1], r[2])
    return dict(coef=(u(g["r_wbyte"]), u(g["r_kvbyte"]), u(g["r_gemm"]), u(g["r_static"])),
                tp=rnd.choice(TIME_ENSEMBLE[g["time_key"]]),
                mfu=rnd.uniform(MFU_PREFILL[1], MFU_PREFILL[2]),
                tau_ar=rnd.uniform(10e-6, 60e-6),
                tp_pen=rnd.uniform(TP_TIME_PENALTY[1], TP_TIME_PENALTY[2]))


def n_gpus_needed(g, m, kv_bytes_total):
    """Smallest TP degree (1,2,4,8) whose memory holds weights + requested KV."""
    wtot = m.total_params() * DTYPE_B
    for n in (1, 2, 4, 8):
        if wtot / n + kv_bytes_total / n <= MEM_UTIL * g["mem"] * 0.97:
            return n
    return None


# ---------------------------------------------------------------------------
# 6. Serving-run prediction (used by the backtest)
# ---------------------------------------------------------------------------
def predict_serving(g, m, P, G, conc=None, rate=None, draw=None):
    """Predict tok/s, avg power, J/token for a closed-loop (conc) or Poisson (rate)
    serving run, given only model, GPU, mean prompt P and gen G lengths."""
    d = draw or central_draw(g)
    L, moe = m.L, is_moe(m)

    def at(n):
        w = serving_iter_work(m, n, P, G)
        return w, iter_predict(g, d["coef"], d["tp"], w, L, moe, d["mfu"])

    if conc is not None:
        n = float(conc)
        w, r = at(n)
        tok_s = n / r["t"]
        power = r["power"]
    else:  # open loop: Little's law fixed point n = λ·G·t_iter(n)
        n = 1.0
        for _ in range(60):
            w, r = at(n)
            n_new = max(1.0, rate * G * r["t"])
            if abs(n_new - n) < 1e-3 * n:
                break
            n = 0.5 * (n + n_new)
        w, r = at(n)
        busy = min(1.0, rate * G * r["t"] / n)         # <1 only if the queue ever empties
        tok_s = min(rate * G, n / r["t"])
        power = busy * (r["power"] - d["coef"][3]) + d["coef"][3]
    mbu = (w["Wb"] + w["KVb"]) / (g["bw"] * r["t"])
    return dict(tok_s=tok_s, power=power, j_tok=power / tok_s, t_iter=r["t"], n=n,
                mbu=mbu, capped=r["capped"])


# ---------------------------------------------------------------------------
# 7. Phase prediction for the recommender
# ---------------------------------------------------------------------------
def phase_predict(g, m, phase, prompt, gen, batch, draw):
    """Energy/time for one phase of a batch of `batch` requests.
    prefill: batch·prompt tokens in PREFILL_CHUNK-token passes (weights re-read
             every pass; causal attention).
    decode : `gen` passes of `batch` tokens (1 weight sweep per generated token),
             mean resident context prompt + gen/2."""
    L, moe = m.L, is_moe(m)
    kvpt = m.kv_bytes_per_token(DTYPE_B)
    kv_total = batch * (prompt + gen) * kvpt
    n = n_gpus_needed(g, m, kv_total)
    if n is None:
        return None
    tau_ar = draw["tau_ar"] if n > 1 else 0.0
    if phase == "prefill":
        tokens = batch * prompt
        n_pass = max(1, int(math.ceil(tokens / float(PREFILL_CHUNK))))
        chunk = tokens / n_pass
        w = dict(Wb=weight_params(m, chunk) * DTYPE_B,
                 KVb=chunk * kvpt,          # KV written once; attention reads are in F
                 Fg=2.0 * m.active_params() * chunk,
                 T=chunk)
        # causal self-attention: sum_i i ≈ prompt/2 context per token
        w["F"] = w["Fg"] + m.attn_flops_per_token(1) * chunk * prompt / 2.0
        reps, out_tokens = n_pass, tokens
    else:
        w = dict(Wb=weight_params(m, batch) * DTYPE_B,
                 KVb=batch * (prompt + gen / 2.0) * kvpt,
                 Fg=2.0 * m.active_params() * batch, T=batch)
        w["F"] = w["Fg"] + m.attn_flops_per_token(1) * batch * (prompt + gen / 2.0)
        reps, out_tokens = gen, batch * gen
    r = iter_predict(g, draw["coef"], draw["tp"], w, L, moe, draw["mfu"], n, tau_ar,
                     draw["tp_pen"] if n > 1 else 1.0)
    return dict(t=r["t"] * reps, energy=r["energy"] * reps, power=r["power"],
                j_tok=r["energy"] * reps / out_tokens, bound=r["bound"],
                capped=r["capped"], n_gpu=n,
                mbu=(w["Wb"] + w["KVb"]) / (g["bw"] * n * r["t"]))


def _pct(xs, q):
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[i]


def compare(gpus, m, phase, prompt, gen, batch, draws=300, seed=0):
    """Monte-Carlo comparison. Returns {gpu: dict(central, p10, p90, p_best, ...)}."""
    rnd = random.Random(seed)
    names = [n for n in gpus]
    cen = {n: phase_predict(gpus[n], m, phase, prompt, gen, batch, central_draw(gpus[n]))
           for n in names}
    names = [n for n in names if cen[n] is not None]
    samples = {n: [] for n in names}
    wins = {n: 0 for n in names}
    for _ in range(draws):
        es = {}
        for n in names:
            r = phase_predict(gpus[n], m, phase, prompt, gen, batch, random_draw(gpus[n], rnd))
            es[n] = r["j_tok"]
            samples[n].append(r["j_tok"])
        wins[min(es, key=es.get)] += 1
    out = {}
    for n in names:
        out[n] = dict(cen[n], p10=_pct(samples[n], 0.1), p90=_pct(samples[n], 0.9),
                      p_best=wins[n] / float(draws))
    return out


# ---------------------------------------------------------------------------
# 8. Win-map (which GPU wins, over model size x batch) — pure python
# ---------------------------------------------------------------------------
WINMAP_MODELS = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct",
                 "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct",
                 "Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-32B-Instruct",
                 "Qwen/Qwen2-72B-Instruct"]
WINMAP_BATCH = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def winmap(gpu_names=("H200", "B200"), phase="decode", prompt=1024, gen=256,
           models=WINMAP_MODELS, batches=WINMAP_BATCH, draws=200, seed=1):
    gpus_all = load_gpus()
    gpus = {n: gpus_all[n] for n in gpu_names if n in gpus_all}
    rows = []
    for mid in models:
        m = resolve_model(mid)
        for b in batches:
            res = compare(gpus, m, phase, prompt, gen, b, draws=draws, seed=seed)
            rows.append(dict(model=mid, params=m.active_params(), batch=b, res=res))
    return rows


# ---------------------------------------------------------------------------
# 9. CLI
# ---------------------------------------------------------------------------
def recommend(model_id, prompt_len=2048, gen_len=256, batch=32, gpu_names=None, draws=300):
    gpus = load_gpus()
    if gpu_names:
        gpus = {k: v for k, v in gpus.items() if k in gpu_names}
    m = resolve_model(model_id)
    print("\n### Workload: %s  prompt=%d gen=%d batch=%d" % (model_id, prompt_len, gen_len, batch))
    print("    active_params=%.2fB  total=%.2fB  layers=%d  %s" % (
        m.active_params() / 1e9, m.total_params() / 1e9, m.L, "MoE" if is_moe(m) else "dense"))
    for phase, label in (("prefill", "PREFILL (prompt ingest)"), ("decode", "DECODE  (token gen)")):
        res = compare(gpus, m, phase, prompt_len, gen_len, batch, draws=draws)
        if not res:
            print("\n  %s: model does not fit on any candidate" % label)
            continue
        rows = sorted(res.items(), key=lambda kv: kv[1]["j_tok"])
        best = rows[0][1]["j_tok"]
        unit = "mJ/tok"
        print("\n %s" % label)
        print("   %-14s%10s%18s%7s%8s%6s%6s%8s  %s" % ("GPU", unit, "[p10, p90]", "rel", "bound",
                                                      "MBU", "nGPU", "P(best)", "confidence"))
        for n, r in rows:
            g = gpus[n]
            print("   %-14s%10.2f  [%6.2f,%7.2f]%6.2fx%8s%5.0f%%%5d%7.0f%%  %s%s" % (
                n, r["j_tok"] * 1e3, r["p10"] * 1e3, r["p90"] * 1e3, r["j_tok"] / best,
                r["bound"][:3] + ("/CAP" if r["capped"] else ""), 100 * r["mbu"], r["n_gpu"],
                100 * r["p_best"], g["confidence"], "" if r["n_gpu"] == 1 else " +TP-extrapolated"))
        top = rows[0][0]
        conf = rows[0][1]["p_best"]
        print("   -> energy-optimal: %s  (P(best)=%.0f%%%s)" % (
            top, 100 * conf, "" if conf >= 0.9 else " — NOT a confident ranking (P<0.9)"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--prompt", type=int, default=2048)
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--gpus", default=None, help="comma list, default all")
    ap.add_argument("--draws", type=int, default=300)
    ap.add_argument("--winmap", action="store_true", help="print H200-vs-B200 win-map")
    ap.add_argument("--list", action="store_true", help="list GPU table")
    a = ap.parse_args()
    if a.list:
        for n, g in load_gpus().items():
            print("%-14s %-6s bw=%.2fTB/s peak=%4.0fTF cap=%4.0fW idle=%5.1fW e_w=%.3g e_kv=%.3g "
                  "e_g=%.3gpJ  %s | time:%s" % (
                      n, g["mem_tech"], g["bw"] / 1e12, g["peak_flops"] / 1e12, g["p_cap"],
                      g["p_static"], g["e_wbyte"], g["e_kvbyte"], g["e_gemm"] * 1e12,
                      g["energy_src"], g["time_key"]))
        return
    if a.winmap:
        for phase in ("decode", "prefill"):
            rows = winmap(phase=phase, draws=min(a.draws, 200))
            print("\n### %s win-map  (cell = B200/H200 energy ratio, * = P(winner)>=0.9)" % phase)
            bs = sorted(set(r["batch"] for r in rows))
            print("%-14s" % "model \\ batch" + "".join("%8d" % b for b in bs))
            for mid in dict.fromkeys(r["model"] for r in rows):
                line = "%-14s" % mid.split("/")[-1].replace("-Instruct", "")[:14]
                for b in bs:
                    r = [x for x in rows if x["model"] == mid and x["batch"] == b][0]["res"]
                    if "H200" in r and "B200" in r:
                        ratio = r["B200"]["j_tok"] / r["H200"]["j_tok"]
                        conf = max(r["B200"]["p_best"], r["H200"]["p_best"]) >= 0.9
                        line += "%7.2f%s" % (ratio, "*" if conf else " ")
                    else:
                        line += "%8s" % "n/a"
                print(line)
        return
    gl = a.gpus.split(",") if a.gpus else None
    recommend(a.model, a.prompt, a.gen, a.batch, gl, a.draws)


_load_time_fit()

if __name__ == "__main__":
    main()
