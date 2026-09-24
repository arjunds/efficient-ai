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

PREFIX CACHE: `--cached-frac f` = fraction of prompt tokens served from vLLM's
prefix cache (not computed): prefill FLOPs/tokens scale by (1-f); KV stays
resident. Default 0 (deployment without shared prefixes). The coefficients in
gpu_coefficients.json are fit on COMPUTED FLOPs (prefix-cache-corrected).

CLI (compatible with v1):
  python3 recommend_gpu.py <hf_model_id> [--prompt 2048 --gen 256 --batch 32] [--cached-frac 0]
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
                 e_wbyte=1.066e-10, e_kvbyte=3.288e-10, e_gemm=0.794e-12,
                 e_wbyte_ci=[1.063e-10, 1.069e-10], e_kvbyte_ci=[3.197e-10, 3.374e-10],
                 e_gemm_ci=[0.773e-12, 0.822e-12],
                 source="builtin: logs/ragged 3-term fit, prefix-cache-corrected FLOPs"),
    "B200": dict(gpu_name="NVIDIA B200", mem_tech="HBM3e", bw=8.0e12, peak_flops=2.25e15,
                 p_cap=1000.0, p_static=237.8, p_static_range=[236.3, 241.7],
                 e_wbyte=1.243e-10, e_kvbyte=2.852e-10, e_gemm=0.642e-12,
                 e_wbyte_ci=[1.238e-10, 1.248e-10], e_kvbyte_ci=[2.567e-10, 3.143e-10],
                 e_gemm_ci=[0.600e-12, 0.694e-12],
                 source="builtin: logs/B200 3-term fit, prefix-cache-corrected FLOPs"),
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
TIME_FIT = {"B200": {"central": {"eta": 0.864160569, "eta_kv": 0.330317884, "form": "additive(k=1)", "held_out": None, "k": 1.0, "lomo_mape_t": 2.651033365, "mu": 0.662729102, "t0": 0.001419045, "tau": 7.8283e-05, "tau_moe": 0.000345665, "tau_moe_note": "scaled from H200 (no MoE run on this GPU)"}, "ensemble": [{"eta": 0.864160569, "eta_kv": 0.330317884, "form": "additive(k=1)", "held_out": None, "k": 1.0, "lomo_mape_t": 2.651033365, "mu": 0.662729102, "t0": 0.001419045, "tau": 7.8283e-05, "tau_moe": 0.000345665, "tau_moe_note": "scaled from H200 (no MoE run on this GPU)"}, {"eta": 0.842749987, "eta_kv": 0.353814315, "form": "additive(k=1)", "held_out": "Qwen/Qwen2-72B-Instruct", "k": 1.0, "mu": 0.623734634, "t0": 0.001526809, "tau": 7.2736e-05, "tau_moe": 0.00032117, "tau_moe_note": "scaled from H200 (no MoE run on this GPU)"}, {"eta": 0.879398997, "eta_kv": 0.250296443, "form": "additive(k=1)", "held_out": "Qwen/Qwen2-7B-Instruct", "k": 1.0, "mu": 0.881919894, "t0": 0.001121936, "tau": 8.7894e-05, "tau_moe": 0.0003881, "tau_moe_note": "scaled from H200 (no MoE run on this GPU)"}, {"eta": 0.883486672, "eta_kv": 0.502179316, "form": "additive(k=1)", "held_out": "Qwen/Qwen2.5-32B-Instruct", "k": 1.0, "mu": 0.509498331, "t0": 0.001319564, "tau": 8.2305e-05, "tau_moe": 0.000363423, "tau_moe_note": "scaled from H200 (no MoE run on this GPU)"}, {"eta": 0.861884731, "eta_kv": 0.171423027, "form": "additive(k=1)", "held_out": "mistralai/Mistral-7B-v0.1", "k": 1.0, "mu": 1.018755824, "t0": 0.00143056, "tau": 7.8676e-05, "tau_moe": 0.0003474, "tau_moe_note": "scaled from H200 (no MoE run on this GPU)"}, {"eta": 0.659830226, "eta_kv": 1.0, "form": "soft(k=3)", "held_out": None, "k": 3.0, "mu": 0.482816549, "t0": 0.0, "tau": 0.00018823, "tau_moe": 0.00083114, "tau_moe_note": "scaled from H200 (no MoE run on this GPU)"}, {"eta": 0.608896374, "eta_kv": 0.353814334, "form": "soft(k=3)", "held_out": "Qwen/Qwen2-72B-Instruct", "k": 3.0, "mu": 0.623734624, "t0": 0.001948729, "tau": 0.000106728, "tau_moe": 0.000471266, "tau_moe_note": "scaled from H200 (no MoE run on this GPU)"}, {"eta": 0.648470383, "eta_kv": 0.384691766, "form": "soft(k=3)", "held_out": "Qwen/Qwen2-7B-Instruct", "k": 3.0, "mu": 0.614715566, "t0": 0.0, "tau": 0.00018348, "tau_moe": 0.000810169, "tau_moe_note": "scaled from H200 (no MoE run on this GPU)"}, {"eta": 0.694609023, "eta_kv": 0.502179145, "form": "soft(k=3)", "held_out": "Qwen/Qwen2.5-32B-Instruct", "k": 3.0, "mu": 0.509498386, "t0": 0.002079829, "tau": 0.00010854, "tau_moe": 0.000479266, "tau_moe_note": "scaled from H200 (no MoE run on this GPU)"}, {"eta": 0.663118646, "eta_kv": 0.15873208, "form": "soft(k=3)", "held_out": "mistralai/Mistral-7B-v0.1", "k": 3.0, "mu": 1.062428082, "t0": 0.0, "tau": 0.000197061, "tau_moe": 0.000870134, "tau_moe_note": "scaled from H200 (no MoE run on this GPU)"}, {"eta": 0.681114146, "eta_kv": 0.323882628, "form": "max(k=50)", "held_out": None, "k": 50.0, "mu": 0.600035081, "t0": 0.00312552, "tau": 8.7628e-05, "tau_moe": 0.000386928, "tau_moe_note": "scaled from H200 (no MoE run on this GPU)"}, {"eta": 0.621518797, "eta_kv": 0.353814294, "form": "max(k=50)", "held_out": "Qwen/Qwen2-72B-Instruct", "k": 50.0, "mu": 0.623734654, "t0": 0.002804618, "tau": 9.8706e-05, "tau_moe": 0.000435842, "tau_moe_note": "scaled from H200 (no MoE run on this GPU)"}, {"eta": 0.692521986, "eta_kv": 0.35334201, "form": "max(k=50)", "held_out": "Qwen/Qwen2-7B-Instruct", "k": 50.0, "mu": 0.614923827, "t0": 0.003458215, "tau": 3e-05, "tau_moe": 0.000132467, "tau_moe_note": "scaled from H200 (no MoE run on this GPU)"}, {"eta": 0.7135839, "eta_kv": 0.502179244, "form": "max(k=50)", "held_out": "Qwen/Qwen2.5-32B-Instruct", "k": 50.0, "mu": 0.509498356, "t0": 0.0030895, "tau": 8.9869e-05, "tau_moe": 0.000396823, "tau_moe_note": "scaled from H200 (no MoE run on this GPU)"}, {"eta": 0.677031297, "eta_kv": 0.121340898, "form": "max(k=50)", "held_out": "mistralai/Mistral-7B-v0.1", "k": 50.0, "mu": 1.307551145, "t0": 0.00311903, "tau": 5.7921e-05, "tau_moe": 0.000255755, "tau_moe_note": "scaled from H200 (no MoE run on this GPU)"}], "lomo": {"Qwen/Qwen2-72B-Instruct": {"eta": 0.842749987, "eta_kv": 0.353814315, "form": "additive(k=1)", "held_out": "Qwen/Qwen2-72B-Instruct", "k": 1.0, "mu": 0.623734634, "t0": 0.001526809, "tau": 7.2736e-05, "tau_moe": 0.00032117, "tau_moe_note": "scaled from H200 (no MoE run on this GPU)"}, "Qwen/Qwen2-7B-Instruct": {"eta": 0.879398997, "eta_kv": 0.250296443, "form": "additive(k=1)", "held_out": "Qwen/Qwen2-7B-Instruct", "k": 1.0, "mu": 0.881919894, "t0": 0.001121936, "tau": 8.7894e-05, "tau_moe": 0.0003881, "tau_moe_note": "scaled from H200 (no MoE run on this GPU)"}, "Qwen/Qwen2.5-32B-Instruct": {"eta": 0.883486672, "eta_kv": 0.502179316, "form": "additive(k=1)", "held_out": "Qwen/Qwen2.5-32B-Instruct", "k": 1.0, "mu": 0.509498331, "t0": 0.001319564, "tau": 8.2305e-05, "tau_moe": 0.000363423, "tau_moe_note": "scaled from H200 (no MoE run on this GPU)"}, "mistralai/Mistral-7B-v0.1": {"eta": 0.861884731, "eta_kv": 0.171423027, "form": "additive(k=1)", "held_out": "mistralai/Mistral-7B-v0.1", "k": 1.0, "mu": 1.018755824, "t0": 0.00143056, "tau": 7.8676e-05, "tau_moe": 0.0003474, "tau_moe_note": "scaled from H200 (no MoE run on this GPU)"}}}, "H200": {"central": {"eta": 0.725632976, "eta_kv": 0.221570365, "form": "max(k=50)", "held_out": None, "k": 50.0, "lomo_mape_t": 5.312194601, "mu": 2.98570262, "t0": 0.002153184, "tau": 0.000109618, "tau_moe": 0.000484026}, "ensemble": [{"eta": 0.95, "eta_kv": 0.266509409, "form": "additive(k=1)", "held_out": None, "k": 1.0, "mu": 5.0, "t0": 0.003241021, "tau": 2.4718e-05, "tau_moe": 0.000295185}, {"eta": 0.95, "eta_kv": 0.278533984, "form": "additive(k=1)", "held_out": "Qwen/Qwen2-7B-Instruct", "k": 1.0, "mu": 5.0, "t0": 0.00349928, "tau": 1.9948e-05, "tau_moe": 0.00029019}, {"eta": 0.95, "eta_kv": 0.252835117, "form": "additive(k=1)", "held_out": "Qwen/Qwen2.5-0.5B-Instruct", "k": 1.0, "mu": 5.0, "t0": 0.002068678, "tau": 5.6593e-05, "tau_moe": 0.000319127}, {"eta": 0.95, "eta_kv": 0.257059, "form": "additive(k=1)", "held_out": "Qwen/Qwen2.5-1.5B-Instruct", "k": 1.0, "mu": 5.0, "t0": 0.002949504, "tau": 3.0227e-05, "tau_moe": 0.000300931}, {"eta": 0.95, "eta_kv": 0.2711848, "form": "additive(k=1)", "held_out": "Qwen/Qwen2.5-14B-Instruct", "k": 1.0, "mu": 5.0, "t0": 0.003070861, "tau": 3.0934e-05, "tau_moe": 0.000298884}, {"eta": 0.95, "eta_kv": 0.273185929, "form": "additive(k=1)", "held_out": "Qwen/Qwen2.5-32B-Instruct", "k": 1.0, "mu": 5.0, "t0": 0.003773063, "tau": 6.413e-06, "tau_moe": 0.000284319}, {"eta": 0.95, "eta_kv": 0.257004627, "form": "additive(k=1)", "held_out": "Qwen/Qwen2.5-3B-Instruct", "k": 1.0, "mu": 3.314488473, "t0": 0.003803956, "tau": 6.54e-07, "tau_moe": 0.000282676}, {"eta": 0.95, "eta_kv": 0.282103186, "form": "additive(k=1)", "held_out": "Qwen/Qwen2.5-7B-Instruct", "k": 1.0, "mu": 4.528616383, "t0": 0.003553271, "tau": 1.8668e-05, "tau_moe": 0.00028908}, {"eta": 0.95, "eta_kv": 0.288233565, "form": "additive(k=1)", "held_out": "google/gemma-7b", "k": 1.0, "mu": 5.0, "t0": 0.003113106, "tau": 2.8088e-05, "tau_moe": 0.000298522}, {"eta": 0.95, "eta_kv": 0.271297235, "form": "additive(k=1)", "held_out": "meta-llama/Meta-Llama-3-8B", "k": 1.0, "mu": 5.0, "t0": 0.003183809, "tau": 2.9412e-05, "tau_moe": 0.000296534}, {"eta": 0.95, "eta_kv": 0.255890947, "form": "additive(k=1)", "held_out": "mistralai/Mistral-7B-v0.1", "k": 1.0, "mu": 5.0, "t0": 0.003187556, "tau": 2.9531e-05, "tau_moe": 0.000295929}, {"eta": 0.760388173, "eta_kv": 0.220001364, "form": "soft(k=3)", "held_out": None, "k": 3.0, "mu": 3.22974631, "t0": 0.001587392, "tau": 0.000125356, "tau_moe": 0.000472272}, {"eta": 0.755584369, "eta_kv": 0.217229109, "form": "soft(k=3)", "held_out": "Qwen/Qwen2-7B-Instruct", "k": 3.0, "mu": 4.454414858, "t0": 0.001652337, "tau": 0.000123338, "tau_moe": 0.000470633}, {"eta": 0.756093149, "eta_kv": 0.220819104, "form": "soft(k=3)", "held_out": "Qwen/Qwen2.5-0.5B-Instruct", "k": 3.0, "mu": 3.260357125, "t0": 0.001553846, "tau": 0.000126006, "tau_moe": 0.000472716}, {"eta": 0.756888767, "eta_kv": 0.220536017, "form": "soft(k=3)", "held_out": "Qwen/Qwen2.5-1.5B-Instruct", "k": 3.0, "mu": 3.357687409, "t0": 0.001593193, "tau": 0.000124169, "tau_moe": 0.000471921}, {"eta": 0.75256793, "eta_kv": 0.218305097, "form": "soft(k=3)", "held_out": "Qwen/Qwen2.5-14B-Instruct", "k": 3.0, "mu": 3.173968263, "t0": 0.001521201, "tau": 0.000127738, "tau_moe": 0.00047299}, {"eta": 0.780986082, "eta_kv": 0.217677071, "form": "soft(k=3)", "held_out": "Qwen/Qwen2.5-32B-Instruct", "k": 3.0, "mu": 3.292889189, "t0": 0.001711141, "tau": 0.000121513, "tau_moe": 0.000471064}, {"eta": 0.769804478, "eta_kv": 0.221824138, "form": "soft(k=3)", "held_out": "Qwen/Qwen2.5-3B-Instruct", "k": 3.0, "mu": 3.243637169, "t0": 0.001838231, "tau": 0.000113366, "tau_moe": 0.000467491}, {"eta": 0.75851774, "eta_kv": 0.223092665, "form": "soft(k=3)", "held_out": "Qwen/Qwen2.5-7B-Instruct", "k": 3.0, "mu": 3.87431865, "t0": 0.001749659, "tau": 0.000120004, "tau_moe": 0.000468823}, {"eta": 0.781587843, "eta_kv": 0.426948962, "form": "soft(k=3)", "held_out": "google/gemma-7b", "k": 3.0, "mu": 1.145202534, "t0": 0.001063619, "tau": 0.000145106, "tau_moe": 0.000488421}, {"eta": 0.757258249, "eta_kv": 0.220154013, "form": "soft(k=3)", "held_out": "meta-llama/Meta-Llama-3-8B", "k": 3.0, "mu": 3.335120674, "t0": 0.001623572, "tau": 0.000124558, "tau_moe": 0.00047124}, {"eta": 0.758241788, "eta_kv": 0.211490591, "form": "soft(k=3)", "held_out": "mistralai/Mistral-7B-v0.1", "k": 3.0, "mu": 3.316542101, "t0": 0.001631837, "tau": 0.000124814, "tau_moe": 0.000470694}, {"eta": 0.725632976, "eta_kv": 0.221570365, "form": "max(k=50)", "held_out": None, "k": 50.0, "lomo_mape_t": 5.312194601, "mu": 2.98570262, "t0": 0.002153184, "tau": 0.000109618, "tau_moe": 0.000484026}, {"eta": 0.72172971, "eta_kv": 0.217664358, "form": "max(k=50)", "held_out": "Qwen/Qwen2-7B-Instruct", "k": 50.0, "mu": 4.082077608, "t0": 0.002172691, "tau": 0.00010911, "tau_moe": 0.000483827}, {"eta": 0.735729452, "eta_kv": 0.224439673, "form": "max(k=50)", "held_out": "Qwen/Qwen2.5-0.5B-Instruct", "k": 50.0, "mu": 2.993604219, "t0": 0.002238045, "tau": 0.000109868, "tau_moe": 0.000482399}, {"eta": 0.724947448, "eta_kv": 0.222009106, "form": "max(k=50)", "held_out": "Qwen/Qwen2.5-1.5B-Instruct", "k": 50.0, "mu": 3.104830833, "t0": 0.002154449, "tau": 0.000110314, "tau_moe": 0.000484078}, {"eta": 0.725916266, "eta_kv": 0.221002763, "form": "max(k=50)", "held_out": "Qwen/Qwen2.5-14B-Instruct", "k": 50.0, "mu": 2.840533354, "t0": 0.002150411, "tau": 0.000109665, "tau_moe": 0.00048398}, {"eta": 0.652407381, "eta_kv": 0.237418272, "form": "max(k=50)", "held_out": "Qwen/Qwen2.5-32B-Instruct", "k": 50.0, "mu": 2.267981467, "t0": 0.001665951, "tau": 0.000126456, "tau_moe": 0.000494421}, {"eta": 0.739080157, "eta_kv": 0.222406833, "form": "max(k=50)", "held_out": "Qwen/Qwen2.5-3B-Instruct", "k": 50.0, "mu": 3.115773465, "t0": 0.002265403, "tau": 0.000101475, "tau_moe": 0.000481791}, {"eta": 0.725388966, "eta_kv": 0.223309186, "form": "max(k=50)", "held_out": "Qwen/Qwen2.5-7B-Instruct", "k": 50.0, "mu": 3.621932567, "t0": 0.002227924, "tau": 0.00010719, "tau_moe": 0.000482814}, {"eta": 0.740375918, "eta_kv": 0.230829428, "form": "max(k=50)", "held_out": "google/gemma-7b", "k": 50.0, "mu": 2.000348346, "t0": 0.002088059, "tau": 0.000111591, "tau_moe": 0.000485079}, {"eta": 0.724355751, "eta_kv": 0.220130434, "form": "max(k=50)", "held_out": "meta-llama/Meta-Llama-3-8B", "k": 50.0, "mu": 3.159796695, "t0": 0.002143235, "tau": 0.00010999, "tau_moe": 0.000484245}, {"eta": 0.728551927, "eta_kv": 0.218405344, "form": "max(k=50)", "held_out": "mistralai/Mistral-7B-v0.1", "k": 50.0, "mu": 2.870906074, "t0": 0.002177038, "tau": 0.000108717, "tau_moe": 0.000483314}], "lomo": {"Qwen/Qwen2-7B-Instruct": {"eta": 0.72172971, "eta_kv": 0.217664358, "form": "max(k=50)", "held_out": "Qwen/Qwen2-7B-Instruct", "k": 50.0, "mu": 4.082077608, "t0": 0.002172691, "tau": 0.00010911, "tau_moe": 0.000483827}, "Qwen/Qwen2.5-0.5B-Instruct": {"eta": 0.735729452, "eta_kv": 0.224439673, "form": "max(k=50)", "held_out": "Qwen/Qwen2.5-0.5B-Instruct", "k": 50.0, "mu": 2.993604219, "t0": 0.002238045, "tau": 0.000109868, "tau_moe": 0.000482399}, "Qwen/Qwen2.5-1.5B-Instruct": {"eta": 0.724947448, "eta_kv": 0.222009106, "form": "max(k=50)", "held_out": "Qwen/Qwen2.5-1.5B-Instruct", "k": 50.0, "mu": 3.104830833, "t0": 0.002154449, "tau": 0.000110314, "tau_moe": 0.000484078}, "Qwen/Qwen2.5-14B-Instruct": {"eta": 0.725916266, "eta_kv": 0.221002763, "form": "max(k=50)", "held_out": "Qwen/Qwen2.5-14B-Instruct", "k": 50.0, "mu": 2.840533354, "t0": 0.002150411, "tau": 0.000109665, "tau_moe": 0.00048398}, "Qwen/Qwen2.5-32B-Instruct": {"eta": 0.652407381, "eta_kv": 0.237418272, "form": "max(k=50)", "held_out": "Qwen/Qwen2.5-32B-Instruct", "k": 50.0, "mu": 2.267981467, "t0": 0.001665951, "tau": 0.000126456, "tau_moe": 0.000494421}, "Qwen/Qwen2.5-3B-Instruct": {"eta": 0.739080157, "eta_kv": 0.222406833, "form": "max(k=50)", "held_out": "Qwen/Qwen2.5-3B-Instruct", "k": 50.0, "mu": 3.115773465, "t0": 0.002265403, "tau": 0.000101475, "tau_moe": 0.000481791}, "Qwen/Qwen2.5-7B-Instruct": {"eta": 0.725388966, "eta_kv": 0.223309186, "form": "max(k=50)", "held_out": "Qwen/Qwen2.5-7B-Instruct", "k": 50.0, "mu": 3.621932567, "t0": 0.002227924, "tau": 0.00010719, "tau_moe": 0.000482814}, "google/gemma-7b": {"eta": 0.740375918, "eta_kv": 0.230829428, "form": "max(k=50)", "held_out": "google/gemma-7b", "k": 50.0, "mu": 2.000348346, "t0": 0.002088059, "tau": 0.000111591, "tau_moe": 0.000485079}, "meta-llama/Meta-Llama-3-8B": {"eta": 0.724355751, "eta_kv": 0.220130434, "form": "max(k=50)", "held_out": "meta-llama/Meta-Llama-3-8B", "k": 50.0, "mu": 3.159796695, "t0": 0.002143235, "tau": 0.00010999, "tau_moe": 0.000484245}, "mistralai/Mistral-7B-v0.1": {"eta": 0.728551927, "eta_kv": 0.218405344, "form": "max(k=50)", "held_out": "mistralai/Mistral-7B-v0.1", "k": 50.0, "mu": 2.870906074, "t0": 0.002177038, "tau": 0.000108717, "tau_moe": 0.000483314}}}}  # generated by recommender_backtest.py --fit


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


def load_gpus(path=COEFF_JSON, uncorrected=False):
    """Return {name: gpu dict}. Measured entries come from gpu_coefficients.json
    (else built-ins); every other GPU gets datasheet specs + memory-tech prior.
    GPUs listed under `_excluded` (e.g. the power-limited A5000) are ignored, so
    their technology keeps its prior. uncorrected=True uses the *_uncorrected
    (logged-prefill, pre prefix-cache fix) coefficients — for the backtest only."""
    measured, excluded = {}, set()
    src = "builtin"
    if os.path.exists(path):
        try:
            js = json.load(open(path))
            excluded = set((js.get("_excluded") or {}).keys())
            measured = {k: v for k, v in js.items() if not k.startswith("_") and isinstance(v, dict)
                        and k not in excluded}
            src = os.path.basename(path)
        except Exception as e:  # malformed json -> builtins
            print("[warn] could not read %s (%r); using built-ins" % (path, e), file=sys.stderr)
    for k, v in BUILTIN_MEASURED.items():
        if k not in excluded:
            measured.setdefault(k, v)

    gpus = {}
    for name, e in measured.items():
        if e.get("e_wbyte") is None:
            continue
        if uncorrected:
            e = dict(e)
            for k in ("e_wbyte", "e_kvbyte", "e_gemm"):
                if e.get(k + "_uncorrected") is not None:
                    e[k] = e[k + "_uncorrected"]
                    e.pop(k + "_ci", None)
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


def serving_iter_work(m, n, P, G, cached_frac=0.0):
    """Mean per-iteration work in steady-state continuous batching: n running
    sequences each generating 1 token, plus prefill chunks for arriving requests
    (n·P/G tokens on average, of which a fraction `cached_frac` is served from the
    prefix cache and never computed); mean resident context P + G/2 (the cached
    prefix is still resident KV and is still read by attention)."""
    T = n * (1.0 + (1.0 - cached_frac) * P / max(G, 1.0))
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
def predict_serving(g, m, P, G, conc=None, rate=None, draw=None, cached_frac=0.0):
    """Predict tok/s, avg power, J/token for a closed-loop (conc) or Poisson (rate)
    serving run, given only model, GPU, mean prompt P and gen G lengths."""
    d = draw or central_draw(g)
    L, moe = m.L, is_moe(m)

    def at(n):
        w = serving_iter_work(m, n, P, G, cached_frac)
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
def phase_predict(g, m, phase, prompt, gen, batch, draw, cached_frac=0.0):
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
        # prefix-cache hits are not computed (cached_frac of the prompt tokens);
        # energy/token is still reported per *prompt* token ingested
        tokens = max(1.0, batch * prompt * (1.0 - cached_frac))
        n_pass = max(1, int(math.ceil(tokens / float(PREFILL_CHUNK))))
        chunk = tokens / n_pass
        w = dict(Wb=weight_params(m, chunk) * DTYPE_B,
                 KVb=chunk * kvpt,          # KV written once; attention reads are in F
                 Fg=2.0 * m.active_params() * chunk,
                 T=chunk)
        # causal self-attention: sum_i i ≈ prompt/2 context per token
        w["F"] = w["Fg"] + m.attn_flops_per_token(1) * chunk * prompt / 2.0
        reps, out_tokens = n_pass, batch * prompt
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


def compare(gpus, m, phase, prompt, gen, batch, draws=300, seed=0, cached_frac=0.0):
    """Monte-Carlo comparison. Returns {gpu: dict(central, p10, p90, p_best, ...)}."""
    rnd = random.Random(seed)
    names = [n for n in gpus]
    cen = {n: phase_predict(gpus[n], m, phase, prompt, gen, batch, central_draw(gpus[n]), cached_frac)
           for n in names}
    names = [n for n in names if cen[n] is not None]
    samples = {n: [] for n in names}
    wins = {n: 0 for n in names}
    for _ in range(draws):
        es = {}
        for n in names:
            r = phase_predict(gpus[n], m, phase, prompt, gen, batch, random_draw(gpus[n], rnd),
                              cached_frac)
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
def recommend(model_id, prompt_len=2048, gen_len=256, batch=32, gpu_names=None, draws=300,
              cached_frac=0.0):
    gpus = load_gpus()
    if gpu_names:
        gpus = {k: v for k, v in gpus.items() if k in gpu_names}
    m = resolve_model(model_id)
    print("\n### Workload: %s  prompt=%d gen=%d batch=%d cached_frac=%.2f" % (
        model_id, prompt_len, gen_len, batch, cached_frac))
    print("    active_params=%.2fB  total=%.2fB  layers=%d  %s" % (
        m.active_params() / 1e9, m.total_params() / 1e9, m.L, "MoE" if is_moe(m) else "dense"))
    for phase, label in (("prefill", "PREFILL (prompt ingest)"), ("decode", "DECODE  (token gen)")):
        res = compare(gpus, m, phase, prompt_len, gen_len, batch, draws=draws, cached_frac=cached_frac)
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
    ap.add_argument("--cached-frac", type=float, default=0.0,
                    help="fraction of prompt tokens served from the prefix cache (not computed)")
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
    recommend(a.model, a.prompt, a.gen, a.batch, gl, a.draws, a.cached_frac)


_load_time_fit()

if __name__ == "__main__":
    main()
