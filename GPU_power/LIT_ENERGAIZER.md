# EnergAIzer (Lee et al., ISPASS 2026) — findings for our energy model

Kyungmi Lee, Zhiye Song, Eun Kyung Lee, Xin Zhang, Tamar Eilam, Anantha Chandrakasan
(MIT + IBM Research). "EnergAIzer: Fast and Accurate GPU Power Estimation Framework for
AI Workloads." ISPASS 2026; arXiv:2604.20105 (v1, 2026-04-22, the only version).
Artifact (MIT license): github.com/kyungmi-lee/energaizer-ispass26-artifact, Zenodo
10.5281/zenodo.18916559. Follow-up: **EnergyLens** (arXiv:2605.14249) — multi-GPU
TensorRT-LLM prefill and decode on 8×A100-SXM, with EnergAIzer as the per-kernel backend.
Read 2026-10-02: the full paper, the artifact appendix, the measurement/fitting code, and
the released measurement database. Notes and an analysis script:
`/shared_data0/adsampat/lit/energaizer/` (`NOTES.md`, `analysis/implied_ebyte2.py`).

## What it is
A per-kernel latency and power **predictor** that needs no simulation or profiling:
- GEMM, softmax, FlashAttention, conv and elementwise kernels get an analytic timeline
  built from tiling, threadblock swizzling and pipelining. The timeline gives DRAM, L2
  and shared-memory traffic and per-module busy time.
- Latency is corrected per phase, `t = λ·t_ideal + ε`.
- Power is `P_dyn = α_DRAM·C_D·V_D²·f_D + Σ α_m·C_m·V²·f`, where `C_m` is fit by
  non-negative least squares to a database measured with NVML.
- End-to-end predictions sum kernels run in sequence (about 1.8 s per prediction).

Scope: one GPU, kernels in sequence, **prefill only** (eager attention), on BERT,
GPT-2, OPT-1.3B, Qwen2-1.5B, ResNet and ViT. **No decode, no serving, no continuous
batching, no optimizer or controller.**

## Measurement protocol (from the artifact code; the paper says only "NVML")
- NVML is polled every 10 ms. The cuBLAS harness uses `GetPowerUsage`; the PyTorch
  harness uses `NVML_FI_DEV_POWER_INSTANT`.
- Each kernel runs back-to-back for **"at least a few seconds"** (~30 s for attention).
  Idle edges are trimmed, **clocks are locked with `nvidia-smi -lgc`**, and the power
  limit is set.
- They sidestep time resolution by measuring only long steady-state loops. There is no
  power-to-timeline alignment and no discussion of NVML averaging, and end-to-end runs
  are reported only as totals.

## Accuracy
- Kernel power: 3.1–3.8% MAPE.
- End-to-end on A100-PCIe / A10: latency 11.0% / 8.8%, power 8.0% / 8.2%.
- SM-clock sweep 510–1410 MHz: 6–9%.
- Forecasting other GPUs from Ampere data: A100-SXM 9.1%, H100 6.7%, **L40S 12.7%**.
  The L40S miss is attributed to GDDR6 not matching their assumed constant pJ/bit.
- Analytic DRAM, L2 and SMEM byte counts agree with Nsight Compute at r ≥ 0.99 (over
  1,200 kernels on A100).
- EnergyLens decode: about 25% latency error and about 13% energy error.

## Energy per byte implied by their released database (derived; never stated in the paper)
Large elementwise kernels (≥1 GB of traffic) behave as a streaming proxy. The figure
below is (P − P_idle) / achieved bandwidth:

| GPU (memory) | SM clock | achieved BW (frac. of peak) | J per byte |
|---|---|---|---|
| A100-PCIe (HBM2) | 510 MHz | 825 GB/s (0.53) | 114 pJ/B |
| A100-PCIe (HBM2) | 705 MHz | 983 GB/s (0.63) | 113 pJ/B |
| A100-PCIe (HBM2) | 900 MHz | 1152 GB/s (0.74) | 113 pJ/B |
| A100-PCIe (HBM2) | 1410 MHz | 1270 GB/s (0.82) | 136 pJ/B (higher core voltage) |
| A10 (GDDR6) | 900 MHz | 453 GB/s (0.76) | ~150 pJ/B (baseline is tiny-kernel power; true idle unknown) |

These include the kernel's own SM and L2 power, so they are an upper bound on DRAM-only
energy.

## How this bears on our results
- **Supports:** energy per byte is flat while bandwidth changes 1.4× (510→900 MHz), the
  same within-GPU invariance we see. Their L40S failure is attributed to non-constant
  pJ/bit across memory technologies, which matches our finding.
- **Challenges our HBM2e prior.** A100 HBM2 at ~113 pJ/B is close to H200 HBM3e
  (107 pJ/B) and below B200 (125 pJ/B). Our HBM2e prior of ~175 pJ/B came from
  *power-capped* A100 runs and is probably inflated by the cap. Across HBM generations
  energy per byte may be roughly constant (~105–135 pJ/B); the large effect may be
  **HBM vs GDDR**, not HBM2 vs HBM3e.
- **Conflicts on GDDR6.** A10 ~150 pJ/B against our A5000 270–310 pJ/B is a ~2× gap.
  Their figure is approximate (idle unknown) and ours comes from a 100 W-capped part
  measured with bursts. This is unresolved: it is either a GPU difference or one of the
  baselines is wrong.
- **Clock and voltage matter:** at 1410 MHz energy per byte rises 20%, because the
  on-chip share scales with V²f. Our serving runs use default boost clocks, so our
  lumped `e_wbyte` includes a clock-dependent on-chip share. Our A5000 L2 data shows the
  same effect.

## What to adopt
1. **Steady-state streaming proxy (the PI's "blast HBM"):** a ≥1 GB read or copy
   kernel, back-to-back for ≥10 s, at several occupancy levels, logging
   POWER_INSTANT at 10 ms. Compute (P − P_idle)/BW and compare with the serving
   `e_wbyte`. This gives a calibration independent of the 200 ms per-bin smear.
2. **Steady-state compute calibration:** loop fixed-shape decode and prefill GEMMs, fit
   `e_gemm` from the loops, then hold it fixed in the serving fit. This removes the
   smear bias on the compute term without needing faster sampling.
3. **Give the fixed-overhead term its own power coefficient** (their `ε` term), mapped
   to our 1.4–2.2 ms/step host overhead.
4. **Lock clocks for calibration.** This needs root (`nvidia-smi -lgc`), which we do
   not have on our cluster; access on the B200 node is unknown. Without it, record
   clocks and stratify by clock state.

## Positioning
EnergAIzer is the closest analytic-utilization power model, but it targets
kernel-level, offline and prefill workloads. It does not model decode or continuous
batching, does not address NVML sampling artifacts, reports power coefficients in watts
rather than J/byte, and does no control. EnergyLens adds decode on static TensorRT-LLM
batches, at about 13% energy error. **Ours is the serving-level, decode-focused
complement:** coefficients in J/byte and J/FLOP, measured on HBM3e and GDDR6, validated
by counters (DRAM/analytic = 1.004 on B200), and used for recommendation and closed-loop
control. Cite both; watch EnergyLens.
