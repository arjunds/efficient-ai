# LIMINAL (arXiv 2507.14397) — power appendix, what was cut, and the "5×" gap

Read 2026-10-02 against both arXiv versions (PDFs + TeX sources, which include
commented-out material). Excerpts and sources: `/shared_data0/adsampat/lit/liminal/`
(`liminal_v{1,2}.txt`, `liminal_excerpts.txt`, `src_v1/`, `src_v2/`).

## Versions
| | date | pages | notes |
|---|---|---|---|
| v1 | 2025-07-18 | 22 | "Efficient LLM Inference: Bandwidth, Compute, Synchronization, and Capacity are all you need" (NeurIPS template). Appendices A (FLOP/byte eqns), B (detailed results), C (CENT PIM), **D Power Modeling (p20–22)**, **E Validation (p22)** |
| v2 | 2025-11-13 | 17 | "LIMINAL: Exploring the Frontiers of LLM Decode Performance" (MLSys template, "under review"). New §5 Validation (p9–10); one appendix: A Power Modeling (p17) |

The user's local PDF is **v2** (byte-identical). No proceedings version and no code release.

## The power "measurements" are a model, in both versions
- STPS/W uses "a simple power model based on disclosed TDP… and memory power": chip
  1 W/mm² (v2: 1.1), DRAM power from JEDEC HBM4 / DRAMPower / CACTI / CoMeT, +300 W
  per 8-chip server, on-wafer and inter-chip communication energy = 0 (v1).
- Commented-out v1 TeX: **HBM4 "30 W per TB/s" ≈ 30 pJ/B (3.75 pJ/bit)**; 3D-DRAM
  ≈ 5.6 pJ/B; a TCO model in which "chips run at TDP all the time."
- **Power does not depend on activity**: there is no per-byte/per-FLOP energy, no
  static/dynamic split and no idle state, and the model is never compared with a
  measurement. STPS/W = throughput ÷ provisioned power.

## The "≈5×" is a latency gap, and it was cut
- **v1 App. E (p22):** an H100 GEMV 1×16384×16384 (BF16, 512 MB) is predicted at
  146 µs and measured at 736 µs, "a gap of ≈5×". They attribute it to exposed CUDA
  launch latency and poor prefetch ("≈51M memory accesses with an L2 hit rate of only
  50%"). Against an anonymized NVIDIA simulator, LIMINAL is 1.6–2.3× optimistic.
  *(Reviewer's reading: 736 µs ⇒ ~0.7 TB/s ≈ 22% of H100 peak — likely an untuned
  microbenchmark, since a tuned GEMV reaches ~85–90%.)*
- **v2 §5 replaces it** with a calibrated fit on 8×H100 vLLM, timed by the PyTorch
  profiler at µs resolution. They add `T_Exposed × L` per layer: **95–138 µs/layer,
  mean 114**, with the slope on memory time **fixed at 1.0 (peak bandwidth)**. A
  commented-out table shows per-model R² of 0.61–0.82; the abstract reports 7.6% MAPE.
  The pure roofline is ~2.4× off at low batch for Llama3-70B TP8 and ~3–7× for small
  or MoE models (reviewer's estimate, not stated in the paper).
- **Time resolution is not involved in their gap.** The 200 ms power-sampling concern
  applies to our methodology, not to LIMINAL.

## Cross-checks against our measurements
| quantity | LIMINAL | ours |
|---|---|---|
| fixed per-layer overhead | 114 µs/layer fitted (95–138), H100 | **~113 µs/layer** of CPU time (sysid, H200); per-layer floor 80–110 µs (recommender); 1.4–2.2 ms/step total |
| bandwidth utilization | forced to 1.0 (all inefficiency in the intercept) | fitted separately: asymptotic MBU **0.72 (H200), 0.86 (B200)**, plus the intercept |
| DRAM energy per byte | assumed ~30 pJ/B (HBM4, from sources, never measured) | measured lumped `e_wbyte` 107 pJ/B (H200) / 124 pJ/B (B200), which includes the on-chip hierarchy (B200 `ncu`: each DRAM byte crosses L2 ~1.9×) |
| power vs activity | none (TDP) | measured: static + per-byte + per-FLOP |

## Balance / overprovisioning in LIMINAL
Qualitative only: "architects must seek to build balanced systems that provide the
right amount of all four" (bandwidth, capacity, compute, sync). The method is
one-at-a-time sweeps, e.g. 8× bandwidth buys only 1.1–4.5× UTPS because sync latency
dominates, and compute utilization is ≤1% at low batch, i.e. compute is overprovisioned
for decode. The one energy-balance point (v2 §4.6) is that bandwidth beyond what
compute can use is wasted power. They never measure utilization against power.

## Implications for us
1. **Positioning:** LIMINAL has no measured energy model. Ours is the first measured,
   activity-dependent energy decomposition for their roofline, and it replaces a TDP
   denominator.
2. **Our time model is stronger:** it fits a utilization slope *and* an overhead
   intercept, where LIMINAL forces the slope to peak. State this explicitly. The
   per-layer overhead agrees with theirs independently (~113 vs 114 µs/layer).
3. **A saturation proxy kernel is worth running:** a constant-read HBM kernel held for
   seconds gives sustained bandwidth and the power plateau, which pins `e_byte` and
   `P_static` independently of NVML smoothing. LIMINAL never measured this.
4. **Measurement protocol:** use ≥5–10 s steady windows per operating point (≫ the
   ~0.4 s smear), regress over many bins, cross-check with instantaneous power or
   DCGM, and report bandwidth, tokens/s (inferences/s) and iteration time alongside
   J/token.
5. **Make "balanced" quantitative:** report the static+overhead / per-byte / per-FLOP
   energy shares at *realized* utilization. Call a system overprovisioned when extra
   peak bandwidth or compute raises static power without lowering J/token at the
   target SLO.
