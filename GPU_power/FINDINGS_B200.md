# B200 cross-GPU energy-coefficient measurement — findings

Run 2026-09-17 on the Penn PARCC cluster (`dgx-b200`), following
`HANDOFF_CROSSGPU.md`. Raw data: `logs/B200/` (16 runs, 3255 bins,
`binned_table.csv` + `run_meta.json` per run). Machine-generated record:
`logs/B200/RECORD.txt`.

## Headline

**The datasheet scaling law does not hold on B200.** Measured coefficients are
64–84% above what scaling the H200 values by datasheet specs predicts.

| coefficient | H200 (published) | B200 (measured) | predicted by law | error |
|---|---|---|---|---|
| `e_wbyte`  | 1.077e-10 J/B | **1.189e-10** [1.18, 1.19] | 0.646e-10 | **+84%** |
| `e_kvbyte` | 3.380e-10 J/B | **3.511e-10** [3.21, 3.76] | 2.028e-10 | **+73%** |
| `e_gemm`   | 0.673 pJ/flop | **0.485 pJ** [0.44, 0.52]  | 0.296 pJ  | **+64%** |

3-term fit, R²=0.677, held-out-by-model MAPE 6.7%. 2-term: `e_bit`=1.210e-10
[1.202, 1.220], `e_flop`=0.635 pJ [0.598, 0.667] (bootstrap 95% CI).

Note the *pattern*: the two memory coefficients are within 4–10% of the **raw
H200 values** (not the scaled ones), while the compute coefficient improved ~28%.
The memory channel behaves as if it were hardware-invariant, not ∝ 1/bandwidth.

## Why — hypothesis: the workload never exercises the bigger GPU

> **SUPERSEDED — see the 72B UPDATE at the end of this document.** This section
> proposed that low utilization explained the deviation. The 72B run doubled
> realized bandwidth (35% -> 62% of peak) and energy per byte did NOT fall, which
> refutes it. The section is kept for the measurements it records.

| | H200 | B200 | ratio |
|---|---|---|---|
| peak HBM bandwidth | 4.8 TB/s | 8.0 TB/s | 1.67× |
| **realized** bandwidth, Qwen2-7B c=1 decode | 2.52 TB/s (53% of peak) | 2.78 TB/s (**35% of peak**) | **1.10×** |
| tokens/s, Qwen2-7B c=1 | 165.6 | 182.8 | 1.10× |

B200 is only **10% faster** than H200 on this workload despite 1.67× the peak
bandwidth and 2.27× the peak FLOPS. Realized bandwidth across the whole sweep is
~2.4 TB/s — about **31% of peak**, at every concurrency from 1 to 64. A 7B model
simply does not saturate a 180 GB B200. The extra capability the datasheet
advertises is idle, so the energy per byte of *actual* traffic does not improve.

At roughly equal realized bandwidth (2.5 vs 2.8 TB/s) the two GPUs cost roughly
equal energy per byte (1.077 vs 1.189e-10) — consistent with DRAM energy/bit
being set by memory technology and interface physics, while bandwidth gains come
from added parallelism that raises throughput without lowering energy per bit.

## Confounds tested and ruled out

1. **Fixed/time-proportional overhead** (the obvious explanation for inflated
   coefficients under low utilization). Adding an explicit per-bin time term to
   the regression fits it to **0 W** and changes the coefficients not at all
   (+84% → +84%). Dynamic energy is genuinely work-proportional; the excess is
   not absorbed overhead.
2. **Power capping** (HANDOFF gotcha #1). Worst observed is **68% of the 1000 W
   cap**. Never capped, so the linear model is valid — unlike the A100-PCIe.
3. **Crash-truncated runs.** 5 of 8 sharegpt runs hit CUDA errors (below).
   Refitting on only the 13 clean runs gives `e_wbyte`=1.189e-10,
   `e_gemm`=0.484 pJ — identical to 3 significant figures, R² improves to 0.708.
4. **Identifiability** (gotcha #2). Bytes-only R² is **negative** (−0.78, −1.17)
   while the 2-term fit is +0.54/+0.62, so the FLOP term earns its place. The two
   models agree closely (`e_bit` 1.238 vs 1.187; `e_flop` 0.634 vs 0.627),
   i.e. coefficients transfer across models on B200 as they did on H200.

## Caveats — read before quoting these numbers

- **Per-concurrency coefficients are not trustworthy individually.** Fitting each
  concurrency separately gives negative R² for c=1/4/16 (regressors barely vary
  within a group). Only the pooled fit is meaningful. The apparent trend
  (`e_gemm` falling 0.62→0.35 pJ with concurrency) is suggestive, not established.
- **Attention backend differs from the H200 runs.** vLLM auto-selected
  FlashInfer + TRTLLM attention on Blackwell (the bundled `vllm_flash_attn 2.7.2`
  has no sm_100 kernels). This should be second-order for the 3-term fit, whose
  channels are weight/KV bytes and dense GEMM flops, but `attn_flops` is affected.
- **`peak_flops` = 2250 TFLOPS dense** (NVIDIA quotes 4.5 PFLOPS *with 2:1
  sparsity*; halved to match the H200 990 TFLOPS dense anchor). Corroborated
  on-hardware: a warmed-up 8192³ fp16 matmul achieved **1520 TFLOPS = 68% of
  2250**, the normal band for real cuBLAS.
- **H200 numbers are the published ones**, not refit here — the H200 raw logs are
  not on this server. `scaling_check.py` anchors on the documented coefficients.
  With `logs/ragged` copied over, `pool_gpus.py H200:logs/ragged B200:logs/B200`
  refits both from raw bins and is strictly better.
- H200's own `e_bit` drifts 1.078 → 1.603e-10 across c=1→64 in
  `sweep_summary_H200.csv`, so "hardware constant" was already approximate.

## Implication for the recommender

`recommend_gpu.py` currently derives non-H200 GPUs' coefficients by scaling
datasheet specs (`scaled_gpu()`), assuming `e_byte ∝ 1/bw`. On this evidence that
assumption is **wrong in the direction that matters**: it would predict a B200
is ~1.67× more energy-efficient per byte than an H200 and recommend it for a 7B
decode workload, when measurement says the two are within 10% — and the B200
draws far more idle power (237 W vs 118 W), so it is likely *worse* in total
energy for this workload.

The defensible revision is to drive the model with **realized** bandwidth/FLOPS
(peak × achievable utilization for that model+batch), not datasheet peak. That
keeps the roofline tie-in in `roofline_energy.py` and makes the recommender
utilization-aware rather than spec-sheet-aware.

## Known issue: CUDA errors on long sequences

5 of 8 `sharegpt` runs (long prompts, up to 23k chars) hit
`CUDA error: an illegal memory access was encountered` in the FlashInfer/TRTLLM
attention path; all 8 `alpaca` runs were clean. Three crashed near the end and
still produced near-full bin counts; two were badly truncated (9 and 39 bins).
Excluding them does not change the coefficients (see confound 3), but this
**RESOLVED.** Root cause: `vllm/utils/flashinfer.py::use_trtllm_attention()`
selects the TRTLLM kernel PER BATCH via `num_tokens <= 256`, so variable-length
sharegpt traffic makes vLLM switch kernels mid-run. A/B/control test
(`b200_attnfix.sbatch`, job 8430582) on the exact failing case:

| config | result | tok/s |
|---|---|---|
| baseline (auto) | **CRASH** | 68 |
| `VLLM_USE_TRTLLM_ATTENTION=0` | OK | **2487** |
| `VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1` | OK | 1825 |
| `VLLM_USE_TRTLLM_ATTENTION=1` | OK | 2428 |

Forcing TRTLLM *on* also works, so neither kernel is broken — the **mid-run
switching** is. Fix: pin it. `=0` is both correct and fastest. Verified on the
72B run: 0 crashes in 6 runs.

## Reproducing

```
sbatch b200_prefetch_models.sbatch   # CPU: cache weights (no GPU charge)
sbatch b200_pull_sif.sbatch          # CPU: build vLLM 0.10.2 .sif
sbatch b200_smoke.sbatch             # 1 GPU, ~3 min: go/no-go + cap check
sbatch b200_sweep.sbatch             # 1 GPU, ~23 min: the 16-run sweep
./b200_analyze.sh logs/B200          # CPU: writes logs/B200/RECORD.txt
```

---

# UPDATE: 72B run (2026-09-17) — the utilization hypothesis FAILS

`logs/B200_large/`, Qwen2-72B-Instruct, 6 runs, c={1,4,16}, both tasks, TP=1.
Attention fix applied (`VLLM_USE_TRTLLM_ATTENTION=0`): **0 crashes in 6 runs**,
versus 5 of 8 sharegpt runs crashing before.

## The clean result — a direct ratio, no regression

At concurrency 1 each decode token is exactly one full weight sweep, so realized
bandwidth and energy/byte are **direct measurements**, immune to the
identifiability problems that plague the fits.

| GPU / model | tok/s | realized BW | % of peak | dyn W | **J/byte** |
|---|---|---|---|---|---|
| H200 Qwen2-7B | 165.6 | 2.52 TB/s | 53% | 251 | **0.997e-10** |
| B200 Qwen2-7B | 182.8 | 2.78 TB/s | 35% | 324 | **1.163e-10** |
| B200 Qwen2-72B | 34.3 | 4.99 TB/s | **62%** | 617 | **1.236e-10** |
| scaling-law prediction for B200 | | | | | 0.646e-10 |

The 72B nearly **doubled** realized bandwidth on B200 (2.78 -> 4.99 TB/s, 35% ->
62% of peak) — the workload now genuinely exercises the GPU. **Energy per byte
did not fall.** It rose slightly.

So the utilization explanation for the 7B deviation is refuted. Energy per byte
is ~invariant (1.0–1.24e-10) across two GPU generations, a 10x model-size range,
and a 2x range of realized bandwidth. It is **not** proportional to 1/peak
bandwidth.

## The 72B *regression* coefficients are NOT usable — and why

The 3-term fit on the 72B gives **R2 = -0.573** (worse than predicting the mean).
Do not quote e_wbyte=1.236e-10 as a fitted coefficient. The cause is a
degenerate regime, not a code bug:

* Dynamic energy per bin barely varies: mean 128.4 J, **std 6.9 J, CV = 0.054**.
* The regressors vary enormously: `gemm_flops_bin` CV = 2.16, `kv_bytes_bin`
  CV = 1.11.
* `weight_bytes_bin` is nearly **constant** (CV = 0.085), because with 145 GB of
  weights almost every bin is dominated by one full weight sweep.

With a near-constant regressor and a near-constant target, `e_wbyte` degenerates
to `mean(dyn)/mean(weight_bytes)` — a ratio of means, not a slope. (Confirmed:
128.4/9.899e11 = 1.297e-10 vs the "fitted" 1.236e-10.) Adding an explicit
intercept changes nothing because `weight_bytes_bin` is already acting as one.

The GPU is in a **near-constant-power regime**: per-bin power p50 = 880 W,
p100 = 943 W against a 1000 W cap. Not hard-clamped (nothing above 950 W, so
gotcha #1's DVFS clamp is not triggered), but power is pinned close enough to the
practical ceiling that energy stops tracking work. This is the large-model
analogue of gotcha #1 and deserves its own entry: **a model that saturates the
GPU destroys coefficient identifiability**, which is the opposite failure mode
from the 7B (too little variance in energy, rather than too little in AI).

Practical consequence: coefficient fitting needs a model big enough to use the
GPU but small enough that power still swings with the workload. The 7B and 72B
bracket that window; something in the 30B class is the likely sweet spot.

## Idle power — matters for the recommender

B200 idles at **237 W** vs H200's **117 W**, roughly 2x. For a workload that
does not saturate it, a B200 loses on total energy before doing any work.

## Counter availability (gotcha #5) — partial

**CORRECTION (2026-09-24): `ncu` IS available.** The probe in the 72B job ran
`command -v ncu` on the HOST PATH and wrongly concluded it was missing. It ships
**inside the vLLM container at `/usr/local/cuda/bin/ncu`** (the image env
advertises `NV_CUDA_NSIGHT_COMPUTE_VERSION=12.8.1`), and a second copy exists on
the host at `/vast/parcc/sw/26.1.b200/.../cuda-13.1.1/bin/ncu`. `dcgmi` is also
present at `/usr/bin/dcgmi`. So `ncu_dram_check.py` IS runnable here; what
remains unverified is whether counter *permission* is granted
(`ERR_NVGPUCTRPERM`) -- though a colleague collected `dram__bytes.sum` on this
cluster in July 2026, which suggests it is.

Our runs recorded `dram_backend: None` because `dram_counter.py` looks for
`dcgmi` *inside the container*, where it is absent. Binding `/usr/bin/dcgmi`
(plus its libraries) into the container should enable DCGM field 1005
(`DRAM_ACTIVE`) sampling and give in-run DRAM traffic. Caveat: DCGM reports an
activity *fraction* scaled by an assumed peak bandwidth, so it partly begs the
question it is being used to answer — it is weaker evidence than NCU byte counts.

## Where this leaves the interpretation

Two explanations were live. The 72B kills one:

1. ~~Utilization~~ — **refuted**. Doubling realized bandwidth did not lower J/byte.
2. **Memory-hierarchy scope** — still standing, and now better supported. Our
   lumped J/byte (1.0–1.24e-10) sits right on vgao's full-hierarchy sum
   (1.12e-10), while their DRAM-only term (0.66e-10) sits on the scaling-law
   prediction (0.646e-10). The law plausibly governs the DRAM channel alone,
   while our analytic byte model conflates DRAM with L2/L1/shared-memory traffic
   that does not scale with HBM bandwidth.

Explanation 2 remains **unverified for a serving workload** — it rests on a
colleague's GEMM microbenchmarks whose own OLS and WLS fits disagree by ~15%,
and without `ncu` we cannot decompose the hierarchy ourselves here.

---

# UPDATE 2: 32B run (2026-09-18) — invariance confirmed on a third point

`logs/B200_32B/`, Qwen2.5-32B-Instruct, 8 runs, c={1,4,16,64}, both tasks, TP=1,
same grid as the 7B sweep. Attention fix applied: **0 crashes in 8 runs**.

## Energy per byte is flat across a 1.8x range of realized bandwidth

Direct c=1 measurement (no regression — each decode token is one full weight
sweep, so this is just dynamic power / byte-rate):

| GPU / model | tok/s | realized BW | % of peak | dyn W | **J/byte** |
|---|---|---|---|---|---|
| H200 Qwen2-7B | 165.6 | 2.52 TB/s | 53% | 251 | **0.997e-10** |
| B200 Qwen2-7B | 182.8 | 2.78 TB/s | 35% | 324 | **1.163e-10** |
| B200 Qwen2.5-32B | 62.7 | **4.11 TB/s** | 51% | 505 | **1.229e-10** |
| B200 Qwen2-72B | 34.3 | 4.99 TB/s | 62% | 617 | **1.236e-10** |
| scaling-law prediction for B200 | | | | | 0.646e-10 |

Three B200 points spanning **2.78 -> 4.99 TB/s** of realized bandwidth give
1.163, 1.229, 1.236e-10 J/byte — flat to within 6%, and about **1.9x above** the
0.646e-10 the datasheet law predicts. Energy per byte does not depend on how
hard the memory system is driven, and does not fall with peak bandwidth.

This closes the argument: two agreeing points could be coincidence, three across
a 1.8x utilization span on identical silicon cannot. **`e_byte` is ~invariant,
not proportional to 1/peak bandwidth.**

## The identifiability mechanism, confirmed

Model size, dynamic-energy variance and fit quality move together exactly as the
72B degeneracy predicted:

| dataset | CV(dyn energy) | R² (3-term) | e_wbyte | e_gemm |
|---|---|---|---|---|
| B200 7B | **0.154** | **0.677** | 1.189e-10 | 0.485 pJ |
| B200 32B | 0.105 | 0.237 | 1.248e-10 | 0.398 pJ |
| B200 72B | **0.054** | **−0.573** | 1.236e-10 | 0.360 pJ |

As the model saturates the GPU, power pins near its ceiling, the variance the
regression needs collapses, and R² follows it down through zero. The 7B remains
the only well-conditioned fit; treat 32B as weak and 72B as unusable.

Note `e_wbyte` is nonetheless stable at 1.19–1.25e-10 across all three *despite*
the collapsing fit quality — because in every case it is really measuring the
same physical ratio. That is reassuring for the headline, and a warning against
reading R² as if it validated the coefficient.

## A nuance: the COMPUTE coefficient may behave differently

`e_gemm` falls monotonically as utilization rises — 0.485 -> 0.398 -> 0.360 pJ
against a 0.296 pJ prediction, i.e. +64%, +34%, **+21%**. The 72B value is close
to the ~20% success criterion. So `e_flop ∝ 1/peak_FLOPS` may hold approximately
once the compute is actually used, even though `e_byte ∝ 1/bandwidth` clearly
does not.

**Treat this as suggestive, not established.** It rests on fits whose R² is
0.677, 0.237 and −0.573 respectively — the trend strengthens exactly as the fit
quality degrades, which is the wrong direction for confidence. Confirming it
needs a model that uses the compute hard while keeping power variance, which is
the same narrow window the coefficient fitting needs.

## Bottom line

* **Memory channel:** the scaling law fails, robustly and by ~1.9x. Established
  by direct measurement, no regression involved.
* **Compute channel:** may follow the law once utilized; evidence is weak.
* **Recommender:** `scaled_gpu()`'s `e_byte ∝ 1/bw` should go. Given B200 also
  idles at 237 W vs H200's 117 W, a datasheet-driven recommender will pick a
  B200 for memory-bound decode and lose on energy.

---

# RECONCILIATION (2026-09-24, H200 cluster) — one consistent convention

The reconstructed `models.py` counted **both** embedding tables in `active_params`
(= total params); the canonical file (the one the whole H200 corpus used, restored
to the repo) counts lm_head once and **excludes the input-embedding gather**, which
reads only the batch's rows (e.g. Qwen2-7B: 7.070B vs 7.615B, exactly one
152064×3584 table). Its "validation anchor" `8.03e9` is the synthetic placeholder in
`energy_profile_load.py::run_self_test`, not the real accounting. All 30 B200 runs
were re-binned with the canonical file (`reconcile_gpus.py`,
`gpu_coefficients.json`, `realized_utilization.csv`).

| | direct c=1 J/byte, Qwen2-7B | 3-term `e_wbyte` [95% CI] | `e_gemm` | R² / held-out | MBU (7B c=1) |
|---|---|---|---|---|---|
| H200 | **1.076e-10** | 1.076e-10 [1.073, 1.080] | 0.680 pJ | 0.80 / 7.0% | 0.49 |
| B200 | **1.252e-10** | 1.252e-10 [1.247, 1.257] | 0.539 pJ | 0.65 / 9.0% | 0.33 |
| 1/BW-law prediction for B200 | 0.646e-10 | | 0.296 pJ | | |

- **B200/H200 energy per byte = 1.16** (law predicted 0.60). The direct-table ratio
  above (1.163/0.997 = 1.17) was already self-consistent; the fitted-coefficient
  comparison (1.189 vs 1.077) mixed conventions and is superseded (1.16).
- Within-GPU invariance holds tightly: H200 7B–32B 1.03–1.08e-10 (MBU 0.41→0.67);
  B200 7B/32B/72B 1.24–1.26e-10 (MBU 0.31→0.62).
- **Caveat:** at very low utilization J/byte inflates (H200 Qwen2.5-0.5B 1.26e-10 at
  MBU 0.05; 1.5B 1.13e-10 at MBU 0.13) — per-iteration fixed energy not
  proportional to bytes. Invariance is a statement about MBU ≳ 0.2.
- Open: why B200 is +16% per byte on the same memory technology (dual-die NV-HBI
  crossing? load-dependent baseline above idle?). Being tested with microbenchmarks.

---

# UPDATE 3: hardware counters (2026-09-25, B200) — gotcha #5 answered

`logs/B200_ncu/`. Nsight Compute on a batch=1 Qwen2-7B decode, 3004 kernels,
5 memory-hierarchy metrics. **Counter permission is GRANTED on the PARCC
cluster** — the `ERR_NVGPUCTRPERM` wall is gone here (unlike the A5000 node).
Numbers below use the CANONICAL `models.py`, so they compose with the
RECONCILIATION section above.

## Q1 — the analytic byte model is correct to 0.4%

| | |
|---|---|
| measured DRAM | 113.6 GB over 8 forward passes = **14.20 GB/pass** |
| canonical analytic | `active_params x 2` = **14.14 GB/pass** |
| **ratio** | **1.004** |

Gate #1 passes. The denominator under every coefficient in this corpus is sound.

**This independently confirms the canonical convention.** Two accounting details
were required, and both match what the canonical file already does:

1. **Pass count.** vLLM's prefill emits the *first* output token, so
   `max_tokens=N` is **N** forward passes, not N+1. (N+1 gave 0.829.)
2. **The input-embedding gather is not streamed.** Using the reconstructed
   file's `active_params` (which counted both tables) gives **0.932**; using the
   canonical one (lm_head once, input embed excluded) gives **1.004**. The
   hardware agrees with the canonical file's own comment, "input embed is a
   lookup" — arrived at from opposite directions.

## Q2 — the lumped coefficient spans the whole memory hierarchy

Measured traffic per canonical analytic byte, on the real serving workload:

| channel | bytes / analytic byte |
|---|---|
| DRAM | 1.004 |
| **L2** | **1.856** |
| TMA | 1.162 |
| L1 | 0.002 |
| shared | 0.004 |

Every weight byte pulled from HBM crosses L2 roughly **1.9x** and the TMA path
~1.2x. Combining those measured ratios with the group's per-channel energies
(`results/coefficients/pooled_phys3_*.json`, B200 GEMM microbenchmarks):

| | value | vs |
|---|---|---|
| **DRAM channel alone** | **0.664e-10** | 1/BW-law prediction 0.646e-10 → **+3%** |
| full hierarchy (OLS) | 1.001e-10 | canonical `e_wbyte` 1.252e-10 → −20% |
| full hierarchy (WLS) | 1.142e-10 | canonical `e_wbyte` 1.252e-10 → −9% |

So the 1/BW law appears to hold for the **DRAM channel** (+3%). What fails is
applying it to a *lumped* coefficient that also carries on-chip traffic which
does not scale with HBM bandwidth. That is consistent with, not contrary to, the
A5000 result: across memory *technologies* (GDDR6 vs HBM3e) the DRAM term itself
shifts, which is why `FINDINGS_A5000.md` finds energy/byte set by technology.

## Bearing on the open "+16% B200 vs H200" question

The RECONCILIATION section asks why B200 costs +16% per byte on the same memory
technology. This decomposition offers a testable answer: only ~53% of B200's
lumped coefficient is DRAM; the rest is on-chip (L2 1.86x, TMA 1.16x). If H200
moves fewer on-chip bytes per analytic byte — plausible given B200's dual-die
layout and NV-HBI crossing — that difference lands entirely in the lumped
number. **Running this same NCU capture on H200 would settle it**, but H200
access was revoked 2026-09-24, so it may only be answerable from the archived
logs or not at all.

## What is ours and what is borrowed

* **Measured here:** the traffic ratios, and the lumped `e_wbyte`. Real serving
  workload, B200.
* **Borrowed:** per-channel *energies* from the colleague's GEMM
  microbenchmarks. NCU replay destroys the power waveform, so traffic and energy
  cannot be captured in one run. Their OLS/WLS fits differ ~15%, which is why
  the reconstruction spans −9% to −20%, and why the +3% DRAM agreement is soft.

Traffic amplification is **established**; energy attribution is **supported but
not independently verified here**.

## Method note (this cost 75 min of B200 to get wrong twice)

NCU **kernel** replay is unusable on an LLM decode: ~20 s/kernel x ~2500 kernels
≈ 14 h; two runs timed out with partial captures. **`--replay-mode application`**
re-runs the process once per metric pass-group and finished in **10.5 min**.
Also required: `--profile-from-start off` with `cudaProfilerStart/Stop` around
only the measured `generate` (otherwise the 15 GB weight upload and FlashInfer's
JIT autotuner land in the counts), a warmup outside the profiled region, and a
**distinct warmup prompt** so vLLM prefix caching does not let the measured
generate skip its prefill. `ncu_hierarchy.py` now refuses to interpret a capture
whose DRAM/analytic ratio is physically impossible — a truncated profile
otherwise yields a small ratio that reads exactly like a real finding.
