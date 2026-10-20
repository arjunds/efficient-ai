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

---

# REVIEW NOTE (H200 side, 2026-09-25) on UPDATE 3's "the law holds for the DRAM channel"

The NCU results are a major win and are adopted as established: analytic bytes =
DRAM bytes to 0.4% (canonical convention), and every DRAM byte also crosses L2
~1.86× and TMA ~1.16×. Modeling DRAM and on-chip channels separately is the right
next step.

**But the "+3%" is not a valid test of the 1/BW law for the DRAM channel.** It compares
B200's *DRAM-only* energy (0.664e-10, borrowed) against the prediction 0.646e-10 =
H200's *lumped* coefficient × 4.8/8.0. A consistent DRAM-channel test needs H200's
DRAM-only energy, and a simple bound argues against the law:

- The law requires `e_DRAM(H200) = e_DRAM(B200) × 8.0/4.8 = 0.664 × 1.667 = 1.107e-10 J/B`.
- H200's **total** measured per-byte energy is 1.066–1.076e-10 J/B (3-term / direct).
- DRAM-only cannot exceed the total, so the law holds on the DRAM channel only if H200
  spends ~nothing on-chip per byte (vs ~47% on B200, ~30% on the A5000). Allowing the
  borrowed DRAM coefficient to be ~13% high (the OLS/WLS spread) still needs H200's
  on-chip share ≈ 9%.

**Status:** "the law holds for the DRAM channel" = *unestablished; a bound argument
disfavors it*. It could be rescued only if H200's on-chip energy per byte is far smaller
than B200's (e.g., NV-HBI die-to-die traffic on B200). The same NCU capture on H200 would
settle it (not possible — H200 access revoked). What is established: the lumped
coefficient is DRAM + on-chip hierarchy energy, and the lumped number is what serving
actually costs.

---

# UPDATE 4: steady-state calibration + microbench (2026-10-07, B200)

> **Reviewed 2026-10-08: see "UPDATE 4 review" below.** The e_flop (0.509 pJ), the
> "balanced (270 vs 281)" ridge and the "serving has ~10% extra (0.903x)" figures all came
> from fits that included power-capped, clock-throttled points. Uncapped, they become
> **0.94 pJ**, **133 vs 281** and **1.006x**. The DRAM-law falsification, the NVML smoothing
> result and the NCU cross-check all stand.

Ran both outstanding handoffs: `microbench/b200_microbench.sbatch` (probe, energy,
ncu_micro, ncu_serve; 1h02) and `microbench/b200_steady_calib.sbatch`
(steady+analyze; 38 min). Artifacts committed alongside.

## My "the law holds for the DRAM channel" was WRONG — now settled with our own numbers

The H200-side REVIEW NOTE was right, and this run confirms it without borrowing
anyone's coefficients. We can now split a streamed byte ourselves:
L2-resident stream J/B = 3.518e-11 against DRAM-streamed J/B, i.e. **on-chip is
~31% of a DRAM-streamed byte**.

| | B200 streamed J/B | → e_DRAM (69%) | law needs e_DRAM(H200) | H200 **total** |
|---|---|---|---|---|
| full-clock (duty-cycle fit) | 1.282e-10 | 0.885e-10 | 1.475e-10 | 1.076e-10 |
| saturated (power-capped) | 1.122e-10 | 0.774e-10 | 1.290e-10 | 1.076e-10 |

Either way the law needs **120–137% of H200's entire per-byte budget to be DRAM
alone**. Impossible. My `+3%` was wrong twice: it compared B200 DRAM-only against
H200 *lumped*-scaled, and it used a borrowed DRAM coefficient rather than a
measured one. **Status: `e_byte ∝ 1/BW` is falsified for the DRAM channel too.**

## Headline measurements

| quantity | value |
|---|---|
| DRAM stream, full clock (duty-cycle, R²=0.993) | **1.2825e-10 ± 3.3e-12 J/B** |
| DRAM stream, saturated | 1.122e-10 J/B at **7.39 TB/s = 92% of peak** |
| L2-resident stream | 3.518e-11 J/B (≈31% on-chip share) |
| `llmgemm` steady 2-feature fit (n=32, R²=0.709) | e_byte **1.374e-10** [1.299, 1.451], e_flop **0.509 pJ** [0.477, 0.537] |
| kernel launch energy | 16.7 µJ/launch (1-elem), 4.8 µJ (4096-elem) |
| idle | 235.3 W |

**B200 reaches 92% of peak bandwidth** — EnergAIzer's A100 managed only 82%.

## The PI's "200 ms sampling is a lot" — answered

Fitted NVML smoothing from the `square_multi` on/off test:

| field | boxcar window | lag | R² |
|---|---|---|---|
| `POWER_INSTANT` (`p_inst`) | **0.10 s** | +0.00 s | 0.880 |
| `GetPowerUsage` (`p_avg`) | **0.85 s** | +0.02 s | 0.965 |

So the per-bin "~0.4 s smear" **is** NVML averaging, as suspected. `GetPowerUsage`
averages over 0.85 s — more than 4× our 200 ms bin, so it cannot resolve them.
`POWER_INSTANT` at 0.10 s can. Matches the A5000 (1.0 s / ~0.1 s). **Use
POWER_INSTANT and long windows from here on.**

## Compute coefficient: serving OVER-estimated it

Steady `e_flop` = **0.509 pJ** vs serving `e_gemm` = 0.642 pJ — ratio **0.79**.
Note this is the *opposite* direction from the A5000 hint (which suggested
smoothing made per-bin fits too low). Refitting serving with `e_gemm` held at the
steady value gives `e_wbyte` 1.253e-10, `e_kvbyte` 3.632e-10 (free 3-term:
1.243e-10 / 2.852e-10 / 0.642 pJ, R² 0.690).

Streaming J/B is **0.903×** the serving `e_wbyte`, so the serving weight-byte
coefficient is close to pure weight-streaming energy, with ~10% extra from
attention/KV/overhead.

## Balanced, not overprovisioned

Energy ridge `e_byte/e_flop` = **270 FLOP/B**; performance ridge `peak/BW` =
**281 FLOP/B**. Within 4% — on B200 the energy balance point and the performance
balance point essentially coincide.

## Power capping is pervasive at saturation — read this before trusting any J/B

Large-working-set DRAM streams **sit at the 1000 W cap** (`capfrac` 1.00, SM clock
throttled 1965 → 1556 MHz). The analysis correctly refuses the occupancy-sweep
fit: *"56% of points ran at the power cap → constant-power regime; this fit is NOT
a valid calibration."* That is also why two DRAM numbers appear above: the
duty-cycle fit keeps bursts short enough to stay at full clock (1.2825e-10), while
saturated streaming is capped and throttled (1.122e-10). The ~14% gap is a
clock/voltage effect, the same phenomenon the A5000 showed at ~2×.

**Clock locking is not permitted on this cluster** (`nvidia-smi -lgc` denied,
established in the smoke test), so the `clocks` stage was not run; per-point
`sm_mhz` is recorded in the json for stratification instead.

## NCU cross-check of UPDATE 3, with 3× more data

`ncu_serve` (24 tokens, 9120 kernels) independently reproduces the UPDATE 3
capture (8 tokens, 3004 kernels):

| | UPDATE 3 | this run |
|---|---|---|
| DRAM/analytic | 1.004 | **1.008** |
| L2/analytic | 1.86 | 1.57 |
| L1/analytic | 0.002 | 0.0013 |

Gate #1 holds at 1.00 across two independent captures. (The L2 difference is the
prefill share: 1/8 of passes here vs 1/24, and prefill is more L2-dense.)

Per-op validation — every microbench op does what its name says:

| op | DRAM/an | L2/an | L1/an |
|---|---|---|---|
| `dram_stream` | 1.001 | 1.503 | 1.001 |
| `l2_stream_half` | 0.001 | 1.004 | 1.002 |
| `l1_stream_8KBperprog` | 0.000 | 0.039 | 1.015 |
| `gemv_b1` | 1.002 | **1.565** | 0.000 |
| `gemm_fp16_4096` | 0.927 | 6.207 | 0.000 |

`gemv_b1`'s L2/analytic of 1.565 matches the serving decode's 1.567 — the
microbenchmark is a faithful stand-in for decode traffic.

## Method note for the other server

`--replay-mode kernel` (as `b200_microbench.sbatch` originally specified) is not
viable on a vLLM decode: ~20 s/kernel × thousands of launches. I switched
`ncu_serve` to `--replay-mode application`, which captured 9120 kernels in 7m53s.
`--cache-control all` requires kernel replay, so the cold-L2 serving variant was
dropped (kept for `ncu_micro`, where ops are single cheap kernels); B200's L2 is
132 MB against 14.1 GB of weights per pass, so warm-vs-cold moves DRAM bytes
<1%. Also: both scripts specified `--cpus-per-task=8`, which PARCC's `dgx-b200`
rejects (`DefCpuPerGPU=28`, `cli_filter` error) — fixed in both.

## UPDATE 4b: per-channel energies, measured here (PI next-step #1 closed)

No new GPU time — a linear solve over data already collected. Three single-channel
ops give three equations in three unknowns: per-op **energy** from the duty-cycle
fits (full clock, R² 0.979–0.998) and per-op **traffic** from `ncu_micro`.

| op | e (J/B) | DRAM | L2 | L1 |
|---|---|---|---|---|
| `dram_stream` | 1.2825e-10 | 1.001 | 1.503 | 1.001 |
| `l2_stream_half` | 3.3099e-11 | 0.001 | 1.004 | 1.002 |
| `l1_stream_8KBperprog` | 1.1005e-11 | 0.000 | 0.039 | 1.015 |

Solving:

| channel | **measured here** | borrowed GEMM-microbench ref |
|---|---|---|
| `e_DRAM` | **8.37e-11 J/B** | 6.62e-11 |
| `e_L2` | **2.29e-11 J/B** | 1.71e-11 |
| `e_L1` | **1.00e-11 J/B** | 1.28e-11 |

**Closure test.** Feeding these channels the *independently measured* serving
traffic (`ncu_serve`: DRAM 1.008, L2 1.567, L1 0.0013) predicts
`e_wbyte` = **1.203e-10** against the measured **1.243e-10** — **−3%**. With the
borrowed coefficients this closed only to −16% (OLS) / −4% (WLS). The lumped
serving coefficient is now reconstructed from first principles out of our own
per-channel energies and our own per-channel traffic.

**The law test is now fully self-contained.** `e_DRAM(B200) = 8.37e-11` requires
`e_DRAM(H200) = 1.395e-10` for `e_byte ∝ 1/BW` to hold — **130% of H200's entire
1.076e-10 per-byte budget**. Falsified without reference to any borrowed number.

This closes PI next-step #1 ("per-channel energies measured on our own GPUs
rather than borrowed"). Next-step #2 (`e_gemm` / NVML smoothing) was closed by
UPDATE 4. Next-steps #3 and #4 are **not runnable on this cluster**: #3 needs
power-limit control (`nvidia-smi -pl` denied here, `POWER_LIMIT_SET=no`) and #4
needs a GDDR6/HBM2e part we do not have.
---

# UPDATE 4 review: capped points contaminated three fits (2026-10-08)

The cause is the analysis script I wrote (`steady_calib_analysis.py`). It only *warned*
when points sat at the 1000 W cap; it did not exclude them. A capped point holds power
constant by throttling SM clock and voltage (here 1.1–1.9 GHz), so J/B and J/FLOP fall
and power stops tracking work. The script now **excludes capped points from every
calibration fit** (`CAPPED = 0.05`), reports them separately, and computes R² on the
same target for the pinned refit. Re-run output: `microbench/steady_calib_B200.json`.

| quantity | UPDATE 4 (capped points in) | uncapped | note |
|---|---|---|---|
| steady e_flop | 0.509 pJ (n=32, 18 capped) | **0.943 pJ [0.74, 1.28]** (n=14, R² 0.85) | 1.09 pJ with e_byte pinned to the stream value; 0.78 with an activity intercept |
| steady / serving e_gemm | 0.79 | **1.47** (range 1.2–1.7) | serving per-bin **under**-estimates compute |
| DRAM J/B | 1.122e-10 (all 4 stream points capped, SM ~1570 MHz) | **1.251e-10** (uncapped occupancy grid, +39 W intercept, R² 0.999); duty-cycle full clock 1.2825e-10 | |
| full-clock stream / serving e_wbyte | 0.903 | **1.006** | the serving weight coefficient *is* pure streaming energy |
| energy ridge e_byte/e_flop | 270 ("balanced") | **133** [98, 169] vs performance ridge 281 | 0.47×; same as H200's 134 |

**Three independent estimates now agree that per-bin serving fits under-weight compute:**
- steady uncapped llmgemm: 0.94 pJ (0.78–1.09 across model forms);
- run-level serving fit, smear-free (`BALANCE.md`): 0.93 pJ (fit root) / 0.84 pJ (all
  roots);
- the A5000 smoothing correction: +26–30%.

The mechanism is the one UPDATE 4 identified. The serving logger records
`GetPowerUsage` (`nvml_logger.py:75`), an **0.85 s boxcar on B200**. Prefill bursts get
smeared into neighbouring bins, and the per-bin regression attenuates the FLOP
coefficient.

**KV bytes cost about the same as weight bytes on B200.** With e_gemm pinned at
0.943 pJ, the serving refit gives e_wbyte 1.220e-10 and **e_kvbyte 1.09e-10**. The
independent run-level fit gives 1.226e-10 / **1.025e-10** / 0.927 pJ. The per-bin
2.85e-10 (2.3× a weight byte) is a smear artifact: energy from FLOP-heavy bins lands on
the KV term. Physically this is what you would expect, since both are DRAM reads.
- Caveat: **H200** run-level still gives e_kv 2.4–2.6e-10. H200 has no steady
  calibration (access revoked), so this is unresolved there.
- Per-bin R² with e_gemm pinned is 0.565, against 0.690 free. Expected: per-bin bins
  are where the smear lives. The run-level fit reaches R² 0.993.

**New: the B200's power limit, not its rooflines, is the binding resource.**
- At full clock, peak BW costs 39 W + 1.251e-10 × 8 TB/s ≈ **1040 W dynamic** and peak
  dense FLOPs 0.943 pJ × 2.25 PF ≈ **2120 W**. The budget above idle is **765 W**.
- So within 1000 W it sustains about **73% of peak BW and 36% of peak FLOPs at full
  clock**. It reaches 92% BW and 1.1–1.3 PFLOP/s (50–58% of dense peak) only by
  throttling SM to 1.1–1.7 GHz.
- **Every LLM-shaped GEMM with M ≥ 512 hits the cap**, and 3 of 4 shapes already hit
  it at M = 128 (mlp_up from M = 32). That covers all prefill chunks.
- In the capped regime the delivered P_dyn/FLOP is 0.57–0.67 pJ (lower V). So e_gemm is
  regime-dependent: about 0.94 pJ uncapped (decode-sized GEMMs) and about 0.6 pJ at the
  cap (prefill). The serving per-bin 0.642 coincides with the capped value, but serving
  bins essentially never reached the cap (≤ 0.2% of bins ≥ 950 W). The smear, not
  capping, explains it.
- For the PI's balance question: **B200's compute is provisioned about 2.8× beyond its
  power envelope.** Decode never gets close (MBU ≤ 0.62, power well under the cap).
  Prefill is power-limited.

**UPDATE 4b still holds.** The per-channel solve uses the full-clock duty-cycle fits, not
the capped streams. Its −3% closure agrees with the uncapped grid (1.006×). PI
next-step #2 (e_gemm and smoothing) is closed, **but with the opposite sign from
UPDATE 4**: per-bin serving fits *under*-estimate e_gemm.

**What stands from UPDATE 4:**
- **1/BW law falsified for the DRAM channel.** This is robust to the on-chip split.
  - With the full-clock 1.251e-10 and a 28% on-chip share, e_DRAM is 0.90e-10, so the
    law needs H200 at 1.50e-10.
  - Counting the 1.5× L2 crossings of a streamed byte (about 42% on-chip), e_DRAM is
    0.73e-10, so the law needs 1.21e-10.
  - Both exceed H200's *total* 1.076e-10.
- The NVML smoothing fit: POWER_INSTANT 0.10 s vs GetPowerUsage 0.85 s.
- NCU DRAM/analytic 1.008.
- 92% of peak BW reached (capped).
- The gap between 1.28e-10 (full clock) and 1.12e-10 (saturated) is partly V/f and partly
  the 39 W activity intercept being amortized over more bytes. It is not only a voltage
  effect.

**Open decision: canonical coefficients.** Should `gpu_coefficients.json` adopt B200
e_gemm ≈ 0.94 pJ and e_kvbyte ≈ e_wbyte? That would replace the per-bin fit with the
steady/run-level one. J/token predictions barely move, since per-bin coefficients already
predict run totals at 1.9% MAPE. But the prefill ranking and the energy ridge move, and
the recommender's constant e_gemm does not model the cap's lower J/FLOP.

## UPDATE 4 — CORRECTIONS ACCEPTED (B200 side, 2026-10-09)

The review above is right on all three counts and I am withdrawing those UPDATE 4
claims. I flagged power capping prominently ("read this before trusting any J/B")
and quoted the script's own warning about the occupancy fit — then reported the
`llmgemm` and balance numbers anyway, without checking whether *those* fits also
contained capped points. They did: 18 of 32. Flagging a confound is not the same
as excluding it, and I had the data to refit.

| my UPDATE 4 claim | status |
|---|---|
| steady `e_flop` 0.509 pJ, serving **over**-estimates compute (0.79×) | **withdrawn** → 0.943 pJ, serving **under**-estimates (1.47×) |
| "B200 is balanced, 270 vs 281 FLOP/B" | **withdrawn** → energy ridge 133 vs 281, i.e. 0.47× |
| streaming is 0.903× serving `e_wbyte` | **withdrawn** → 1.006× |

The direction matters: I reported the serving per-bin fit as *over*-weighting
compute, which contradicted the A5000 smoothing hint. Corrected, three
independent estimates agree it **under**-weights compute (0.94 / 0.93 / +26–30%),
and the mechanism is the one UPDATE 4 did get right — `GetPowerUsage` is an
0.85 s boxcar, so prefill bursts smear across bins and attenuate the FLOP term.

**Unaffected: UPDATE 4b (per-channel energies).** That solve used only the
duty-cycle fits, whose 36 points all ran at `sm=1965 MHz, cap=0.00` (verified in
the job log) — short low-duty bursts never reach the cap. So `e_DRAM` 8.37e-11,
`e_L2` 2.29e-11, `e_L1` 1.00e-11 stand, as does the −3% closure against serving
and the self-contained DRAM-law falsification. Likewise UPDATE 3/NCU and the
NVML smoothing fit, which the review also preserves.



### NCU counter availability — verified, for the record

Counters are readable on PARCC `dgx-b200` (contrast `FINDINGS_A5000.md`, where
node-d1 returns `ERR_NVGPUCTRPERM`). Verified four ways:

* `/usr/local/cuda/bin/ncu` ships **inside the vLLM container**. An early probe
  wrongly reported it missing by checking only the host PATH.
* Zero `ERR_NVGPUCTRPERM` across every job log.
* Confirmed on **dgx009, dgx011, dgx020**, and re-probed on dgx009 on 2026-10-09,
  14 days after the first success — same value, so not transient or node-specific.
* Strongest check: the counters **discriminate by op design**. `l2_stream_half`
  reads DRAM/an 0.001, `l1_stream_8KBperprog` reads DRAM 0.000 / L2 0.039, and
  `gemm_fp16_4096` reads L2/an 6.207. A denied or stubbed counter returns zeros
  or a constant; it cannot reproduce each op's intended level, and these values
  matched the handoff's predictions made before the run.

**Still denied here** (do not conflate with the above): clock locking
(`nvidia-smi -lgc`) and power-limit setting (`nvidia-smi -pl`). Hence the skipped
`clocks` stage and the blocked PI next-step #3.

---

# UPDATE 5: provisioning runs (2026-10-16, B200)

`logs/B200_prov/`. Both handoff jobs: `b200_provisioning` (1 GPU, 1h02 — relog,
graphs, highconc, prefill) and `b200_provisioning_tp2` (2 GPUs, 30 min — tp2,
tp1ref). All bins used the **NVML energy counter** (`bin energy source: ['energy']`),
so these are smear-free. Decode arms were essentially uncapped (`cap%` 0–9%;
32B c=64 at 27%); prefill was 100% capped by design.

## The headline: a third of the "fixed per-step overhead" was eager mode

Every H200/B200 serving run before this used `enforce_eager=True`. With CUDA
graphs on, same model/task/concurrency:

| | step time | tok/s | J/tok | MBU |
|---|---|---|---|---|
| 7B alpaca c=1 | 6.8 → **4.6 ms** | ×1.48 | ×0.85 | 0.26 → **0.39** |
| 7B alpaca c=16 | 7.2 → 4.9 ms | ×1.49 | ×0.84 | 0.25 → 0.36 |
| 7B alpaca c=64 | 8.0 → 5.9 ms | ×1.39 | ×0.87 | 0.22 → 0.31 |
| 7B sharegpt c=16 | 7.3 → 4.8 ms | ×1.53 | ×0.84 | 0.24 → 0.36 |

So ~2.2 ms of a 6.8 ms step was eager-mode launch cost, and **bandwidth
utilization rises from 0.26 to 0.39 at c=1**. Part of the "B200 is badly
underutilized" story was a software artifact, not the hardware.

**But the per-byte coefficient barely moves: `e_wbyte` 1.260e-10 (eager) vs
1.257e-10 (graphs).** Energy per byte is unchanged by a 48% throughput gain, so
the invariance result is robust to launch overhead — exactly the right control
to have run. J/tok improves only 13–16%, nowhere near the ~1.9× the 1/BW law
would need.

## Smear-free refit: one hypothesis confirmed, one refuted

| | `e_wbyte` | `e_kvbyte` | `e_gemm` | R² |
|---|---|---|---|---|
| per-bin, GetPowerUsage (Sept) | 1.243e-10 | 2.852e-10 | 0.642 pJ | 0.690 |
| **relog** (energy counter, n=1861) | **1.260e-10** [1.252, 1.270] | **3.981e-10** | **0.713 pJ** [0.655, 0.793] | 0.594 |
| **graphs** (n=1839) | 1.257e-10 [1.243, 1.273] | 3.342e-10 | 0.744 pJ [0.667, 0.809] | 0.225 |
| steady microbench (UPDATE 4 rev.) | — | — | 0.943 pJ | 0.85 |

* **Confirmed:** removing the 0.85 s boxcar raises `e_gemm` (0.642 → 0.713–0.744),
  the predicted direction. But it lands **short of the steady 0.943 pJ**, so
  smoothing explains only part of the gap; the rest is still open.
* **Refuted:** the UPDATE 4 review predicted `e_kv ≈ e_wbyte ≈ 1.0–1.1e-10` once
  the smear was removed. Measured, `e_kvbyte` went the other way — **up** to
  3.98e-10 (relog) / 3.34e-10 (graphs), i.e. 2.7–3.2× a weight byte. The
  smear-artifact explanation for the large KV coefficient does not survive
  smear-free logging.

## Recommender validated on B200; B200 only wins at very high batch

B200 predictions are accurate — measured vs predicted mJ/tok: 2934 vs 3118 (c=1),
206 vs 209 (c=16), 68 vs 64 (c=64), 42 vs 40 (c=128). The cross-GPU ratio moves
monotonically with load:

| load | measured B200 / predicted H200 |
|---|---|
| 7B alpaca c=1 | 1.31 |
| 7B alpaca c=64 | 1.33 |
| 7B sharegpt c=256 | **0.96** |
| 7B sharegpt c=512 | **0.87** |
| 32B alpaca c=256 | 0.95 |
| 32B alpaca c=512 | **0.93** |

This **confirms the provisioning model's central claim**: B200 beats H200 per
token only at very large batch (c ≳ 256), and is 1.2–1.3× worse at the
concurrencies all our earlier sweeps used. Caveat: the H200 column is a
prediction (access revoked 2026-09-24), and the time model was fitted on *eager*
runs at c ≤ 64, so the `graphs` rows show its bias.

## Prefill is power-capped, and that is the finding

| run | J/prompt-token | P | at cap | effective pJ/FLOP |
|---|---|---|---|---|
| in2048_c4 | 0.0190 | 879 W | 100% | 0.685 |
| in2048_c16 | 0.0143 | 961 W | 100% | 0.664 |
| in8192_c4 | 0.0177 | 951 W | 100% | 0.838 |
| in8192_c16 | 0.0165 | 973 W | 100% | 0.830 |

Prefill-dominated serving sits at the 1000 W cap **100% of the time**, and the
effective compute energy (0.66–0.84 pJ/FLOP) lands between the uncapped steady
value (0.943) and the capped GEMM-loop range (0.57–0.67). So "is prefill cheaper
on B200?" — it is cheaper *per FLOP*, but only because the part is throttling.

## TP=2 costs 3–13% more energy per token, buys ~1.5× throughput

First TP measurement with **both GPUs metered** (earlier TP runs would have
metered one):

| model | c | TP2 mJ/tok (tok/s) | TP1 mJ/tok (tok/s) | J/tok |
|---|---|---|---|---|
| 72B | 1 | 25635 (60) | 24852 (37) | ×1.03 |
| 72B | 16 | 1837 (879) | 1724 (572) | ×1.07 |
| 32B | 1 | 12840 (101) | 11528 (71) | ×1.11 |
| 32B | 16 | 898 (1503) | 828 (1031) | ×1.09 |
| 32B | 64 | 293 (5103) | 260 (3733) | ×1.13 |

The all-reduce/sync overhead is a modest **3–13% energy premium** for ~1.4–1.6×
throughput — the first direct measurement of LIMINAL's sync limit on this
hardware.

## Operational notes

* `energy_model.py` was re-verified to bin **old** runs bit-identically
  (`e_bit 1.2767467419697012e-10`, `power_source: avg` on `logs/B200`), so the
  refactor does not disturb any earlier comparison.
* The 32B/72B weights had been deleted in the earlier cleanup; re-prefetched on
  CPU (212 GB, ~2 min) before submitting, otherwise `graphs`/`highconc` would
  have silently lost their 32B arms and the TP job would have run 32B-only.
* `b200_provisioning_tp2.sbatch` specified `--cpus-per-task=56` for `gpu:2`,
  which PARCC's `cli_filter` rejects. The filter requires **exactly** 28
  (`DefCpuPerGPU`) and SLURM multiplies by GPU count ("using 56 processors").
  Fixed in the script with the reason recorded.

## UPDATE 5 review (2026-10-16)

**The KV-vs-compute split is undetermined, neither refuted nor confirmed.** The free
fit uses Qwen2-7B only, where per-bin KV bytes and GEMM FLOPs correlate at 0.74–0.76
(both grow with concurrency). Moving compute energy to KV costs almost nothing in fit
quality:

| fit (smear-free bins, energy counter) | e_wbyte | e_kvbyte | e_gemm | R² |
|---|---|---|---|---|
| relog, free | 1.260e-10 | 3.98e-10 | 0.713 pJ | 0.594 |
| relog, e_gemm pinned at steady 0.943 | 1.250e-10 | **1.22e-10** | 0.943 | 0.560 |
| relog, whole-run (8 runs) | 1.252e-10 | 0.52e-10 | 0.990 | — |
| graphs, free / pinned | 1.257 / 1.249e-10 | 3.34 / **0.97e-10** | 0.744 / 0.943 | 0.225 / 0.201 |

- The 2.3× KV premium (September) is therefore *not shown to be* a smoothing artifact.
- My UPDATE 4 review claim that a KV byte costs about the same as a weight byte was
  equally unsupported: it rested on the same collinearity.
- For 7B, KV is about 4% of dynamic energy, so it barely matters. It matters for long
  context.
- **Settling it needs KV varied independently of FLOPs:** a long-context sweep at
  fixed concurrency (context 1k → 32k, as logs/longctx did on H200).

**What is robust across every fit:**
- e_wbyte = 1.25–1.26e-10 J/B (eager or graphs, free or pinned, per-bin or whole-run).
- e_gemm lies between 0.71 and 0.99 pJ, depending on how KV is treated; the steady
  uncapped value is 0.943.

**Cross-GPU ratio at low concurrency is understated.** "Measured B200 / predicted H200"
puts a CUDA-graphs B200 measurement against an H200 prediction from a time model
fitted on *eager* runs. At c ≤ 16, graphs made B200 1.5× faster than eager, and the
model under-predicts graphs B200 throughput by 15–24%. An H200 with graphs would gain
similarly, so the true low-concurrency B200 penalty is probably larger than 1.3×. At
c ≥ 256 the model matches graphs B200 throughput (within 4%) and the bias is small.

**The crossover depends on the workload, not only on concurrency.**
- B200 wins per token at c ≥ 256 for long-context traffic (sharegpt 7B: 0.87–0.96;
  32B alpaca: 0.93–0.95).
- It never wins for short-prompt 7B (alpaca: 1.23 at c = 512, where the model also
  over-predicts B200 throughput by 28%).

**New provisioning fact: at high load the B200 is power-limited.** 32B at c ≥ 128
runs 95–99% of samples at the 1000 W cap, and prefill runs 100%.

**Possible logger overhead.** The eager relog runs are ~20% slower than the September
runs with the same config (7B c = 1: 6.8 vs 5.4 ms/step). The new logger makes more
NVML calls per sample (sample p99 interval 19–21 ms vs 10 ms) in the client process.
Energy per byte is unaffected (1.26 vs 1.24e-10), but absolute throughput and J/token
of the new runs may be slightly pessimistic. To settle it: one relog c = 1 run with
the new columns disabled.

---

# UPDATE 6: long-context sweep (2026-10-16, B200)

`logs/B200_prov/{longctx,longctx32,loggerab}`, `microbench/ncu_B200/serve_ctx*`.
Job 9085503, 54 min, 1 GPU. Context 1k→30k at **fixed** concurrency, which is
what breaks the KV/FLOP collinearity that made UPDATE 5's conclusion (and the
UPDATE 4 review's) unsupportable.

## The design worked: collinearity broken

`corr(KV bytes, GEMM FLOPs)` per bin: **0.74–0.76** in the concurrency sweeps →
**0.34** (7B decode-only) and **0.13** (32B). KV now varies ~30× while weight
bytes and FLOPs per step stay fixed.

## Byte model holds at long context — a third and fourth confirmation

| capture | context | DRAM/analytic | L2/analytic |
|---|---|---|---|
| UPDATE 3 | short (8 tok) | 1.004 | 1.86 |
| UPDATE 4 | short (24 tok) | 1.008 | 1.57 |
| **this run** | **8,192** | **1.0118** | 1.575 |
| **this run** | **30,720** | **1.0119** | 1.571 |

This is the stronger test. At 30k, KV is ~11–33% of decode bytes, so (per the
handoff) a 2× KV miscount would move DRAM/analytic by ~10%. It moved by **0.1%**.
So **both weight and KV analytic byte counts are validated** — not just weights.
The analytic byte model is good to ~1% across a 30× context range.

## KV energy: ~1.4–1.5× a weight byte. Both earlier claims were wrong.

| | `e_kv / e_w` |
|---|---|
| UPDATE 4 review predicted | ~1.0 |
| September per-bin implied | 2.3–3.2 |
| **7B, free fit** | **1.51** [1.45, 1.59] |
| **7B, `e_gemm` pinned at 0.943 pJ** | **1.53** |
| **32B, `e_gemm` pinned** | **1.39** |

Two models agree on ~1.4–1.5, and for the 7B the ratio is insensitive to how
`e_gemm` is treated (1.51 free vs 1.53 pinned) — which is the reassuring part,
since `e_gemm` is the badly-determined term here.

**Caveat, and it is a real one: the free fits are degenerate.** 7B decode-only
gives R² = **−6.84** with `e_gemm` = **−0.141 pJ**; the 32B free fit is worse
(R² −18.6, `e_gemm` −144 pJ, `e_w` 7.06e-10 with CI [3.9e-10, 1.9e-9]). Decode-only
bins have little energy variance relative to the regressor spread — the same
failure mode as the 72B run. So:

* the **32B free fit is unusable**; its `e_kv/e_w` of 0.24 should be ignored;
* the 32B figure of 1.39 **depends on importing `e_gemm` = 0.943 pJ** from the
  steady microbench, so it is not self-contained;
* treat **1.4–1.5 as the best current estimate, provisional**, not as a settled
  coefficient.

What is now well established across every fit in this corpus: **`e_wbyte` =
1.25–1.34e-10 J/B**.

## Prefill `e_gemm` from these bins is not usable

7B prefill bins give 0.422 pJ, 32B gives 0.004 pJ — both **below** even the
capped GEMM-loop range (0.57–0.67), and the 32B value is nonsense. Prefill ran at
`cap% = 100%`, so these bins are constant-power and cannot calibrate compute.
The steady uncapped 0.943 pJ remains the reference.

## Operational: the new logger costs ~20% throughput (confirmed)

| | full logger | slim logger | September (old logger) |
|---|---|---|---|
| 7B alpaca c=1 | 148 | **178** (×1.21) | 182 |
| 7B alpaca c=64 | 7474 | **9244** (×1.24) | ~8900 |

The slim logger recovers September's throughput almost exactly, so the ~20%
slowdown is the full logger's extra NVML calls, not the node or environment.
**The slim logger should become the default.** Note this means UPDATE 5's
`relog` throughput figures are depressed ~20%; its *coefficients* are less
affected (energy and work fall together) and `e_w` 1.260e-10 there agrees with
1.266e-10 here.

## Power limit binds at high load (corroborates the UPDATE 5 review)

32B `highconc`: `cap%` **95 / 98 / 99%** at c = 128 / 256 / 512. Prefill 100%.
7B stays under 25% at every concurrency. So on B200 the 1000 W limit, not the
rooflines, is the binding resource for a 32B at scale.

## No second GPU type is reachable from this account

The `probe_gpus` stage (read-only, `--test-only`) found **no A100/H100/H200/L40S
GRES** in any partition available to `leebcc-systems-architecture-int` — only
`dgx` (B200), `mig`, and CPU partitions. PI next-step #4 (an uncapped GDDR6 or
HBM2e point) is therefore not reachable here, confirmed rather than assumed.
