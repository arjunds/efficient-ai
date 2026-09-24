# Handoff: B200 results → the H200 server

You are the agent on the **H200** box picking up the B200 measurement. This tells
you what arrived, what to check first, and what to run. Science and caveats are
in `FINDINGS_B200.md`; this is the operational part.

---

## ⚠ CHECK THIS FIRST: `models.py` was reconstructed

`GPU_power/models.py` was committed to this repo as a **0-byte file** (commit
a73675d). On the old cluster the real file lived at `$HOME/models.py` and was
bind-mounted into the container by `ragged_sweep.sbatch`:

```
-B "$HOME/models.py:/workspace/models.py"
```

so it never travelled with the repo. I reconstructed it from the interface its
consumers require. **If the original still exists on the H200 box (check
`~/models.py`), diff it against the committed one before comparing any
coefficients** — if the two disagree on `active_params()` or
`kv_bytes_per_token()`, then H200 and B200 numbers are not comparable and every
cross-GPU conclusion is void.

```bash
diff <(python3 -c "import sys;sys.path.insert(0,'$HOME');import models;\
print(models.MODELS['Qwen2-7B-Instruct'].active_params(),\
models.MODELS['Qwen2-7B-Instruct'].kv_bytes_per_token(2))") \
     <(python3 -c "import sys;sys.path.insert(0,'GPU_power');import models;\
print(models.MODELS['Qwen2-7B-Instruct'].active_params(),\
models.MODELS['Qwen2-7B-Instruct'].kv_bytes_per_token(2))")
```

The reconstruction is validated four independent ways, so I believe it is right,
but a direct diff beats my confidence:

1. `models.py` self-check (`python3 GPU_power/models.py`) reproduces the two
   constants hard-coded in the harness — `energy_profile_load.py:430`
   (`8.03e9 * 2`) and `:431` (`2*32*8*128*2 = 131072`) — exactly, and every table
   entry lands within 2% of its published parameter count.
2. The hand-written table and the `model_from_hf_id()` auto-ingest path agree to
   **0.00%** on all four models testable without an HF token.
3. In production the harness independently computed
   `weight_bytes = 15,230,566,400` for Qwen2-7B = exactly 2× the reconstruction's
   `active_params()`, with `kv/tok = 57,344` and `attn_flops = 401,408`.
4. It predicted 145.4 GB for Qwen2-72B; the actual download was 145.4 GB.

It also fixed a real error found along the way: Qwen3-30B-A3B has **untied**
embeddings (unlike smaller Qwen3 models), which corrects both its total (30.53B
vs 30.5B published) and its active count (3.35B, the "A3B").

---

## What is in this commit

| path | what |
|---|---|
| `GPU_power/models.py` | reconstructed analytic byte/FLOP model (see above) |
| `GPU_power/logs/B200/` | **7B sweep raw data** — 16 runs, 3255 bins, `binned_table.csv` + `run_meta.json` + `specs.json` + `RECORD.txt` |
| `GPU_power/logs/B200_large/` | **72B raw data** — 6 runs, 1528 bins |
| `GPU_power/FINDINGS_B200.md` | the full writeup, results and caveats |
| `GPU_power/scaling_check.py` | scaling test anchored on *published* H200 coefficients (for when the H200 raw logs are not to hand) |
| `GPU_power/pool_gpus.py` | added a `B200` entry to `GPU_SPECS` |
| `GPU_power/run_ragged_sweep.py` | env overrides for the sweep grid; defaults unchanged so the original H200 protocol still reproduces exactly |
| `GPU_power/b200_*.sbatch` | the job scripts (Penn PARCC specific, unlikely to be directly useful to you) |

The logs are **force-added** — the repo `.gitignore` excludes `logs/` and
`*.csv`, which would otherwise have dropped the entire deliverable.

---

## The one command you want

With the H200 raw logs in hand on your side, this is the analysis that was not
possible from the B200 box:

```bash
python3 pool_gpus.py H200:logs/ragged B200:logs/B200
```

It refits **both** GPUs from raw bins through one code path, which is strictly
better than `scaling_check.py` (that anchors on the published H200 numbers
because `logs/ragged` was never on the B200 server). `pool_gpus.py` already knows
B200's specs; `logs/B200/specs.json` also carries them, read from the hardware.

Worth also computing, since it is the cleanest result and needs no fit: H200's
**realized bandwidth and J/byte at c=1** (`tok/s × weight_bytes`, and
`(avg_power − idle) / that`). I only had one such H200 point, derived from
`sweep_summary_H200.csv`. The B200 comparison table in `FINDINGS_B200.md` hangs
off it.

---

## Headline, so you know what you are checking

The datasheet scaling law **`e_byte ∝ 1/HBM_bandwidth` does not hold**:

| | realized BW | J/byte |
|---|---|---|
| H200 Qwen2-7B c=1 | 2.52 TB/s (53% of peak) | 0.997e-10 |
| B200 Qwen2-7B c=1 | 2.78 TB/s (35% of peak) | 1.163e-10 |
| B200 Qwen2-72B c=1 | 4.99 TB/s (62% of peak) | 1.236e-10 |
| *prediction for B200* | | *0.646e-10* |

Energy per byte is roughly **invariant** across two GPU generations, a 10× model
size range and a 2× realized-bandwidth range. The obvious confound — that a 7B
never exercises a B200 — was tested with the 72B and **refuted**: doubling
realized bandwidth did not lower J/byte.

Two things to be careful about when you re-analyse:

* **Do not quote the 72B fitted coefficients.** R² = −0.573. A model that
  saturates the GPU pins power near its ceiling (p50 880 W of a 1000 W cap) and
  destroys identifiability — `weight_bytes_bin` goes nearly constant (CV 0.085)
  while energy varies only 5.4%, so `e_wbyte` degenerates into a ratio of means.
  This is a *new* failure mode, the mirror of gotcha #1; the c=1 direct ratios
  above are unaffected because they involve no regression.
* **B200 idles at 237 W vs H200's 117 W.** Roughly 2×. It matters for any total-
  energy comparison and for the recommender.

---

## Implication for `recommend_gpu.py`

`scaled_gpu()` derives non-H200 coefficients assuming `e_byte ∝ 1/bw`. On this
evidence that is wrong in the direction that matters: it would call a B200 ~1.67×
more energy-efficient per byte and recommend it for 7B decode, when measurement
puts the two within ~17% — and the B200's 2× idle power likely makes it *worse*
in total energy for that workload. I did **not** change `recommend_gpu.py`,
because the fix should wait until you have refit H200 from raw bins and confirmed
the premise.

---

## Also fixed here (may bite you if you ever run on Blackwell)

`vllm/utils/flashinfer.py::use_trtllm_attention()` selects the TRTLLM kernel
**per batch** via `num_tokens <= 256`, so variable-length sharegpt traffic makes
vLLM switch kernels mid-run → `CUDA error: an illegal memory access`. It killed
5 of 8 sharegpt runs. Controls:

| config | result | tok/s |
|---|---|---|
| baseline (auto) | **CRASH** | 68 |
| `VLLM_USE_TRTLLM_ATTENTION=0` | OK | **2487** |
| `VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1` | OK | 1825 |
| `VLLM_USE_TRTLLM_ATTENTION=1` | OK | 2428 |

Forcing TRTLLM *on* also works — neither kernel is broken, the **switching** is.
Not applicable to H200 (sm_90 uses FlashAttention), but the 7B B200 sweep was
collected under the buggy auto config while the 72B used the pinned one. Excluding
the crashed runs changes the 7B coefficients by nothing (1.189e-10 → 1.189e-10,
R² improves 0.677 → 0.708), so I judged a re-run unnecessary — but that is a
config inconsistency you should know about.

## Counter availability (gotcha #5)

`ncu` IS available inside the vLLM container (`/usr/local/cuda/bin/ncu`) -- an
earlier note in this file wrongly said otherwise after probing only the host
PATH. The analytic-vs-hardware byte validation has still not been RUN, but it is
not blocked by tooling; only counter permission is unverified. `dcgmi` *is* present on the host but not
inside the container, which is why every run shows `dram_backend: None`; a bind
mount would enable DCGM field 1005. If NCU is available on the H200 box, that
validation is still the highest-value unrun experiment — a colleague's B200 GEMM
data (`/vast/projects/leebcc/systems-architecture-int/results/coefficients/`)
suggests our lumped `e_wbyte` may be summing the whole memory hierarchy
(DRAM+L2+L1+smem ≈ 1.12e-10, which matches our 1.19e-10) while the scaling law
governs only the DRAM component (0.66e-10, which matches the 0.646e-10
prediction). That is the leading unverified explanation.
