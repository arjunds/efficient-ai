# Handoff — cross-GPU energy-coefficient runs (for a fresh Claude session)

You are picking up an LLM-serving **energy model** project to run on a **new
server with different GPUs**. This doc is self-contained. Repo: `efficient-ai`,
branch `vllm_ragged`, work dir `GPU_power/`.

## The one job
We fit an interpretable per-iteration energy model on H200 and showed its
coefficients behave as **hardware constants**. The missing piece — and your whole
task — is **measuring those coefficients on 2–3 more GPUs** to test the scaling law
that turns a proposal into a result:

> **e_byte ∝ 1/HBM_bandwidth  and  e_flop ∝ 1/peak_FLOPS.**

If a GPU's measured coefficients match what you predict by scaling the H200 values
by its datasheet specs, the model generalizes across hardware zero-shot (→ an
energy-optimal GPU recommender). That's the deliverable.

## The model (what you're measuring)
Per 200 ms bin, dynamic energy is fit by OLS (coeffs ≥ 0):
```
E_bin − P_static·Δt  =  e_wbyte·weight_bytes + e_kvbyte·kv_bytes + e_gemm·gemm_flops
```
`P_static` = measured idle power. Bytes/FLOPs are **analytic** (from model config,
computed in `models.py`); energy is **measured** (NVML). Also report the 2-term
lumped form `e_bit·bytes + e_flop·flops` for comparison.

**H200 reference values (fp16), to check scaling against:**
| quantity | H200 |
|---|---|
| HBM bandwidth β | 4.8 TB/s |
| peak fp16 (dense) π | 990 TFLOPS |
| idle P_static | ~118 W |
| power cap | 700 W |
| **e_wbyte** | **1.077e-10 J/byte** |
| **e_kvbyte** | **3.38e-10 J/byte** |
| **e_gemm** | **0.673 pJ/flop** |
| e_bit / e_flop (2-term) | 1.12e-10 / 0.87 pJ |

So predict, e.g., for a GPU with β=2 TB/s: `e_wbyte ≈ 1.077e-10 × (4.8/2) ≈ 2.6e-10`.

## Pipeline (per GPU) — reproduce this
Everything runs in a container/venv with **vLLM 0.10.2, transformers ~4.56,
pynvml, torch**. The analysis is pure Python + numpy.

**0. One-time: dump prompt pools (KEEP datasets OUT of the vLLM env — see gotcha 3).**
```
PYTHONPATH=<dir-with-datasets> python3 prompts.py dump alpaca   2000 prompts_alpaca.json
PYTHONPATH=<dir-with-datasets> python3 prompts.py dump sharegpt 2000 prompts_sharegpt.json
```

**1. Idle baseline (per GPU — gives P_static):**
```
python3 energy_profile_load.py --mode idle --model <hf_id> --dtype float16 \
  --run_dir logs/<GPU>/<model>/idle --duration_s 15 --max_model_len 8192
```

**2. Sweep — MUST run both tasks and a concurrency range (see gotcha 2):**
```
for task in alpaca sharegpt; do for c in 1 4 16 64; do
  python3 energy_profile_load.py --mode load --model <hf_id> --task $task \
    --concurrency $c --output_len 256 --duration_s 45 --max_model_len 8192 \
    --run_dir logs/<GPU>/<model>/${task}_c${c}
done; done
```
The whole sweep is automated in `run_ragged_sweep.py` (set `RAGGED_LOG_ROOT` and
`RAGGED_MODELS`); adapt its sbatch wrapper `ragged_sweep.sbatch` to the new
scheduler. Models we used: Qwen/Qwen2-7B-Instruct, meta-llama/Meta-Llama-3-8B,
mistralai/Mistral-7B-v0.1, google/gemma-7b (dense). 2–3 models is enough per GPU.

**3. Analyze — produces binned_table.csv + coefficients:**
```
for rd in logs/<GPU>/<model>/{alpaca,sharegpt}_c*; do
  python3 energy_model.py --two_term --run_dir $rd    # writes binned_table.csv
done
python3 fit_channels.py  logs/<GPU>     # 2/3/4-term fit, CIs, held-out MAPE
python3 diagnose_fit.py  logs/<GPU>     # bytes-only vs 2-term, bootstrap CI, held-out
python3 roofline_energy.py logs/<GPU>   # MBU/MFU + power validation (edit β,π,coeffs for this GPU)
```

## Record for each GPU (this is what you hand back)
- **GPU name + datasheet β (HBM BW) + π (peak fp16 FLOPS) + TDP.**
- `nvidia-smi -q -d POWER` → **power cap** and whether the run **saturates it** (gotcha 1).
- Measured **idle P_static**.
- The fitted **e_wbyte, e_kvbyte, e_gemm** (3-term) and **e_bit, e_flop** (2-term),
  with R² and held-out MAPE.
- **Best: the raw `binned_table.csv` + `run_meta.json` per run** (has
  `idle_power_w`) so coefficients can be re-fit centrally and pooled with H200.

## GOTCHAS (learned the hard way — read before running)
1. **Power capping breaks the linear model.** If avg power hits the cap *even at
   concurrency 1* (we saw this on an A100-80GB-PCIe, 300 W), the GPU DVFS-clamps to
   the cap, energy ≈ P_cap·t, and the fit gives **negative R²** — a constant-power
   regime, not a coefficient. Check early: run c1 and look at avg power vs cap. If
   capped, either raise the limit (`nvidia-smi -pl <watts>` if permitted) or prefer
   SXM/higher-TDP parts. Record capped GPUs anyway (they validate the cap term
   `P = min(e·rates + P_static, P_cap)`), but their e_* won't be clean.
2. **Identifiability needs arithmetic-intensity spread — run BOTH tasks.** sharegpt
   (long prompts) gives high-AI/prefill-heavy bins; alpaca (short) gives low-AI
   decode bins. With only one, `bytes` and `flops` are collinear and e_bit/e_flop
   can't separate. Confirm `diagnose_fit.py` reports `identifiable=True` and an AI
   span of ~100–2000×.
3. **`datasets` breaks vLLM if on the same PYTHONPATH.** It pulls
   `huggingface-hub 1.x`, which is incompatible with transformers/vLLM (needs
   <1.0). Fix: dump prompts to JSON in a separate step (step 0), then run vLLM with
   a clean env; `energy_profile_load.py --task` reads the JSON with stdlib json.
4. **Model cache is huge.** Put HF cache on scratch/shared storage and **make sure
   the container bind-mounts that path** (a dangling symlink silently falls back to
   home and blows the quota). Set `HF_HOME`, `HF_XET_CACHE`, `HF_DATASETS_CACHE` to
   the shared location.
5. **Hardware DRAM-byte counters are blocked** (`ERR_NVGPUCTRPERM`, admin-only) on
   our cluster — we use analytic bytes only. If the new server *allows* perf
   counters (DCGM/NCU), run `ncu_dram_check.py` to validate the analytic byte
   counts against hardware — that would be a bonus result we've never been able to get.
6. **New models auto-ingest from HF config** via `models.py::model_from_hf_id` (dense
   GQA/SwiGLU). MoE/MLA need hand entries. Validate any new dense model reconstructs
   sane active/total params before trusting its coefficients.
7. Keep **dtype=float16** and **enforce_eager** (default) to match H200 runs, so
   coefficients are comparable.

## Key files
`energy_profile_load.py` (harness: NVML power + per-iter scheduler log on one
clock) · `models.py` (analytic bytes/FLOPs) · `energy_model.py` (binning +
byte/FLOP accounting, `--two_term` writes binned_table.csv) · `fit_channels.py` /
`fit_two_term.py` / `diagnose_fit.py` (fits) · `roofline_energy.py` (roofline
tie-in) · `run_ragged_sweep.py` + `ragged_sweep.sbatch` (full sweep driver) ·
`recommend_gpu.py` (the recommender the new coefficients feed) ·
`PROPOSAL.md` (full context + results).

## Re-integration (when data comes back)
`pool_gpus.py` ingests multiple GPUs' logs, fits per-GPU coefficients, tests the
scaling law, and writes `plots_proposal/fig7_gpu_scaling.png`:
```
python3 pool_gpus.py H200:logs/ragged L40S:logs/L40S A100-SXM:logs/A100
```
Each arg is `LABEL:log_dir` (log_dir scanned for binned_table.csv at any depth).
Specs come from its built-in `GPU_SPECS` table (add your GPU there) or a
`specs.json` in the log dir `{"bw":..,"peak_flops":..,"p_cap":..}`. It auto-flags
cap-saturated GPUs and excludes them from the scaling fit. **So the fastest path:
just get the new GPUs' `binned_table.csv`+`run_meta.json` into `logs/<GPU>/...` and
run this — it does the rest.** (Requires the channel-column binning, i.e. run
`energy_model.py --two_term` with the current code, not an old checkout.)

## What success looks like
For each new GPU, measured `e_wbyte` ≈ `1.077e-10 × (4.8e12 / β_new)` and
`e_gemm` ≈ `0.673e-12 × (990e12 / π_new)` to within ~10–20%. Drop the new
`binned_table.csv` files back into `logs/<GPU>/…` and re-run `fit_channels.py`
pooled to confirm. If the scaling holds across 2–3 GPUs, the recommender becomes
zero-shot from datasheets — the headline result.
```
```
