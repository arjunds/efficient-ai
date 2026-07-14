# Session log — offered-load energy harness (branch `vllm_ragged`)

A readable narrative of what was built and run, for catching up without scrolling
the terminal. Newest section (the sweep) is at the bottom. All code is committed
on branch `vllm_ragged`; run outputs live under `GPU_power/logs/` (gitignored).

## Goal
Server-side data collection for the LIMINAL energy extension:
`E_iter = e_bit·(weight_bytes + Σ_seq KV_bytes(ctx)) + P_static·t_iter`. Drive
vLLM with N concurrent clients (real ragged continuous-batching, NOT static
batch), log NVML power + per-iteration scheduler state on one clock, calibrate
`e_bit`/`P_static`, and validate by predicting the measured power waveform.

## Environment (see also memory: cluster-env)
- Login node has no GPU stack. Everything runs via `sbatch` on a compute node.
- Container: `apptainer exec --nv /shared_data0/fwzhang/vllm.sif` — python3.12,
  torch 2.8+cu128, **vLLM 0.10.2 (V1 engine)**, transformers 4.56, pynvml, ncu.
  No `dcgmi`.
- A100 is a ~4-day queue; **runs are on H200** (700 W, 143 GB) per user choice.
- HF token stored at `~/.cache/huggingface/token` (600, not in git).

## What was fixed to get vLLM 0.10.2 working (chronological)
1. Container interpreter is `python3`, not `python`.
2. Load path decoupled from the old `energy_profile*.py` (they import
   pandas/datasets, absent in the sif); helpers inlined into `engine_compat.py`.
3. `AsyncEngineArgs` in 0.10.2 dropped `disable_log_requests`.
4. **Async engine is multiprocessing-only.** Setting `VLLM_ENABLE_V1_MULTIPROCESSING=0`
   broke core init; removed it. Per-iteration logging rewritten from an
   in-process scheduler monkeypatch to a proper **`StatLoggerBase`** wired via
   `AsyncLLM(stat_loggers=[...])`. `kv_tokens_resident = kv_cache_usage ·
   num_gpu_blocks · block_size` (exact — verified against vLLM's own log).
5. **Binned energy attribution.** The stat-logger `record()` timestamps are
   async-jittery and finer than the power sampling, so per-iteration energy was
   noise-dominated. Calibration + validation now bin over ~200 ms; the fit
   `E_bin = e_bit·bytes_bin + P_static·Δ` is the sound roofline relation.

## Results so far (Qwen2-7B-Instruct on H200)
- Idle baseline **P_static = 118 W**.
- Fixed load: c=1 → 165 tok/s, 367 W; c=8 → 1263 tok/s, 372 W. Decode is
  HBM-bound (weights reloaded per iter) so power is ~flat across concurrency
  while throughput scales — the expected memory-bound regime.
- **`e_bit ≈ 1.07–1.18e-10 J/byte`**, consistent across loads.
- Fixed-load waveform MAPE 1.6–2.4% (R² not meaningful on a flat signal).
- **Bursty active↔idle run (`logs/wave/`): waveform R² = 0.83, MAPE 9%** — the
  strong "predict the power waveform" validation, on a trace that actually moves.
- Overlay plots: matplotlib import is flaky in the sif, so
  `waveform_validation.py` also writes `waveform_series.csv`
  (`t_rel_s,measured_w,predicted_w`) — plot that on the analysis side.

## How to read outputs
- Per run dir (`logs/<...>/`): `run_meta.json` (config + baselines + segments),
  `results.json` (throughput/power), `power_trace.csv`, `iter_log.csv`
  (per-iteration scheduler state), `calibration.json` (e_bit, P_static, r2),
  `waveform_validation.json` + `waveform_series.csv`.
- Sweep roll-up: `logs/load_sweep/sweep_summary.csv` (one row per operating
  point) and `logs/load_sweep/sweep_console.log` (live progress).
- SLURM stdout per job: `GPU_power/egy-*-<jobid>.out`.

## Jobs run (for reference)
- 62598 probe (SLURM env), 62619–62623 first-run iterations (fixed the above),
  62622 API diagnostic, 62624/62625 reanalysis (binning), 62626 bursty waveform.

## THE SWEEP (running now)
`sweep.sbatch` → `run_load_sweep.py` inside the container on H200:
- MODELS: Qwen2-7B-Instruct (open), Meta-Llama-3-8B, Mistral-7B-v0.1 (gated — run
  if the HF token has access, else skipped).
- CONCURRENCY: 1,2,4,8,16,32,64 (stops climbing when throughput saturates).
- WORKLOADS: (input 512, output 128) and (input 2048, output 128).
- Per model: idle + prefill-ceiling baselines, DRAM gate at c=1 (non-fatal here
  since no dcgmi), then each point → calibrate + validate.
- Resume-safe: re-submit `sweep.sbatch` to continue if preempted.
- Watch: `logs/load_sweep/sweep_summary.csv` grows as points complete.
