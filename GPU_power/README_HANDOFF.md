# Offered-load energy harness — build handoff

Built overnight on branch `vllm_ragged` to the spec in `~/PROFILING_HANDOFF.md`.
Implements the LIMINAL energy extension data collection:

```
E_iter = e_bit · (weight_bytes + Σ_seq KV_bytes(ctx_seq)) + P_static · t_iter
```

Drives vLLM with **N concurrent clients** (not `batch_size=N`) so the iteration
scheduler forms real ragged running batches, captures time-resolved NVML power +
per-iteration scheduler state on one shared clock, reconciles a live DRAM-byte
counter, then **calibrates the model and validates it by predicting the measured
power waveform**.

## ⚠️ Read first: environment decision (needs your call)

The prior vLLM runs worked on **RunAI/Kubernetes** (`~/sweep_job`,
`nvcr.io/nvidia/pytorch:24.01-py3`). The SLURM side of this cluster exposes **no**
CUDA/apptainer/python modules and no container runtime on the login node, so a
working vLLM under SLURM is **unconfirmed**. I did not spin up RunAI pods (real
GPU resource; you asked for SLURM) and ran nothing on the login node.

- **Fastest to data: RunAI** (known-good). `./launch_runai.sh`.
- **SLURM: probe first.** `sbatch probe.sbatch` reports whether apptainer/
  singularity exists on compute nodes, whether PyPI is reachable to pip-install
  vLLM into `~/.local`, and whether `dcgmi` is present. I submitted this probe —
  check `egy-probe-*.out` (and `probe_result.json`) for the verdict.

Either way the **Python harness is identical**; only the launcher differs.

## Quickstart

```bash
# inside the GPU container (RunAI pod or SLURM+apptainer shell), from this dir:
bash setup_load.sh                                   # deps (+ your known-good vLLM)
python probe_env.py --dump-vllm-api                  # confirm env + scheduler API

# prove the math with NO GPU (works anywhere, incl. now):
python energy_profile_load.py --mode self_test --run_dir logs/selftest
python energy_model.py     --run_dir logs/selftest   # should recover e_bit,P_static
python validate_waveform.py --run_dir logs/selftest  # expect R^2 ~ 0.99

# one real run (smoke), then the sweep:
python energy_profile_load.py --mode idle   --model meta-llama/Meta-Llama-3-8B --run_dir logs/m/idle --duration_s 20
python energy_profile_load.py --mode load   --model meta-llama/Meta-Llama-3-8B --concurrency 1 --input_len 512 --output_len 128 --run_dir logs/m/c1
python gate_dram.py --run_dir logs/m/c1              # GATE #1 must pass
python energy_model.py --run_dir logs/m/c1 && python validate_waveform.py --run_dir logs/m/c1
python run_load_sweep.py                             # full resume-safe sweep
```

## Files

| file | role |
|------|------|
| `energy_profile_load.py` | driver: modes `load` / `idle` / `prefill_ceiling` / `self_test` |
| `engine_compat.py`       | vLLM V0/V1 async-engine shim; forces in-process engine core |
| `nvml_logger.py`         | in-process NVML → `power_trace.csv` (power/clocks/util/temp) |
| `iter_logger.py`         | monkeypatches `Scheduler.schedule` → `iter_log.csv` per step |
| `dram_counter.py`        | DCGM `DRAM_ACTIVE`×peak-BW → `dram_trace.csv`, window integrals |
| `gate_dram.py`           | handoff gate #1 via `~/models.py` (abort if counter broken) |
| `energy_model.py`        | calibrate `e_bit` (idle-anchored `P_static`) → `calibration.json` |
| `validate_waveform.py`   | replay iters → predicted power(t) vs measured; RMSE/R² + overlay |
| `run_load_sweep.py`      | resume-safe grid: baselines → gate → concurrency×workload sweep |
| `probe_env.py` / `probe.sbatch` | environment probe (in-container / SLURM node) |
| `launch_runai.sh` / `setup_load.sh` | RunAI launcher / in-container deps |

## Artifacts per run (matches handoff schema)

- `power_trace.csv`: `t_wall,power_w,sm_mhz,mem_mhz,gpu_util,mem_util,temp_c`
- `iter_log.csv`: `t_start,t_end,phase,n_running,n_waiting,kv_tokens_resident,kv_cache_pct,prefill_tokens,decode_tokens,dram_bytes`
- `dram_trace.csv`: `t_wall,dram_active_frac,inst_bytes_per_s`
- `run_meta.json`, `results.json`, `calibration.json`, `waveform_validation.{json,png}`
- `iter_ctx.jsonl` sidecar: per-request context lengths per step (for re-deriving Σ KV)

## Validation gates (handoff)

1. **DRAM counter** — `gate_dram.py`: c=1 decode `dram_bytes/token` within 0.8–1.5×
   the model's `active_params·dtype + KV(ctx)`. The sweep runs this first and
   **skips any model that fails or isn't in `models.py`**.
2. **Clock alignment** — power_trace and iter_log share `time.time()`; the
   waveform join in `validate_waveform.py` fails loudly if they don't overlap.
3. **Idle baseline** — `--mode idle` captured per model; anchors `P_static`.

## Known risks / TODO (confirm with the probe, then fix)

- **vLLM scheduler API** (`iter_logger.py`): written defensively for V1 (target,
  per the repo's 0.10.x references) with a V0 fallback, but the exact attribute
  names (`num_computed_tokens`, `num_scheduled_tokens`, kv-cache-manager blocks)
  are **unverified against the installed build**. Run `probe_env.py --dump-vllm-api`;
  correct `_SCHED_PATHS` and the extractors in `iter_logger._record` if the dump
  differs. If `iter_log.csv` has blank `kv_tokens_resident`/`phase`, this is why.
- **In-process engine core**: `engine_compat.py` sets
  `VLLM_ENABLE_V1_MULTIPROCESSING=0` so the scheduler monkeypatch is visible. If
  vLLM ignores it in your version, iter_log will be empty — switch to registering
  a `StatLoggerBase` (partial fields only; no per-seq ctx).
- **DRAM counter**: needs `dcgmi`. If absent, `dram_bytes` is blank and gate #1
  cannot pass — the alternative is offline CUPTI `dram__bytes_read/write.sum`
  calibration (can't run under load). Peak-HBM-BW table in `dram_counter.py` must
  include the actual GPU (check the probe's GPU name).
- **`min_tokens`/`ignore_eos`**: used to force exact output_len; if unsupported,
  decode windows still work but output lengths vary.
- **models.py coverage**: `deepseek-llm-7b-base` is NOT in `models.py` (that's the
  dense 7B, not DeepSeek-V3) — it'll be flagged/skipped. Only sweep models with a
  config.

## Status

Code complete and committed on `vllm_ragged`. `self_test` path is fully
exercisable with no GPU. No GPU run has executed yet (env unresolved). The SLURM
probe was submitted; results in `egy-probe-*.out`.
