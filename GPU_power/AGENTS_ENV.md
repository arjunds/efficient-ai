# Working environment (read first) — shared brief for parallel agents

Repo: `~/efficient-ai`, branch `vllm_ragged`, work dir `~/efficient-ai/GPU_power`.
Project context: `PROPOSAL.md` (the model + results), `FINDINGS_B200.md` (B200
cross-GPU result: e_byte ~invariant, 1/BW scaling law FALSIFIED),
`CONTROL_THEORY_NOTES.md` (MPC/state-space viability), `HANDOFF_CROSSGPU.md`
(pipeline + gotchas), `SESSION_LOG.md` (history).

## Compute access (changed 2026-09-24 — important)
- **Login node:** no numpy/scipy in any system python. Do NOT run numeric work here.
- **Only usable partition: `debug`** (account `exwong`), node `node-d1`: 48 CPU,
  250 GB RAM, **4× RTX A5000 (24 GB GDDR6, 230 W TDP, sm_86)**, **MaxTime 4 h**.
  Shared with other users — request only what you need (≤1 GPU per job, `-c 4-8`).
  `standby` / H200 / L40S / A100 partitions now **reject** our account.
- **Fast blocking runs (use these to iterate):**
  ```
  srun -p debug -A exwong -t 00:20:00 -c 4 --mem 16G \
    apptainer exec -B $HOME/efficient-ai/GPU_power:/workspace \
      -B $HOME/models.py:/workspace/models.py -B /shared_data0 \
      /shared_data0/fwzhang/vllm.sif bash -lc 'cd /workspace && python3 your_script.py'
  ```
  ~6 s overhead. Add `--gres=gpu:a5000:1` for GPU. For long jobs use `sbatch`
  with the same partition/account (write `#SBATCH -p debug -A exwong`).
- **Container** `/shared_data0/fwzhang/vllm.sif`: python 3.12, numpy 2.2.6,
  scipy 1.16.2, torch 2.8+cu128, vLLM 0.10.2, transformers 4.56, pynvml.
  **No cvxpy/osqp/matplotlib/pandas.** For plots: `export
  PYTHONPATH=/shared_data0/adsampat/pydeps:$PYTHONPATH` (matplotlib) — but
  **never** put pydeps on PYTHONPATH for a vLLM run (its huggingface-hub 1.x
  breaks transformers). For optimization use scipy (linprog/HiGHS, minimize/SLSQP,
  milp) or hand-rolled solvers.
- `/tmp` and the session scratchpad are NOT visible on compute nodes. Put files
  jobs need under the repo (bound as `/workspace`) or `/shared_data0/adsampat/`.
- `models.py`: jobs bind `~/models.py` (canonical) over `/workspace/models.py`.
  The repo copy `GPU_power/models.py` is being reconciled — always bind ~/models.py.
- HF cache on shared storage (cached already: Qwen2-7B-Instruct, Llama-3-8B,
  Mistral-7B, gemma-7b, Qwen2.5-{0.5,1.5,3,7,14,32}B-Instruct, Qwen3-30B-A3B).
  Env: `APPTAINERENV_HF_HOME=$HOME/.cache/huggingface`,
  `APPTAINERENV_HF_XET_CACHE=/shared_data0/adsampat/hf_cache/xet`,
  `APPTAINERENV_HF_TOKEN=$(cat ~/.cache/huggingface/token)`.
- Prompt pools already dumped: `prompts_alpaca.json`, `prompts_sharegpt.json`
  (gitignored — contain third-party text; never commit them).

## Data on disk (all under GPU_power/logs/, per-run dirs)
Each run dir: `iter_log.csv` (per vLLM iteration: `t_start,t_end,phase,n_running,
n_waiting,kv_tokens_resident,kv_cache_pct,prefill_tokens,decode_tokens,dram_bytes`),
`power_trace.csv` (NVML ~10 ms), `binned_table.csv` (200 ms bins: energy + analytic
weight/KV bytes + GEMM/attn FLOPs), `run_meta.json` (config, idle_power_w,
window times), `results.json` (tok/s, avg power, completed requests).
- `logs/ragged/` H200, 4 dense models × {alpaca,sharegpt} × c{1,4,16,64} + poisson8
- `logs/ragged_ladder/` H200 Qwen2.5 0.5–32B; `logs/ragged_moe/` H200 Qwen3-30B-A3B
- `logs/ragged_a100/` A100-PCIe (power-capped at 300 W — linear model invalid)
- `logs/B200/`, `logs/B200_32B/`, `logs/B200_large/` B200 (7B/32B/72B)
- Per-request latency (TTFT/ITL) is **not** currently persisted — only counts.

## Rules for parallel agents
- **Do not run git commands** (no add/commit/checkout/stash). The lead commits.
- Write only the files your brief assigns you; don't edit other agents' files.
- If you must change a shared file (e.g. the harness), make it additive and
  backward-compatible, and say so in your report.
- Report honestly: what worked, what didn't, numbers with uncertainty.
