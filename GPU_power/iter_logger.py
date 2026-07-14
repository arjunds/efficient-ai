#!/usr/bin/env python3
"""
iter_logger.py

Per-iteration vLLM scheduler-state logger for the V1 engine, implemented as a
StatLoggerBase passed via AsyncLLM(stat_loggers=[...]). This is the correct hook
for vLLM's multiprocessing engine core: SchedulerStats/IterationStats are shipped
from the core process to the frontend, where record() is called once per engine
step. (An in-process monkeypatch cannot see the core process — that approach was
abandoned after confirming vllm 0.10.2's async engine is multiprocessing-only.)

Emits iter_log.csv (PROFILING_HANDOFF.md schema):
    t_start, t_end, phase, n_running, n_waiting, kv_tokens_resident,
    kv_cache_pct, prefill_tokens, decode_tokens, dram_bytes

kv_tokens_resident (= Σ_seq ctx) is derived exactly from the fraction:
    kv_cache_usage · num_gpu_blocks · block_size
(confirmed against vLLM's own "GPU KV cache size" log). dram_bytes is filled later
by dram_counter reconciliation.

`make_iter_logger_factory(out_csv)` returns a factory Callable[[VllmConfig, int],
StatLoggerBase] and imports vLLM lazily, so importing this module never requires
vLLM (self_test / non-GPU paths stay clean).
"""

import csv
import time


def make_iter_logger_factory(out_csv):
    from vllm.v1.metrics.loggers import StatLoggerBase

    class IterationStatsLogger(StatLoggerBase):
        def __init__(self, vllm_config, engine_index: int = 0):
            self.engine_index = engine_index
            self._vllm_config = vllm_config
            self._kv_capacity_tokens = self._resolve_kv_capacity(vllm_config)
            self._prev_t = None
            self._f = None
            self._writer = None
            self.n_rows = 0
            # Only the first engine writes the shared CSV.
            if engine_index == 0:
                import os
                os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
                self._f = open(out_csv, "w", newline="")
                self._writer = csv.writer(self._f)
                self._writer.writerow([
                    "t_start", "t_end", "phase", "n_running", "n_waiting",
                    "kv_tokens_resident", "kv_cache_pct", "prefill_tokens",
                    "decode_tokens", "dram_bytes"])
                self._f.flush()

        @staticmethod
        def _resolve_kv_capacity(vllm_config):
            try:
                cc = vllm_config.cache_config
                nb = getattr(cc, "num_gpu_blocks", None)
                bs = getattr(cc, "block_size", None)
                if nb and bs:
                    return int(nb) * int(bs)
            except Exception:
                pass
            return None

        # StatLoggerBase abstract surface
        def log_engine_initialized(self):
            # num_gpu_blocks is finalized by now; capture if we missed it at init.
            if self._kv_capacity_tokens is None:
                self._kv_capacity_tokens = self._resolve_kv_capacity(self._vllm_config)

        def log(self, *args, **kwargs):
            pass

        def record(self, scheduler_stats, iteration_stats=None, engine_idx: int = 0):
            if self._writer is None or scheduler_stats is None:
                return
            t = time.time()
            t_start = self._prev_t if self._prev_t is not None else t
            self._prev_t = t

            nr = getattr(scheduler_stats, "num_running_reqs", "")
            nw = getattr(scheduler_stats, "num_waiting_reqs", "")
            kv_usage = getattr(scheduler_stats, "kv_cache_usage", None)

            if self._kv_capacity_tokens is None:
                self._kv_capacity_tokens = self._resolve_kv_capacity(self._vllm_config)
            kv_tokens = ""
            if kv_usage is not None and self._kv_capacity_tokens:
                kv_tokens = int(kv_usage * self._kv_capacity_tokens)

            pref = _getattr_num(iteration_stats, "num_prompt_tokens")
            dec = _getattr_num(iteration_stats, "num_generation_tokens")
            phase = _phase_tag(pref, dec)

            self._writer.writerow([
                f"{t_start:.6f}", f"{t:.6f}", phase, nr, nw,
                kv_tokens,
                "" if kv_usage is None else f"{kv_usage:.4f}",
                "" if pref is None else int(pref),
                "" if dec is None else int(dec),
                ""])  # dram_bytes filled by reconciliation
            self.n_rows += 1
            self._f.flush()

    def factory(vllm_config, engine_index: int = 0):
        return IterationStatsLogger(vllm_config, engine_index)

    return factory


def _getattr_num(obj, name):
    if obj is None:
        return None
    v = getattr(obj, name, None)
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _phase_tag(prefill_tokens, decode_tokens) -> str:
    p = prefill_tokens or 0
    d = decode_tokens or 0
    if p and d:
        return "mixed"
    if p:
        return "prefill"
    if d:
        return "decode"
    return "idle"
