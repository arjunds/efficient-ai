#!/usr/bin/env python3
"""
iter_logger.py

Per-iteration vLLM scheduler-state logger. One row per LLMEngine/EngineCore
scheduling step, feeding the `Σ_seq KV_bytes(ctx_seq)` term of the energy model.

Emits iter_log.csv (matches PROFILING_HANDOFF.md schema):
    t_start, t_end, phase, n_running, n_waiting, kv_tokens_resident,
    kv_cache_pct, prefill_tokens, decode_tokens, dram_bytes

`dram_bytes` is left blank here and filled in later by dram_counter.py
reconciliation (joined on the [t_start, t_end] window). Optionally also writes a
sidecar iter_ctx.jsonl with the raw per-request context-length list per step, so
the analysis side can recompute Σ KV under any KV model (e.g. Llama4 chunked).

MECHANISM
---------
We monkeypatch the scheduler's `schedule()` method on its CLASS (so it applies to
the already-constructed engine instance — Python binds methods at call time). The
wrapper times the call and, from the scheduler instance's own `running` / `waiting`
queues plus the returned scheduler output, extracts every field.

vLLM has two very different engines. We try V1 first (default since ~0.8; the
repo's energy_profile_vllm.py targets 0.10.x), then V0. Every attribute access is
defensive: a shape we don't recognize yields a blank cell, never a crash — a
degraded row is better than killing the serving run. The probe job
(probe_env.py --dump-vllm-api) prints the real installed API so these extractors
can be corrected against ground truth.
"""

import csv
import json
import os
import threading
import time
from typing import Any, List, Optional, Tuple


# Candidate import paths for the scheduler class, newest-first.
_SCHED_PATHS = [
    ("vllm.v1.core.sched.scheduler", "Scheduler", "v1"),
    ("vllm.v1.core.scheduler", "Scheduler", "v1"),
    ("vllm.core.scheduler", "Scheduler", "v0"),
]


def _resolve_scheduler_class():
    import importlib
    for mod_path, cls_name, ver in _SCHED_PATHS:
        try:
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name, None)
            if cls is not None and hasattr(cls, "schedule"):
                return cls, ver
        except Exception:
            continue
    return None, None


class IterationStatsLogger:
    def __init__(self, out_csv: str = "iter_log.csv",
                 ctx_sidecar: Optional[str] = "iter_ctx.jsonl"):
        self.out_csv = out_csv
        self.ctx_sidecar = ctx_sidecar

        self._cls = None
        self._ver = None
        self._orig_schedule = None
        self._lock = threading.Lock()
        self._file = None
        self._writer = None
        self._sidecar_f = None
        self._n_rows = 0
        self.attached = False

    # ---------------- attach / detach ----------------
    def attach(self):
        cls, ver = _resolve_scheduler_class()
        if cls is None:
            raise RuntimeError(
                "Could not locate a vLLM Scheduler class to hook. Run "
                "`python probe_env.py --dump-vllm-api` and update _SCHED_PATHS."
            )
        self._cls, self._ver = cls, ver
        self._orig_schedule = cls.schedule

        os.makedirs(os.path.dirname(self.out_csv) or ".", exist_ok=True)
        self._file = open(self.out_csv, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow([
            "t_start", "t_end", "phase", "n_running", "n_waiting",
            "kv_tokens_resident", "kv_cache_pct", "prefill_tokens",
            "decode_tokens", "dram_bytes",
        ])
        if self.ctx_sidecar:
            self._sidecar_f = open(self.ctx_sidecar, "w")

        logger = self
        orig = self._orig_schedule
        ver = self._ver

        def patched_schedule(sched_self, *args, **kwargs):
            t0 = time.time()
            out = orig(sched_self, *args, **kwargs)
            t1 = time.time()
            try:
                logger._record(sched_self, out, t0, t1, ver)
            except Exception:
                # Never let logging break scheduling.
                pass
            return out

        cls.schedule = patched_schedule
        self.attached = True
        return self

    def detach(self):
        if self._cls is not None and self._orig_schedule is not None:
            self._cls.schedule = self._orig_schedule
        with self._lock:
            if self._file is not None:
                self._file.flush()
                self._file.close()
            if self._sidecar_f is not None:
                self._sidecar_f.flush()
                self._sidecar_f.close()
        self.attached = False
        return self

    def __enter__(self):
        return self.attach()

    def __exit__(self, *exc):
        self.detach()

    # ---------------- extraction ----------------
    def _record(self, sched, out, t0, t1, ver):
        running = getattr(sched, "running", None) or []
        waiting = getattr(sched, "waiting", None) or []
        n_running = _safe_len(running)
        n_waiting = _safe_len(waiting)

        ctx_lengths = self._context_lengths(running, ver)
        kv_tokens_resident = sum(ctx_lengths) if ctx_lengths else None

        prefill_tokens, decode_tokens = self._token_split(out, running, ver)
        phase = _phase_tag(prefill_tokens, decode_tokens)
        kv_cache_pct = self._kv_cache_pct(sched, out)

        with self._lock:
            self._writer.writerow([
                f"{t0:.6f}", f"{t1:.6f}", phase, n_running, n_waiting,
                "" if kv_tokens_resident is None else int(kv_tokens_resident),
                "" if kv_cache_pct is None else f"{kv_cache_pct:.4f}",
                "" if prefill_tokens is None else int(prefill_tokens),
                "" if decode_tokens is None else int(decode_tokens),
                "",  # dram_bytes filled by reconciliation
            ])
            self._n_rows += 1
            if self._sidecar_f is not None and ctx_lengths is not None:
                self._sidecar_f.write(json.dumps(
                    {"t_start": t0, "t_end": t1, "ctx": ctx_lengths}) + "\n")

    def _context_lengths(self, running, ver) -> Optional[List[int]]:
        """Resident context length (tokens already in KV) for each running req."""
        out: List[int] = []
        for item in running:
            n = _first_attr(item, [
                "num_computed_tokens",     # v1 Request: tokens with KV resident
            ])
            if n is None:
                # v0 SequenceGroup -> first seq length
                try:
                    seqs = item.get_seqs()
                    if seqs:
                        n = seqs[0].get_len()
                except Exception:
                    n = None
            if n is None:
                n = _first_attr(item, ["num_tokens", "seq_len", "context_len"])
            if n is not None:
                try:
                    out.append(int(n))
                except Exception:
                    pass
        return out or None

    def _token_split(self, out, running, ver) -> Tuple[Optional[int], Optional[int]]:
        """(prefill_tokens, decode_tokens) scheduled this iteration."""
        # v1 SchedulerOutput: num_scheduled_tokens: dict[req_id -> int];
        # a request is prefilling while computed < prompt length.
        nst = _first_attr(out, ["num_scheduled_tokens"])
        if isinstance(nst, dict) and nst:
            prefill = decode = 0
            by_id = {}
            for item in running:
                rid = _first_attr(item, ["request_id", "req_id"])
                if rid is not None:
                    by_id[rid] = item
            for rid, ntok in nst.items():
                try:
                    ntok = int(ntok)
                except Exception:
                    continue
                # decode step schedules exactly 1 new token per request.
                if ntok <= 1:
                    decode += ntok
                else:
                    prefill += ntok
            return prefill, decode

        # v1 alt: explicit counters sometimes present on the output.
        nb = _first_attr(out, ["num_batched_tokens", "total_num_scheduled_tokens"])
        npref = _first_attr(out, ["num_prefill_tokens"])
        if npref is not None and nb is not None:
            return int(npref), int(nb) - int(npref)

        # v0 SchedulerOutputs tuple: (seq_group_metadata_list, scheduler_outputs, ...)
        so = None
        if isinstance(out, tuple) and len(out) >= 2:
            so = out[1]
        if so is not None:
            nb = _first_attr(so, ["num_batched_tokens"])
            num_prefill_groups = _first_attr(so, ["num_prefill_groups"])
            if nb is not None and num_prefill_groups is not None:
                # Prefill groups carry chunked prompt tokens; decode groups carry 1
                # token each. Without per-group chunk sizes we approximate decode as
                # (num running - prefill groups) and prefill as the remainder.
                n_run = _safe_len(running)
                decode = max(0, (n_run or 0) - int(num_prefill_groups))
                prefill = max(0, int(nb) - decode)
                return prefill, decode
        return None, None

    def _kv_cache_pct(self, sched, out) -> Optional[float]:
        # Prefer a value carried on the scheduler output / stats.
        for obj in (out, sched):
            v = _first_attr(obj, ["gpu_cache_usage", "gpu_cache_usage_perc",
                                   "kv_cache_usage"])
            if isinstance(v, (int, float)):
                return float(v)
        # Try the v1 KV-cache manager block accounting.
        kvm = _first_attr(sched, ["kv_cache_manager"])
        if kvm is not None:
            free = _first_attr(kvm, ["num_free_blocks", "free_block_count"])
            total = _first_attr(kvm, ["num_gpu_blocks", "num_total_blocks",
                                      "total_num_blocks"])
            try:
                if free is not None and total:
                    return 1.0 - (float(free) / float(total))
            except Exception:
                pass
        return None

    @property
    def n_rows(self):
        return self._n_rows


# ---------------- module-level helpers ----------------
def _safe_len(x) -> Optional[int]:
    try:
        return len(x)
    except Exception:
        return None


def _first_attr(obj: Any, names) -> Any:
    for n in names:
        if isinstance(obj, dict):
            if n in obj:
                return obj[n]
            continue
        v = getattr(obj, n, None)
        if v is not None:
            return v
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
