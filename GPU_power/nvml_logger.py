#!/usr/bin/env python3
"""
nvml_logger.py

In-process NVML power / clock / utilization / temperature time-series logger.

Why in-process (vs the nvidia-smi subprocess in energy_profile.py):
  - Shares one wall clock (time.time()) with the per-iteration scheduler log, so
    power windows can be joined to the iterations that ran in them. Clock
    alignment is a hard validation gate for waveform prediction (PROFILING_HANDOFF.md).
  - No CSV timestamp-string parsing; higher, more regular sample rate.

Emits power_trace.csv with columns (matches handoff schema):
    t_wall, power_w, sm_mhz, mem_mhz, gpu_util, mem_util, temp_c

Usage:
    log = NvmlPowerLogger(gpu_id=0, out_csv="run/power_trace.csv", interval_ms=10)
    log.start()
    ... workload ...
    log.stop()
    print(log.gpu_name, log.power_limit_w)
    print(log.summary())   # mean/median power etc. over the whole trace
"""

import csv
import threading
import time
from typing import List, Optional

try:
    import pynvml
    _HAVE_NVML = True
except Exception:  # pragma: no cover - import guard for login/dev boxes
    pynvml = None
    _HAVE_NVML = False


class NvmlPowerLogger:
    def __init__(self, gpu_id: int = 0, out_csv: str = "power_trace.csv",
                 interval_ms: int = 10):
        if not _HAVE_NVML:
            raise ImportError(
                "pynvml (nvidia-ml-py) is not importable. Install with "
                "`pip install nvidia-ml-py` inside the GPU container."
            )
        self.gpu_id = gpu_id
        self.out_csv = out_csv
        self.interval_s = interval_ms / 1000.0

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._handle = None
        self._file = None
        self._writer = None
        self._powers_w: List[float] = []
        self._n_rows = 0

        # populated at start()
        self.gpu_name: Optional[str] = None
        self.power_limit_w: Optional[float] = None
        self.t_start: Optional[float] = None
        self.t_stop: Optional[float] = None

    # -- small helpers that never raise; a missing metric becomes None --
    @staticmethod
    def _try(fn, scale=1.0):
        try:
            v = fn()
            return float(v) * scale if v is not None else None
        except Exception:
            return None

    def _read_row(self):
        h = self._handle
        power_w = self._try(lambda: pynvml.nvmlDeviceGetPowerUsage(h), 1e-3)  # mW -> W
        sm_mhz = self._try(lambda: pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))
        mem_mhz = self._try(lambda: pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM))
        temp_c = self._try(lambda: pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))

        gpu_util = mem_util = None
        try:
            u = pynvml.nvmlDeviceGetUtilizationRates(h)
            gpu_util, mem_util = float(u.gpu), float(u.memory)
        except Exception:
            pass

        return power_w, sm_mhz, mem_mhz, gpu_util, mem_util, temp_c

    def _loop(self):
        next_t = time.perf_counter()
        while not self._stop.is_set():
            t_wall = time.time()
            power_w, sm_mhz, mem_mhz, gpu_util, mem_util, temp_c = self._read_row()

            self._writer.writerow([
                f"{t_wall:.6f}",
                "" if power_w is None else f"{power_w:.3f}",
                "" if sm_mhz is None else int(sm_mhz),
                "" if mem_mhz is None else int(mem_mhz),
                "" if gpu_util is None else f"{gpu_util:.1f}",
                "" if mem_util is None else f"{mem_util:.1f}",
                "" if temp_c is None else f"{temp_c:.1f}",
            ])
            self._n_rows += 1
            if power_w is not None:
                self._powers_w.append(power_w)

            # Fixed-rate schedule that tolerates slow reads without drifting.
            next_t += self.interval_s
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                self._stop.wait(sleep)
            else:
                next_t = time.perf_counter()

    def start(self):
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)

        name = self._try_str(lambda: pynvml.nvmlDeviceGetName(self._handle))
        self.gpu_name = name
        # Enforced limit is what actually caps draw; fall back to management limit.
        lim_mw = self._try(lambda: pynvml.nvmlDeviceGetEnforcedPowerLimit(self._handle))
        if lim_mw is None:
            lim_mw = self._try(lambda: pynvml.nvmlDeviceGetPowerManagementLimit(self._handle))
        self.power_limit_w = None if lim_mw is None else lim_mw * 1e-3

        import os
        os.makedirs(os.path.dirname(self.out_csv) or ".", exist_ok=True)
        self._file = open(self.out_csv, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(
            ["t_wall", "power_w", "sm_mhz", "mem_mhz", "gpu_util", "mem_util", "temp_c"]
        )

        self._stop.clear()
        self.t_start = time.time()
        self._thread = threading.Thread(target=self._loop, name="nvml-logger", daemon=True)
        self._thread.start()
        return self

    @staticmethod
    def _try_str(fn):
        try:
            v = fn()
            return v.decode() if isinstance(v, bytes) else str(v)
        except Exception:
            return None

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.t_stop = time.time()
        if self._file is not None:
            self._file.flush()
            self._file.close()
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        return self

    def summary(self) -> dict:
        ps = self._powers_w
        ps_sorted = sorted(ps)
        n = len(ps_sorted)
        median = ps_sorted[n // 2] if n else None
        return {
            "gpu_name": self.gpu_name,
            "power_limit_w": self.power_limit_w,
            "n_samples": self._n_rows,
            "duration_s": (self.t_stop - self.t_start) if (self.t_start and self.t_stop) else None,
            "mean_power_w": (sum(ps) / n) if n else None,
            "median_power_w": median,
            "max_power_w": max(ps) if ps else None,
            "min_power_w": min(ps) if ps else None,
        }
