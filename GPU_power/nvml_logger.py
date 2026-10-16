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
  + trailing (additive; readers use column names, so old code is unaffected):
    power_inst_w  NVML field 186 POWER_INSTANT (B200: 0.10 s window, no lag)
    energy_mj     nvmlDeviceGetTotalEnergyConsumption, cumulative mJ (updates
                  ~every 100 ms on B200). Differencing it gives TRUE energy over a
                  window; power_w (GetPowerUsage) is a 0.85 s moving average on
                  B200 and smears 200 ms bins (FINDINGS_B200 UPDATE 4).
    clk_reasons   OR of nvmlDeviceGetCurrentClocksEventReasons bitmasks
                  (0x4 = SW power cap, 0x8 = HW slowdown, 0x80 = HW power brake):
                  marks samples where the GPU throttled to hold its power limit.

NVML_LOGGER_SLIM=1 (env): read only power_w, power_inst_w and energy_mj (no clocks,
util, temperature or throttle reasons) - to test whether the logger's NVML calls slow
the client process (B200 relog runs were ~20% slower than September's).

Multi-GPU (tensor parallel): pass gpu_id as a list; power_w, power_inst_w and
energy_mj are SUMMED across the GPUs (whole-job energy), clk_reasons is OR'ed,
and clocks / util / temp come from the first GPU.

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
    def __init__(self, gpu_id=0, out_csv: str = "power_trace.csv",
                 interval_ms: int = 10):
        if not _HAVE_NVML:
            raise ImportError(
                "pynvml (nvidia-ml-py) is not importable. Install with "
                "`pip install nvidia-ml-py` inside the GPU container."
            )
        self.gpu_ids = list(gpu_id) if isinstance(gpu_id, (list, tuple)) else [int(gpu_id)]
        self.gpu_id = self.gpu_ids[0]
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
        self.power_limit_w: Optional[float] = None        # per GPU (first GPU)
        self.power_limit_total_w: Optional[float] = None  # summed over metered GPUs
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

    @staticmethod
    def _clk_reasons(h):
        for fn in ("nvmlDeviceGetCurrentClocksEventReasons",
                   "nvmlDeviceGetCurrentClocksThrottleReasons"):
            f = getattr(pynvml, fn, None)
            if f is not None:
                try:
                    return int(f(h))
                except Exception:
                    pass
        return None

    def _sum(self, fn, scale=1.0):
        """Sum a per-GPU reading over all metered GPUs (None if any is missing)."""
        tot = 0.0
        for h in self._handles:
            v = self._try(lambda: fn(h), scale)
            if v is None:
                return None
            tot += v
        return tot

    def _read_row(self):
        h = self._handle
        slim = getattr(self, "_slim", False)
        power_w = self._sum(pynvml.nvmlDeviceGetPowerUsage, 1e-3)  # mW -> W
        sm_mhz = mem_mhz = temp_c = None
        if not slim:
            sm_mhz = self._try(lambda: pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))
            mem_mhz = self._try(lambda: pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM))
            temp_c = self._try(lambda: pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))

        # Additive (A5000 agent, 2026-09-24): NVML field 186 POWER_INSTANT. On
        # Ampere-non-GA100 and newer, nvmlDeviceGetPowerUsage is a ~1 s moving
        # average, which smears 200 ms bins; the instant field is not. Kept as an
        # extra trailing column so existing readers of power_w are unaffected.
        p_inst = None
        if getattr(self, "_have_inst", False):
            tot = 0.0
            try:
                for hh in self._handles:
                    v = pynvml.nvmlDeviceGetFieldValues(hh, [186])[0]
                    if v.nvmlReturn != 0:
                        tot = None
                        break
                    tot += v.value.uiVal * 1e-3
                p_inst = tot
            except Exception:
                p_inst = None
        self._last_inst = p_inst

        # Additive (2026-10-16): cumulative energy counter (mJ) and throttle reasons.
        self._last_energy = (self._sum(pynvml.nvmlDeviceGetTotalEnergyConsumption)
                             if getattr(self, "_have_energy", False) else None)
        reasons = None
        for hh in ([] if slim else self._handles):
            r = self._clk_reasons(hh)
            if r is not None:
                reasons = (reasons or 0) | r
        self._last_reasons = reasons

        gpu_util = mem_util = None
        if not slim:
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
                "" if self._last_inst is None else f"{self._last_inst:.3f}",
                "" if self._last_energy is None else f"{self._last_energy:.0f}",
                "" if self._last_reasons is None else self._last_reasons,
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
        self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in self.gpu_ids]
        self._handle = self._handles[0]

        name = self._try_str(lambda: pynvml.nvmlDeviceGetName(self._handle))
        self.gpu_name = name
        # Enforced limit is what actually caps draw; fall back to management limit.
        lim_mw = self._try(lambda: pynvml.nvmlDeviceGetEnforcedPowerLimit(self._handle))
        if lim_mw is None:
            lim_mw = self._try(lambda: pynvml.nvmlDeviceGetPowerManagementLimit(self._handle))
        self.power_limit_w = None if lim_mw is None else lim_mw * 1e-3
        tot = self._sum(pynvml.nvmlDeviceGetEnforcedPowerLimit, 1e-3)
        self.power_limit_total_w = tot if tot is not None else (
            None if self.power_limit_w is None else self.power_limit_w * len(self._handles))

        self._have_inst = False
        self._last_inst = None
        self._last_energy = None
        self._last_reasons = None
        import os as _os
        self._slim = _os.environ.get("NVML_LOGGER_SLIM") == "1"
        self._have_energy = self._sum(pynvml.nvmlDeviceGetTotalEnergyConsumption) is not None
        try:
            v = pynvml.nvmlDeviceGetFieldValues(self._handle, [186])[0]
            self._have_inst = (v.nvmlReturn == 0 and v.value.uiVal > 0)
        except Exception:
            self._have_inst = False

        import os
        os.makedirs(os.path.dirname(self.out_csv) or ".", exist_ok=True)
        self._file = open(self.out_csv, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(
            ["t_wall", "power_w", "sm_mhz", "mem_mhz", "gpu_util", "mem_util", "temp_c",
             "power_inst_w", "energy_mj", "clk_reasons"]
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
            "metered_gpus": self.gpu_ids,
            "power_limit_total_w": self.power_limit_total_w,
            "have_power_instant": bool(getattr(self, "_have_inst", False)),
            "have_energy_counter": bool(getattr(self, "_have_energy", False)),
            "n_samples": self._n_rows,
            "duration_s": (self.t_stop - self.t_start) if (self.t_start and self.t_stop) else None,
            "mean_power_w": (sum(ps) / n) if n else None,
            "median_power_w": median,
            "max_power_w": max(ps) if ps else None,
            "min_power_w": min(ps) if ps else None,
        }
