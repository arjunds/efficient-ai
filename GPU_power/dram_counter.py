#!/usr/bin/env python3
"""
dram_counter.py

Live DRAM-traffic sampler for the energy window. Fixes the #1 problem with the
previous data (a byte counter that captured <3% of real traffic).

The previous NCU byte metric (smsp__sass_data_bytes_mem_global.sum) required
kernel replay, which cannot run under realistic concurrent load without
destroying the timing/power waveform we need. Instead we sample DCGM's
`DCGM_FI_PROF_DRAM_ACTIVE` (field 1005) live during the run.

DRAM_ACTIVE is the *fraction* of cycles the HBM interface was active, not bytes.
We convert per sample:

    inst_bytes_per_s = dram_active_frac * peak_hbm_bytes_per_s
    bytes_in_window  = integral of inst_bytes_per_s over [t0, t1]

`peak_hbm_bytes_per_s` is device-specific (PEAK_HBM_BW below). The concurrency-1
decode gate (gate_dram.py) independently checks that the resulting bytes/token
matches the model's weight+KV bytes, which validates both the counter and this
peak-bandwidth constant in one shot.

Backends, tried in order:
  1. `dcgmi dmon -e 1005` subprocess (no python bindings needed).
  2. pydcgm bindings, if importable.

Emits dram_trace.csv: t_wall, dram_active_frac, inst_bytes_per_s
"""

import csv
import os
import re
import shutil
import subprocess
import threading
import time
from typing import List, Optional, Tuple


# Peak HBM bandwidth (bytes/s) by NVML device name substring. Extend as needed.
PEAK_HBM_BW = {
    "A100-SXM4-80GB": 2039e9,
    "A100-SXM-80GB": 2039e9,
    "A100 80GB PCIe": 1935e9,
    "A100-PCIE-40GB": 1555e9,
    "A100-SXM4-40GB": 1555e9,
    "A100": 1555e9,            # conservative fallback for unlabeled A100
    "H100": 3350e9,
    "H200": 4800e9,
    "L40S": 864e9,
    "L40": 864e9,
    "A5000": 768e9,
    "B200": 8000e9,
}

DRAM_ACTIVE_FIELD = 1005  # DCGM_FI_PROF_DRAM_ACTIVE


def peak_hbm_bw_for(gpu_name: Optional[str]) -> Optional[float]:
    if not gpu_name:
        return None
    # Longest key match wins (so "A100-SXM4-80GB" beats bare "A100").
    best = None
    for key, bw in PEAK_HBM_BW.items():
        if key in gpu_name and (best is None or len(key) > len(best[0])):
            best = (key, bw)
    return best[1] if best else None


class DramCounter:
    def __init__(self, gpu_id: int = 0, out_csv: str = "dram_trace.csv",
                 interval_ms: int = 100, peak_bw_bytes_per_s: Optional[float] = None,
                 gpu_name: Optional[str] = None):
        self.gpu_id = gpu_id
        self.out_csv = out_csv
        self.interval_ms = interval_ms
        self.peak_bw = peak_bw_bytes_per_s or peak_hbm_bw_for(gpu_name)
        self.gpu_name = gpu_name

        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._file = None
        self._writer = None
        self._samples: List[Tuple[float, float]] = []  # (t_wall, inst_bytes_per_s)
        self.backend: Optional[str] = None
        self.error: Optional[str] = None

    # ---------------- lifecycle ----------------
    def start(self):
        os.makedirs(os.path.dirname(self.out_csv) or ".", exist_ok=True)
        self._file = open(self.out_csv, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["t_wall", "dram_active_frac", "inst_bytes_per_s"])

        if shutil.which(os.environ.get("DCGMI_BIN", "dcgmi")):
            self.backend = "dcgmi"
            self._start_dcgmi()
        else:
            self.error = "dcgmi not found on PATH; DRAM counter disabled"
        return self

    def _start_dcgmi(self):
        # Additive (A5000 agent, 2026-09-24): DCGM enumerates HOST GPUs, so under
        # SLURM/containers the CUDA index (0) is usually NOT the DCGM id. Set
        # DCGM_GPU_ID (map via `dcgmi discovery -l` vs nvidia-smi pci.bus_id);
        # DCGMI_BIN overrides the binary path. Defaults preserve old behaviour.
        gid = os.environ.get("DCGM_GPU_ID", str(self.gpu_id))
        cmd = [os.environ.get("DCGMI_BIN", "dcgmi"), "dmon", "-e", str(DRAM_ACTIVE_FIELD),
               "-d", str(self.interval_ms), "-i", gid]
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except Exception as e:  # pragma: no cover
            self.error = f"failed to launch dcgmi dmon: {e}"
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._read_dcgmi, daemon=True)
        self._thread.start()

    def _read_dcgmi(self):
        assert self._proc is not None and self._proc.stdout is not None
        num_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Expected: "GPU <id>  <value>"; be liberal about extra columns.
            if not line.upper().startswith("GPU"):
                continue
            nums = num_re.findall(line)
            if len(nums) < 2:
                continue
            # nums[0] is the GPU id; the DRAM_ACTIVE ratio is the last float.
            try:
                frac = float(nums[-1])
            except ValueError:
                continue
            frac = max(0.0, min(1.0, frac))
            t_wall = time.time()
            inst_bps = frac * self.peak_bw if self.peak_bw else float("nan")
            self._samples.append((t_wall, inst_bps))
            self._writer.writerow([f"{t_wall:.6f}", f"{frac:.6f}",
                                   "" if self.peak_bw is None else f"{inst_bps:.3e}"])

    def stop(self):
        self._stop.set()
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._file is not None:
            self._file.flush()
            self._file.close()
        return self

    # ---------------- integration ----------------
    def bytes_in_window(self, t0: float, t1: float) -> Optional[float]:
        """Trapezoidal integral of inst_bytes_per_s over [t0, t1]."""
        if self.peak_bw is None:
            return None
        pts = [(t, b) for (t, b) in self._samples if t0 <= t <= t1]
        if len(pts) < 2:
            return None
        total = 0.0
        for i in range(len(pts) - 1):
            (ta, ba), (tb, bb) = pts[i], pts[i + 1]
            dt = tb - ta
            if dt > 0:
                total += 0.5 * (ba + bb) * dt
        return total

    def total_bytes(self) -> Optional[float]:
        if not self._samples:
            return None
        return self.bytes_in_window(self._samples[0][0], self._samples[-1][0])
