#!/usr/bin/env python3
"""
energy_microbench.py -- portable GPU energy microbenchmarks (any NVIDIA GPU, Volta+).

Purpose: measure energy per byte at each level of the memory hierarchy (L1, L2,
DRAM) and energy per FLOP for tensor-core GEMM, with the same idle-subtraction
convention the serving pipeline uses, so the serving "e_wbyte" (lumped J/byte of
analytic weight bytes) can be compared against physically distinct traffic.

Power/energy measurement (background thread, ~10 ms):
  * NVML total-energy counter (nvmlDeviceGetTotalEnergyConsumption, mJ) -- PRIMARY.
    Hardware-integrated; on GA102 it updates every ~100 ms. Window energy is taken
    between the first and last counter *update events* detected inside the window,
    so quantization is ~ +-10 ms / window (0.1% for 10 s).
  * NVML field 186 (POWER_INSTANT) if the driver supports it -- cross-check.
  * nvmlDeviceGetPowerUsage -- on Ampere-non-GA100 and newer this is a 1 s moving
    average (NVML docs); reported for completeness only.
  * SM/mem clocks, temperature, clock-event (throttle) reasons -> power-cap flag.

Idle: measured with the CUDA context alive and no work, before and after the
suite ("idle_pre", "idle_post"); dynamic = measured - idle_pre.
Also reports, for DRAM streaming, a grid-size sweep (fraction of SMs issuing
loads) fitted as P = P0 + e_marg * BW -- the marginal J/byte, independent of the
idle-subtraction convention (P0 = "clocks-up static" floor).

Tests
  stream   Triton read-only reduction, working set 64 KB .. 4 GB; ".ca" loads
           (L1-cacheable) for tiny sets, ".cg" (L2-only) otherwise.
           bytes = bytes delivered to the SMs (working_set * passes).
  grid     DRAM-resident (1 GB) stream at varying #programs -> P vs BW slope.
  vendor   torch.sum (DRAM read) and tensor copy (DRAM read+write) cross-checks.
  gemv     fp16 F.linear(x[b,K], W[N,K]) with 512 MB weight, b in {1,16}.
           bytes = weight bytes (+x,y negligible)  -- decode-like.
  gemm     fp16 square matmul n in {1024,2048,4096,8192}; flops = 2 n^3.
  square   0.5 s on / 0.5 s off DRAM stream -> which NVML signal tracks power.

Usage (inside the vLLM container, 1 GPU):
  python3 microbench/energy_microbench.py --out microbench/results_A5000.json
  python3 microbench/energy_microbench.py --tests stream,gemm --steady_s 10 --repeats 3
"""

import argparse
import json
import math
import os
import statistics as st
import sys
import threading
import time

import pynvml
import torch

try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except Exception:  # pragma: no cover
    _HAVE_TRITON = False

import random as _random
_rng = _random.Random(7)
FI_POWER_INSTANT = getattr(pynvml, "NVML_FI_DEV_POWER_INSTANT", 186)
BIT_SW_POWER_CAP = 0x4
BIT_HW_SLOWDOWN = 0x8
BIT_SW_THERMAL = 0x20
BIT_HW_THERMAL = 0x40
BIT_HW_POWER_BRAKE = 0x80


# ---------------------------------------------------------------- sampler
class Sampler:
    """Background NVML sampler. Rows: (t, energy_mJ, p_inst_W, p_avg_W, sm, mem,
    temp, reasons)."""

    def __init__(self, handle, interval_s=0.01):
        self.h = handle
        self.dt = interval_s
        self.rows = []
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.have_inst = self._probe_inst()
        self.have_energy = self._probe_energy()

    def _probe_inst(self):
        try:
            v = pynvml.nvmlDeviceGetFieldValues(self.h, [FI_POWER_INSTANT])[0]
            return v.nvmlReturn == 0 and v.value.uiVal > 0
        except Exception:
            return False

    def _probe_energy(self):
        try:
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self.h)
            return True
        except Exception:
            return False

    def _read(self):
        h = self.h
        t = time.time()
        e = p_inst = p_avg = sm = mem = temp = reasons = None
        if self.have_energy:
            try:
                e = float(pynvml.nvmlDeviceGetTotalEnergyConsumption(h))
            except Exception:
                pass
        if self.have_inst:
            try:
                v = pynvml.nvmlDeviceGetFieldValues(h, [FI_POWER_INSTANT])[0]
                if v.nvmlReturn == 0:
                    p_inst = v.value.uiVal * 1e-3
            except Exception:
                pass
        try:
            p_avg = pynvml.nvmlDeviceGetPowerUsage(h) * 1e-3
        except Exception:
            pass
        try:
            sm = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
            mem = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM)
        except Exception:
            pass
        try:
            temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            pass
        try:
            reasons = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(h)
        except Exception:
            pass
        return (t, e, p_inst, p_avg, sm, mem, temp, reasons)

    def _loop(self):
        nxt = time.perf_counter()
        while not self._stop.is_set():
            r = self._read()
            with self._lock:
                self.rows.append(r)
            nxt += self.dt
            s = nxt - time.perf_counter()
            if s > 0:
                self._stop.wait(s)
            else:
                nxt = time.perf_counter()

    def start(self):
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        return self

    def stop(self):
        self._stop.set()
        self._th.join(timeout=5)

    def window(self, t0, t1):
        with self._lock:
            return [r for r in self.rows if t0 <= r[0] <= t1]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def window_stats(rows, avg_skip_s=1.0):
    """Power over a window from the three NVML signals + clocks/cap flags."""
    out = {"n_samples": len(rows)}
    if not rows:
        return out
    t0w, t1w = rows[0][0], rows[-1][0]
    # energy counter: use update events (value changes) to avoid 100 ms quantization
    ev = []
    prev = None
    for r in rows:
        if r[1] is None:
            continue
        if prev is not None and r[1] != prev:
            ev.append((r[0], r[1]))
        prev = r[1]
    if len(ev) >= 2 and ev[-1][0] > ev[0][0]:
        out["p_energy_w"] = (ev[-1][1] - ev[0][1]) * 1e-3 / (ev[-1][0] - ev[0][0])
        out["energy_counter_updates"] = len(ev)
        out["energy_counter_period_s"] = (ev[-1][0] - ev[0][0]) / max(1, len(ev) - 1)
    out["p_inst_w"] = _mean([r[2] for r in rows])
    out["p_avg_w"] = _mean([r[3] for r in rows if r[0] >= t0w + avg_skip_s])
    out["sm_mhz"] = _mean([r[4] for r in rows])
    out["mem_mhz"] = _mean([r[5] for r in rows])
    out["temp_c"] = _mean([r[6] for r in rows])
    rs = [r[7] for r in rows if r[7] is not None]
    if rs:
        out["frac_sw_power_cap"] = sum(1 for x in rs if x & BIT_SW_POWER_CAP) / len(rs)
        out["frac_thermal"] = sum(1 for x in rs if x & (BIT_SW_THERMAL | BIT_HW_THERMAL)) / len(rs)
        out["frac_hw_slowdown"] = sum(1 for x in rs if x & (BIT_HW_SLOWDOWN | BIT_HW_POWER_BRAKE)) / len(rs)
    # PRIMARY = POWER_INSTANT field if present, else the energy counter, else the
    # 1 s-averaged reading. (On RTX A5000 / driver 565 the energy counter was found
    # to under-read the board power by 4-20x -- see FINDINGS_A5000.md -- so it is
    # never trusted without the consistency ratio below.)
    out["p_w"] = out.get("p_inst_w") or out.get("p_energy_w") or out.get("p_avg_w")
    if out.get("p_energy_w") and out.get("p_inst_w"):
        out["energy_counter_over_inst"] = out["p_energy_w"] / out["p_inst_w"]
    return out


# ---------------------------------------------------------------- kernels
if _HAVE_TRITON:
    # do_not_specialize: otherwise Triton recompiles whenever `passes` changes
    # divisibility class (e.g. 1 vs 16), and the compile lands inside a timed burst.
    @triton.jit(do_not_specialize=["n_chunks", "passes"])
    def _stream_kernel(x_ptr, out_ptr, n_chunks, passes,
                       BLOCK: tl.constexpr, L1: tl.constexpr):
        pid = tl.program_id(0)
        nprog = tl.num_programs(0)
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for p in range(passes):
            for c in range(pid, n_chunks, nprog):
                offs = c * BLOCK + tl.arange(0, BLOCK)
                if L1:
                    x = tl.load(x_ptr + offs, cache_modifier=".ca")
                else:
                    x = tl.load(x_ptr + offs, cache_modifier=".cg")
                acc += x
        tl.store(out_ptr + pid * BLOCK + tl.arange(0, BLOCK), acc)


STREAM_BLOCK = 2048          # fp32 elements per program-iteration (8 KB)


class StreamOp:
    """Read-only reduction over a working set, `passes` times per call."""

    def __init__(self, ws_bytes, l1=False, nprog=None, sm_count=None):
        n = max(STREAM_BLOCK, (ws_bytes // 4) // STREAM_BLOCK * STREAM_BLOCK)
        self.x = torch.rand(n, device="cuda", dtype=torch.float32)
        self.n_chunks = n // STREAM_BLOCK
        self.ws = n * 4
        sm_count = sm_count or torch.cuda.get_device_properties(0).multi_processor_count
        self.nprog = nprog or min(self.n_chunks, sm_count * 8)
        self.out = torch.empty(self.nprog * STREAM_BLOCK, device="cuda", dtype=torch.float32)
        self.l1 = bool(l1)
        self.passes = 1

    def __call__(self):
        _stream_kernel[(self.nprog,)](self.x, self.out, self.n_chunks, self.passes,
                                      BLOCK=STREAM_BLOCK, L1=self.l1, num_warps=8)

    @property
    def bytes_per_call(self):
        return float(self.ws) * self.passes

    flops_per_call = 0.0

    def tune(self, target_s=0.03):
        self.passes = 1
        self(); self(); torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(10):
            self()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t) / 10
        self.passes = max(1, int(round(target_s / max(dt, 1e-7))))


class TorchOp:
    def __init__(self, fn, bytes_per_call, flops_per_call):
        self.fn = fn
        self._b = bytes_per_call
        self._f = flops_per_call
        self.reps = 1

    def __call__(self):
        for _ in range(self.reps):
            self.fn()

    @property
    def bytes_per_call(self):
        return self._b * self.reps

    @property
    def flops_per_call(self):
        return self._f * self.reps

    def tune(self, target_s=0.03):
        self.reps = 1
        self(); torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(5):
            self.fn()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t) / 5
        self.reps = max(1, int(round(target_s / max(dt, 1e-7))))


# ---------------------------------------------------------------- runner
class Bench:
    def __init__(self, args):
        self.args = args
        pynvml.nvmlInit()
        # Under SLURM/containers CUDA device 0 == NVML index 0 of the visible set
        # when CUDA_VISIBLE_DEVICES is honoured by the cgroup; match by UUID to be safe.
        self.h = self._match_handle()
        self.sampler = Sampler(self.h, args.sample_ms / 1000.0).start()
        self.idle = None
        self.idle_p0 = None

    def _match_handle(self):
        try:
            uuid = str(torch.cuda.get_device_properties(0).uuid)
            for i in range(pynvml.nvmlDeviceGetCount()):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                u = pynvml.nvmlDeviceGetUUID(h)
                u = u.decode() if isinstance(u, bytes) else u
                if uuid in u or u.replace("GPU-", "") == uuid:
                    return h
        except Exception:
            pass
        return pynvml.nvmlDeviceGetHandleByIndex(0)

    def gpu_info(self):
        h = self.h
        p = torch.cuda.get_device_properties(0)

        def q(fn, scale=1.0):
            try:
                v = fn()
                return v * scale if isinstance(v, (int, float)) else v
            except Exception:
                return None
        name = q(lambda: pynvml.nvmlDeviceGetName(h))
        return {
            "name": name.decode() if isinstance(name, bytes) else name,
            "driver": q(pynvml.nvmlSystemGetDriverVersion),
            "power_limit_enforced_w": q(lambda: pynvml.nvmlDeviceGetEnforcedPowerLimit(h), 1e-3),
            "power_limit_default_w": q(lambda: pynvml.nvmlDeviceGetPowerManagementDefaultLimit(h), 1e-3),
            "power_limit_range_w": q(lambda: [x * 1e-3 for x in pynvml.nvmlDeviceGetPowerManagementLimitConstraints(h)]),
            "max_sm_mhz": q(lambda: pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_SM)),
            "max_mem_mhz": q(lambda: pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_MEM)),
            "mem_bus_width_bits": q(lambda: pynvml.nvmlDeviceGetMemoryBusWidth(h)),
            "sm_count": p.multi_processor_count,
            "l2_bytes": getattr(p, "L2_cache_size", None),
            "total_mem_bytes": p.total_memory,
            "cc": f"{p.major}.{p.minor}",
            "have_power_instant_field": self.sampler.have_inst,
            "have_energy_counter": self.sampler.have_energy,
            "torch": torch.__version__,
            "triton": getattr(triton, "__version__", None) if _HAVE_TRITON else None,
        }

    def measure_idle(self, dur):
        torch.cuda.synchronize()
        time.sleep(1.0)
        t0 = time.time()
        time.sleep(dur)
        t1 = time.time()
        s = window_stats(self.sampler.window(t0, t1))
        s["duration_s"] = t1 - t0
        return s

    def measure_idle_deep(self, dur, max_wait_s=30.0):
        """Wait (no CUDA work) until the SM clock falls to its idle P-state, then
        measure. This is the state the vLLM 'idle' baseline usually lands in."""
        torch.cuda.synchronize()
        t_w = time.time()
        while time.time() - t_w < max_wait_s:
            time.sleep(1.0)
            r = self.sampler.window(time.time() - 1.0, time.time())
            sm = _mean([x[4] for x in r])
            if sm is not None and sm < 600:
                break
        time.sleep(2.0)
        t0 = time.time(); time.sleep(dur); t1 = time.time()
        s = window_stats(self.sampler.window(t0, t1))
        s.update({"duration_s": t1 - t0, "waited_s": t0 - t_w})
        return s

    def measure_idle_p0(self, dur, period_s=0.005):
        """Clocks-up idle: a 1-thread no-op kernel every 5 ms keeps the GPU in its
        compute P-state at full clocks while doing ~no work. This is the static
        floor under which the duty-cycle slopes are taken."""
        a = torch.zeros(1, device="cuda")
        t_end = time.time() + 2.0
        while time.time() < t_end:
            a.add_(0.0); torch.cuda.synchronize(); time.sleep(period_s)
        t0 = time.time()
        while time.time() < t0 + dur:
            a.add_(0.0); torch.cuda.synchronize(); time.sleep(period_s)
        t1 = time.time()
        s = window_stats(self.sampler.window(t0, t1))
        s["duration_s"] = t1 - t0
        return s

    def run_duty(self, op, duty, steady_s, burst_s=None, settle_s=2.0, jitter=False,
                 period_s=None):
        """Duty-cycled load: one burst (~burst_s of kernel time) every
        burst_s/duty seconds, host-sleeping in between. Keeps average power below a
        low power cap so the clocks stay at their uncapped maximum; the slope of
        mean power vs mean work-rate is then the marginal energy per unit work at
        full clocks. duty=0 -> the keepalive no-op (clocks-up floor)."""
        op.tune(burst_s or self.args.burst_s)
        # measure the burst's own duration / rate (warm call first: no compile inside)
        op(); torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(10):
            op()
        torch.cuda.synchronize()
        t_burst = (time.perf_counter() - t) / 10
        keep = torch.zeros(1, device="cuda")
        if period_s:                         # fixed period: burst length = duty * period
            period = period_s
        else:
            period = max(t_burst / duty, 0.002) if duty > 0 else 0.005

        def cycle():
            ts = time.perf_counter()
            if duty > 0:
                op()
            else:
                keep.add_(0.0)
            torch.cuda.synchronize()
            t_op = time.perf_counter() - ts
            rem = period - t_op
            if jitter:                       # break phase-lock with the power sensor
                rem *= 2.0 * _rng.random()   # mean preserved
            if rem > 0:
                time.sleep(rem)
            return t_op
        t_end = time.time() + settle_s
        while time.time() < t_end:
            cycle()
        t0 = time.time(); nb = nf = 0.0; nburst = 0; t_on = 0.0
        while time.time() < t0 + steady_s:
            t_op = cycle()
            if duty > 0:
                nb += op.bytes_per_call; nf += op.flops_per_call; nburst += 1
                t_on += t_op
        t1 = time.time()
        if nburst:
            t_burst = t_on / nburst          # in-window burst time (detects DVFS in bursts)
        s = window_stats(self.sampler.window(t0, t1))
        dur = t1 - t0
        s.update({"duty_target": duty, "period_s": period, "t_burst_s": t_burst,
                  "duty_actual": nburst * t_burst / dur, "duration_s": dur,
                  "bytes_per_s": nb / dur, "flops_per_s": nf / dur,
                  "burst_bytes_per_s": op.bytes_per_call / t_burst,
                  "burst_flops_per_s": op.flops_per_call / t_burst})
        return s

    def run_point(self, op, settle_s, steady_s):
        """Warm up + settle, then a steady window. Returns window stats + rates."""
        op.tune(self.args.chunk_s)
        t_end = time.time() + settle_s
        while time.time() < t_end:
            op()
            torch.cuda.synchronize()
        torch.cuda.synchronize()
        t0 = time.time()
        nb = nf = 0.0
        t_end = t0 + steady_s
        while True:
            op()
            torch.cuda.synchronize()
            nb += op.bytes_per_call
            nf += op.flops_per_call
            if time.time() >= t_end:
                break
        t1 = time.time()
        s = window_stats(self.sampler.window(t0, t1))
        dur = t1 - t0
        s.update({"duration_s": dur, "bytes_per_s": nb / dur, "flops_per_s": nf / dur,
                  "t0": t0, "t1": t1})
        return s

    def repeat(self, label, make_op, meta, repeats=None, settle_s=None, steady_s=None):
        repeats = repeats or self.args.repeats
        settle_s = self.args.settle_s if settle_s is None else settle_s
        steady_s = self.args.steady_s if steady_s is None else steady_s
        op = make_op()
        reps = []
        for k in range(repeats):
            s = self.run_point(op, settle_s, steady_s)
            p_idle = self.idle["p_w"]
            dyn = s["p_w"] - p_idle
            s["p_dyn_w"] = dyn
            if self.idle_p0:
                s["p_dyn_vs_p0_w"] = s["p_w"] - self.idle_p0["p_w"]
            if s["bytes_per_s"] > 0:
                s["j_per_byte_dyn"] = dyn / s["bytes_per_s"]
                if self.idle_p0:
                    s["j_per_byte_dyn_vs_p0"] = s["p_dyn_vs_p0_w"] / s["bytes_per_s"]
                s["j_per_byte_gross"] = s["p_w"] / s["bytes_per_s"]
            if s["flops_per_s"] > 0:
                s["j_per_flop_dyn"] = dyn / s["flops_per_s"]
                if self.idle_p0:
                    s["j_per_flop_dyn_vs_p0"] = s["p_dyn_vs_p0_w"] / s["flops_per_s"]
                s["j_per_flop_gross"] = s["p_w"] / s["flops_per_s"]
            reps.append(s)
            print(f"  [{label} r{k}] P={s['p_w']:.1f} W (inst {s.get('p_inst_w') or float('nan'):.1f}, "
                  f"avg {s.get('p_avg_w') or float('nan'):.1f})  dyn={dyn:.1f} W  "
                  f"BW={s['bytes_per_s']/1e9:.1f} GB/s  TF={s['flops_per_s']/1e12:.2f}  "
                  f"sm={s.get('sm_mhz') or 0:.0f} mem={s.get('mem_mhz') or 0:.0f} MHz  "
                  f"T={s.get('temp_c') or 0:.0f}C  cap={s.get('frac_sw_power_cap', 0):.2f}"
                  + (f"  J/B={s['j_per_byte_dyn']:.3e}" if 'j_per_byte_dyn' in s else "")
                  + (f"  pJ/flop={s['j_per_flop_dyn']*1e12:.3f}" if 'j_per_flop_dyn' in s else ""),
                  flush=True)
        del op
        torch.cuda.empty_cache()
        summ = {"label": label, **meta, "repeats": reps}
        for key in ("j_per_byte_dyn", "j_per_flop_dyn", "j_per_byte_dyn_vs_p0",
                    "j_per_flop_dyn_vs_p0", "j_per_byte_gross", "j_per_flop_gross", "p_w", "p_dyn_w", "bytes_per_s",
                    "flops_per_s", "sm_mhz", "mem_mhz", "temp_c", "frac_sw_power_cap"):
            vals = [r[key] for r in reps if r.get(key) is not None]
            if vals:
                summ[key + "_mean"] = sum(vals) / len(vals)
                summ[key + "_std"] = st.pstdev(vals) if len(vals) > 1 else 0.0
        return summ


# ---------------------------------------------------------------- tests
def test_stream(b, results):
    sm = b.info["sm_count"]
    # L1-cacheable (.ca): nprog = 8/SM, each program owns 1 (or 2) 8 KB chunks
    l1_sizes = [sm * 8 * STREAM_BLOCK * 4, sm * 16 * STREAM_BLOCK * 4]
    l2 = b.info.get("l2_bytes") or 6 * 2**20
    cg_sizes = [256 * 2**10, 1 * 2**20, 2 * 2**20, int(l2 * 0.5), int(l2 * 0.75),
                l2, 2 * l2, 4 * l2, 64 * 2**20, 256 * 2**20, 1 * 2**30, 4 * 2**30]
    free = torch.cuda.mem_get_info()[0]
    out = []
    for ws in l1_sizes:
        out.append(b.repeat(f"stream_L1ca_{ws/2**20:.1f}MB", lambda: StreamOp(ws, l1=True),
                            {"test": "stream", "cache": ".ca", "ws_bytes": ws,
                             "per_sm_footprint_bytes": ws // sm}))
    for ws in sorted(set(cg_sizes)):
        if ws > 0.8 * free:
            continue
        out.append(b.repeat(f"stream_cg_{ws/2**20:.2f}MB", lambda: StreamOp(ws, l1=False),
                            {"test": "stream", "cache": ".cg", "ws_bytes": ws}))
    results["stream"] = out


def test_grid(b, results):
    sm = b.info["sm_count"]
    ws = 1 * 2**30
    out = []
    for f in [0.125, 0.25, 0.5, 1, 2, 4, 8]:
        nprog = max(1, int(round(sm * f)))
        out.append(b.repeat(f"grid_{nprog}prog", lambda: StreamOp(ws, nprog=nprog),
                            {"test": "grid", "nprog": nprog, "ws_bytes": ws}, repeats=1))
    # linear fit P = P0 + e*BW over points not flagged power-capped
    pts = [(r["bytes_per_s_mean"], r["p_w_mean"]) for r in out
           if r.get("frac_sw_power_cap_mean", 0) < 0.05]
    fit = None
    if len(pts) >= 3:
        xs, ys = zip(*pts)
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        e = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
        p0 = my - e * mx
        res = [y - (p0 + e * x) for x, y in zip(xs, ys)]
        se = math.sqrt(sum(r * r for r in res) / max(1, len(xs) - 2) / sxx)
        fit = {"e_marginal_j_per_byte": e, "e_se": se, "p0_w": p0, "n_points": len(pts),
               "bw_range": [min(xs), max(xs)]}
    results["grid"] = {"points": out, "fit": fit}


def test_vendor(b, results):
    n = 256 * 2**20                     # 256M fp32 = 1 GB
    out = []

    def mk_sum():
        x = torch.rand(n, device="cuda")
        return TorchOp(lambda: x.sum(), 4.0 * n, 0.0)
    out.append(b.repeat("torch_sum_1GB", mk_sum, {"test": "vendor", "kind": "read"}))

    def mk_copy():
        a = torch.rand(n // 2, device="cuda"); c = torch.empty_like(a)
        return TorchOp(lambda: c.copy_(a), 2 * 4.0 * (n // 2), 0.0)
    out.append(b.repeat("torch_copy_512MB", mk_copy,
                        {"test": "vendor", "kind": "read+write", "note": "bytes = read + write"}))
    results["vendor"] = out


def test_gemv(b, results):
    N = K = 16384                       # 512 MB fp16 weight
    out = []
    for bs in (1, 16):
        def mk(bs=bs):
            W = torch.randn(N, K, device="cuda", dtype=torch.float16) * 0.01
            x = torch.randn(bs, K, device="cuda", dtype=torch.float16)
            fn = lambda: torch.nn.functional.linear(x, W)
            return TorchOp(fn, 2.0 * N * K + 2.0 * bs * (N + K), 2.0 * bs * N * K)
        out.append(b.repeat(f"gemv_b{bs}_512MB", mk,
                            {"test": "gemv", "batch": bs, "weight_bytes": 2 * N * K}))
    results["gemv"] = out


def test_gemm(b, results):
    out = []
    for n in (1024, 2048, 4096, 8192):
        def mk(n=n):
            A = torch.randn(n, n, device="cuda", dtype=torch.float16)
            B = torch.randn(n, n, device="cuda", dtype=torch.float16)
            C = torch.empty(n, n, device="cuda", dtype=torch.float16)
            return TorchOp(lambda: torch.matmul(A, B, out=C), 3 * 2.0 * n * n, 2.0 * n ** 3)
        out.append(b.repeat(f"gemm_fp16_{n}", mk, {"test": "gemm", "n": n}))
    results["gemm"] = out


def _linfit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    e = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - e * mx
    res = [y - (a + e * x) for x, y in zip(xs, ys)]
    s2 = sum(r * r for r in res) / max(1, n - 2)
    se_e = math.sqrt(s2 / sxx)
    se_a = math.sqrt(s2 * (1.0 / n + mx * mx / sxx))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1 - sum(r * r for r in res) / ss_tot if ss_tot > 0 else None
    return a, e, se_a, se_e, r2


def _l2():
    return getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 0) or 6 * 2**20


def _sm():
    return torch.cuda.get_device_properties(0).multi_processor_count


def dram_ws():
    """DRAM-resident working set: >= 256 MB and >= 16x L2 (B200 L2 is ~126 MB)."""
    return max(256 * 2**20, 16 * _l2())


def _mk_gemv(bs=1):
    K = 16384
    wbytes = max(512 * 2**20, 8 * _l2())
    N = int(wbytes // (2 * K)) // 256 * 256
    W = torch.randn(N, K, device="cuda", dtype=torch.float16) * 0.01
    x = torch.randn(bs, K, device="cuda", dtype=torch.float16)
    return TorchOp(lambda: torch.nn.functional.linear(x, W),
                   2.0 * N * K + 2.0 * bs * (N + K), 2.0 * bs * N * K)


def _mk_gemm(n=4096):
    A = torch.randn(n, n, device="cuda", dtype=torch.float16)
    B = torch.randn(n, n, device="cuda", dtype=torch.float16)
    C = torch.empty(n, n, device="cuda", dtype=torch.float16)
    return TorchOp(lambda: torch.matmul(A, B, out=C), 3 * 2.0 * n * n, 2.0 * n ** 3)


# name: (factory, unit)   -- sizes are relative to this GPU's L2 so the same
# names mean the same hierarchy level on A5000 (6 MB L2) and B200 (~126 MB L2).
DUTY_OPS = {
    "dram_stream": (lambda: StreamOp(dram_ws()), "byte"),
    "l2_stream_half": (lambda: StreamOp(_l2() // 2), "byte"),
    # small L2 set: on multi-die parts (B200) mostly near-die L2 vs half-L2 spanning both dies
    "l2_stream_sixteenth": (lambda: StreamOp(max(_l2() // 16, _sm() * 2 * STREAM_BLOCK * 4)), "byte"),
    # L1: every program re-reads its own 8 KB chunk (nprog = 8/SM -> 64 KB/SM < L1)
    "l1_stream_8KBperprog": (lambda: StreamOp(_sm() * 8 * STREAM_BLOCK * 4, l1=True), "byte"),
    "gemv_b1": (_mk_gemv, "byte"),
    "gemm_fp16_4096": (_mk_gemm, "flop"),
}


def test_duty(b, results):
    """Duty-cycle slope method (valid under a low power cap). For each op, sweep
    duty in args.duties; fit P = a + e*rate over points that were NOT power-capped
    and kept near-max SM clock."""
    duties = [float(x) for x in b.args.duties.split(",")]
    out = {}
    only = [x for x in (b.args.duty_ops or "").split(",") if x]
    for name, (fac, unit) in DUTY_OPS.items():
        if only and name not in only:
            continue
        op = fac()
        sweeps = []
        for rep in range(b.args.duty_repeats):
            pts = []
            for d in duties:
                s = b.run_duty(op, d, b.args.duty_steady_s)
                pts.append(s)
                rate = s["bytes_per_s"] if unit == "byte" else s["flops_per_s"]
                print(f"  [duty {name} r{rep} d={d:.3f}] P={s['p_w']:.2f} W  rate={rate:.3e}/s  "
                      f"duty_act={s['duty_actual']:.3f}  burst_rate="
                      f"{(s['burst_bytes_per_s'] if unit=='byte' else s['burst_flops_per_s']):.3e}  "
                      f"sm={s.get('sm_mhz') or 0:.0f} mem={s.get('mem_mhz') or 0:.0f}  "
                      f"cap={s.get('frac_sw_power_cap', 0):.2f}  T={s.get('temp_c') or 0:.0f}", flush=True)
            sweeps.append(pts)
        # fit over clean points (pooled over repeats) and per repeat
        bk = "burst_bytes_per_s" if unit == "byte" else "burst_flops_per_s"
        bmax = max([p[bk] for pts in sweeps for p in pts if p["duty_target"] > 0] or [0])
        cap_w = b.info.get("power_limit_enforced_w") or 1e9

        def clean(p):
            # clean = clocks near max, the burst ran at full speed (no DVFS inside
            # bursts), and mean power safely below the enforced cap. The NVML
            # SW-power-cap flag alone is too eager (it trips on ms-scale bursts
            # that are not actually slowed), so it is recorded but not used.
            full = p["duty_target"] == 0 or p[bk] >= 0.97 * bmax
            return (full and (p.get("sm_mhz") or 0) >= 0.8 * (b.info.get("max_sm_mhz") or 0)
                    and p["p_w"] < 0.95 * cap_w)
        rk = "bytes_per_s" if unit == "byte" else "flops_per_s"
        fits = []
        for pts in sweeps:
            cp = [p for p in pts if clean(p)]
            if len(cp) >= 3:
                a, e, sa, se, r2 = _linfit([p[rk] for p in cp], [p["p_w"] for p in cp])
                fits.append({"a_w": a, "e": e, "se_e": se, "se_a": sa, "r2": r2, "n": len(cp)})
        allc = [p for pts in sweeps for p in pts if clean(p)]
        pooled = None
        if len(allc) >= 3:
            a, e, sa, se, r2 = _linfit([p[rk] for p in allc], [p["p_w"] for p in allc])
            pooled = {"a_w": a, "e": e, "se_e": se, "se_a": sa, "r2": r2, "n": len(allc)}
        out[name] = {"unit": unit, "sweeps": sweeps, "fits_per_repeat": fits, "fit_pooled": pooled,
                     "burst_rate_mean": _mean([p["burst_bytes_per_s" if unit == "byte" else "burst_flops_per_s"]
                                               for pts in sweeps for p in pts if p["duty_target"] > 0])}
        if pooled:
            print(f"  ==> {name}: e = {pooled['e']:.4e} ± {pooled['se_e']:.1e} J/{unit}  "
                  f"floor a = {pooled['a_w']:.2f} W  R2={pooled['r2']:.4f}  "
                  f"per-repeat e = {[round(f['e'], 16) for f in fits]}", flush=True)
        del op
        torch.cuda.empty_cache()
    results["duty"] = out


def test_burstlen(b, results):
    """How fast does the power-cap controller act? DRAM stream at 3% duty with
    bursts of increasing length: if the in-window burst rate drops, bursts that
    long get DVFS-throttled (relevant for duty-cycled serving)."""
    op = StreamOp(dram_ws())
    out = []
    for bl in (0.0015, 0.005, 0.02, 0.05, 0.1, 0.2):
        s = b.run_duty(op, 0.03, max(6.0, 40 * bl / 0.03 * 0 + 6.0), burst_s=bl)
        out.append({"burst_s_target": bl, **{k: s.get(k) for k in (
            "t_burst_s", "burst_bytes_per_s", "p_w", "sm_mhz", "frac_sw_power_cap", "duty_actual")}})
        print(f"  [burstlen {bl*1e3:.1f} ms] t_burst={s['t_burst_s']*1e3:.2f} ms  "
              f"burst BW={s['burst_bytes_per_s']/1e9:.1f} GB/s  P={s['p_w']:.1f} W  "
              f"sm={s.get('sm_mhz') or 0:.0f}  cap={s.get('frac_sw_power_cap', 0):.2f}", flush=True)
    del op
    torch.cuda.empty_cache()
    results["burstlen"] = out


def test_burstscan(b, results):
    """Discriminates per-burst overhead from in-burst DVFS. Fixed mean work rate
    (duty b.args.scan_duty), burst length varied; floor (keepalive) re-measured
    before every point to cancel drift. Per-burst overhead => P_dyn linear in
    bursts/s (smooth 1/burst_len trend, even for sub-ms bursts); in-burst DVFS
    (firmware lowering V/f once a burst outlasts its reaction time) => flat for
    short bursts, drop beyond a threshold, with memory-bound burst BW unchanged."""
    lens = [float(x) * 1e-3 for x in b.args.scan_lens_ms.split(",")]
    out = {}
    for name in [x for x in b.args.scan_ops.split(",") if x]:
        fac, unit = DUTY_OPS[name]
        op = fac()
        rk = "bytes_per_s" if unit == "byte" else "flops_per_s"
        pts = []
        for rep in range(b.args.duty_repeats):
            for bl in lens:
                fl = b.run_duty(op, 0.0, 5.0, settle_s=1.0)
                s = b.run_duty(op, b.args.scan_duty, b.args.duty_steady_s, burst_s=bl)
                dyn = s["p_w"] - fl["p_w"]
                rec = {"rep": rep, "burst_s_target": bl, "t_burst_s": s["t_burst_s"],
                       "bursts_per_s": s["duty_actual"] / s["t_burst_s"] if s["t_burst_s"] else None,
                       "rate": s[rk], "burst_rate": s["burst_bytes_per_s" if unit == "byte" else "burst_flops_per_s"],
                       "p_w": s["p_w"], "p_avg_w": s.get("p_avg_w"), "floor_w": fl["p_w"], "p_dyn_w": dyn,
                       "e_per_unit": dyn / s[rk] if s[rk] else None, "sm_mhz": s.get("sm_mhz"),
                       "frac_sw_power_cap": s.get("frac_sw_power_cap")}
                pts.append(rec)
                print(f"  [scan {name} r{rep} {bl*1e3:.1f} ms] t_burst={s['t_burst_s']*1e3:.2f} ms  "
                      f"burst_rate={rec['burst_rate']:.3e}  floor={fl['p_w']:.2f}  P={s['p_w']:.2f} "
                      f"(avg1s {s.get('p_avg_w') or 0:.2f})  dyn={dyn:.2f} W  e={rec['e_per_unit']:.3e} J/{unit}  "
                      f"sm={s.get('sm_mhz') or 0:.0f} cap={s.get('frac_sw_power_cap', 0):.2f}", flush=True)
        out[name] = {"unit": unit, "duty": b.args.scan_duty, "points": pts}
        del op
        torch.cuda.empty_cache()
    results["burstscan"] = out


class RepOp:
    """Run a small op k times per burst so burst length = duty * period."""
    def __init__(self, op):
        self.op = op; self.k = 1
    def __call__(self):
        for _ in range(self.k):
            self.op()
    @property
    def bytes_per_call(self):
        return self.op.bytes_per_call * self.k
    @property
    def flops_per_call(self):
        return self.op.flops_per_call * self.k
    def tune(self, target_s):
        self.op.tune(1e-9)                   # minimum unit (1 pass / 1 rep)
        self.op(); torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(20):
            self.op()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t) / 20
        self.k = max(1, int(round(target_s / dt)))


DUTY10_OPS = {
    "dram_stream_64MB": (lambda: StreamOp(max(64 * 2**20, 10 * _l2())), "byte"),
    "l2_stream_half": (lambda: StreamOp(_l2() // 2), "byte"),
    "gemv_b1_64MB": (lambda: _mk_gemv_small(), "byte"),
    "gemm_fp16_2048": (lambda: _mk_gemm(2048), "flop"),
}


def _mk_gemv_small():
    K = 8192
    wbytes = max(64 * 2**20, 10 * _l2())
    N = int(wbytes // (2 * K)) // 256 * 256
    W = torch.randn(N, K, device="cuda", dtype=torch.float16) * 0.01
    x = torch.randn(1, K, device="cuda", dtype=torch.float16)
    return TorchOp(lambda: torch.nn.functional.linear(x, W), 2.0 * N * K + 2.0 * (N + K), 2.0 * N * K)


def test_duty10(b, results):
    """Fixed-period duty cycle (period b.args.period_ms, default 10 ms): gaps are
    always short, so the GPU cannot sink into a lower idle state between bursts
    (the failure mode found by 'burstscan'/'aliasing': with >=50 ms gaps the floor
    drops and burst energy disappears from floor-subtracted power). Floor is a
    keepalive at the SAME period, re-measured before every point; duty order is
    shuffled per repeat."""
    per = b.args.period_ms * 1e-3
    duties = [float(x) for x in b.args.duties10.split(",")]
    out = {}
    only = [x for x in (b.args.duty_ops or "").split(",") if x]
    for name, (fac, unit) in DUTY10_OPS.items():
        if only and name not in only:
            continue
        base = fac()
        pts = []
        for rep in range(b.args.duty_repeats):
            order = duties[:]
            _rng.shuffle(order)
            for d in order:
                op = RepOp(base)
                fl = b.run_duty(op, 0.0, 5.0, settle_s=1.0, period_s=per)
                s = b.run_duty(op, d, b.args.duty_steady_s, burst_s=d * per, period_s=per)
                rk = "bytes_per_s" if unit == "byte" else "flops_per_s"
                dyn = s["p_w"] - fl["p_w"]
                rec = {"rep": rep, "duty": d, "duty_actual": s["duty_actual"], "t_burst_s": s["t_burst_s"],
                       "rate": s[rk], "burst_rate": s["burst_bytes_per_s" if unit == "byte" else "burst_flops_per_s"],
                       "floor_w": fl["p_w"], "p_w": s["p_w"], "p_dyn_w": dyn,
                       "e_point": dyn / s[rk] if s[rk] else None, "sm_mhz": s.get("sm_mhz"),
                       "floor_sm_mhz": fl.get("sm_mhz"), "frac_sw_power_cap": s.get("frac_sw_power_cap"),
                       "temp_c": s.get("temp_c")}
                pts.append(rec)
                print(f"  [duty10 {name} r{rep} d={d:.3f}] burst={s['t_burst_s']*1e3:.2f} ms  floor={fl['p_w']:.2f} "
                      f"(sm {fl.get('sm_mhz') or 0:.0f})  P={s['p_w']:.2f} (sm {s.get('sm_mhz') or 0:.0f})  "
                      f"dyn={dyn:.2f} W  rate={rec['rate']:.3e}  burst_rate={rec['burst_rate']:.3e}  "
                      f"e={rec['e_point']:.3e} J/{unit}  cap={s.get('frac_sw_power_cap', 0):.2f}", flush=True)
        # slope through origin of dyn vs rate (floor already subtracted per point) + OLS with intercept
        xs = [p["rate"] for p in pts]; ys = [p["p_dyn_w"] for p in pts]
        e0 = sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)
        a, e, sa, se, r2 = _linfit(xs, ys) if len(set(xs)) > 2 else (None, None, None, None, None)
        eps = [p["e_point"] for p in pts]
        out[name] = {"unit": unit, "period_s": per, "points": pts,
                     "e_origin": e0, "fit_ols": {"a_w": a, "e": e, "se_e": se, "r2": r2},
                     "e_point_mean": sum(eps) / len(eps), "e_point_std": st.pstdev(eps)}
        print(f"  ==> {name}: e(origin)={e0:.4e}  e(OLS)={e if e is None else f'{e:.4e}'} ± {se if se is None else f'{se:.1e}'} "
              f"(intercept {a if a is None else f'{a:.2f}'} W)  per-point mean {out[name]['e_point_mean']:.4e} "
              f"± {out[name]['e_point_std']:.1e} J/{unit}", flush=True)
        del base
        torch.cuda.empty_cache()
    results["duty10"] = out


def test_aliasing(b, results):
    """Does NVML see sparse bursts? Same mean rate (3% duty), burst length and
    period-jitter varied. Physically dyn power must be ~independent of burst
    length at fixed mean rate and full burst speed; a collapse for long, regular
    bursts that jitter restores = sensor phase-locking/under-sampling."""
    out = []
    for name in ("dram_stream", "gemm_fp16_4096"):
        fac, unit = DUTY_OPS[name]
        op = fac()
        rk = "bytes_per_s" if unit == "byte" else "flops_per_s"
        for bl in (0.0004, 0.0015, 0.006):
            for jit in (False, True):
                fl = b.run_duty(op, 0.0, 5.0, settle_s=1.0)
                s = b.run_duty(op, 0.03, b.args.duty_steady_s, burst_s=bl, jitter=jit)
                dyn = s["p_w"] - fl["p_w"]
                rec = {"op": name, "unit": unit, "burst_s": s["t_burst_s"], "jitter": jit,
                       "period_s": s["period_s"], "rate": s[rk], "floor_w": fl["p_w"], "p_w": s["p_w"],
                       "p_avg_w": s.get("p_avg_w"), "p_dyn_w": dyn, "e": dyn / s[rk] if s[rk] else None,
                       "sm_mhz": s.get("sm_mhz"), "frac_sw_power_cap": s.get("frac_sw_power_cap"),
                       "burst_rate": s["burst_bytes_per_s" if unit == "byte" else "burst_flops_per_s"]}
                out.append(rec)
                print(f"  [alias {name} {bl*1e3:.1f} ms jitter={jit}] period={s['period_s']*1e3:.1f} ms  "
                      f"floor={fl['p_w']:.2f} P={s['p_w']:.2f} dyn={dyn:.2f} W  e={rec['e']:.3e} J/{unit}  "
                      f"burst_rate={rec['burst_rate']:.3e} cap={s.get('frac_sw_power_cap', 0):.2f}", flush=True)
        del op
        torch.cuda.empty_cache()
    results["aliasing"] = out


def test_lowgrid(b, results):
    """Steady (non-bursty, sensor-safe) DRAM streams on 1..6 programs: low enough
    power to stay under a 100 W cap at full clocks. Slope P vs BW = upper bound on
    marginal J/byte (per-active-SM power is attributed to bytes)."""
    ws = dram_ws()
    pts = []
    for nprog in (1, 2, 3, 4, 6):
        r = b.repeat(f"lowgrid_{nprog}prog", lambda: StreamOp(ws, nprog=nprog),
                     {"test": "lowgrid", "nprog": nprog, "ws_bytes": ws}, repeats=1,
                     settle_s=2.0, steady_s=b.args.steady_s)
        pts.append(r)
    fl = b.measure_idle_p0(8.0)
    xs = [0.0] + [p["bytes_per_s_mean"] for p in pts]
    ys = [fl["p_w"]] + [p["p_w_mean"] for p in pts]
    a, e, sa, se, r2 = _linfit(xs, ys)
    a2, e2, sa2, se2, r22 = _linfit(xs[1:], ys[1:])
    results["lowgrid"] = {"points": pts, "floor_p0": fl,
                          "fit_with_floor": {"a_w": a, "e": e, "se_e": se, "r2": r2},
                          "fit_active_only": {"a_w": a2, "e": e2, "se_e": se2, "r2": r22}}
    print(f"  lowgrid fit (incl. keepalive floor): e={e:.3e}±{se:.1e} J/B a={a:.2f} W R2={r2:.4f}; "
          f"active-only: e={e2:.3e}±{se2:.1e} a={a2:.2f} W", flush=True)


def test_square(b, results, trace_path):
    op = StreamOp(1 * 2**30)
    op.tune(0.02)
    for _ in range(20):
        op()
    torch.cuda.synchronize()
    time.sleep(2.0)
    t0 = time.time()
    marks = []
    for _ in range(10):
        a = time.time()
        while time.time() < a + 0.5:
            op(); torch.cuda.synchronize()
        m = time.time(); time.sleep(0.5)
        marks.append((a, m, time.time()))
    time.sleep(1.0)
    rows = b.sampler.window(t0 - 1.0, time.time())
    with open(trace_path, "w") as f:
        f.write("t,energy_mj,p_inst_w,p_avg_w,sm_mhz,mem_mhz,temp_c,reasons,on\n")
        for r in rows:
            on = int(any(a <= r[0] < m for a, m, _ in marks))
            f.write(",".join("" if v is None else str(v) for v in r) + f",{on}\n")
    # correlation of each signal with on/off; energy-counter derived power per 100 ms
    on = [int(any(a <= r[0] < m for a, m, _ in marks)) for r in rows]

    def corr(xs, ys):
        pr = [(x, y) for x, y in zip(xs, ys) if x is not None]
        if len(pr) < 3:
            return None
        xs, ys = zip(*pr)
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        sx = math.sqrt(sum((x - mx) ** 2 for x in xs)); sy = math.sqrt(sum((y - my) ** 2 for y in ys))
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy) if sx and sy else None
    # energy-derived power: differentiate between update events
    pe = [None] * len(rows)
    last = None
    for i, r in enumerate(rows):
        if r[1] is None:
            continue
        if last is not None and r[1] != rows[last][1]:
            p = (r[1] - rows[last][1]) * 1e-3 / (r[0] - rows[last][0])
            for j in range(last + 1, i + 1):
                pe[j] = p
            last = i
        elif last is None:
            last = i
    results["square"] = {
        "trace": os.path.basename(trace_path),
        "corr_on_vs_p_inst": corr([r[2] for r in rows], on),
        "corr_on_vs_p_avg": corr([r[3] for r in rows], on),
        "corr_on_vs_p_energy_diff": corr(pe, on),
        "note": "0.5 s on / 0.5 s off DRAM stream; higher corr = signal resolves sub-second power",
    }
    print("  [square]", {k: v for k, v in results["square"].items() if k.startswith("corr")}, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", default="duty,duty10,burstlen,burstscan,stream,grid,vendor,gemv,gemm,square")
    ap.add_argument("--duties", default="0,0.015,0.03,0.045,0.06,0.075")
    ap.add_argument("--duty_repeats", type=int, default=2)
    ap.add_argument("--burst_s", type=float, default=0.0015)
    ap.add_argument("--duty_ops", default="", help="comma list subset of DUTY_OPS")
    ap.add_argument("--scan_ops", default="dram_stream,gemv_b1,gemm_fp16_4096")
    ap.add_argument("--scan_lens_ms", default="0.4,0.7,1,1.5,2.5,4,6")
    ap.add_argument("--scan_duty", type=float, default=0.03)
    ap.add_argument("--period_ms", type=float, default=10.0)
    ap.add_argument("--duties10", default="0.015,0.03,0.045")
    ap.add_argument("--duty_steady_s", type=float, default=8.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--steady_s", type=float, default=10.0)
    ap.add_argument("--settle_s", type=float, default=3.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--idle_s", type=float, default=10.0)
    ap.add_argument("--chunk_s", type=float, default=0.03)
    ap.add_argument("--sample_ms", type=float, default=10.0)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    b = Bench(args)
    b.info = b.gpu_info()
    print("GPU:", json.dumps(b.info), flush=True)
    gtag = (b.info["name"] or "gpu").replace("NVIDIA ", "").replace("RTX ", "").replace(" ", "_")
    out_path = args.out or f"microbench/results_{gtag}.json"
    results = {"gpu": b.info, "args": vars(args), "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    # touch CUDA so the context exists during idle
    torch.zeros(1, device="cuda"); torch.cuda.synchronize()
    # Two idle conventions (they differ by ~4x on consumer-class parts):
    #   idle_deep = GPU fell back to its idle P-state (what vLLM's idle baseline sees)
    #   idle_p0   = clocks-up floor (keepalive no-op); intercept of the duty fits
    b.idle = b.measure_idle_deep(args.idle_s)
    results["idle_pre"] = b.idle
    print(f"idle_deep(pre): {json.dumps(b.idle)}", flush=True)
    b.idle_p0 = b.measure_idle_p0(args.idle_s)
    results["idle_p0_pre"] = b.idle_p0
    print(f"idle_p0(pre): {json.dumps(b.idle_p0)}", flush=True)

    tests = [t.strip() for t in args.tests.split(",") if t.strip()]

    def save():
        with open(out_path, "w") as f:
            json.dump(results, f, indent=1)

    for t in tests:
        print(f"=== {t} ===", flush=True)
        try:
            if t == "duty":
                test_duty(b, results)
            elif t == "duty10":
                test_duty10(b, results)
            elif t == "aliasing":
                test_aliasing(b, results)
            elif t == "lowgrid":
                test_lowgrid(b, results)
            elif t == "burstscan":
                test_burstscan(b, results)
            elif t == "burstlen":
                test_burstlen(b, results)
            elif t == "stream":
                test_stream(b, results)
            elif t == "grid":
                test_grid(b, results)
            elif t == "vendor":
                test_vendor(b, results)
            elif t == "gemv":
                test_gemv(b, results)
            elif t == "gemm":
                test_gemm(b, results)
            elif t == "square":
                test_square(b, results, out_path.replace(".json", "_square_trace.csv"))
        except Exception as e:
            import traceback; traceback.print_exc()
            results.setdefault("errors", {})[t] = repr(e)
        save()

    results["idle_p0_post"] = b.measure_idle_p0(args.idle_s)
    results["idle_post"] = b.measure_idle_deep(args.idle_s)
    print(f"idle_post: {json.dumps(results['idle_post'])}", flush=True)
    results["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    b.sampler.stop()
    save()
    print_table(results)
    print("wrote", out_path)


def print_table(res):
    print("\n%-26s %9s %8s %8s %7s %6s %5s %12s %10s" % (
        "point", "BW GB/s", "TFLOPS", "P W", "dyn W", "SMMHz", "cap", "J/byte dyn", "pJ/flop"))
    rows = []
    for k in ("stream", "vendor", "gemv", "gemm"):
        rows += res.get(k, [])
    rows += (res.get("grid") or {}).get("points", [])
    for r in rows:
        jb = r.get("j_per_byte_dyn_mean")
        jf = r.get("j_per_flop_dyn_mean")
        print("%-26s %9.1f %8.2f %8.1f %7.1f %6.0f %5.2f %12s %10s" % (
            r["label"], r.get("bytes_per_s_mean", 0) / 1e9, r.get("flops_per_s_mean", 0) / 1e12,
            r.get("p_w_mean", 0), r.get("p_dyn_w_mean", 0), r.get("sm_mhz_mean") or 0,
            r.get("frac_sw_power_cap_mean", 0) or 0,
            f"{jb:.3e}±{r.get('j_per_byte_dyn_std', 0):.1e}" if jb else "-",
            f"{jf*1e12:.3f}" if jf else "-"))
    d = res.get("duty") or {}
    if d:
        print("\nduty-cycle slope fits (full clocks, below cap):")
        print("%-20s %14s %10s %9s %7s %4s %14s" % ("op", "e (marginal)", "se", "floor W", "R2", "n", "burst rate"))
        for k, v in d.items():
            f = v.get("fit_pooled")
            if f:
                print("%-20s %14.4e %10.1e %9.2f %7.4f %4d %14.3e  J/%s" % (
                    k, f["e"], f["se_e"], f["a_w"], f["r2"] or 0, f["n"], v.get("burst_rate_mean") or 0, v["unit"]))
    g = (res.get("grid") or {}).get("fit")
    if g:
        print(f"\ngrid fit: e_marginal={g['e_marginal_j_per_byte']:.3e} ± {g['e_se']:.1e} J/B, "
              f"P0={g['p0_w']:.1f} W over {g['n_points']} pts")


if __name__ == "__main__":
    main()
