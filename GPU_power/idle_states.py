#!/usr/bin/env python3
"""
idle_states.py — GPU power in each "idle" state, which matters for provisioning
(the idle term dominates J/token at low load, and the A100 showed 49 W with no
process vs 99 W with a CUDA context open). NVML sampled at 10 ms throughout.

  1 no_process     NVML only, no CUDA context (what an unused, provisioned GPU draws)
  2 cuda_context   torch CUDA context open, no kernels (clocks may stay up)
  3 vllm_loaded    vLLM engine with the model loaded, no traffic (= the serving
                   harness's idle_power_w; what we use as P_static)
  4 after_shutdown engine deleted, process alive
  5 decay          same process after the context is torn down... the CUDA context
                   cannot be destroyed in-process, so this is 4 continued for 60 s:
                   does the GPU fall back to a low P-state on its own?
  python3 idle_states.py --model Qwen/Qwen2-7B-Instruct --out <dir>/idle_states.json
"""
import argparse, json, os, threading, time
import pynvml


class Sampler:
    def __init__(self, idx=0, dt=0.01):
        pynvml.nvmlInit(); self.h = pynvml.nvmlDeviceGetHandleByIndex(idx); self.dt = dt
        self.rows = []; self._stop = threading.Event()

    def _read(self):
        h = self.h
        try:
            pi = pynvml.nvmlDeviceGetFieldValues(h, [186])[0]
            pi = pi.value.uiVal * 1e-3 if pi.nvmlReturn == 0 else None
        except Exception:
            pi = None
        return (time.time(), pynvml.nvmlDeviceGetPowerUsage(h) * 1e-3, pi,
                pynvml.nvmlDeviceGetTotalEnergyConsumption(h),
                pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
                pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM),
                pynvml.nvmlDeviceGetPerformanceState(h))

    def start(self):
        def loop():
            while not self._stop.is_set():
                self.rows.append(self._read()); time.sleep(self.dt)
        self.t = threading.Thread(target=loop, daemon=True); self.t.start(); return self

    def stop(self):
        self._stop.set(); self.t.join()

    def window(self, t0, t1):
        r = [x for x in self.rows if t0 <= x[0] <= t1]
        if len(r) < 2:
            return {}
        e = (r[-1][3] - r[0][3]) * 1e-3 / (r[-1][0] - r[0][0])
        m = lambda i: sum(x[i] for x in r if x[i] is not None) / max(1, sum(1 for x in r if x[i] is not None))
        return dict(p_avg_w=m(1), p_inst_w=m(2), p_energy_w=e, sm_mhz=m(4), mem_mhz=m(5),
                    pstate_min=min(x[6] for x in r), pstate_max=max(x[6] for x in r),
                    duration_s=r[-1][0] - r[0][0], n=len(r))


def hold(s, label, dur, out):
    time.sleep(2.0)                      # let the transition settle
    t0 = time.time(); time.sleep(dur); t1 = time.time()
    out[label] = s.window(t0, t1)
    print(f"[{label}] {json.dumps(out[label])}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-7B-Instruct")
    ap.add_argument("--dur", type=float, default=30.0)
    ap.add_argument("--decay_s", type=float, default=90.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s = Sampler().start()
    out = {"gpu": pynvml.nvmlDeviceGetName(s.h)}
    hold(s, "1_no_process", a.dur, out)
    import torch
    torch.zeros(1, device="cuda"); torch.cuda.synchronize()
    hold(s, "2_cuda_context", a.dur, out)
    from vllm import LLM
    llm = LLM(model=a.model, dtype="float16", gpu_memory_utilization=0.9, max_model_len=4096)
    llm.generate(["warm up"], use_tqdm=False)
    hold(s, "3_vllm_loaded", a.dur, out)
    del llm
    import gc; gc.collect(); torch.cuda.empty_cache()
    hold(s, "4_after_shutdown", a.dur, out)
    # decay: is there a timeout after which clocks drop on their own?
    t0 = time.time(); trace = []
    while time.time() - t0 < a.decay_s:
        time.sleep(10); w = s.window(time.time() - 10, time.time())
        trace.append(dict(t=round(time.time() - t0), p_w=w.get("p_inst_w") or w.get("p_avg_w"), sm_mhz=w.get("sm_mhz")))
    out["5_decay_trace"] = trace
    print(f"[5_decay_trace] {json.dumps(trace)}", flush=True)
    s.stop()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
