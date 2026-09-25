#!/usr/bin/env python3
"""
controls/live_load.py -- live closed-loop admission-control experiment on one GPU
(designed for the 100-W-capped RTX A5000s on node-d1; power limits are root-only,
so the actuator is an admission gate in front of vLLM).

  open-loop MMPP arrivals -> FIFO -> [admission gate: max in flight] -> vLLM AsyncLLM
                                          ^ 1 Hz controller (static|pi|polca|mpc)
                                          |  obs: NVML power, queue/ages, in-flight,
                                          |  tokens generated, announced *virtual*
                                          |  budget B(t) (below the hardware cap)

One process loads the engine once and runs every (controller, rep) episode back to
back (order shuffled per rep, idle gap between episodes), so all episodes share
one iter_log.csv / power_trace.csv and one set of weights in memory.

Per episode it persists
  requests.csv     request_id, t_arrival, t_admit, t_first_token, t_finish,
                   prompt_tokens, gen_tokens (open loop: arrival != admit)
  control_log.csv  one row per tick: budget, measured power (1 s, 10 s), queue,
                   oldest wait, in flight, tokens, TPOT, decision, controller info
  episode.json     window, config, summary metrics
and at the top level run_meta.json (sysid-compatible fields), gpu_query.csv
(nvidia-smi index,pci.bus_id,power.limit,enforced.power.limit), summary.csv.

Reuses the shared harness pieces WITHOUT modifying them: engine_compat
(build_async_engine, make_sampling_params), iter_logger (via build_async_engine),
nvml_logger.NvmlPowerLogger, prompts_<task>.json.

  # CPU dry run (mock engine + mock power), ~2 min:
  python3 -m controls.live_load --mock --controllers static64,pi,polca,mpc \
      --duration_s 40 --reps 1 --run_dir /tmp/live_mock
  # real (inside the vLLM container, on an A5000):
  python3 -m controls.live_load --model Qwen/Qwen2.5-3B-Instruct --task alpaca \
      --table_json logs/A5000_live/a5000_table.json --run_dir logs/A5000_live/live
  # analysis only:
  python3 -m controls.live_load --analyze logs/A5000_live/live
"""
import argparse
import asyncio
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from controls.live_controllers import (Throughput, default_cfg, make_live_controller,  # noqa
                                       mmpp_arrivals, parse_schedule, budget_at, solve_mu)


# ============================================================ mock engine / power
class _MockOut:
    def __init__(self, toks, prompt_ids):
        self.token_ids = toks


class _MockRequestOutput:
    def __init__(self, toks, prompt_ids, finished):
        self.outputs = [_MockOut(toks, prompt_ids)]
        self.prompt_token_ids = prompt_ids
        self.finished = finished


class MockEngine:
    """Continuous-batching stand-in: every iteration each running request gets
    one token; iteration time n/thr(n) from the throughput table; busy power
    pinned at p_busy, idle p_idle (the 100-W-capped A5000 behaviour)."""

    def __init__(self, cfg, seed=0):
        self.thr = Throughput(cfg["table"])
        self.cfg = cfg
        self.active = {}
        self.rng = random.Random(seed)
        self.mu = solve_mu(cfg["gen_mean"], cfg["gen_cap"])
        self.power = None
        self._task = None

    async def _loop(self):
        while True:
            n = len(self.active)
            if n == 0:
                await asyncio.sleep(0.005)
                continue
            await asyncio.sleep(n / max(self.thr(n), 1e-6))
            for rid, st in list(self.active.items()):
                st["g"] += 1
                done = st["g"] >= st["G"]
                st["q"].put_nowait((st["g"], done))
                if done:
                    del self.active[rid]

    async def generate(self, prompt, sp, request_id):
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(self._loop())
        G = int(min(self.cfg["gen_cap"], max(1, self.rng.expovariate(1.0 / self.mu))))
        q = asyncio.Queue()
        self.active[request_id] = dict(g=0, G=G, q=q)
        pid = list(range(20))
        while True:
            g, done = await q.get()
            yield _MockRequestOutput(list(range(g)), pid, done)
            if done:
                return

    def shutdown(self):
        if self._task is not None:
            self._task.cancel()


class MockPower:
    """Mimics NvmlPowerLogger's surface (start/stop/summary/_powers_w)."""

    def __init__(self, engine, out_csv, interval_ms=10):
        self.engine, self.out_csv, self.dt = engine, out_csv, interval_ms / 1000.0
        self._powers_w = []
        self.gpu_name = "MOCK (capped A5000 model)"
        self.power_limit_w = 100.0
        self._stop = False
        self._thread = None

    def start(self):
        import threading
        self._f = open(self.out_csv, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(["t_wall", "power_w", "sm_mhz", "mem_mhz", "gpu_util", "mem_util",
                          "temp_c", "power_inst_w"])

        def loop():
            while not self._stop:
                c = self.engine.cfg
                p = c["p_busy"] if self.engine.active else c["p_idle"]
                p += random.uniform(-0.5, 0.5)
                self._powers_w.append(p)
                self._w.writerow([f"{time.time():.6f}", f"{p:.3f}", "", "", "", "", "", f"{p:.3f}"])
                time.sleep(self.dt)
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop = True
        if self._thread:
            self._thread.join(timeout=2)
        self._f.close()
        return self

    def summary(self):
        ps = self._powers_w
        return dict(gpu_name=self.gpu_name, power_limit_w=self.power_limit_w,
                    n_samples=len(ps), mean_power_w=sum(ps) / max(len(ps), 1))


# ============================================================ helpers
def gpu_query(path, mock=False):
    """Record index,pci.bus_id,power.limit,enforced.power.limit per run."""
    if mock:
        txt = "index, pci.bus_id, power.limit [W], enforced.power.limit [W]\n0, MOCK, 100.00 W, 100.00 W\n"
    else:
        try:
            txt = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,pci.bus_id,power.limit,enforced.power.limit,"
                 "power.default_limit,power.min_limit,power.max_limit", "--format=csv"],
                capture_output=True, text=True, timeout=20).stdout
        except Exception as e:
            txt = f"nvidia-smi unavailable: {e!r}\n"
    with open(path, "w") as f:
        f.write(txt)
    return txt


def load_table(path):
    """a5000_table.json written by controls/a5000_sysid.py table."""
    with open(path) as f:
        js = json.load(f)
    tab = {int(k): float(v) for k, v in js["table"].items()}
    return tab, js


class Episode:
    """State + logs of one controller episode."""

    def __init__(self, name, rundir):
        self.name, self.dir = name, rundir
        os.makedirs(rundir, exist_ok=True)
        self.waiting = deque()          # dicts: rid, t_arr, prompt
        self.inflight = {}              # rid -> dict
        self.done = []
        self.cap = 0
        self.tok_tick = 0
        self.gap_sum = 0.0
        self.gap_n = 0
        self.arr_tick = 0
        self.ctrl_rows = []


async def run_request(ep, engine, req, sp):
    req["t_admit"] = time.time()
    req["g"] = 0
    req["t_first"] = None
    last_t = None
    ep.inflight[req["rid"]] = req
    try:
        async for out in engine.generate(req["prompt"], sp, req["rid"]):
            try:
                g = len(out.outputs[0].token_ids)
            except Exception:
                g = req["g"] + 1
            now = time.time()
            d = g - req["g"]
            if d > 0:
                if req["t_first"] is None:
                    req["t_first"] = now
                elif last_t is not None:
                    ep.gap_sum += (now - last_t)
                    ep.gap_n += d
                last_t = now
                ep.tok_tick += d
                req["g"] = g
            try:
                req["prompt_tokens"] = len(out.prompt_token_ids or [])
            except Exception:
                pass
    except Exception as e:
        req["error"] = repr(e)
    req["t_finish"] = time.time()
    ep.inflight.pop(req["rid"], None)
    ep.done.append(req)


async def run_episode(args, engine, power, ep, ctrl_name, ctrl_kw, cfg, prompts, sp, rng_seed):
    """One controller episode: arrivals for duration_s, then drain (bounded)."""
    ctrl = make_live_controller(ctrl_name, **ctrl_kw)
    ctrl.reset(cfg)
    sched = parse_schedule(args.budget_schedule)
    T = args.duration_s
    arr_t = list(mmpp_arrivals(T, args.rate_lo, args.rate_hi, args.dur_lo, args.dur_hi,
                               rng_seed))
    prng = random.Random(rng_seed)
    t0 = time.time()
    ep.t0 = t0
    tasks = []
    samples = getattr(power, "_inst_w", power._powers_w)
    p_idx = len(samples)
    p_hist = deque(maxlen=cfg["budget_window"])
    arr_hist = []
    stop = {"arrivals": False}

    async def arrivals():
        k = 0
        for ta in arr_t:
            delay = t0 + ta - time.time()
            if delay > 0:
                await asyncio.sleep(delay)
            ep.waiting.append(dict(rid=f"{ep.name}-{k}", t_arr=time.time(),
                                   prompt=prng.choice(prompts)))
            ep.arr_tick += 1
            k += 1
        stop["arrivals"] = True

    async def dispatcher():
        while not (stop["arrivals"] and not ep.waiting) and time.time() < t0 + T + args.drain_s:
            while ep.waiting and len(ep.inflight) < ep.cap:
                req = ep.waiting.popleft()
                tasks.append(asyncio.create_task(run_request(ep, engine, req, sp)))
                await asyncio.sleep(0)
            await asyncio.sleep(0.005)

    async def controller():
        nonlocal p_idx
        k = 0
        while time.time() < t0 + T + args.drain_s and not (
                stop["arrivals"] and not ep.waiting and not ep.inflight):
            now = time.time()
            ps = samples[p_idx:]
            p_idx = len(samples)
            p1 = sum(ps) / len(ps) if ps else (p_hist[-1] if p_hist else cfg["p_idle"])
            p_hist.append(p1)
            arr_hist.append(ep.arr_tick)
            tel = now - t0
            B = budget_at(sched, tel)
            obs = dict(t=tel, dt=1.0, q=len(ep.waiting), n_run=len(ep.inflight),
                       ages=[now - r["t_arr"] for r in ep.waiting],
                       run_done_tok=[r.get("g", 0) for r in ep.inflight.values()],
                       p_1s=p1, p_10s=sum(p_hist) / len(p_hist), budget=B,
                       budget_future=[budget_at(sched, tel + j) for j in range(cfg["horizon"])],
                       arr_hist=arr_hist[-120:], tok_tick=ep.tok_tick,
                       tpot_tick=(ep.gap_sum / ep.gap_n) if ep.gap_n else 0.0)
            t_c = time.perf_counter()
            try:
                m = int(ctrl.act(obs))
            except Exception as e:           # a controller bug must not hang the GPU run
                print(f"[warn] controller {ctrl_name} failed: {e!r}; holding cap", flush=True)
                m = ep.cap
            ctrl_ms = 1e3 * (time.perf_counter() - t_c)
            if tel >= T:                      # arrivals over: drain what is queued
                m = max(m, cfg["m_max"])
            ep.cap = max(0, m)
            ep.ctrl_rows.append(dict(t=round(tel, 3), budget=B, p_1s=round(p1, 3),
                                     p_10s=round(obs["p_10s"], 3), q=obs["q"],
                                     oldest=round(obs["ages"][0], 3) if obs["ages"] else 0.0,
                                     n_run=obs["n_run"], arrivals=ep.arr_tick,
                                     tokens=ep.tok_tick, tpot=round(obs["tpot_tick"], 5),
                                     cap=ep.cap, ctrl_ms=round(ctrl_ms, 3),
                                     info=json.dumps(ctrl.info(), default=str)))
            ep.tok_tick = 0
            ep.gap_sum, ep.gap_n = 0.0, 0
            ep.arr_tick = 0
            k += 1
            await asyncio.sleep(max(0.0, t0 + k - time.time()))

    await asyncio.gather(arrivals(), dispatcher(), controller())
    # hard stop: anything still running after the drain budget is cancelled
    for t in tasks:
        if not t.done():
            t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    ep.t1 = time.time()
    return ctrl


def write_episode(ep, ctrl, cfg, args):
    with open(os.path.join(ep.dir, "requests.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["request_id", "t_arrival", "t_admit", "t_first_token", "t_finish",
                    "prompt_tokens", "gen_tokens", "ttft_s", "queue_s", "tpot_s", "error"])
        for r in sorted(ep.done, key=lambda r: r["t_arr"]):
            tf = r.get("t_first")
            tp = ((r["t_finish"] - tf) / max(r["g"] - 1, 1)) if tf else None
            w.writerow([r["rid"], f"{r['t_arr']:.6f}", f"{r['t_admit']:.6f}",
                        "" if tf is None else f"{tf:.6f}", f"{r['t_finish']:.6f}",
                        r.get("prompt_tokens", ""), r["g"],
                        "" if tf is None else f"{tf - r['t_arr']:.6f}",
                        f"{r['t_admit'] - r['t_arr']:.6f}",
                        "" if tp is None else f"{tp:.6f}", r.get("error", "")])
        for r in ep.waiting:        # never admitted
            w.writerow([r["rid"], f"{r['t_arr']:.6f}", "", "", "", "", 0, "", "", "", "not_admitted"])
    if ep.ctrl_rows:
        with open(os.path.join(ep.dir, "control_log.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(ep.ctrl_rows[0].keys()))
            w.writeheader()
            w.writerows(ep.ctrl_rows)
    js = dict(controller=ep.name, window_wall_t0=ep.t0, window_wall_t1=ep.t1,
              duration_s=args.duration_s, budget_schedule=args.budget_schedule,
              arrivals=dict(rate_lo=args.rate_lo, rate_hi=args.rate_hi, dur_lo=args.dur_lo,
                            dur_hi=args.dur_hi),
              cfg={k: v for k, v in cfg.items() if k != "table"}, table=cfg["table"],
              n_done=len(ep.done), n_left=len(ep.waiting))
    with open(os.path.join(ep.dir, "episode.json"), "w") as f:
        json.dump(js, f, indent=1)


# ============================================================ analysis
def _read_power(path):
    ts, ps = [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                t = float(r["t_wall"])
                p = r.get("power_inst_w") or r.get("power_w")
                p = float(p)
            except (TypeError, ValueError, KeyError):
                continue
            ts.append(t)
            ps.append(p)
    return ts, ps


def analyze(root):
    """Summarise every episode under root (energy from the NVML trace)."""
    import numpy as np
    rows = []
    top = root
    ptr = os.path.join(top, "power_trace.csv")
    if not os.path.exists(ptr):
        print("no power_trace.csv under", root)
        return []
    ts, ps = map(np.asarray, _read_power(ptr))
    for d in sorted(os.listdir(top)):
        ed = os.path.join(top, d)
        epj = os.path.join(ed, "episode.json")
        if not os.path.exists(epj):
            continue
        E = json.load(open(epj))
        t0, t1 = E["window_wall_t0"], E["window_wall_t1"]
        m = (ts >= t0) & (ts <= t1)
        tt, pp = ts[m], ps[m]
        energy = float(np.trapz(pp, tt)) if len(tt) > 1 else float("nan")
        req = list(csv.DictReader(open(os.path.join(ed, "requests.csv"))))
        toks = sum(int(r["gen_tokens"] or 0) for r in req)
        ttft = np.array([float(r["ttft_s"]) for r in req if r["ttft_s"]])
        tpot = np.array([float(r["tpot_s"]) for r in req if r["tpot_s"]])
        cfg = E["cfg"]
        sched = parse_schedule(E["budget_schedule"])
        # budget compliance on 1 s and budget_window moving averages of NVML power
        sec = np.arange(math.floor(t0), math.ceil(min(t1, t0 + E["duration_s"])))
        p1 = np.array([pp[(tt >= s) & (tt < s + 1)].mean() if ((tt >= s) & (tt < s + 1)).any()
                       else np.nan for s in sec])
        W = cfg["budget_window"]
        pw = np.convolve(np.nan_to_num(p1, nan=cfg["p_idle"]), np.ones(W) / W, mode="full")[:len(p1)]
        B = np.array([budget_at(sched, s - t0) for s in sec])
        cl = list(csv.DictReader(open(os.path.join(ed, "control_log.csv")))) \
            if os.path.exists(os.path.join(ed, "control_log.csv")) else []
        ms = np.array([float(r["ctrl_ms"]) for r in cl]) if cl else np.array([0.0])
        rows.append(dict(episode=d, controller=E["controller"], energy_j=energy, tokens=toks,
                         j_per_tok=energy / max(toks, 1), done=E["n_done"], left=E["n_left"],
                         mean_w=energy / max(t1 - t0, 1e-9),
                         ttft_p50=float(np.median(ttft)) if len(ttft) else float("nan"),
                         ttft_p99=float(np.percentile(ttft, 99)) if len(ttft) else float("nan"),
                         ttft_viol=float(np.mean(ttft > cfg["slo_ttft"])) if len(ttft) else float("nan"),
                         tpot_p50=float(np.median(tpot)) if len(tpot) else float("nan"),
                         tpot_p99=float(np.percentile(tpot, 99)) if len(tpot) else float("nan"),
                         bviol_1s=float(np.nanmean(p1 > B * 1.01)),
                         bviol_win=float(np.mean(pw > B * 1.01)),
                         over_j=float(np.sum(np.maximum(pw - B, 0))),
                         ctrl_ms_mean=float(ms.mean()), ctrl_ms_max=float(ms.max())))
    if rows:
        with open(os.path.join(top, "summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        for r in rows:
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    return rows


# ============================================================ main
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--task", default="alpaca", choices=["alpaca", "sharegpt"])
    ap.add_argument("--prompt_file", default=None)
    ap.add_argument("--output_len", type=int, default=256)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--enforce_eager", action="store_true", default=True)
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--run_dir", default="logs/A5000_live/live")
    ap.add_argument("--controllers", default="static64,pi,polca,mpc")
    ap.add_argument("--ctrl_kw", default="{}", help='JSON {"mpc": {...}, "pi": {...}}')
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--duration_s", type=float, default=300.0)
    ap.add_argument("--drain_s", type=float, default=90.0)
    ap.add_argument("--gap_s", type=float, default=20.0, help="idle gap between episodes")
    ap.add_argument("--rate_lo", type=float, default=0.6)
    ap.add_argument("--rate_hi", type=float, default=2.0)
    ap.add_argument("--dur_lo", type=float, default=40.0)
    ap.add_argument("--dur_hi", type=float, default=12.0)
    ap.add_argument("--budget_schedule", default="100:90,88:60,78:60,100:90",
                    help="virtual budget W:dur segments (announced to the MPC)")
    ap.add_argument("--slo_ttft", type=float, default=30.0)
    ap.add_argument("--slo_tpot", type=float, default=0.25)
    ap.add_argument("--budget_window", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--m_max", type=int, default=64)
    ap.add_argument("--table_json", default=None,
                    help="identified plant table (controls/a5000_sysid.py table)")
    ap.add_argument("--p_idle", type=float, default=None)
    ap.add_argument("--p_busy", type=float, default=None)
    ap.add_argument("--gen_mean", type=float, default=235.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mock", action="store_true", help="CPU dry run: mock engine + power")
    ap.add_argument("--analyze", default=None, help="only summarise an existing run dir")
    return ap


def main():
    args = build_parser().parse_args()
    if args.analyze:
        analyze(args.analyze)
        return
    os.makedirs(args.run_dir, exist_ok=True)
    cfg = default_cfg(slo_ttft=args.slo_ttft, slo_tpot=args.slo_tpot,
                      budget_window=args.budget_window, horizon=args.horizon,
                      m_max=args.m_max, gen_mean=args.gen_mean, gen_cap=args.output_len)
    table_meta = None
    if args.table_json:
        tab, table_meta = load_table(args.table_json)
        cfg["table"] = tab
        cfg["p_idle"] = table_meta.get("p_idle", cfg["p_idle"])
        cfg["p_busy"] = table_meta.get("p_busy", cfg["p_busy"])
        cfg["gen_mean"] = table_meta.get("gen_mean", cfg["gen_mean"])
    if args.p_idle:
        cfg["p_idle"] = args.p_idle
    if args.p_busy:
        cfg["p_busy"] = args.p_busy
    q = gpu_query(os.path.join(args.run_dir, "gpu_query.csv"), mock=args.mock)
    print("[gpu]", q.strip().replace("\n", " | "), flush=True)
    ctrl_kw = json.loads(args.ctrl_kw)

    if args.mock:
        engine = MockEngine(cfg, seed=args.seed)
        power = MockPower(engine, os.path.join(args.run_dir, "power_trace.csv")).start()
        prompts = [f"mock prompt {i}" for i in range(100)]
        sp = None
        emeta = dict(vllm_version="mock", engine_api="mock")
    else:
        from engine_compat import build_async_engine, make_sampling_params
        from nvml_logger import NvmlPowerLogger

        class InstPowerLogger(NvmlPowerLogger):
            """Also keeps POWER_INSTANT (field 186) samples in memory: on Ampere
            nvmlDeviceGetPowerUsage is a ~1 s moving average, too laggy for a
            1 Hz controller. Falls back to power_w when the field is absent."""

            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self._inst_w = []

            def _read_row(self):
                row = super()._read_row()
                v = getattr(self, "_last_inst", None)
                v = v if v is not None else row[0]
                if v is not None:
                    self._inst_w.append(v)
                return row
        pf = args.prompt_file or os.path.join(ROOT, f"prompts_{args.task}.json")
        with open(pf) as f:
            prompts = json.load(f)
        engine, emeta = build_async_engine(args, iter_log_csv=os.path.join(args.run_dir,
                                                                           "iter_log.csv"))
        sp = make_sampling_params(args.output_len, force_exact=False)
        power = InstPowerLogger(args.gpu_id, os.path.join(args.run_dir, "power_trace.csv"),
                                interval_ms=10).start()
    names = [c.strip() for c in args.controllers.split(",") if c.strip()]
    order = []
    for rep in range(args.reps):
        o = list(names)
        random.Random(1000 + rep).shuffle(o)
        order += [(rep, c) for c in o]
    episodes = []

    async def run_all():
        # warmup (2 requests) so the first episode does not pay compilation / cache cold start
        warm = Episode("warmup", os.path.join(args.run_dir, "_warmup"))
        warm.cap = 2
        for i in range(2):
            await run_request(warm, engine, dict(rid=f"warm-{i}", t_arr=time.time(),
                                                 prompt=prompts[i]), sp)
        for rep, c in order:
            ep = Episode(c, os.path.join(args.run_dir, f"{c}_rep{rep}"))
            print(f"[episode] {c} rep{rep} start", flush=True)
            ctrl = await run_episode(args, engine, power, ep, c, ctrl_kw.get(c.rstrip("0123456789"),
                                                                           ctrl_kw.get(c, {})),
                                     cfg, prompts, sp, rng_seed=args.seed + rep)
            write_episode(ep, ctrl, cfg, args)
            episodes.append(ep)
            print(f"[episode] {c} rep{rep} done: {len(ep.done)} done, {len(ep.waiting)} left",
                  flush=True)
            await asyncio.sleep(args.gap_s)

    t_start = time.time()
    try:
        asyncio.run(run_all())
    finally:
        power.stop()
        try:
            engine.shutdown()
        except Exception:
            pass
    s = power.summary()
    meta = dict(gpu_name=s.get("gpu_name"), power_limit_w=s.get("power_limit_w"),
                model=args.model, dtype=args.dtype, task=args.task, mode="live_admission",
                arrival_rate=0.0, concurrency_schedule=None,
                segments=[dict(controller=e.name, t0=e.t0, t1=e.t1) for e in episodes],
                window_wall_t0=t_start, window_wall_t1=time.time(),
                idle_power_w=cfg["p_idle"], vllm=emeta, table_json=args.table_json,
                table_meta={k: v for k, v in (table_meta or {}).items() if k != "rows"},
                args=vars(args))
    with open(os.path.join(args.run_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    analyze(args.run_dir)


if __name__ == "__main__":
    main()
