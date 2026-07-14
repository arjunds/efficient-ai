#!/usr/bin/env python3
"""
probe_env.py

Environment probe for the offered-load energy harness. Reports exactly what the
harness needs and what's missing, so the RunAI-vs-SLURM decision and the
version-specific hooks (iter_logger scheduler API, DRAM counter field) can be
confirmed against ground truth instead of assumptions.

Dependency-light and Python 3.6-safe: every heavy import is guarded, so it still
produces a useful report on a bare node. Writes probe_result.json.

  python probe_env.py                 # full report
  python probe_env.py --dump-vllm-api # also print scheduler class + schedule sig
"""

import json
import os
import shutil
import subprocess
import sys


def sh(cmd, timeout=20):
    try:
        out = subprocess.run(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, timeout=timeout)
        return out.returncode, out.stdout.decode(errors="replace").strip()
    except Exception as e:
        return -1, str(e)


def probe_python():
    return {"version": sys.version.split()[0], "executable": sys.executable}


def probe_cmd(name):
    path = shutil.which(name)
    info = {"path": path}
    if path and name in ("nvidia-smi", "dcgmi", "ncu"):
        if name == "nvidia-smi":
            rc, out = sh([path, "--query-gpu=name,power.limit,memory.total",
                          "--format=csv,noheader"])
            info["query"] = out
        elif name == "ncu":
            rc, out = sh([path, "--version"])
            info["version"] = out.splitlines()[0] if out else ""
        elif name == "dcgmi":
            rc, out = sh([path, "dmon", "-e", "1005", "-c", "2"])
            info["dram_active_sample"] = out
    return info


def probe_import(mod):
    try:
        m = __import__(mod)
        return {"ok": True, "version": getattr(m, "__version__", "?")}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def probe_nvml():
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(h)
        name = name.decode() if isinstance(name, bytes) else name
        pw = pynvml.nvmlDeviceGetPowerUsage(h) * 1e-3
        pynvml.nvmlShutdown()
        return {"ok": True, "gpu_name": name, "power_w": pw}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def probe_pypi():
    # Can this node reach PyPI to pip-install torch/vllm into ~/.local?
    rc, out = sh([sys.executable, "-m", "pip", "index", "versions", "pip"], 25)
    if rc != 0:
        rc2, _ = sh([sys.executable, "-c",
                     "import urllib.request,sys;"
                     "urllib.request.urlopen('https://pypi.org',timeout=10);"
                     "print('ok')"], 25)
        return {"reachable": rc2 == 0, "via": "urllib"}
    return {"reachable": True, "via": "pip index"}


def probe_vllm_api(dump=False):
    info = {}
    try:
        import vllm
        info["vllm_version"] = getattr(vllm, "__version__", "?")
    except Exception as e:
        info["import_error"] = str(e)
        return info
    # Locate scheduler class (mirrors iter_logger._SCHED_PATHS).
    import importlib
    import inspect
    for mod_path, cls_name, ver in [
        ("vllm.v1.core.sched.scheduler", "Scheduler", "v1"),
        ("vllm.v1.core.scheduler", "Scheduler", "v1"),
        ("vllm.core.scheduler", "Scheduler", "v0"),
    ]:
        try:
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name, None)
            if cls is not None and hasattr(cls, "schedule"):
                info["scheduler_class"] = "%s.%s" % (mod_path, cls_name)
                info["engine_api"] = ver
                if dump:
                    try:
                        info["schedule_signature"] = str(
                            inspect.signature(cls.schedule))
                    except Exception:
                        pass
                break
        except Exception:
            continue
    for cand in ("vllm.v1.engine.async_llm", "vllm.engine.async_llm_engine"):
        try:
            importlib.import_module(cand)
            info["async_engine_module"] = cand
            break
        except Exception:
            continue
    return info


def main():
    dump = "--dump-vllm-api" in sys.argv
    report = {
        "hostname": sh(["hostname"])[1],
        "in_slurm": bool(os.environ.get("SLURM_JOB_ID")),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python": probe_python(),
        "container_runtimes": {n: shutil.which(n) for n in
                               ("apptainer", "singularity", "enroot", "podman",
                                "docker", "runai")},
        "commands": {n: probe_cmd(n) for n in
                     ("nvidia-smi", "dcgmi", "ncu")},
        "imports": {m: probe_import(m) for m in
                    ("torch", "vllm", "transformers", "pynvml", "pandas",
                     "matplotlib")},
        "nvml": probe_nvml(),
        "pypi": probe_pypi(),
        "vllm_api": probe_vllm_api(dump=dump),
    }
    print(json.dumps(report, indent=2))
    try:
        with open("probe_result.json", "w") as f:
            json.dump(report, f, indent=2)
        print("\n[saved] probe_result.json")
    except Exception:
        pass

    # One-line verdict for quick scanning.
    ok_gpu = report["nvml"].get("ok")
    ok_vllm = report["imports"]["vllm"].get("ok")
    ok_dcgm = report["commands"]["dcgmi"].get("path") is not None
    print("\nVERDICT: gpu=%s vllm=%s dcgmi=%s -> %s" % (
        ok_gpu, ok_vllm, ok_dcgm,
        "READY" if (ok_gpu and ok_vllm) else "NEEDS SETUP (see report)"))


if __name__ == "__main__":
    main()
