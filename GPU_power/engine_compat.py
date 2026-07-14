#!/usr/bin/env python3
"""
engine_compat.py

Thin compatibility shim over vLLM's async engine so the load driver doesn't care
whether the installed vLLM uses the V1 (AsyncLLM) or V0 (AsyncLLMEngine) API.

Both expose an async-generator `.generate(prompt, sampling_params, request_id)`
yielding RequestOutput; only construction differs. We detect and adapt.

The tokenizer/CUDA-version compatibility shims live in energy_profile_vllm.py and
are imported (and applied) before vLLM is touched.
"""

import os
from typing import Any, List, Optional

# Reuse the cluster compat shims + engine-kwargs helper already in the repo.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
# CRITICAL for iter_logger: the scheduler monkeypatch only sees the engine core
# if it runs in THIS process. V1 defaults to a separate engine-core process, so
# disable that. NVML/DRAM sampling read the physical GPU and are unaffected.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


def build_async_engine(args):
    """Return (engine, meta) where meta carries version/config for run_meta.json."""
    from energy_profile_vllm import (
        _install_transformers_tokenizer_compat, vllm_engine_kwargs,
    )
    _install_transformers_tokenizer_compat()

    import vllm
    vllm_version = getattr(vllm, "__version__", "unknown")

    from vllm import AsyncEngineArgs
    extra = vllm_engine_kwargs(args)  # gemma rope_theta override etc.

    engine_args = AsyncEngineArgs(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        trust_remote_code=True,
        disable_log_stats=False,
        disable_log_requests=True,
        **extra,
    )

    engine = None
    api = None
    # V1 first.
    try:
        from vllm.v1.engine.async_llm import AsyncLLM
        engine = AsyncLLM.from_engine_args(engine_args)
        api = "v1"
    except Exception:
        engine = None
    if engine is None:
        from vllm import AsyncLLMEngine
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        api = "v0"

    meta = {
        "vllm_version": vllm_version,
        "engine_api": api,
        "max_num_seqs": getattr(engine_args, "max_num_seqs", None),
        "block_size": getattr(engine_args, "block_size", None),
    }
    return engine, meta


def make_sampling_params(output_len: int, force_exact: bool = True):
    from vllm import SamplingParams
    kw = dict(max_tokens=output_len, temperature=0.0)
    if force_exact:
        # Force exactly output_len tokens so decode-window accounting is clean.
        kw["ignore_eos"] = True
        try:
            SamplingParams(min_tokens=1)  # probe support
            kw["min_tokens"] = output_len
        except Exception:
            pass
    return SamplingParams(**kw)


def build_prompt(tokenizer, input_len: Optional[int], text: Optional[str]) -> Any:
    """Controlled-length token prompt if input_len given, else a text prompt."""
    if input_len and tokenizer is not None:
        base = tokenizer("the quick brown fox jumps over the lazy dog ",
                         add_special_tokens=False)["input_ids"]
        if not base:
            base = [1]
        ids = (base * (input_len // len(base) + 1))[:input_len]
        # dict prompt form is accepted across recent vLLM versions
        return {"prompt_token_ids": ids}
    return text if text is not None else "Hello"


async def drain_generate(engine, prompt, sampling_params, request_id) -> dict:
    """Run one request to completion; return token/timing summary."""
    import time
    t0 = time.time()
    last = None
    async for out in engine.generate(prompt, sampling_params, request_id):
        last = out
    t1 = time.time()

    prompt_tokens = 0
    gen_tokens = 0
    if last is not None:
        try:
            prompt_tokens = len(last.prompt_token_ids or [])
        except Exception:
            pass
        try:
            gen_tokens = sum(len(o.token_ids) for o in last.outputs)
        except Exception:
            pass
    return {"t0": t0, "t1": t1, "prompt_tokens": prompt_tokens,
            "gen_tokens": gen_tokens, "request_id": request_id}
