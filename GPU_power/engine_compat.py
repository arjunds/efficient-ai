#!/usr/bin/env python3
"""
engine_compat.py

Thin compatibility shim over vLLM's async engine. Confirmed against the cluster's
vllm 0.10.2 (V1 engine): AsyncLLM.from_engine_args accepts a `stat_loggers` list
of factories Callable[[VllmConfig, int], StatLoggerBase]; the async engine runs
its core in a separate process (multiprocessing is mandatory), so per-iteration
logging goes through the stat-logger API (iter_logger.make_iter_logger_factory),
NOT an in-process monkeypatch.
"""

import os

# NOTE: do NOT set VLLM_ENABLE_V1_MULTIPROCESSING=0 — the V1 AsyncLLM requires the
# multiprocessing engine-core client and setting that var makes core init fail.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")


def _install_transformers_tokenizer_compat():
    """vLLM expects all_special_tokens_extended, which newer transformers
    tokenizers hide behind __getattr__. Inlined so the load path has NO dependency
    on the old pandas/datasets-importing modules."""
    from transformers import (PreTrainedTokenizer, PreTrainedTokenizerBase,
                              PreTrainedTokenizerFast)
    from transformers.tokenization_utils_base import (
        PreTrainedTokenizerBase as TokenizerBaseImpl)
    for cls in (PreTrainedTokenizerBase, PreTrainedTokenizer, PreTrainedTokenizerFast):
        if not hasattr(cls, "all_special_tokens_extended"):
            cls.all_special_tokens_extended = property(
                lambda self: self.all_special_tokens)
    original_getattr = TokenizerBaseImpl.__getattr__
    if getattr(original_getattr, "_vllm_compat_patched", False):
        return

    def compat_getattr(self, key):
        if key == "all_special_tokens_extended":
            return self.all_special_tokens
        return original_getattr(self, key)

    compat_getattr._vllm_compat_patched = True
    TokenizerBaseImpl.__getattr__ = compat_getattr


def vllm_engine_kwargs(args) -> dict:
    kwargs = {}
    if getattr(args, "model", None) and "gemma" in args.model.lower():
        kwargs["hf_overrides"] = {"rope_theta": 10000.0}
    return kwargs


def build_async_engine(args, iter_log_csv=None):
    """Return (engine, meta). If iter_log_csv is given, a per-iteration stat
    logger is attached via stat_loggers."""
    _install_transformers_tokenizer_compat()

    import vllm
    vllm_version = getattr(vllm, "__version__", "unknown")

    from vllm import AsyncEngineArgs
    extra = vllm_engine_kwargs(args)

    engine_args = AsyncEngineArgs(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        trust_remote_code=True,
        **extra,
    )

    stat_loggers = None
    if iter_log_csv:
        from iter_logger import make_iter_logger_factory
        stat_loggers = [make_iter_logger_factory(iter_log_csv)]

    try:
        from vllm.v1.engine.async_llm import AsyncLLM
        _v1 = True
    except Exception:
        _v1 = False

    if _v1:
        kw = {}
        if stat_loggers is not None:
            kw["stat_loggers"] = stat_loggers
        engine = AsyncLLM.from_engine_args(engine_args, **kw)
        api = "v1"
    else:
        from vllm import AsyncLLMEngine
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        api = "v0"

    meta = {"vllm_version": vllm_version, "engine_api": api,
            "max_num_seqs": getattr(engine_args, "max_num_seqs", None),
            "block_size": getattr(engine_args, "block_size", None)}
    return engine, meta


def make_sampling_params(output_len: int, force_exact: bool = True):
    from vllm import SamplingParams
    kw = dict(max_tokens=output_len, temperature=0.0)
    if force_exact:
        kw["ignore_eos"] = True
        try:
            SamplingParams(min_tokens=1)
            kw["min_tokens"] = output_len
        except Exception:
            pass
    return SamplingParams(**kw)


def build_prompt(tokenizer, input_len, text=None):
    """Controlled-length token prompt if input_len given, else a text prompt."""
    if input_len and tokenizer is not None:
        base = tokenizer("the quick brown fox jumps over the lazy dog ",
                         add_special_tokens=False)["input_ids"]
        if not base:
            base = [1]
        ids = (base * (input_len // len(base) + 1))[:input_len]
        return {"prompt_token_ids": ids}
    return text if text is not None else "Hello"


async def drain_generate(engine, prompt, sampling_params, request_id) -> dict:
    """Run one request to completion; return token/timing summary."""
    import time
    t0 = time.time()
    last = None
    t_first = None          # additive: wall time the first generated token arrived
    async for out in engine.generate(prompt, sampling_params, request_id):
        if t_first is None:
            try:
                if out.outputs and len(out.outputs[0].token_ids) > 0:
                    t_first = time.time()
            except Exception:
                t_first = time.time()
        last = out
    t1 = time.time()
    prompt_tokens = gen_tokens = 0
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
            "gen_tokens": gen_tokens, "request_id": request_id,
            "t_first": t_first}
