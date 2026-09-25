#!/usr/bin/env python3
"""
ncu_decode_probe.py

Run a batch=1 vLLM decode with ONLY the steady-state decode inside the profiled
region, for `ncu --profile-from-start off`.

Why this exists rather than using ncu_dram_check.py --mode generate directly:
profiling the whole process also captures (a) the ~15 GB of weight upload during
model load and (b) FlashInfer's JIT autotuner, which benchmarks many kernel
variants. Both are DRAM traffic that has nothing to do with a decode step, and
under kernel replay the autotuner alone ran for 30+ minutes. Wrapping only the
measured generate in cudaProfilerStart/Stop excludes both.

The warmup generate before the marker is what forces load, JIT and autotuning to
happen OUTSIDE the profiled window.

Writes n_tokens.txt (tokens actually generated inside the profiled region).
"""
import argparse
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # keep kernels in-proc
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-7B-Instruct")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--run_dir", default="logs/B200_ncu")
    ap.add_argument("--prompt_len", type=int, default=8)
    ap.add_argument("--gen_tokens", type=int, default=8)
    ap.add_argument("--max_model_len", type=int, default=2048)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.6)
    args = ap.parse_args()
    os.makedirs(args.run_dir, exist_ok=True)

    import torch
    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, dtype=args.dtype, enforce_eager=True,
              max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization)
    fixed = dict(ignore_eos=True, temperature=0.0)
    # DISTINCT prompts for warmup vs measurement. vLLM enables automatic prefix
    # caching, so reusing the same token ids would let the measured generate hit
    # the warmup's cached KV and skip the prefill entirely -- silently changing
    # the number of weight sweeps in the profiled region (9 -> 8, an 11% error).
    warm   = {"prompt_token_ids": list(range(1000, 1000 + args.prompt_len))}
    prompt = {"prompt_token_ids": list(range(10, 10 + args.prompt_len))}

    # Warmup OUTSIDE the profiled region: forces JIT, autotuning and any lazy
    # allocation to happen before the counters start.
    llm.generate(warm, SamplingParams(max_tokens=4, min_tokens=4, **fixed))
    torch.cuda.synchronize()

    torch.cuda.profiler.start()
    out = llm.generate(prompt, SamplingParams(max_tokens=args.gen_tokens,
                                              min_tokens=args.gen_tokens, **fixed))
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()

    n = len(out[0].outputs[0].token_ids)
    with open(os.path.join(args.run_dir, "n_tokens.txt"), "w") as f:
        f.write(str(n))
    print(f"[probe] profiled region: prompt_len={args.prompt_len} "
          f"generated={n} tokens (1 prefill + {n} decode passes)")


if __name__ == "__main__":
    main()
