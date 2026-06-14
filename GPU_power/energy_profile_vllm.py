#!/usr/bin/env python3
"""
energy_profile_vllm.py

Energy + throughput benchmarking using vLLM offline inference.
Reuses power logging and data helpers from energy_profile.py.

Example energy:
  python energy_profile_vllm.py \
      --mode energy \
      --model Qwen/Qwen2-7B-Instruct \
      --task alpaca \
      --dtype float16 \
      --limit 50 \
      --max_new_tokens 64 \
      --run_dir logs/vllm_qwen2_alpaca_float16

Example NCU/intensity:
  python energy_profile_vllm.py --mode ncu --model Qwen/Qwen2-7B-Instruct --task alpaca --run_dir logs/vllm_qwen2_alpaca_float16
  python energy_profile_vllm.py --mode intensity --run_dir logs/vllm_qwen2_alpaca_float16
"""

import argparse
import json
import os
import subprocess
import sys
import time
from typing import List, Optional

# Avoid FlashInfer sampler JIT compilation on cluster nodes without nvcc.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
# Avoid vLLM/Torch internals invoking TorchInductor on clusters where Triton
# helper compilation cannot link libcuda cleanly.
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast
from transformers.tokenization_utils_base import PreTrainedTokenizerBase as TokenizerBaseImpl


def _install_transformers_tokenizer_compat() -> None:
    """
    vLLM 0.10.x expects all_special_tokens_extended, which newer transformers
    tokenizers can hide behind __getattr__. Add a narrow fallback before vLLM
    imports and initializes tokenizers.
    """
    for tokenizer_cls in (PreTrainedTokenizerBase, PreTrainedTokenizer, PreTrainedTokenizerFast):
        if not hasattr(tokenizer_cls, "all_special_tokens_extended"):
            tokenizer_cls.all_special_tokens_extended = property(
                lambda self: self.all_special_tokens
            )

    original_getattr = TokenizerBaseImpl.__getattr__

    if getattr(original_getattr, "_vllm_compat_patched", False):
        return

    def compat_getattr(self, key):
        if key == "all_special_tokens_extended":
            return self.all_special_tokens
        return original_getattr(self, key)

    compat_getattr._vllm_compat_patched = True
    TokenizerBaseImpl.__getattr__ = compat_getattr


_install_transformers_tokenizer_compat()

try:
    from vllm import LLM, SamplingParams
except ImportError as exc:
    if "libcudart.so.13" in str(exc):
        raise ImportError(
            "vLLM failed to import because its installed binary expects CUDA 13 "
            "(missing libcudart.so.13). This cluster node exposes CUDA 12.7, so "
            "install a CUDA 12.x-compatible vLLM build/version."
        ) from exc
    raise

from energy_profile import (
    load_task_examples,
    start_smi_logger,
    stop_smi_logger,
    parse_power_samples,
    energy_j_from_samples,
    write_json,
    maybe_mkdir,
    now_tag,
    requested_ncu_metrics,
    export_ncu_rep_to_csv,
    newest_file,
    build_intensity_results,
    safe_mean,
    evaluate_example,
)


def load_vllm_engine(args) -> LLM:
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN")

    kwargs = vllm_engine_kwargs(args)
    if token:
        kwargs["download_dir"] = None  # use default
    os.environ.setdefault("HF_TOKEN", token or "")

    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        trust_remote_code=True,
        **kwargs,
    )
    return llm


def vllm_engine_kwargs(args) -> dict:
    kwargs = {}

    # vLLM 0.10.x expects GemmaConfig.rope_theta, but some transformers
    # versions omit it for google/gemma-7b. Gemma's default RoPE theta is 10000.
    if args.model and "gemma" in args.model.lower():
        kwargs["hf_overrides"] = {"rope_theta": 10000.0}

    return kwargs


def generate_vllm_single(
    llm: LLM,
    prompt: str,
    max_new_tokens: int,
) -> dict:
    """Generate one prompt at a time so we can measure per-example timing."""
    params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0,
    )

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    outputs = llm.generate([prompt], params)

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    out = outputs[0]
    completion = out.outputs[0]

    return {
        "text": completion.text,
        "prompt_tokens": len(out.prompt_token_ids),
        "generated_tokens": len(completion.token_ids),
        "elapsed_s": t1 - t0,
    }


def generate_vllm_batch(
    llm: LLM,
    prompts: List[str],
    max_new_tokens: int,
) -> dict:
    """Generate a batch of prompts in one vLLM call."""
    params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0,
    )

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    outputs = llm.generate(prompts, params)

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    texts = []
    prompt_tokens_total = 0
    gen_tokens_total = 0
    per_example_gen_tokens = []

    for out in outputs:
        completion = out.outputs[0]
        texts.append(completion.text)
        prompt_tokens_total += len(out.prompt_token_ids)
        n_gen = len(completion.token_ids)
        gen_tokens_total += n_gen
        per_example_gen_tokens.append(n_gen)

    return {
        "texts": texts,
        "elapsed_s": t1 - t0,
        "prompt_tokens_total": prompt_tokens_total,
        "generated_tokens_total": gen_tokens_total,
        "per_example_gen_tokens": per_example_gen_tokens,
    }


def cuda_profiler_start(enabled: bool) -> None:
    if not enabled or not torch.cuda.is_available():
        return

    try:
        torch.cuda.cudart().cudaProfilerStart()
    except Exception as exc:
        print(f"[warn] cudaProfilerStart failed: {exc}", flush=True)


def cuda_profiler_stop(enabled: bool) -> None:
    if not enabled or not torch.cuda.is_available():
        return

    try:
        torch.cuda.cudart().cudaProfilerStop()
    except Exception as exc:
        print(f"[warn] cudaProfilerStop failed: {exc}", flush=True)


def run_energy_pass(args) -> dict:
    maybe_mkdir(args.run_dir)

    under_ncu = os.environ.get("UNDER_NCU", "") == "1"

    power_csv = os.path.join(args.run_dir, "power.csv")
    anchor_path = os.path.join(args.run_dir, "time_anchor.json")
    window_path = os.path.join(args.run_dir, "window_times.json")
    results_path = os.path.join(args.run_dir, "results_under_ncu.json" if under_ncu else "results.json")
    predictions_path = os.path.join(args.run_dir, "predictions.jsonl")

    examples = load_task_examples(args.task, args.limit)
    if not examples:
        raise RuntimeError(f"No examples loaded for task={args.task}")

    llm = load_vllm_engine(args)

    smi_proc = None
    smi_file = None
    dt = args.power_interval_ms / 1000.0

    if not under_ncu:
        smi_proc, smi_file, dt = start_smi_logger(power_csv, args.gpu_id, args.power_interval_ms)

        write_json(anchor_path, {
            "wall_time_s": time.time(),
            "perf_counter_s": time.perf_counter(),
        })

    total_examples = 0
    total_prompt_tokens = 0
    total_generated_tokens = 0

    end_to_end_time_list: List[float] = []
    tokens_per_sec_list: List[float] = []
    example_accs: List[bool] = []

    win_t0_wall: Optional[float] = None
    win_t1_wall: Optional[float] = None

    overall_t0 = time.time()

    pred_f = None
    if args.save_predictions and not under_ncu:
        pred_f = open(predictions_path, "w")

    try:
        if args.batch_size == 1:
            for i, ex in enumerate(examples):
                prompt = ex["prompt"]
                gold = ex["answer"]

                in_nvtx = (
                    args.nvtx_count > 0
                    and i >= args.nvtx_start
                    and i < args.nvtx_start + args.nvtx_count
                )

                if in_nvtx and torch.cuda.is_available():
                    if win_t0_wall is None:
                        win_t0_wall = time.time()
                    cuda_profiler_start(under_ncu)
                    torch.cuda.nvtx.range_push("PROFILE_WINDOW")

                try:
                    out = generate_vllm_single(llm, prompt, args.max_new_tokens)
                finally:
                    if in_nvtx and torch.cuda.is_available():
                        torch.cuda.nvtx.range_pop()
                        cuda_profiler_stop(under_ncu)
                        win_t1_wall = time.time()

                prompt_tokens = out["prompt_tokens"]
                gen_tokens = out["generated_tokens"]
                e2e_s = out["elapsed_s"]
                pred_text = out["text"]

                total_examples += 1
                total_prompt_tokens += prompt_tokens
                total_generated_tokens += gen_tokens
                end_to_end_time_list.append(e2e_s)

                if e2e_s > 0:
                    tokens_per_sec_list.append(gen_tokens / e2e_s)

                acc = evaluate_example(args.task, pred_text, gold)
                if acc is not None:
                    example_accs.append(acc)

                if pred_f is not None:
                    pred_f.write(json.dumps({
                        "idx": i,
                        "task": args.task,
                        "prompt": prompt,
                        "prediction": pred_text,
                        "gold": gold,
                        "prompt_tokens": prompt_tokens,
                        "generated_tokens": gen_tokens,
                        "end_to_end_time_s": e2e_s,
                        "correct": acc,
                    }) + "\n")

                if total_examples % 10 == 0:
                    elapsed = time.time() - overall_t0
                    agg_tps = total_generated_tokens / max(elapsed, 1e-9)
                    print(f"[{total_examples}/{len(examples)}] agg_tok_per_s={agg_tps:.2f}", flush=True)

        else:
            all_prompts = [ex["prompt"] for ex in examples]
            all_answers = [ex["answer"] for ex in examples]

            win_t0_wall = time.time()
            if args.nvtx_count > 0 and torch.cuda.is_available():
                cuda_profiler_start(under_ncu)
                torch.cuda.nvtx.range_push("PROFILE_WINDOW")

            try:
                out = generate_vllm_batch(llm, all_prompts, args.max_new_tokens)
            finally:
                if args.nvtx_count > 0 and torch.cuda.is_available():
                    torch.cuda.nvtx.range_pop()
                    cuda_profiler_stop(under_ncu)
                win_t1_wall = time.time()

            elapsed_s = out["elapsed_s"]
            total_examples = len(examples)
            total_prompt_tokens = out["prompt_tokens_total"]
            total_generated_tokens = out["generated_tokens_total"]
            end_to_end_time_list.append(elapsed_s)

            if elapsed_s > 0:
                tokens_per_sec_list.append(total_generated_tokens / elapsed_s)

            for j, pred_text in enumerate(out["texts"]):
                acc = evaluate_example(args.task, pred_text, all_answers[j])
                if acc is not None:
                    example_accs.append(acc)

                if pred_f is not None:
                    pred_f.write(json.dumps({
                        "idx": j,
                        "task": args.task,
                        "prompt": all_prompts[j],
                        "prediction": pred_text,
                        "gold": all_answers[j],
                        "generated_tokens": out["per_example_gen_tokens"][j],
                        "end_to_end_time_s": elapsed_s,
                        "correct": acc,
                    }) + "\n")

    finally:
        total_wall_seconds = time.time() - overall_t0
        stop_smi_logger(smi_proc, smi_file)

        if pred_f is not None:
            pred_f.close()

    if not under_ncu:
        write_json(window_path, {
            "nvtx_start": args.nvtx_start,
            "nvtx_count": args.nvtx_count,
            "window_wall_t0": win_t0_wall,
            "window_wall_t1": win_t1_wall,
        })

    results = {
        "backend": "vllm",
        "model": args.model,
        "task": args.task,
        "dtype": args.dtype,
        "device": "cuda",
        "limit": args.limit,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "enforce_eager": args.enforce_eager,
        "tensor_parallel_size": args.tensor_parallel_size,
        "examples": total_examples,
        "seconds": total_wall_seconds,
        "prompt_tokens": total_prompt_tokens,
        "generated_tokens": total_generated_tokens,
        "avg_ttft_s": None,
        "avg_decode_time_s": None,
        "avg_end_to_end_time_s": safe_mean(end_to_end_time_list),
        "avg_tokens_per_sec": safe_mean(tokens_per_sec_list),
        "aggregate_tokens_per_sec": (
            total_generated_tokens / max(total_wall_seconds, 1e-9)
            if total_generated_tokens > 0 else None
        ),
        "accuracy": (
            float(sum(1 for x in example_accs if x)) / len(example_accs)
            if example_accs else None
        ),
        "num_accuracy_examples": len(example_accs),
        "power_csv": power_csv if os.path.exists(power_csv) else None,
        "time_anchor_json": anchor_path if os.path.exists(anchor_path) else None,
        "window_times_json": window_path if os.path.exists(window_path) else None,
        "predictions_jsonl": predictions_path if args.save_predictions and not under_ncu else None,
        "sampling_dt_s": dt,
        "under_ncu": under_ncu,
    }

    if not under_ncu and os.path.exists(power_csv):
        samples = parse_power_samples(power_csv)
        powers = [p for _, p in samples]

        energy_total_j = energy_j_from_samples(powers, dt)
        avg_power_w = float(sum(powers) / len(powers))

        results["energy_total_j"] = energy_total_j
        results["avg_power_w"] = avg_power_w
        results["energy_per_generated_token_j"] = (
            energy_total_j / total_generated_tokens if total_generated_tokens > 0 else None
        )

    write_json(results_path, results)
    print(f"[saved] {results_path}", flush=True)

    return results


def build_ncu_child_command(args, ncu_base: str) -> List[str]:
    cmd = [
        "ncu",
        "--target-processes", "all",
        "--replay-mode", args.ncu_replay_mode,
        "--profile-from-start", args.ncu_profile_from_start,
        "--launch-count", str(args.ncu_launch_count),
        "--metrics", ",".join(requested_ncu_metrics()),
        "-o", ncu_base,
        sys.executable,
        os.path.abspath(__file__),
        "--mode", "energy",
        "--model", args.model,
        "--task", args.task,
        "--dtype", args.dtype,
        "--limit", str(args.limit),
        "--batch_size", str(args.batch_size),
        "--max_new_tokens", str(args.max_new_tokens),
        "--run_dir", args.run_dir,
        "--gpu_id", str(args.gpu_id),
        "--power_interval_ms", str(args.power_interval_ms),
        "--nvtx_start", str(args.nvtx_start),
        "--nvtx_count", str(args.nvtx_count),
        "--tensor_parallel_size", str(args.tensor_parallel_size),
        "--gpu_memory_utilization", str(args.gpu_memory_utilization),
        "--max_model_len", str(args.max_model_len),
    ]

    if args.enforce_eager:
        cmd.append("--enforce_eager")

    if args.save_predictions:
        cmd.append("--save_predictions")

    return cmd


def run_ncu_pass(args) -> dict:
    maybe_mkdir(args.run_dir)

    if not args.enforce_eager:
        args.enforce_eager = True
        print("[ncu] enabling --enforce_eager for vLLM profiling", flush=True)

    ncu_base = os.path.join(args.run_dir, f"ncu_{now_tag()}")
    cmd = build_ncu_child_command(args, ncu_base)

    env = os.environ.copy()
    env["UNDER_NCU"] = "1"
    env.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    print("[ncu] launching:")
    print("  " + " ".join(cmd), flush=True)

    ret = subprocess.run(cmd, env=env)

    rep_path = f"{ncu_base}.ncu-rep"
    csv_path = None

    if os.path.exists(rep_path):
        csv_path = export_ncu_rep_to_csv(rep_path)
        print(f"[ncu] exported csv -> {csv_path}", flush=True)
    else:
        newest_rep = newest_file(args.run_dir, ".ncu-rep", prefix="ncu_")
        if newest_rep is not None:
            csv_path = export_ncu_rep_to_csv(newest_rep)
            print(f"[ncu] exported newest csv -> {csv_path}", flush=True)
        else:
            print("[warn] no NCU report found", flush=True)

    summary = {
        "returncode": ret.returncode,
        "ncu_base": ncu_base,
        "ncu_rep": rep_path if os.path.exists(rep_path) else None,
        "ncu_csv": csv_path,
        "status": "ok" if ret.returncode == 0 and csv_path is not None else "failed_or_partial",
    }

    out_path = os.path.join(args.run_dir, "ncu_status.json")
    write_json(out_path, summary)
    print(f"[saved] {out_path}", flush=True)

    if ret.returncode != 0:
        print(f"[warn] NCU exited with return code {ret.returncode}", flush=True)

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="vLLM energy profiling")

    ap.add_argument("--mode", default="energy", choices=["energy", "ncu", "intensity"])

    ap.add_argument("--model")
    ap.add_argument("--task", choices=["alpaca", "sharegpt", "math", "multi_turn"])

    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=0,
                    help="0 = send all prompts in one vLLM call (recommended); "
                         "1 = one-at-a-time for per-example timing")
    ap.add_argument("--max_new_tokens", type=int, default=64)

    ap.add_argument("--run_dir", default="logs/vllm_run")
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--power_interval_ms", type=int, default=20)

    ap.add_argument("--nvtx_start", type=int, default=0)
    ap.add_argument("--nvtx_count", type=int, default=10)

    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--max_model_len", type=int, default=2048)
    ap.add_argument("--enforce_eager", action="store_true", default=True,
                    help="Disable CUDA graphs/TorchInductor compilation")
    ap.add_argument("--no_enforce_eager", dest="enforce_eager", action="store_false",
                    help="Allow vLLM CUDA graphs/TorchInductor compilation")

    ap.add_argument("--ncu_launch_count", type=int, default=20000)
    ap.add_argument("--ncu_replay_mode", default="kernel", choices=["application", "kernel"])
    ap.add_argument("--ncu_profile_from_start", default="no", choices=["yes", "no"])

    ap.add_argument("--save_predictions", action="store_true")

    return ap


def validate_args(args) -> None:
    if args.mode in ("energy", "ncu"):
        missing = []
        if not args.model:
            missing.append("--model")
        if not args.task:
            missing.append("--task")

        if missing:
            raise ValueError(f"{args.mode} mode requires: {', '.join(missing)}")

    if args.mode == "intensity" and not args.run_dir:
        raise ValueError("intensity mode requires --run_dir")


def main() -> None:
    args = build_arg_parser().parse_args()
    validate_args(args)

    if args.batch_size == 0:
        args.batch_size = args.limit

    if args.mode == "energy":
        run_energy_pass(args)
    elif args.mode == "ncu":
        run_ncu_pass(args)
    elif args.mode == "intensity":
        build_intensity_results(args.run_dir)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
