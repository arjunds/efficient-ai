#!/usr/bin/env python3
"""
energy_profile_vllm.py

Energy + throughput benchmarking using vLLM offline inference.
Reuses power logging and data helpers from energy_profile.py.

Example:
  python energy_profile_vllm.py \
      --model Qwen/Qwen2-7B-Instruct \
      --task alpaca \
      --dtype float16 \
      --limit 50 \
      --max_new_tokens 64 \
      --run_dir logs/vllm_qwen2_alpaca_float16
"""

import argparse
import json
import os
import time
from typing import List, Optional

import torch
from vllm import LLM, SamplingParams

from energy_profile import (
    load_task_examples,
    start_smi_logger,
    stop_smi_logger,
    parse_power_samples,
    energy_j_from_samples,
    write_json,
    maybe_mkdir,
    safe_mean,
    evaluate_example,
)


def load_vllm_engine(args) -> LLM:
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN")

    kwargs = {}
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
    )
    return llm


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


def run_energy_pass(args) -> dict:
    maybe_mkdir(args.run_dir)

    power_csv = os.path.join(args.run_dir, "power.csv")
    anchor_path = os.path.join(args.run_dir, "time_anchor.json")
    window_path = os.path.join(args.run_dir, "window_times.json")
    results_path = os.path.join(args.run_dir, "results.json")
    predictions_path = os.path.join(args.run_dir, "predictions.jsonl")

    examples = load_task_examples(args.task, args.limit)
    if not examples:
        raise RuntimeError(f"No examples loaded for task={args.task}")

    llm = load_vllm_engine(args)

    smi_proc = None
    smi_file = None
    dt = args.power_interval_ms / 1000.0

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
    if args.save_predictions:
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
                    torch.cuda.nvtx.range_push("PROFILE_WINDOW")

                try:
                    out = generate_vllm_single(llm, prompt, args.max_new_tokens)
                finally:
                    if in_nvtx and torch.cuda.is_available():
                        torch.cuda.nvtx.range_pop()
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
                torch.cuda.nvtx.range_push("PROFILE_WINDOW")

            try:
                out = generate_vllm_batch(llm, all_prompts, args.max_new_tokens)
            finally:
                if args.nvtx_count > 0 and torch.cuda.is_available():
                    torch.cuda.nvtx.range_pop()
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
        "predictions_jsonl": predictions_path if args.save_predictions else None,
        "sampling_dt_s": dt,
    }

    if os.path.exists(power_csv):
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


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="vLLM energy profiling")

    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, choices=["alpaca", "sharegpt", "math", "multi_turn"])

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
    ap.add_argument("--enforce_eager", action="store_true",
                    help="Disable CUDA graphs (required for NCU profiling)")

    ap.add_argument("--save_predictions", action="store_true")

    return ap


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.batch_size == 0:
        args.batch_size = args.limit

    run_energy_pass(args)


if __name__ == "__main__":
    main()
