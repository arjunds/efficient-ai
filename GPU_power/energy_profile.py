#!/usr/bin/env python3
"""
energy_profile_modular.py

Modular benchmark runner for LLM energy + Nsight Compute intensity.

Modes:
  energy      Run model generation, power logging, latency/token collection.
  ncu         Launch this same script under Nsight Compute and export CSV.
  intensity   Combine existing power/window files with newest NCU CSV.

Example energy:
  python energy_profile_modular.py \
      --mode energy \
      --model google/gemma-7b \
      --task alpaca \
      --dtype float16 \
      --limit 50 \
      --batch_size 1 \
      --max_new_tokens 64 \
      --run_dir logs/gemma_alpaca_float16

Example NCU:
  python energy_profile_modular.py \
      --mode ncu \
      --model google/gemma-7b \
      --task alpaca \
      --dtype float16 \
      --limit 50 \
      --batch_size 1 \
      --max_new_tokens 64 \
      --run_dir logs/gemma_alpaca_float16

Example intensity only:
  python energy_profile_modular.py \
      --mode intensity \
      --run_dir logs/gemma_alpaca_float16
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# -------------------------
# GENERAL HELPERS
# -------------------------

def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def maybe_mkdir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def safe_mean(xs: List[float]) -> Optional[float]:
    return float(sum(xs) / len(xs)) if xs else None


def safe_sum(xs: List[float]) -> float:
    return float(sum(xs)) if xs else 0.0


def batched(lst: List[dict], n: int):
    for i in range(0, len(lst), n):
        yield i, lst[i:i + n]


def read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def write_json(path: str, obj: dict) -> None:
    maybe_mkdir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def require_file(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {label}: {path}")


# -------------------------
# PROMPT FORMATTING
# -------------------------

def alpaca_prompt(ex: dict) -> str:
    instr = ex["instruction"]
    inp = ex.get("input", "")
    if inp and inp.strip():
        return f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n"
    return f"### Instruction:\n{instr}\n\n### Response:\n"


def sharegpt_prompt(ex: dict) -> str:
    conv = ex.get("conversations", [])
    if not conv:
        return ""

    pieces = []
    for turn in conv:
        role = turn.get("from", "user")
        value = turn.get("value", "")
        pieces.append(f"{role}: {value}")

    return "\n".join(pieces) + "\nassistant:"


def load_task_examples(task: str, limit: int) -> List[dict]:
    """
    Return examples in this shape:
      {
        "prompt": str,
        "answer": Optional[str],
        "meta": dict
      }
    """
    examples: List[dict] = []

    if task == "alpaca":
        ds = load_dataset("tatsu-lab/alpaca", split="train").select(range(limit))
        for ex in ds:
            examples.append({
                "prompt": alpaca_prompt(ex),
                "answer": None,
                "meta": {"source": "alpaca"},
            })

    elif task == "sharegpt":
        ds = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train").select(range(limit))
        for ex in ds:
            prompt = sharegpt_prompt(ex)
            if prompt.strip():
                examples.append({
                    "prompt": prompt,
                    "answer": None,
                    "meta": {"source": "sharegpt"},
                })

    elif task == "math":
        ds = load_dataset("gsm8k", "main", split="test").select(range(limit))
        for ex in ds:
            prompt = (
                "Solve the following math problem. "
                "Give the final answer clearly.\n\n"
                f"Question: {ex['question']}\nAnswer:"
            )
            examples.append({
                "prompt": prompt,
                "answer": ex["answer"],
                "meta": {"source": "gsm8k"},
            })

    elif task == "multi_turn":
        # Fallback: use ShareGPT as a multi-turn proxy.
        ds = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train").select(range(limit))
        for ex in ds:
            conv = ex.get("conversations", [])
            if len(conv) < 2:
                continue

            prompt = sharegpt_prompt(ex)
            if prompt.strip():
                examples.append({
                    "prompt": prompt,
                    "answer": None,
                    "meta": {"source": "multi_turn_proxy"},
                })

    else:
        raise ValueError(f"Unsupported task: {task}")

    return examples[:limit]


# -------------------------
# POWER LOGGING
# -------------------------

def find_nvidia_smi() -> str:
    path = shutil.which("nvidia-smi")
    if path:
        return path

    for candidate in (
        "/usr/bin/nvidia-smi",
        "/usr/local/cuda/bin/nvidia-smi",
        "/opt/bin/nvidia-smi",
    ):
        if os.path.exists(candidate) and os.access(candidate, os.X_OK):
            return candidate

    raise FileNotFoundError(
        "Could not find nvidia-smi. On a cluster, run this inside a GPU allocation "
        "and load the site CUDA/NVIDIA module if required."
    )


def start_smi_logger(log_path: str, gpu_id: int, interval_ms: int) -> Tuple[subprocess.Popen, Any, float]:
    cmd = [
        find_nvidia_smi(),
        "-i", str(gpu_id),
        "--query-gpu=timestamp,power.draw,utilization.gpu,utilization.memory,temperature.gpu",
        "--format=csv",
        "-lms", str(interval_ms),
    ]

    maybe_mkdir(os.path.dirname(log_path))
    f = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.DEVNULL)

    return proc, f, interval_ms / 1000.0


def stop_smi_logger(proc: Optional[subprocess.Popen], f: Optional[Any]) -> None:
    if proc is not None:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    if f is not None:
        try:
            f.close()
        except Exception:
            pass


def parse_power_samples(power_csv: str) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []

    with open(power_csv, newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError(f"{power_csv} is empty")

        pcol = next((c for c in reader.fieldnames if "power.draw" in c), None)
        tcol = next((c for c in reader.fieldnames if "timestamp" in c), None)

        if pcol is None or tcol is None:
            raise RuntimeError(f"Missing power/timestamp columns in {power_csv}")

        for row in reader:
            ts = (row.get(tcol, "") or "").strip()
            pw = (row.get(pcol, "") or "").replace(" W", "").strip()

            try:
                out.append((ts, float(pw)))
            except ValueError:
                continue

    if not out:
        raise RuntimeError(f"No valid power samples found in {power_csv}")

    return out


def energy_j_from_samples(powers_w: List[float], dt_s: float) -> float:
    return float(sum(p * dt_s for p in powers_w))


def parse_nvidia_smi_timestamp(ts: str) -> float:
    dt = datetime.strptime(ts.strip(), "%Y/%m/%d %H:%M:%S.%f")
    return dt.timestamp()


def integrate_energy_in_window(power_csv: str, t0_wall: float, t1_wall: float) -> dict:
    """
    Integrate power inside [t0_wall, t1_wall] with trapezoidal integration.
    """
    samples = parse_power_samples(power_csv)
    parsed = []

    for ts_str, p in samples:
        try:
            parsed.append((parse_nvidia_smi_timestamp(ts_str), p))
        except Exception:
            continue

    if len(parsed) < 2:
        return {
            "energy_window_j": None,
            "avg_power_window_w": None,
            "num_samples_in_window": 0,
        }

    in_range = [(t, p) for (t, p) in parsed if t0_wall <= t <= t1_wall]

    # Fallback: use a small neighborhood if exact timestamps do not align.
    if len(in_range) < 2:
        in_range = [(t, p) for (t, p) in parsed if (t0_wall - 1.0) <= t <= (t1_wall + 1.0)]

    if len(in_range) < 2:
        return {
            "energy_window_j": None,
            "avg_power_window_w": None,
            "num_samples_in_window": len(in_range),
        }

    energy_j = 0.0
    weighted_power_sum = 0.0
    total_dt = 0.0

    for i in range(len(in_range) - 1):
        t_a, p_a = in_range[i]
        t_b, p_b = in_range[i + 1]

        seg_start = max(t_a, t0_wall)
        seg_end = min(t_b, t1_wall)

        if seg_end <= seg_start:
            continue

        dt = seg_end - seg_start
        p_avg = 0.5 * (p_a + p_b)

        energy_j += p_avg * dt
        weighted_power_sum += p_avg * dt
        total_dt += dt

    avg_power = weighted_power_sum / total_dt if total_dt > 0 else None

    return {
        "energy_window_j": float(energy_j),
        "avg_power_window_w": float(avg_power) if avg_power is not None else None,
        "num_samples_in_window": len(in_range),
    }


# -------------------------
# NCU PARSING
# -------------------------

def requested_ncu_metrics() -> List[str]:
    return [
        "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
        "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
        "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
        "smsp__sass_thread_inst_executed_op_hadd_pred_on.sum",
        "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum",
        "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum",
        "sm__inst_executed_pipe_tensor.sum",
        "smsp__sass_data_bytes_mem_global.sum",
        "dram__bytes.sum",
        "lts__t_bytes.sum",
    ]


def parse_ncu_csv(ncu_csv: str) -> Dict[str, float]:
    """
    Parse Nsight Compute CSV.

    Supports:
      1. Wide CSV: one kernel per row, metric names are columns.
      2. Long/raw CSV: one metric per row with Metric/Value columns.

    Your uploaded CSV is wide format, so we sum metric columns across kernels.
    """
    try:
        df = pd.read_csv(ncu_csv)
    except Exception:
        df = None

    if df is not None and len(df.columns) > 0:
        metrics: Dict[str, float] = {}

        for col in df.columns:
            col_name = str(col).strip()
            metric_name, unit_from_col = split_ncu_metric_column(col_name)
            if not (
                metric_name.startswith("sm__")
                or metric_name.startswith("smsp__")
                or metric_name.startswith("dram__")
                or metric_name.startswith("l2__")
                or metric_name.startswith("lts__")
            ):
                continue

            vals = pd.to_numeric(df[col], errors="coerce")
            total = vals.sum(skipna=True)
            unit = unit_from_col or infer_ncu_unit_from_series(df[col])

            if pd.notna(total):
                metrics[metric_name] = metrics.get(metric_name, 0.0) + (
                    float(total) * ncu_unit_scale(metric_name, unit)
                )

        if metrics:
            return metrics

    # Fallback for long CSV layouts.
    metrics = {}

    with open(ncu_csv, newline="") as f:
        rows = list(csv.reader(f))

    header_i = None
    for i, row in enumerate(rows):
        cells = [c.strip() for c in row]
        if any(("Metric" in c or c == "Name") for c in cells) and any("Value" in c for c in cells):
            header_i = i
            break

    if header_i is None:
        return metrics

    header = rows[header_i]
    name_idx = next((j for j, c in enumerate(header) if "Metric" in c or c.strip() == "Name"), None)
    val_idx = next((j for j, c in enumerate(header) if "Value" in c), None)
    unit_idx = next((j for j, c in enumerate(header) if "Unit" in c), None)

    if name_idx is None or val_idx is None:
        return metrics

    for row in rows[header_i + 1:]:
        if len(row) <= max(name_idx, val_idx):
            continue

        name = row[name_idx].strip()
        raw_val = row[val_idx].strip()
        unit = row[unit_idx].strip() if unit_idx is not None and len(row) > unit_idx else ""
        val = parse_ncu_metric_value(raw_val)

        if val is None:
            continue

        metrics[name] = metrics.get(name, 0.0) + val * ncu_unit_scale(name, unit)

    return metrics


def split_ncu_metric_column(col_name: str) -> Tuple[str, str]:
    stripped = col_name.strip()
    match = re.match(r"^(.+?)\s*[\[(]([^)\]]+)[\])]$", stripped)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return stripped, ""


def infer_ncu_unit_from_series(series) -> str:
    for value in series:
        if value is None:
            continue
        text = str(value).strip()
        if not text or parse_ncu_metric_value(text) is not None:
            continue
        unit = text.replace(",", "")
        if ncu_unit_scale("", unit) != 1.0:
            return unit
    return ""


def parse_ncu_metric_value(raw_val: str) -> Optional[float]:
    cleaned = raw_val.strip().replace(",", "")
    if cleaned in ("", "-", "--", "nan", "NaN", "N/A"):
        return None

    try:
        return float(cleaned)
    except Exception:
        return None


def ncu_unit_scale(metric_name: str, unit: str) -> float:
    unit_norm = unit.strip().lower().replace(" ", "")

    if "byte" not in metric_name.lower() and "byte" not in unit_norm:
        return 1.0

    byte_scales = {
        "byte": 1.0,
        "bytes": 1.0,
        "b": 1.0,
        "kbyte": 1e3,
        "kbytes": 1e3,
        "kb": 1e3,
        "mbyte": 1e6,
        "mbytes": 1e6,
        "mb": 1e6,
        "gbyte": 1e9,
        "gbytes": 1e9,
        "gb": 1e9,
        "tbyte": 1e12,
        "tbytes": 1e12,
        "tb": 1e12,
        "kibyte": 1024.0,
        "mibyte": 1024.0 ** 2,
        "gibyte": 1024.0 ** 3,
        "tibyte": 1024.0 ** 4,
    }

    return byte_scales.get(unit_norm, 1.0)


def compute_flops_and_bytes_from_metrics(metrics: dict) -> dict:
    """
    Estimate scalar FLOPs and bytes from NCU metrics.

    Tensor core instructions are kept separately because converting tensor
    instructions to FLOPs depends on HMMA mode, datatype, and kernel shape.
    """
    fadd = float(metrics.get("smsp__sass_thread_inst_executed_op_fadd_pred_on.sum", 0.0))
    fmul = float(metrics.get("smsp__sass_thread_inst_executed_op_fmul_pred_on.sum", 0.0))
    ffma = float(metrics.get("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum", 0.0))

    hadd = float(metrics.get("smsp__sass_thread_inst_executed_op_hadd_pred_on.sum", 0.0))
    hmul = float(metrics.get("smsp__sass_thread_inst_executed_op_hmul_pred_on.sum", 0.0))
    hfma = float(metrics.get("smsp__sass_thread_inst_executed_op_hfma_pred_on.sum", 0.0))

    tensor_insts = float(metrics.get("sm__inst_executed_pipe_tensor.sum", 0.0))
    global_bytes = float(metrics.get("smsp__sass_data_bytes_mem_global.sum", 0.0))
    dram_bytes = float(metrics.get("dram__bytes.sum", 0.0))
    l2_bytes = float(metrics.get("lts__t_bytes.sum", 0.0))
    bytes_window = global_bytes or dram_bytes or l2_bytes
    selected_bytes_metric = (
        "smsp__sass_data_bytes_mem_global.sum"
        if global_bytes > 0 else
        "dram__bytes.sum"
        if dram_bytes > 0 else
        "lts__t_bytes.sum"
        if l2_bytes > 0 else
        None
    )
    bytes_scale_note = None

    scalar_flops = (
        fadd
        + fmul
        + 2.0 * ffma
        + hadd
        + hmul
        + 2.0 * hfma
    )

    if (
        selected_bytes_metric is not None
        and 0.0 < bytes_window < 1e6
        and (tensor_insts > 1e6 or scalar_flops > 1e6)
    ):
        bytes_window *= 1e9
        if selected_bytes_metric == "smsp__sass_data_bytes_mem_global.sum":
            global_bytes *= 1e9
        elif selected_bytes_metric == "dram__bytes.sum":
            dram_bytes *= 1e9
        elif selected_bytes_metric == "lts__t_bytes.sum":
            l2_bytes *= 1e9
        bytes_scale_note = "scaled suspicious byte metric from Gbyte display value"

    return {
        "scalar_flops": float(scalar_flops),
        "tensor_insts": float(tensor_insts),
        "bytes_window": float(bytes_window),
        "metric_debug": {
            "fadd": fadd,
            "fmul": fmul,
            "ffma": ffma,
            "hadd": hadd,
            "hmul": hmul,
            "hfma": hfma,
            "tensor_insts": tensor_insts,
            "global_bytes": global_bytes,
            "dram_bytes": dram_bytes,
            "l2_bytes": l2_bytes,
            "selected_bytes_metric": selected_bytes_metric,
            "bytes_scale_note": bytes_scale_note,
        },
    }


def newest_file(run_dir: str, suffix: str, prefix: Optional[str] = None) -> Optional[str]:
    if not os.path.exists(run_dir):
        return None

    files = []
    for fn in os.listdir(run_dir):
        if prefix is not None and not fn.startswith(prefix):
            continue
        if fn.endswith(suffix):
            files.append(os.path.join(run_dir, fn))

    if not files:
        return None

    return max(files, key=os.path.getmtime)


def export_ncu_rep_to_csv(rep_path: str) -> str:
    csv_path = rep_path.replace(".ncu-rep", ".csv")

    with open(csv_path, "w") as f:
        subprocess.run(
            ["ncu", "--import", rep_path, "--csv", "--page", "raw"],
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )

    return csv_path


# -------------------------
# GENERATION / ACCURACY
# -------------------------

def extract_last_number(text: str) -> Optional[str]:
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return matches[-1] if matches else None


def normalize_gsm8k_answer(ans: Optional[str]) -> Optional[str]:
    if ans is None:
        return None

    marker = "####"
    if marker in ans:
        ans = ans.split(marker)[-1]

    return extract_last_number(ans)


def evaluate_example(task: str, prediction_text: str, gold_answer: Optional[str]) -> Optional[bool]:
    if task != "math":
        return None

    pred = extract_last_number(prediction_text)
    gold = normalize_gsm8k_answer(gold_answer)

    if pred is None or gold is None:
        return False

    return pred == gold


@torch.no_grad()
def generate_with_ttft(model, tok, prompt: str, max_new_tokens: int) -> dict:
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Prefill timing.
    sync_cuda()
    prefill_start = time.perf_counter()

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )

    sync_cuda()
    prefill_end = time.perf_counter()

    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    generated = [next_token]
    past_key_values = outputs.past_key_values

    ttft_s = prefill_end - prefill_start

    # Decode timing.
    sync_cuda()
    decode_start = time.perf_counter()

    eos_token_id = tok.eos_token_id
    cur_token = next_token

    for _ in range(max_new_tokens - 1):
        outputs = model(
            input_ids=cur_token,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )

        past_key_values = outputs.past_key_values
        cur_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated.append(cur_token)

        if eos_token_id is not None and int(cur_token.item()) == int(eos_token_id):
            break

    sync_cuda()
    decode_end = time.perf_counter()

    full_gen_ids = torch.cat([input_ids] + generated, dim=1)
    text = tok.decode(full_gen_ids[0, input_ids.shape[1]:], skip_special_tokens=True)

    return {
        "text": text,
        "prompt_tokens": input_ids.shape[1],
        "generated_tokens": len(generated),
        "ttft_s": ttft_s,
        "decode_time_s": decode_end - decode_start,
    }


@torch.no_grad()
def generate_batch_no_ttft(model, tok, prompts: List[str], max_new_tokens: int) -> dict:
    inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

    sync_cuda()
    t0 = time.perf_counter()

    gen_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )

    sync_cuda()
    t1 = time.perf_counter()

    prompt_lens = inputs["attention_mask"].sum(dim=1)
    gen_lens = (gen_ids.shape[1] - prompt_lens).clamp_min(0)

    texts = []
    for i in range(gen_ids.shape[0]):
        texts.append(tok.decode(gen_ids[i, int(prompt_lens[i]):], skip_special_tokens=True))

    return {
        "texts": texts,
        "elapsed_s": float(t1 - t0),
        "generated_tokens_total": int(gen_lens.sum().item()),
        "prompt_tokens_total": int(inputs["attention_mask"].sum().item()),
        "per_example_gen_tokens": [int(x.item()) for x in gen_lens],
    }


def load_model_and_tokenizer(args):
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN")

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        token=token,
    )

    model.to(args.device)
    model.eval()

    return model, tok


# -------------------------
# MODE: ENERGY
# -------------------------

def run_energy_pass(args) -> dict:
    """
    Run model generation.

    Normal mode:
      - logs power.csv
      - writes window_times.json
      - writes results.json

    UNDER_NCU mode:
      - does NOT start nvidia-smi logger
      - does NOT overwrite normal results.json
      - writes results_under_ncu.json for debugging only
    """
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

    model, tok = load_model_and_tokenizer(args)

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

    ttft_list: List[float] = []
    decode_time_list: List[float] = []
    end_to_end_time_list: List[float] = []
    tokens_per_sec_list: List[float] = []
    example_accs: List[bool] = []

    win_t0_wall = None
    win_t1_wall = None

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
                    torch.cuda.nvtx.range_push("PROFILE_WINDOW")

                t_start = time.perf_counter()

                try:
                    out = generate_with_ttft(
                        model=model,
                        tok=tok,
                        prompt=prompt,
                        max_new_tokens=args.max_new_tokens,
                    )
                finally:
                    if in_nvtx and torch.cuda.is_available():
                        torch.cuda.nvtx.range_pop()
                        win_t1_wall = time.time()

                t_end = time.perf_counter()

                prompt_tokens = out["prompt_tokens"]
                gen_tokens = out["generated_tokens"]
                e2e_s = t_end - t_start
                pred_text = out["text"]

                total_examples += 1
                total_prompt_tokens += prompt_tokens
                total_generated_tokens += gen_tokens

                ttft_list.append(out["ttft_s"])
                decode_time_list.append(out["decode_time_s"])
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
                        "ttft_s": out["ttft_s"],
                        "decode_time_s": out["decode_time_s"],
                        "end_to_end_time_s": e2e_s,
                        "correct": acc,
                    }) + "\n")

                if total_examples % 10 == 0:
                    elapsed = time.time() - overall_t0
                    agg_tps = total_generated_tokens / max(elapsed, 1e-9)
                    print(f"[{total_examples}/{len(examples)}] agg_tok_per_s={agg_tps:.2f}", flush=True)

        else:
            for batch_idx, batch in batched(examples, args.batch_size):
                batch_prompts = [x["prompt"] for x in batch]
                batch_answers = [x["answer"] for x in batch]

                ex_start = batch_idx
                ex_end = batch_idx + len(batch)
                in_nvtx = (
                    args.nvtx_count > 0
                    and ex_end > args.nvtx_start
                    and ex_start < args.nvtx_start + args.nvtx_count
                )

                if in_nvtx and torch.cuda.is_available():
                    if win_t0_wall is None:
                        win_t0_wall = time.time()
                    torch.cuda.nvtx.range_push("PROFILE_WINDOW")

                try:
                    out = generate_batch_no_ttft(
                        model=model,
                        tok=tok,
                        prompts=batch_prompts,
                        max_new_tokens=args.max_new_tokens,
                    )
                finally:
                    if in_nvtx and torch.cuda.is_available():
                        torch.cuda.nvtx.range_pop()
                        win_t1_wall = time.time()

                elapsed_s = out["elapsed_s"]
                batch_gen_tokens = out["generated_tokens_total"]
                batch_prompt_tokens = out["prompt_tokens_total"]

                total_examples += len(batch)
                total_prompt_tokens += batch_prompt_tokens
                total_generated_tokens += batch_gen_tokens
                end_to_end_time_list.append(elapsed_s)

                if elapsed_s > 0:
                    tokens_per_sec_list.append(batch_gen_tokens / elapsed_s)

                for j, pred_text in enumerate(out["texts"]):
                    acc = evaluate_example(args.task, pred_text, batch_answers[j])
                    if acc is not None:
                        example_accs.append(acc)

                    if pred_f is not None:
                        pred_f.write(json.dumps({
                            "idx": batch_idx + j,
                            "task": args.task,
                            "prompt": batch_prompts[j],
                            "prediction": pred_text,
                            "gold": batch_answers[j],
                            "prompt_tokens": None,
                            "generated_tokens": out["per_example_gen_tokens"][j],
                            "ttft_s": None,
                            "decode_time_s": None,
                            "end_to_end_time_s": elapsed_s,
                            "correct": acc,
                        }) + "\n")

                if total_examples % 10 == 0:
                    elapsed = time.time() - overall_t0
                    agg_tps = total_generated_tokens / max(elapsed, 1e-9)
                    print(f"[{total_examples}/{len(examples)}] agg_tok_per_s={agg_tps:.2f}", flush=True)

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
        "model": args.model,
        "task": args.task,
        "dtype": args.dtype,
        "device": args.device,
        "limit": args.limit,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "examples": total_examples,
        "seconds": total_wall_seconds,
        "prompt_tokens": total_prompt_tokens,
        "generated_tokens": total_generated_tokens,
        "avg_ttft_s": safe_mean(ttft_list),
        "avg_decode_time_s": safe_mean(decode_time_list),
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


# -------------------------
# MODE: NCU
# -------------------------

def build_ncu_child_command(args, ncu_base: str) -> List[str]:
    cmd = [
        "ncu",
        "--target-processes", "all",
        "--replay-mode", args.ncu_replay_mode,
        "--profile-from-start", "yes",
        "--launch-count", str(args.ncu_launch_count),
        "--metrics", ",".join(requested_ncu_metrics()),
        "-o", ncu_base,
        sys.executable,
        os.path.abspath(__file__),
        "--mode", "energy",
        "--model", args.model,
        "--task", args.task,
        "--dtype", args.dtype,
        "--device", args.device,
        "--limit", str(args.limit),
        "--batch_size", str(args.batch_size),
        "--max_new_tokens", str(args.max_new_tokens),
        "--run_dir", args.run_dir,
        "--gpu_id", str(args.gpu_id),
        "--power_interval_ms", str(args.power_interval_ms),
        "--nvtx_start", str(args.nvtx_start),
        "--nvtx_count", str(args.nvtx_count),
    ]

    if args.save_predictions:
        cmd.append("--save_predictions")

    return cmd


def run_ncu_pass(args) -> dict:
    maybe_mkdir(args.run_dir)

    ncu_base = os.path.join(args.run_dir, f"ncu_{now_tag()}")
    cmd = build_ncu_child_command(args, ncu_base)

    env = os.environ.copy()
    env["UNDER_NCU"] = "1"

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


# -------------------------
# MODE: INTENSITY
# -------------------------

def build_intensity_results(run_dir: str) -> dict:
    power_csv = os.path.join(run_dir, "power.csv")
    window_path = os.path.join(run_dir, "window_times.json")

    require_file(power_csv, "power CSV")
    require_file(window_path, "window_times.json")

    window_info = read_json(window_path)
    t0_wall = window_info.get("window_wall_t0")
    t1_wall = window_info.get("window_wall_t1")

    if t0_wall is None or t1_wall is None:
        raise RuntimeError(
            "window_wall_t0/window_wall_t1 missing. "
            "Run energy mode with nvtx_count > 0 first."
        )

    newest_csv = newest_file(run_dir, ".csv", prefix="ncu_")

    if newest_csv is None:
        newest_rep = newest_file(run_dir, ".ncu-rep", prefix="ncu_")
        if newest_rep is None:
            raise FileNotFoundError(f"No ncu_*.csv or ncu_*.ncu-rep found in {run_dir}")
        newest_csv = export_ncu_rep_to_csv(newest_rep)

    metrics = parse_ncu_csv(newest_csv)
    compute = compute_flops_and_bytes_from_metrics(metrics)
    energy = integrate_energy_in_window(power_csv, t0_wall, t1_wall)

    scalar_flops = compute["scalar_flops"]
    tensor_insts = compute["tensor_insts"]
    bytes_window = compute["bytes_window"]
    energy_window_j = energy["energy_window_j"]
    window_duration_s = float(t1_wall - t0_wall)

    results = {
        "ncu_csv": newest_csv,
        "window_wall_t0": t0_wall,
        "window_wall_t1": t1_wall,
        "window_duration_s": window_duration_s,
        "energy_window_j": energy_window_j,
        "avg_power_window_w": energy["avg_power_window_w"],
        "num_samples_in_window": energy["num_samples_in_window"],
        "scalar_flops": scalar_flops,
        "tensor_insts": tensor_insts,
        "bytes_window": bytes_window,
        "scalar_gflops_per_s": (
            scalar_flops / window_duration_s / 1e9
            if window_duration_s > 0 else None
        ),
        "global_gb_per_s": (
            bytes_window / window_duration_s / 1e9
            if window_duration_s > 0 else None
        ),
        "nj_per_scalar_flop": (
            energy_window_j * 1e9 / scalar_flops
            if energy_window_j is not None and scalar_flops > 0 else None
        ),
        "nj_per_byte": (
            energy_window_j * 1e9 / bytes_window
            if energy_window_j is not None and bytes_window > 0 else None
        ),
        "metric_debug": compute["metric_debug"],
        "num_metrics_parsed": len(metrics),
    }

    out_path = os.path.join(run_dir, "results_intensity.json")
    write_json(out_path, results)
    print(f"[saved] {out_path}", flush=True)

    # Fail loudly if the CSV parsed but key metrics are zero.
    if scalar_flops == 0.0 and tensor_insts == 0.0 and bytes_window == 0.0:
        print(
            "[warn] Parsed NCU metrics are all zero. "
            "Check ncu CSV format, selected kernels, or metric names.",
            flush=True,
        )

    return results


# -------------------------
# ARGUMENTS / DISPATCH
# -------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    ap.add_argument("--mode", required=True, choices=["energy", "ncu", "intensity"])

    # Required for energy/ncu. Not required for intensity.
    ap.add_argument("--model")
    ap.add_argument("--task", choices=["alpaca", "sharegpt", "math", "multi_turn"])

    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=64)

    ap.add_argument("--run_dir", default="logs/run")
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--power_interval_ms", type=int, default=20)

    # NVTX profiling window in terms of example indices.
    ap.add_argument("--nvtx_start", type=int, default=0)
    ap.add_argument("--nvtx_count", type=int, default=10)

    # NCU options.
    ap.add_argument("--ncu_launch_count", type=int, default=20000)
    ap.add_argument("--ncu_replay_mode", default="application", choices=["application", "kernel"])

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
    ap = build_arg_parser()
    args = ap.parse_args()
    validate_args(args)

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
