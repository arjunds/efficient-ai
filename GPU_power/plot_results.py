#!/usr/bin/env python3

import argparse
import fnmatch
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


LOG_ROOT = "logs"
OUT_DIR = "plots"

# Put run directories or wildcard patterns here when you want to leave them out.
# Examples:
#   "logs/vllm_Mistral-7B-v0.1_alpaca_float16"
#   "*old*"
#   "*failed*"
EXCLUDE_RUNS = [
    "logs/vllm_deepseek-llm-7b-base_alpaca_float16",
]


METRIC_PLOTS = [
    ("aggregate_tokens_per_sec", "Aggregate Throughput (tokens/s)", "aggregate_throughput.png"),
    ("avg_tokens_per_sec", "Average Throughput (tokens/s)", "average_throughput.png"),
    ("seconds", "Runtime (s)", "runtime.png"),
    ("avg_power_w", "Average Power (W)", "average_power.png"),
    ("energy_total_j", "Total Energy (J)", "total_energy.png"),
    ("energy_per_token_j", "Energy / Generated Token (J/token)", "energy_per_token.png"),
    ("tokens_per_joule", "Generated Tokens / Joule", "tokens_per_joule.png"),
    ("throughput_per_watt", "Aggregate Tokens/s/W", "throughput_per_watt.png"),
    ("energy_window_j", "Profile Window Energy (J)", "window_energy.png"),
    ("avg_power_window_w", "Profile Window Average Power (W)", "window_power.png"),
    ("scalar_flops", "Scalar FLOPs", "scalar_flops.png"),
    ("tensor_insts", "Tensor Instructions", "tensor_insts.png"),
    ("bytes_window", "Profile Window Bytes", "bytes_window.png"),
    ("nj_per_scalar_flop", "nJ / Scalar FLOP", "nj_per_scalar_flop.png"),
    ("nj_per_byte", "nJ / Byte", "nj_per_byte.png"),
]


SCATTER_PLOTS = [
    (
        "aggregate_tokens_per_sec",
        "energy_total_j",
        "Aggregate Throughput (tokens/s)",
        "Total Energy (J)",
        "throughput_vs_energy.png",
    ),
    (
        "aggregate_tokens_per_sec",
        "energy_per_token_j",
        "Aggregate Throughput (tokens/s)",
        "Energy / Generated Token (J/token)",
        "throughput_vs_energy_per_token.png",
    ),
    (
        "aggregate_tokens_per_sec",
        "tokens_per_joule",
        "Aggregate Throughput (tokens/s)",
        "Generated Tokens / Joule",
        "throughput_vs_tokens_per_joule.png",
    ),
    (
        "avg_power_window_w",
        "nj_per_byte",
        "Profile Window Average Power (W)",
        "nJ / Byte",
        "window_power_vs_nj_per_byte.png",
    ),
]


def safe_load_json(path: Path) -> dict:
    try:
        with path.open() as f:
            return json.load(f)
    except Exception:
        return {}


def is_excluded(run_dir: Path, patterns: list[str]) -> bool:
    run_dir_str = run_dir.as_posix()
    run_name = run_dir.name

    return any(
        fnmatch.fnmatch(run_dir_str, pattern) or fnmatch.fnmatch(run_name, pattern)
        for pattern in patterns
    )


def short_model_name(model: object) -> object:
    if isinstance(model, str):
        return model.split("/")[-1]
    return model


def build_record(run_dir: Path) -> dict | None:
    results = safe_load_json(run_dir / "results.json")
    if not results:
        return None

    intensity = safe_load_json(run_dir / "results_intensity.json")
    model = results.get("model")

    record = {
        "run_dir": run_dir.as_posix(),
        "model": model,
        "model_short": short_model_name(model),
        "task": results.get("task"),
        "dtype": results.get("dtype"),
        "batch_size": results.get("batch_size"),
        "max_new_tokens": results.get("max_new_tokens"),
        "examples": results.get("examples"),
        "seconds": results.get("seconds"),
        "avg_ttft_s": results.get("avg_ttft_s"),
        "avg_tokens_per_sec": results.get("avg_tokens_per_sec"),
        "aggregate_tokens_per_sec": results.get("aggregate_tokens_per_sec"),
        "prompt_tokens": results.get("prompt_tokens"),
        "generated_tokens": results.get("generated_tokens"),
        "energy_total_j": results.get("energy_total_j"),
        "avg_power_w": results.get("avg_power_w"),
        "energy_per_token_j": results.get("energy_per_generated_token_j"),
        "accuracy": results.get("accuracy"),
        "num_accuracy_examples": results.get("num_accuracy_examples"),
        "energy_window_j": intensity.get("energy_window_j"),
        "avg_power_window_w": intensity.get("avg_power_window_w"),
        "scalar_flops": intensity.get("scalar_flops"),
        "tensor_insts": intensity.get("tensor_insts"),
        "bytes_window": intensity.get("bytes_window"),
        "nj_per_scalar_flop": intensity.get("nj_per_scalar_flop"),
        "nj_per_byte": intensity.get("nj_per_byte"),
    }

    return record


def load_results(log_root: Path, excludes: list[str]) -> tuple[pd.DataFrame, list[str]]:
    records = []
    skipped = []

    for results_path in sorted(log_root.rglob("results.json")):
        run_dir = results_path.parent
        if is_excluded(run_dir, excludes):
            skipped.append(run_dir.as_posix())
            continue

        record = build_record(run_dir)
        if record is not None:
            records.append(record)

    if not records:
        return pd.DataFrame(), skipped

    df = pd.DataFrame(records)

    numeric_cols = [col for col in df.columns if col not in {"run_dir", "model", "model_short", "task", "dtype"}]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["tokens_per_joule"] = df["generated_tokens"] / df["energy_total_j"]
    df["throughput_per_watt"] = df["aggregate_tokens_per_sec"] / df["avg_power_w"]
    df = df.replace([math.inf, -math.inf], pd.NA)

    return df.sort_values(["model_short", "task", "dtype", "run_dir"]), skipped


def make_label(row: pd.Series) -> str:
    bits = [
        str(row.get("model_short") or "unknown"),
        str(row.get("task") or "task"),
        str(row.get("dtype") or "dtype"),
    ]

    max_new_tokens = row.get("max_new_tokens")
    examples = row.get("examples")
    if pd.notna(max_new_tokens):
        bits.append(f"{int(max_new_tokens)}tok")
    if pd.notna(examples):
        bits.append(f"{int(examples)}ex")

    return "\n".join(bits)


def save_bar_plot(df: pd.DataFrame, metric: str, ylabel: str, out_path: Path) -> bool:
    if metric not in df.columns:
        return False

    plot_df = df[["run_dir", "model_short", "task", "dtype", "max_new_tokens", "examples", metric]].copy()
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df = plot_df.dropna(subset=[metric])
    if plot_df.empty:
        return False

    labels = [make_label(row) for _, row in plot_df.iterrows()]
    width = max(8, min(24, 1.2 * len(plot_df)))

    plt.figure(figsize=(width, 5.5))
    plt.bar(labels, plot_df[metric])
    plt.ylabel(ylabel)
    plt.title(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return True


def bar_plot_data(df: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, list[str]]:
    plot_df = df[["run_dir", "model_short", "task", "dtype", "max_new_tokens", "examples", metric]].copy()
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df = plot_df.dropna(subset=[metric])
    labels = [make_label(row).replace("\n", " ") for _, row in plot_df.iterrows()]
    return plot_df, labels


def save_scatter_plot(
    df: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
) -> bool:
    if x_metric not in df.columns or y_metric not in df.columns:
        return False

    plot_df = df.copy()
    plot_df[x_metric] = pd.to_numeric(plot_df[x_metric], errors="coerce")
    plot_df[y_metric] = pd.to_numeric(plot_df[y_metric], errors="coerce")
    plot_df = plot_df.dropna(subset=[x_metric, y_metric])
    if plot_df.empty:
        return False

    plt.figure(figsize=(8, 5.5))
    plt.scatter(plot_df[x_metric], plot_df[y_metric], s=90)

    for _, row in plot_df.iterrows():
        plt.annotate(
            str(row.get("model_short") or "unknown"),
            (row[x_metric], row[y_metric]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs {xlabel}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return True


def save_all_comparisons_plot(df: pd.DataFrame, out_path: Path) -> bool:
    available_plots = []

    for metric, ylabel, _ in METRIC_PLOTS:
        if metric not in df.columns:
            continue

        plot_df, labels = bar_plot_data(df, metric)
        if not plot_df.empty:
            available_plots.append((metric, ylabel, plot_df, labels))

    if not available_plots:
        return False

    cols = 3
    rows = math.ceil(len(available_plots) / cols)
    fig_width = max(14, 4.8 * cols)
    fig_height = max(4.2 * rows, 5)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), squeeze=False)
    axes_flat = axes.flatten()

    for ax, (metric, ylabel, plot_df, labels) in zip(axes_flat, available_plots):
        x_positions = range(len(plot_df))
        ax.bar(x_positions, plot_df[metric])
        ax.set_title(ylabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xticks(list(x_positions))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.25)

    for ax in axes_flat[len(available_plots):]:
        ax.axis("off")

    fig.suptitle("Benchmark Comparison Summary", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return True


def parse_unit_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(r"[^0-9eE+\-.]", "", regex=True),
        errors="coerce",
    )


def load_power_trace(run_dir: Path) -> pd.DataFrame | None:
    power_csv = run_dir / "power.csv"
    if not power_csv.exists():
        return None

    try:
        power = pd.read_csv(power_csv)
    except Exception:
        return None

    power.columns = [str(col).strip() for col in power.columns]
    timestamp_col = next((col for col in power.columns if "timestamp" in col.lower()), None)
    power_col = next((col for col in power.columns if "power.draw" in col.lower()), None)
    if power_col is None and "power_w" in power.columns:
        power_col = "power_w"

    if timestamp_col is None or power_col is None:
        return None

    power["timestamp"] = pd.to_datetime(power[timestamp_col], errors="coerce")
    power["power_w"] = parse_unit_float(power[power_col])
    power = power.dropna(subset=["timestamp", "power_w"])
    if power.empty:
        return None

    power["time_s"] = (power["timestamp"] - power["timestamp"].iloc[0]).dt.total_seconds()
    return power


def safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "run"


def save_power_plots(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    saved = []
    traces = []

    for _, row in df.iterrows():
        run_dir = Path(row["run_dir"])
        trace = load_power_trace(run_dir)
        if trace is None:
            continue

        label = make_label(row).replace("\n", "_")
        traces.append((label, trace))

        plt.figure(figsize=(9, 4.5))
        plt.plot(trace["time_s"], trace["power_w"], linewidth=1.8)
        plt.xlabel("Time (s)")
        plt.ylabel("Power (W)")
        plt.title(f"Power Trace: {label}")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        out_path = out_dir / f"power_trace_{safe_filename(label)}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        saved.append(out_path)

    if traces:
        plt.figure(figsize=(10, 6))
        for label, trace in traces:
            plt.plot(trace["time_s"], trace["power_w"], label=label, linewidth=1.5)

        plt.xlabel("Time (s)")
        plt.ylabel("Power (W)")
        plt.title("Power Trace Overlay")
        plt.legend(fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        out_path = out_dir / "power_trace_overlay.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        saved.append(out_path)

    return saved


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot all benchmark runs under logs/")
    parser.add_argument("--log_root", default=LOG_ROOT)
    parser.add_argument("--out_dir", default=OUT_DIR)
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Run dir/name wildcard to exclude. Can be passed multiple times.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    log_root = Path(args.log_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    excludes = EXCLUDE_RUNS + args.exclude
    df, skipped = load_results(log_root, excludes)

    if df.empty:
        print(f"[warn] no results.json files found under {log_root}")
        if skipped:
            print(f"[info] skipped {len(skipped)} excluded runs")
        return

    results_csv = out_dir / "plotted_results.csv"
    df.to_csv(results_csv, index=False)

    saved = [results_csv]

    for metric, ylabel, filename in METRIC_PLOTS:
        if save_bar_plot(df, metric, ylabel, out_dir / filename):
            saved.append(out_dir / filename)

    all_comparisons_path = out_dir / "all_comparisons.png"
    if save_all_comparisons_plot(df, all_comparisons_path):
        saved.append(all_comparisons_path)

    for x_metric, y_metric, xlabel, ylabel, filename in SCATTER_PLOTS:
        if save_scatter_plot(df, x_metric, y_metric, xlabel, ylabel, out_dir / filename):
            saved.append(out_dir / filename)

    saved.extend(save_power_plots(df, out_dir))

    print(f"[info] plotted {len(df)} runs from {log_root}")
    if skipped:
        print("[info] excluded runs:")
        for run_dir in skipped:
            print(f"  - {run_dir}")

    print("[saved]")
    for path in saved:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
