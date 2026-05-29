#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import shlex
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator
import numpy as np
import torch
import yaml

from datasets import load_dataset
from diagnostics.kernel_alignment import (
    CenteredKernel,
    apply_propagation,
    centered_anchor_kernel,
    centered_kernel_alignment,
    centered_label_kernel,
    centered_representation_kernel,
    normalized_adjacency,
)
from hparams_search_scripts.multi_pool_scheduler import MultiPoolScheduler, SchedulerTask, TaskResult


DEFAULT_INPUT_ROOT = Path("paper_experiments/figure10")
DEFAULT_OUTPUT_DIR = Path("final_data/figure10")
DEFAULT_X_STEPS = [0, 2, 4, 8, 16, 32, 64]
DEFAULT_FEATURE_DIMS = [
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
]
DEFAULT_BASE_SEED = 12345
DEFAULT_REPEAT_COUNT = 10
DEFAULT_BOOTSTRAP_SAMPLES = 1000
DEFAULT_BOOTSTRAP_SEED = 12345
STEP_COLORS = {
    0: "#111111",
    2: "#234A6F",
    4: "#A7583B",
    8: "#5E7D4D",
    16: "#6E5A8A",
    32: "#8C6D31",
    64: "#3F6B6E",
}


@dataclass(frozen=True)
class SelectedJob:
    k: int
    d: int
    row: dict[str, str]
    job_dir: Path
    final_repeat10_dir: Path


def parse_int_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Figure 10 repeat-10 RS/KLA/accuracy metrics.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--x-steps", type=parse_int_list, default=list(DEFAULT_X_STEPS))
    parser.add_argument("--feature-dims", type=parse_int_list, default=list(DEFAULT_FEATURE_DIMS))
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--repeat-count", type=int, default=DEFAULT_REPEAT_COUNT)
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--metric-dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--skip-repeat10", action="store_true", help="Do not launch missing repeat-10 jobs.")
    parser.add_argument("--python-bin", type=str, default=None, help="Override the Python executable in repeat commands.")
    parser.add_argument("--gpu-ids", type=parse_int_list, default=None, help="GPU ids for repeat-10 scheduler.")
    parser.add_argument("--max-parallel-per-gpu", type=int, default=None)
    parser.add_argument("--gpu-launch-interval-sec", type=float, default=None)
    parser.add_argument("--poll-interval-sec", type=float, default=1.0)
    parser.add_argument("--dry-run-repeat10", action="store_true", help="Print missing repeat-10 tasks without running them.")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def metric_dtype(name: str) -> torch.dtype:
    if name == "float64":
        return torch.float64
    return torch.float32


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def parse_step_name(path: Path) -> int:
    name = path.parent.name.removesuffix(".yaml") if path.name == "manifest.csv" else path.name.removesuffix(".yaml")
    try:
        return int(name.split("=", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Unexpected x_steps path: {path}") from exc


def require_float(row: dict[str, str], field: str, context: str) -> float:
    raw = str(row.get(field, "")).strip()
    if raw == "":
        raise ValueError(f"{context}: missing {field}")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{context}: invalid {field}={raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{context}: non-finite {field}={raw!r}")
    return value


def require_int(row: dict[str, str], field: str, context: str) -> int:
    raw = str(row.get(field, "")).strip()
    if raw == "":
        raise ValueError(f"{context}: missing {field}")
    try:
        return int(float(raw))
    except ValueError as exc:
        raise ValueError(f"{context}: invalid {field}={raw!r}") from exc


def choose_best_rows(rows: list[dict[str, str]], *, k: int, feature_dims: list[int]) -> list[dict[str, str]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("search_status") not in {"completed", "skipped_existing_result"}:
            continue
        context = f"K={k} feature_dim={row.get('feature_dim', '')} scale={row.get('scale', '')}"
        feature_dim = require_int(row, "feature_dim", context)
        require_float(row, "best_verify_val_acc_mean", context)
        require_float(row, "best_verify_test_acc_mean", context)
        require_float(row, "scale", context)
        grouped[feature_dim].append(row)

    found_dims = sorted(grouped)
    expected_dims = sorted(feature_dims)
    if found_dims != expected_dims:
        raise ValueError(f"K={k}: expected feature_dims={expected_dims}, found={found_dims}")

    selected: list[dict[str, str]] = []
    for feature_dim in expected_dims:
        candidates = grouped[feature_dim]
        if len(candidates) != 10:
            raise ValueError(f"K={k}, d={feature_dim}: expected 10 scale rows, found {len(candidates)}")

        def score(row: dict[str, str]) -> tuple[float, float, float]:
            context = f"K={k} d={feature_dim} scale={row.get('scale', '')}"
            return (
                require_float(row, "best_verify_val_acc_mean", context),
                require_float(row, "best_verify_test_acc_mean", context),
                -require_float(row, "scale", context),
            )

        selected.append(max(candidates, key=score))
    return selected


def load_selected_jobs(input_root: Path, *, x_steps: list[int], feature_dims: list[int]) -> list[SelectedJob]:
    jobs: list[SelectedJob] = []
    for k in sorted(x_steps):
        manifest_path = input_root / f"x_steps={k}.yaml" / "manifest.csv"
        rows = load_csv(manifest_path)
        for row in choose_best_rows(rows, k=k, feature_dims=feature_dims):
            d = require_int(row, "feature_dim", f"K={k}")
            best_x_steps = require_int(row, "best_x_steps", f"K={k} d={d}")
            if best_x_steps != k:
                raise ValueError(f"K={k} d={d}: best_x_steps={best_x_steps}, expected {k}")
            job_dir = Path(row["job_dir"])
            jobs.append(
                SelectedJob(
                    k=k,
                    d=d,
                    row=row,
                    job_dir=job_dir,
                    final_repeat10_dir=job_dir / "final_repeat10",
                )
            )
    return jobs


def read_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML must contain a mapping: {path}")
    return payload


def scheduler_settings(input_root: Path, args: argparse.Namespace) -> tuple[list[int], int, float]:
    if args.gpu_ids is not None and args.max_parallel_per_gpu is not None and args.gpu_launch_interval_sec is not None:
        return list(args.gpu_ids), int(args.max_parallel_per_gpu), float(args.gpu_launch_interval_sec)

    config_paths = sorted(input_root.glob("x_steps=*.yaml/input_config.yaml"), key=parse_step_name)
    if not config_paths:
        raise FileNotFoundError(f"No input_config.yaml files found under {input_root}")
    device_cfg = read_yaml(config_paths[0]).get("device")
    if not isinstance(device_cfg, dict):
        raise ValueError(f"Missing device config in {config_paths[0]}")

    gpu_ids = args.gpu_ids if args.gpu_ids is not None else [int(value) for value in device_cfg.get("gpu_ids", [0])]
    max_parallel = args.max_parallel_per_gpu
    if max_parallel is None:
        max_parallel = int(device_cfg.get("max_parallel_per_gpu", 1))
    interval = args.gpu_launch_interval_sec
    if interval is None:
        interval = float(device_cfg.get("gpu_launch_interval_sec", 0.1))
    return list(gpu_ids), int(max_parallel), float(interval)


def replace_option(parts: list[str], names: Iterable[str], value: str) -> None:
    names = tuple(names)
    for index, part in enumerate(parts):
        if part in names:
            if index + 1 >= len(parts):
                raise ValueError(f"Option {part} has no value in command: {parts}")
            parts[index + 1] = value
            return
    parts.extend([names[0], value])


def repeat_command(job: SelectedJob, *, args: argparse.Namespace) -> list[str]:
    command_path = Path(job.row["recommended_command_path"])
    if not command_path.is_file():
        raise FileNotFoundError(command_path)
    parts = shlex.split(command_path.read_text(encoding="utf-8"))
    if args.python_bin:
        parts[0] = args.python_bin
    replace_option(parts, ("-s", "--seed"), str(int(args.base_seed)))
    replace_option(parts, ("-r", "--repeats"), str(int(args.repeat_count)))
    replace_option(parts, ("-o", "--output-dir"), str(job.final_repeat10_dir))
    return parts


def result_csv_candidates(output_dir: Path) -> list[Path]:
    if not output_dir.is_dir():
        return []
    return sorted(path for path in output_dir.glob("*.csv") if path.is_file())


def row_matches_job(row: dict[str, str], job: SelectedJob, *, base_seed: int) -> bool:
    try:
        return (
            int(float(row.get("x_steps", "nan"))) == job.k
            and int(float(row.get("feature_dim", "nan"))) == job.d
            and int(float(row.get("seed", "nan"))) == int(base_seed)
            and math.isclose(float(row.get("learning_rate", "nan")), float(job.row["best_learning_rate"]))
            and math.isclose(float(row.get("weight_decay", "nan")), float(job.row["best_weight_decay"]))
            and math.isclose(float(row.get("dropout", "nan")), float(job.row["best_dropout"]))
        )
    except (TypeError, ValueError):
        return False


def find_repeat_result_csv(job: SelectedJob, *, repeat_count: int, base_seed: int) -> Path | None:
    for csv_path in result_csv_candidates(job.final_repeat10_dir):
        rows = load_csv(csv_path)
        if len(rows) != int(repeat_count):
            continue
        if rows and row_matches_job(rows[0], job, base_seed=base_seed):
            return csv_path
    return None


def run_missing_repeat10(jobs: list[SelectedJob], args: argparse.Namespace) -> None:
    missing = [
        job
        for job in jobs
        if find_repeat_result_csv(job, repeat_count=args.repeat_count, base_seed=args.base_seed) is None
    ]
    if not missing:
        print("[Info] All repeat-10 outputs already exist.")
        return

    print(f"[Info] Missing repeat-{args.repeat_count} outputs: {len(missing)}")
    if args.skip_repeat10:
        raise RuntimeError("Missing repeat-10 outputs and --skip-repeat10 was set.")
    if args.dry_run_repeat10:
        for job in missing:
            print(" ".join(shlex.quote(part) for part in repeat_command(job, args=args)))
        return

    gpu_ids, max_parallel, launch_interval = scheduler_settings(args.input_root.resolve(), args)
    scheduler = MultiPoolScheduler(
        worker_ids=gpu_ids,
        max_parallel_per_worker=max_parallel,
        poll_interval_sec=float(args.poll_interval_sec),
        launch_interval_sec=launch_interval,
        device="gpu",
    )
    log_root = args.output_dir.resolve() / "repeat10_logs"
    log_index_path = log_root / "task_index.csv"
    for job in missing:
        scheduler.enqueue(
            SchedulerTask(
                task_id=f"repeat10_K={job.k}_d={job.d}",
                combo_key=f"K={job.k}_d={job.d}",
                pool="repeat10",
                command=repeat_command(job, args=args),
                env={},
                cwd=REPO_ROOT,
                log_path=log_root / f"K={job.k}_d={job.d}.log",
                retry_count=0,
                label=f"K={job.k} d={job.d}",
                log_index_path=log_index_path,
            )
        )

    failed: list[str] = []

    def on_result(result: TaskResult) -> list[SchedulerTask]:
        if result.success:
            print(f"[Done] {result.task.label} repeat10 in {result.total_seconds:.1f}s")
        else:
            failed.append(result.error_message or result.task.label)
            print(f"[Fail] {result.error_message}")
        return []

    scheduler.run(on_result)
    if failed:
        raise RuntimeError("Repeat-10 tasks failed: " + "; ".join(failed[:5]))

    still_missing = [
        job
        for job in jobs
        if find_repeat_result_csv(job, repeat_count=args.repeat_count, base_seed=args.base_seed) is None
    ]
    if still_missing:
        preview = ", ".join(f"K={job.k},d={job.d}" for job in still_missing[:10])
        raise RuntimeError(f"Repeat-10 scheduler finished but outputs are still missing: {preview}")


def load_repeat_rows(job: SelectedJob, *, repeat_count: int, base_seed: int) -> list[dict[str, str]]:
    csv_path = find_repeat_result_csv(job, repeat_count=repeat_count, base_seed=base_seed)
    if csv_path is None:
        raise FileNotFoundError(f"Missing repeat-{repeat_count} result for K={job.k}, d={job.d}")
    rows = load_csv(csv_path)
    rows.sort(key=lambda row: int(float(row.get("version", "0"))))
    return rows


def generate_random_normal_features(
    *,
    num_nodes: int,
    feature_dim: int,
    mean: float,
    std: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    features = torch.normal(
        mean=float(mean),
        std=float(std),
        size=(int(num_nodes), int(feature_dim)),
        generator=generator,
        device=torch.device("cpu"),
        dtype=dtype,
    )
    features.div_(math.sqrt(float(feature_dim)))
    return features.to(device=device)


def bootstrap_ci(values: list[float], *, samples: int, seed: int) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("bootstrap_ci expects a non-empty 1D array")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, array.size, size=(int(samples), array.size), dtype=np.intp)
    means = array[indices].mean(axis=1)
    low, high = np.nanpercentile(means, [2.5, 97.5])
    return float(low), float(high)


def format_float(value: float) -> str:
    return f"{float(value):.12f}"


def compute_metric_rows(jobs: list[SelectedJob], args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    device = resolve_device(args.device)
    dtype = metric_dtype(args.metric_dtype)
    print(f"[Info] metric_device={device} metric_dtype={dtype}")

    data = load_dataset(
        dataset="cora",
        data_dir="./datasets",
        data_range=(0.0, 1.0),
        val_ratio=0.25,
        test_ratio=0.25,
    )
    labels = data.y if data.y.dim() == 1 else data.y.argmax(dim=1)
    labels = labels.to(dtype=torch.long).cpu()
    num_nodes = int(data.num_nodes)
    normalized_adj_t = normalized_adjacency(data.adj_t.cpu(), device=device, dtype=dtype)
    centered_label = centered_label_kernel(labels, device=device)

    smoothers_by_k: dict[int, str] = {}
    for job in jobs:
        smoother = str(job.row.get("smoother", "hoa")).strip().lower()
        existing = smoothers_by_k.get(job.k)
        if existing is not None and existing != smoother:
            raise ValueError(f"K={job.k}: mixed smoothers {existing!r} and {smoother!r}")
        smoothers_by_k[job.k] = smoother

    anchor_by_k: dict[int, CenteredKernel] = {}
    kla_anchor_by_k: dict[int, float] = {}
    for k in sorted(smoothers_by_k):
        print(f"[Info] building anchor kernel K={k}")
        anchor = centered_anchor_kernel(
            data.adj_t.cpu(),
            smoother=smoothers_by_k[k],
            x_steps=k,
            num_nodes=num_nodes,
            device=device,
            dtype=dtype,
        )
        anchor_by_k[k] = anchor
        kla_anchor_by_k[k] = centered_kernel_alignment(anchor, centered_label)

    by_repeat_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for job_index, job in enumerate(jobs, start=1):
        repeat_rows = load_repeat_rows(job, repeat_count=args.repeat_count, base_seed=args.base_seed)
        mean = require_float(job.row, "random_normal_mean", f"K={job.k} d={job.d}")
        std = require_float(job.row, "random_normal_std", f"K={job.k} d={job.d}")
        smoother = str(job.row.get("smoother", "hoa")).strip().lower()
        anchor_kernel = anchor_by_k[job.k]
        kla_anchor = kla_anchor_by_k[job.k]
        print(f"[Info] metrics {job_index}/{len(jobs)} K={job.k} d={job.d}")

        for repeat_row in repeat_rows:
            repeat = int(float(repeat_row.get("version", "0")))
            seed = int(args.base_seed) + repeat
            acc = require_float(repeat_row, "test/acc", f"K={job.k} d={job.d} repeat={repeat}")
            features = generate_random_normal_features(
                num_nodes=num_nodes,
                feature_dim=job.d,
                mean=mean,
                std=std,
                seed=seed,
                device=device,
                dtype=dtype,
            )
            z = apply_propagation(normalized_adj_t, features, smoother=smoother, x_steps=job.k)
            rep_kernel = centered_representation_kernel(z, name=f"representation kernel K={job.k} d={job.d} seed={seed}")
            rs_z_a = centered_kernel_alignment(rep_kernel, anchor_kernel)
            kla_z_l = centered_kernel_alignment(rep_kernel, centered_label)
            by_repeat_rows.append(
                {
                    "K": job.k,
                    "d": job.d,
                    "repeat": repeat,
                    "seed": seed,
                    "acc": format_float(acc),
                    "rs_z_a": format_float(rs_z_a),
                    "kla_z_l": format_float(kla_z_l),
                    "kla_a_l": format_float(kla_anchor),
                    "scale": job.row["scale"],
                    "learning_rate": job.row["best_learning_rate"],
                    "weight_decay": job.row["best_weight_decay"],
                    "dropout": job.row["best_dropout"],
                    "job_dir": str(job.job_dir),
                    "repeat_result_dir": str(job.final_repeat10_dir),
                }
            )
            del features, z, rep_kernel
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"[Info] metric computation elapsed={time.perf_counter() - started:.1f}s")
    summary_rows = summarize_by_repeat_rows(
        by_repeat_rows,
        x_steps=sorted({job.k for job in jobs}),
        feature_dims=sorted({job.d for job in jobs}),
        repeat_count=int(args.repeat_count),
        bootstrap_samples=int(args.bootstrap_samples),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    return by_repeat_rows, summary_rows


def summarize_by_repeat_rows(
    by_repeat_rows: list[dict[str, object]],
    *,
    x_steps: list[int],
    feature_dims: list[int],
    repeat_count: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for row in by_repeat_rows:
        grouped[(int(row["K"]), int(row["d"]))].append(row)

    summary_rows: list[dict[str, object]] = []
    metric_map = {
        "acc": "mean_acc",
        "rs_z_a": "mean_rs_z_a",
        "kla_z_l": "mean_kla_z_l",
        "kla_a_l": "mean_kla_a_l",
    }
    for k in x_steps:
        for d in feature_dims:
            rows = sorted(grouped[(k, d)], key=lambda row: int(row["repeat"]))
            if len(rows) != repeat_count:
                raise RuntimeError(f"K={k}, d={d}: expected {repeat_count} repeat rows, found {len(rows)}")
            payload: dict[str, object] = {"K": k, "d": d}
            for source_field, output_field in metric_map.items():
                values = [float(row[source_field]) for row in rows]
                mean_value = float(np.mean(values))
                ci_low, ci_high = bootstrap_ci(
                    values,
                    samples=bootstrap_samples,
                    seed=bootstrap_seed + k * 100000 + d + len(output_field),
                )
                payload[output_field] = format_float(mean_value)
                payload[f"{output_field}_ci_low"] = format_float(ci_low)
                payload[f"{output_field}_ci_high"] = format_float(ci_high)
            summary_rows.append(payload)
    return summary_rows


def validate_outputs(
    by_repeat_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    *,
    x_steps: list[int],
    feature_dims: list[int],
    repeat_count: int,
) -> None:
    expected_summary = len(x_steps) * len(feature_dims)
    expected_repeat = expected_summary * repeat_count
    if len(summary_rows) != expected_summary:
        raise RuntimeError(f"Expected {expected_summary} summary rows, found {len(summary_rows)}")
    if len(by_repeat_rows) != expected_repeat:
        raise RuntimeError(f"Expected {expected_repeat} by-repeat rows, found {len(by_repeat_rows)}")

    for row in by_repeat_rows:
        for field in ("acc", "rs_z_a", "kla_z_l", "kla_a_l"):
            value = float(row[field])
            if not math.isfinite(value):
                raise RuntimeError(f"Non-finite {field}: {row}")
            if field != "acc" and (value < -1.000001 or value > 1.000001):
                raise RuntimeError(f"Out-of-range {field}: {row}")


def plot_summary(summary_rows: list[dict[str, object]], *, output_png: Path, output_pdf: Path) -> None:
    rows_by_k: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        rows_by_k[int(row["K"])].append(row)

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0), sharex=True)
    panel_specs = [
        (axes[0], "mean_acc", "Accuracy (%)"),
        (axes[1], "mean_rs_z_a", "RS"),
        (axes[2], "mean_kla_z_l", "KLA"),
    ]
    all_dims = sorted({int(row["d"]) for row in summary_rows})

    for ax, field, ylabel in panel_specs:
        values_for_limits: list[float] = []
        for k in sorted(rows_by_k):
            rows = sorted(rows_by_k[k], key=lambda item: int(item["d"]))
            x_vals = np.asarray([int(row["d"]) for row in rows], dtype=float)
            y_vals = np.asarray([float(row[field]) for row in rows], dtype=float)
            low_vals = np.asarray([float(row[f"{field}_ci_low"]) for row in rows], dtype=float)
            high_vals = np.asarray([float(row[f"{field}_ci_high"]) for row in rows], dtype=float)
            color = STEP_COLORS.get(k)
            ax.plot(x_vals, y_vals, color=color, linewidth=1.8, label=f"K={k}")
            ax.fill_between(x_vals, low_vals, high_vals, color=color, alpha=0.14, linewidth=0)
            values_for_limits.extend(low_vals.tolist())
            values_for_limits.extend(high_vals.tolist())

            if field == "mean_kla_z_l":
                ref_vals = np.asarray([float(row["mean_kla_a_l"]) for row in rows], dtype=float)
                ax.plot(x_vals, ref_vals, color=color, linewidth=1.2, linestyle="--", alpha=0.9)
                values_for_limits.extend(ref_vals.tolist())

        y_min = min(values_for_limits)
        y_max = max(values_for_limits)
        margin = max(y_max - y_min, 1e-6) * 0.08
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Feature Dimension")
        ax.set_ylabel(ylabel)
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.xaxis.set_major_locator(FixedLocator(all_dims))
        ax.xaxis.set_major_formatter(FixedFormatter([str(value) for value in all_dims]))
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        ax.tick_params(axis="y", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].legend(loc="best", frameon=False, fontsize=8, handlelength=2.2)
    axes[2].legend(
        handles=[
            Line2D([0], [0], color="black", linewidth=1.8, linestyle="-", label="KLA(Z,L)"),
            Line2D([0], [0], color="black", linewidth=1.2, linestyle="--", label="KLA(A,L)"),
        ],
        loc="best",
        frameon=False,
        fontsize=8,
        handlelength=2.2,
    )
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight", dpi=220)
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.repeat_count <= 0:
        raise ValueError("--repeat-count must be > 0")
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be > 0")
    if args.max_parallel_per_gpu is not None and args.max_parallel_per_gpu <= 0:
        raise ValueError("--max-parallel-per-gpu must be > 0")

    args.input_root = args.input_root.resolve()
    args.output_dir = args.output_dir.resolve()
    x_steps = sorted({int(value) for value in args.x_steps})
    feature_dims = sorted({int(value) for value in args.feature_dims})

    plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42})

    jobs = load_selected_jobs(args.input_root, x_steps=x_steps, feature_dims=feature_dims)
    print(f"[Info] selected_jobs={len(jobs)}")
    run_missing_repeat10(jobs, args)
    if args.dry_run_repeat10:
        return 0

    by_repeat_rows, summary_rows = compute_metric_rows(jobs, args)
    validate_outputs(
        by_repeat_rows,
        summary_rows,
        x_steps=x_steps,
        feature_dims=feature_dims,
        repeat_count=int(args.repeat_count),
    )

    summary_path = args.output_dir / "figure10_summary.csv"
    by_repeat_path = args.output_dir / "figure10_by_repeat.csv"
    png_path = args.output_dir / "figure10_metrics_panels.png"
    pdf_path = args.output_dir / "figure10_metrics_panels.pdf"

    write_csv(
        summary_path,
        summary_rows,
        fieldnames=[
            "K",
            "d",
            "mean_acc",
            "mean_acc_ci_low",
            "mean_acc_ci_high",
            "mean_rs_z_a",
            "mean_rs_z_a_ci_low",
            "mean_rs_z_a_ci_high",
            "mean_kla_z_l",
            "mean_kla_z_l_ci_low",
            "mean_kla_z_l_ci_high",
            "mean_kla_a_l",
            "mean_kla_a_l_ci_low",
            "mean_kla_a_l_ci_high",
        ],
    )
    write_csv(
        by_repeat_path,
        by_repeat_rows,
        fieldnames=[
            "K",
            "d",
            "repeat",
            "seed",
            "acc",
            "rs_z_a",
            "kla_z_l",
            "kla_a_l",
            "scale",
            "learning_rate",
            "weight_decay",
            "dropout",
            "job_dir",
            "repeat_result_dir",
        ],
    )
    plot_summary(summary_rows, output_png=png_path, output_pdf=pdf_path)

    print(f"[Done] wrote {summary_path}")
    print(f"[Done] wrote {by_repeat_path}")
    print(f"[Done] wrote {png_path}")
    print(f"[Done] wrote {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
