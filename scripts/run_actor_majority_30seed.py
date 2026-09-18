#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import load_dataset
from hparams_search_scripts import mechanism_stage_utils
from hparams_search_scripts.mechanism_stage_context import (
    build_stage_command,
    resolve_job_context,
)
from main import (
    build_parser,
    build_sim_epoch_refresh_callback,
    finalize_parsed_args,
    seed_everything,
)
from models import NodeClassifier
from pre_smoothing_feature_cache import prepare_pre_smoothing_input
from trainer import Trainer
from utils import from_args


DEFAULT_PLOT_DATA = REPO_ROOT / "rebuttal_figure" / "figure1_heter_plot_data.csv"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "diagnostics" / "actor_majority_30run_figure1_eps10"
)
METHOD_ORDER = ("LPGNN", "PrivGE", "UPGNet-MBM", "UPGNet-PM", "FeatFree")
BACKBONE_ORDER = ("gcn", "sage", "gat")
BACKBONE_LABELS = {"gcn": "GCN", "sage": "GraphSAGE", "gat": "GAT"}
PIPELINE_TO_METHOD = {
    "figure3_pipeline1": "LPGNN",
    "figure3_pipeline2": "PrivGE",
    "figure3_pipeline3": "UPGNet-MBM",
    "figure3_pipeline4": "UPGNet-PM",
    "featfree": "FeatFree",
}
RUN_FIELDNAMES = (
    "method",
    "backbone",
    "pipeline",
    "candidate_id",
    "source_job_dir",
    "repeat_id",
    "seed",
    "best_epoch",
    "epochs_ran",
    "test_n",
    "num_classes",
    "global_majority_class",
    "global_majority_class_count",
    "global_node_count",
    "majority_class_test_count",
    "majority_baseline_accuracy_pct",
    "test_accuracy_pct",
    "predicted_majority_count",
    "predicted_majority_share_pct",
    "dominant_predicted_class",
    "dominant_predicted_count",
    "dominant_predicted_share_pct",
    "dominant_matches_global_majority",
    "mean_max_probability_pct",
    "correct_predictions",
    "elapsed_sec",
)


@dataclass(frozen=True)
class PlotPoint:
    method: str
    backbone: str
    pipeline: str
    candidate_id: int
    source_job_dir: Path

    @property
    def key(self) -> tuple[str, str]:
        return self.method, self.backbone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rerun the Actor epsilon=10 Figure 1 points and audit how often "
            "their test predictions equal Actor's global majority class."
        )
    )
    parser.add_argument("--mode", choices=("run", "aggregate"), default="run")
    parser.add_argument("--plot-data", type=Path, default=DEFAULT_PLOT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument("--shard-id", type=int)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_plot_points(path: Path) -> list[PlotPoint]:
    frame = pd.read_csv(path)
    selected = frame[
        (frame["dataset"] == "actor")
        & (frame["x_index"] == 9)
        & (frame["pipeline"].isin(PIPELINE_TO_METHOD))
    ].copy()
    selected["method"] = selected["pipeline"].map(PIPELINE_TO_METHOD)
    selected = selected.drop_duplicates(
        subset=["method", "backbone", "candidate_id", "source_job_dir"]
    )

    points = [
        PlotPoint(
            method=str(row.method),
            backbone=str(row.backbone),
            pipeline=str(row.pipeline),
            candidate_id=int(row.candidate_id),
            source_job_dir=Path(str(row.source_job_dir)),
        )
        for row in selected.itertuples(index=False)
    ]
    method_rank = {method: index for index, method in enumerate(METHOD_ORDER)}
    backbone_rank = {
        backbone: index for index, backbone in enumerate(BACKBONE_ORDER)
    }
    points.sort(key=lambda point: (method_rank[point.method], backbone_rank[point.backbone]))

    expected_keys = {
        (method, backbone) for method in METHOD_ORDER for backbone in BACKBONE_ORDER
    }
    actual_keys = {point.key for point in points}
    if actual_keys != expected_keys or len(points) != len(expected_keys):
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise RuntimeError(
            f"Expected 15 Figure 1 points; found {len(points)} "
            f"(missing={missing}, extra={extra})"
        )

    for point in points:
        best_config_path = point.source_job_dir / "best_config.yaml"
        if not best_config_path.is_file():
            raise FileNotFoundError(best_config_path)
        best_config = yaml.safe_load(best_config_path.read_text(encoding="utf-8"))
        best_candidate_id = int(best_config["best_candidate"]["candidate_id"])
        if best_candidate_id != point.candidate_id:
            raise RuntimeError(
                f"Plot/config candidate mismatch for {point.key}: "
                f"plot={point.candidate_id}, best_config={best_candidate_id}"
            )
    return points


def build_model_args(
    point: PlotPoint,
    *,
    seed: int,
    scratch_dir: Path,
) -> argparse.Namespace:
    context = resolve_job_context(str(point.source_job_dir))
    candidate = mechanism_stage_utils.job_candidate_by_id(
        context.job_spec, point.candidate_id
    )
    verify_defaults = context.defaults["stage"]["verify"]
    command = build_stage_command(
        context,
        candidate,
        stage_name="verify",
        output_dir=scratch_dir,
        seed=seed,
        max_epochs=int(verify_defaults["max_epochs"]),
        patience=int(verify_defaults["patience"]),
        repeats=1,
    )
    parser = build_parser()
    model_args = parser.parse_args(command[2:])
    finalize_parsed_args(parser, model_args)
    return model_args


def run_one(
    point: PlotPoint,
    *,
    repeat_id: int,
    base_seed: int,
    scratch_dir: Path,
) -> dict[str, Any]:
    started = time.time()
    seed = base_seed + repeat_id
    args = build_model_args(point, seed=seed, scratch_dir=scratch_dir)
    seed_everything(seed)

    dataset = from_args(load_dataset, args)
    data = dataset.clone().to(args.device)
    data, _ = prepare_pre_smoothing_input(data, args, rewrite_seed=seed)
    input_dim = int(getattr(data, "operator_num_features", data.num_features))
    model = from_args(
        NodeClassifier,
        args,
        input_dim=input_dim,
        num_classes=data.num_classes,
    )
    trainer = from_args(Trainer, args, logger=None)

    target = model._target_indices(data.y)
    global_counts = torch.bincount(target, minlength=data.num_classes)
    global_majority_class = int(global_counts.argmax().item())
    epoch_stats: list[dict[str, Any]] = []

    def validation_step_with_counts(
        eval_data,
        gnn_adj_t=None,
        smoother_adj_t=None,
    ):
        logits = model._forward_logits(
            eval_data,
            gnn_adj_t=gnn_adj_t,
            smoother_adj_t=smoother_adj_t,
        )
        eval_target = model._target_indices(eval_data.y)
        test_logits = logits[eval_data.test_mask]
        test_target = eval_target[eval_data.test_mask]
        prediction = test_logits.argmax(dim=1)
        test_true_counts = torch.bincount(
            test_target, minlength=eval_data.num_classes
        )
        prediction_counts = torch.bincount(
            prediction, minlength=eval_data.num_classes
        )
        dominant_predicted_class = int(prediction_counts.argmax().item())
        correct_predictions = int((prediction == test_target).sum().item())
        epoch_stats.append(
            {
                "test_n": int(test_target.numel()),
                "num_classes": int(eval_data.num_classes),
                "global_majority_class": global_majority_class,
                "global_majority_class_count": int(
                    global_counts[global_majority_class].item()
                ),
                "global_node_count": int(target.numel()),
                "majority_class_test_count": int(
                    test_true_counts[global_majority_class].item()
                ),
                "predicted_majority_count": int(
                    prediction_counts[global_majority_class].item()
                ),
                "dominant_predicted_class": dominant_predicted_class,
                "dominant_predicted_count": int(
                    prediction_counts[dominant_predicted_class].item()
                ),
                "mean_max_probability_pct": float(
                    test_logits.softmax(dim=1)
                    .max(dim=1)
                    .values.mean()
                    .item()
                    * 100.0
                ),
                "correct_predictions": correct_predictions,
            }
        )
        return {
            "val/loss": F.cross_entropy(
                logits[eval_data.val_mask], eval_target[eval_data.val_mask]
            ),
            "val/acc": model.accuracy(
                pred=logits[eval_data.val_mask],
                target=eval_target[eval_data.val_mask],
            )
            * 100,
            "test/acc": model.accuracy(
                pred=test_logits,
                target=test_target,
            )
            * 100,
        }

    model.validation_step = validation_step_with_counts
    best_metrics = trainer.fit(
        model,
        data,
        epoch_end_data_refresh_fn=build_sim_epoch_refresh_callback(args),
    )
    best_epoch = int(best_metrics["epoch"])
    if best_epoch < 1 or best_epoch > len(epoch_stats):
        raise RuntimeError(
            f"Invalid best epoch {best_epoch}; captured {len(epoch_stats)} epochs"
        )
    stat = epoch_stats[best_epoch - 1]
    test_n = int(stat["test_n"])
    counted_accuracy_pct = 100.0 * stat["correct_predictions"] / test_n
    metric_accuracy_pct = float(best_metrics["test/acc"].detach().item())
    if abs(metric_accuracy_pct - counted_accuracy_pct) > 1e-5:
        raise RuntimeError(
            f"Accuracy mismatch: metric={metric_accuracy_pct}, "
            f"counted={counted_accuracy_pct}"
        )

    return {
        "method": point.method,
        "backbone": point.backbone,
        "pipeline": point.pipeline,
        "candidate_id": point.candidate_id,
        "source_job_dir": str(point.source_job_dir),
        "repeat_id": repeat_id,
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_ran": len(epoch_stats),
        **stat,
        "majority_baseline_accuracy_pct": (
            100.0 * stat["majority_class_test_count"] / test_n
        ),
        "test_accuracy_pct": counted_accuracy_pct,
        "predicted_majority_share_pct": (
            100.0 * stat["predicted_majority_count"] / test_n
        ),
        "dominant_predicted_share_pct": (
            100.0 * stat["dominant_predicted_count"] / test_n
        ),
        "dominant_matches_global_majority": int(
            stat["dominant_predicted_class"] == global_majority_class
        ),
        "elapsed_sec": time.time() - started,
    }


def completed_task_keys(path: Path) -> set[tuple[str, str, int]]:
    if not path.is_file() or path.stat().st_size == 0:
        return set()
    frame = pd.read_csv(path)
    return {
        (str(row.method), str(row.backbone), int(row.repeat_id))
        for row in frame.itertuples(index=False)
    }


def run_shard(args: argparse.Namespace, points: list[PlotPoint]) -> None:
    if args.shard_id is None or args.num_shards is None:
        raise ValueError("--shard-id and --num-shards are required in run mode")
    if args.num_shards <= 0 or args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("Require 0 <= shard-id < num-shards")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")

    tasks = [
        (point, repeat_id)
        for point in points
        for repeat_id in range(args.repeats)
    ]
    tasks = [
        task
        for task_index, task in enumerate(tasks)
        if task_index % args.num_shards == args.shard_id
    ]
    if args.dry_run:
        for point, repeat_id in tasks:
            print(
                f"shard={args.shard_id} method={point.method} "
                f"backbone={point.backbone} candidate={point.candidate_id} "
                f"repeat={repeat_id} seed={args.base_seed + repeat_id}"
            )
        print(f"Dry run: {len(tasks)} tasks")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"shard_{args.shard_id:02d}.csv"
    completed = completed_task_keys(output_path)
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_FIELDNAMES)
        if write_header:
            writer.writeheader()
        for point, repeat_id in tasks:
            task_key = (point.method, point.backbone, repeat_id)
            if task_key in completed:
                continue
            try:
                row = run_one(
                    point,
                    repeat_id=repeat_id,
                    base_seed=args.base_seed,
                    scratch_dir=args.output_dir / "_training_scratch",
                )
                writer.writerow(row)
                handle.flush()
                print(
                    f"shard={args.shard_id} method={point.method} "
                    f"backbone={point.backbone} seed={row['seed']} "
                    f"acc={row['test_accuracy_pct']:.4f} "
                    f"pred_majority={row['predicted_majority_share_pct']:.4f}",
                    flush=True,
                )
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


def aggregate_results(args: argparse.Namespace, points: list[PlotPoint]) -> None:
    shard_paths = sorted(args.output_dir.glob("shard_*.csv"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard CSVs found in {args.output_dir}")
    frame = pd.concat((pd.read_csv(path) for path in shard_paths), ignore_index=True)
    key_columns = ["method", "backbone", "repeat_id"]
    duplicate_mask = frame.duplicated(subset=key_columns, keep=False)
    if bool(duplicate_mask.any()):
        duplicates = frame.loc[duplicate_mask, key_columns].to_dict("records")
        raise RuntimeError(f"Duplicate run keys: {duplicates[:10]}")

    expected_keys = {
        (point.method, point.backbone, repeat_id)
        for point in points
        for repeat_id in range(args.repeats)
    }
    actual_keys = {
        (str(row.method), str(row.backbone), int(row.repeat_id))
        for row in frame.itertuples(index=False)
    }
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise RuntimeError(
            f"Incomplete run set: expected={len(expected_keys)}, actual={len(actual_keys)}, "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )

    method_rank = {method: index for index, method in enumerate(METHOD_ORDER)}
    backbone_rank = {
        backbone: index for index, backbone in enumerate(BACKBONE_ORDER)
    }
    frame["_method_rank"] = frame["method"].map(method_rank)
    frame["_backbone_rank"] = frame["backbone"].map(backbone_rank)
    frame = frame.sort_values(
        ["_method_rank", "_backbone_rank", "repeat_id"]
    ).drop(columns=["_method_rank", "_backbone_rank"])

    summary = (
        frame.groupby(["method", "backbone"], sort=False)
        .agg(
            n_runs=("repeat_id", "count"),
            predicted_majority_share_pct_mean=(
                "predicted_majority_share_pct",
                "mean",
            ),
            predicted_majority_share_pct_std=(
                "predicted_majority_share_pct",
                "std",
            ),
            test_accuracy_pct_mean=("test_accuracy_pct", "mean"),
            test_accuracy_pct_std=("test_accuracy_pct", "std"),
            majority_baseline_accuracy_pct_mean=(
                "majority_baseline_accuracy_pct",
                "mean",
            ),
            dominant_matches_majority_runs=(
                "dominant_matches_global_majority",
                "sum",
            ),
        )
        .reset_index()
    )
    summary["_method_rank"] = summary["method"].map(method_rank)
    summary["_backbone_rank"] = summary["backbone"].map(backbone_rank)
    summary = summary.sort_values(["_method_rank", "_backbone_rank"]).drop(
        columns=["_method_rank", "_backbone_rank"]
    )

    paper_table = summary.pivot(
        index="method",
        columns="backbone",
        values="predicted_majority_share_pct_mean",
    ).reindex(index=METHOD_ORDER, columns=BACKBONE_ORDER)
    paper_table = paper_table.rename(columns=BACKBONE_LABELS)
    paper_table["Average"] = paper_table.mean(axis=1)
    paper_table = paper_table.reset_index()

    plot_frame = pd.read_csv(args.plot_data)
    figure_reference = plot_frame[
        (plot_frame["dataset"] == "actor")
        & (plot_frame["x_index"] == 9)
        & (plot_frame["pipeline"].isin(PIPELINE_TO_METHOD))
    ].copy()
    figure_reference["method"] = figure_reference["pipeline"].map(
        PIPELINE_TO_METHOD
    )
    figure_reference = figure_reference[
        ["method", "backbone", "candidate_id", "test_acc_mean"]
    ].drop_duplicates()
    rerun_first_ten = (
        frame[frame["repeat_id"] < 10]
        .groupby(["method", "backbone"], sort=False)
        .agg(
            rerun_first_10_test_accuracy_pct_mean=("test_accuracy_pct", "mean"),
            rerun_first_10_n=("repeat_id", "count"),
        )
        .reset_index()
    )
    reproduction = figure_reference.merge(
        rerun_first_ten,
        on=["method", "backbone"],
        how="inner",
        validate="one_to_one",
    ).merge(
        summary[["method", "backbone", "test_accuracy_pct_mean"]],
        on=["method", "backbone"],
        how="inner",
        validate="one_to_one",
    )
    reproduction = reproduction.rename(
        columns={
            "test_acc_mean": "figure_test_accuracy_pct_mean",
            "test_accuracy_pct_mean": "rerun_30_test_accuracy_pct_mean",
        }
    )
    reproduction["first_10_minus_figure_pp"] = (
        reproduction["rerun_first_10_test_accuracy_pct_mean"]
        - reproduction["figure_test_accuracy_pct_mean"]
    )
    if len(reproduction) != len(points):
        raise RuntimeError(
            f"Expected {len(points)} reproduction rows, found {len(reproduction)}"
        )
    if not bool((reproduction["rerun_first_10_n"] == 10).all()):
        raise RuntimeError("Figure reproduction requires exactly 10 original seeds per point")
    max_reproduction_error = float(
        reproduction["first_10_minus_figure_pp"].abs().max()
    )
    if max_reproduction_error > 1e-4:
        raise RuntimeError(
            f"Figure point reproduction failed: max error={max_reproduction_error} pp"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        args.output_dir / "all_runs.csv",
        index=False,
        float_format="%.6f",
    )
    summary.to_csv(
        args.output_dir / "summary_by_backbone.csv",
        index=False,
        float_format="%.6f",
    )
    paper_table.to_csv(
        args.output_dir / "paper_table_predicted_majority_pct.csv",
        index=False,
        float_format="%.6f",
    )
    reproduction.to_csv(
        args.output_dir / "figure_point_reproduction.csv",
        index=False,
        float_format="%.9f",
    )
    print(f"Aggregated {len(frame)} runs from {len(shard_paths)} shards")
    print(
        "Figure first-10-seed reproduction max absolute error: "
        f"{max_reproduction_error:.9f} pp"
    )
    print(paper_table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def main() -> None:
    args = parse_args()
    points = load_plot_points(args.plot_data)
    if args.mode == "run":
        run_shard(args, points)
    else:
        aggregate_results(args, points)


if __name__ == "__main__":
    main()
