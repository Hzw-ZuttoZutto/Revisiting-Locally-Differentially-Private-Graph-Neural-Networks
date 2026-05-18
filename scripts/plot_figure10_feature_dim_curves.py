#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator


EXPECTED_FEATURE_DIMS = [
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

VALID_SEARCH_STATUSES = {"completed", "skipped_existing_result"}
STEP_COLORS = {
    "x_steps=2": "#234A6F",
    "x_steps=4": "#A7583B",
    "x_steps=8": "#5E7D4D",
    "x_steps=16": "#6E5A8A",
    "x_steps=32": "#8C6D31",
    "x_steps=64": "#3F6B6E",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate aggregated Figure 10 feature-dimension plots."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("paper_experiments/figure10"),
        help="Directory containing x_steps=*.yaml subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper_experiments/figure10/plots"),
        help="Directory where PDF plots will be written.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_step_name(manifest_path: Path) -> str:
    return manifest_path.parent.name.removesuffix(".yaml")


def step_sort_key(step_name: str) -> int:
    try:
        return int(step_name.split("=", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Unexpected step directory name: {step_name}") from exc


def require_float(row: dict[str, str], field_name: str, context: str) -> float:
    raw_value = row.get(field_name, "").strip()
    if raw_value == "":
        raise ValueError(f"{context}: missing {field_name}")
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{context}: invalid {field_name}={raw_value}") from exc


def require_int(row: dict[str, str], field_name: str, context: str) -> int:
    raw_value = row.get(field_name, "").strip()
    if raw_value == "":
        raise ValueError(f"{context}: missing {field_name}")
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{context}: invalid {field_name}={raw_value}") from exc


def choose_best_rows(rows: list[dict[str, str]], step_name: str) -> list[dict[str, str]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        context = f"{step_name} row feature_dim={row.get('feature_dim', '')} scale={row.get('scale', '')}"
        if row.get("search_status") not in VALID_SEARCH_STATUSES:
            continue
        feature_dim = require_int(row, "feature_dim", context)
        require_float(row, "best_verify_val_acc_mean", context)
        require_float(row, "best_verify_test_acc_mean", context)
        require_float(row, "best_verify_test_acc_std", context)
        require_float(row, "best_verify_sanity_e_pg_mean", context)
        grouped[feature_dim].append(row)

    dims = sorted(grouped)
    if dims != EXPECTED_FEATURE_DIMS:
        raise ValueError(f"{step_name}: expected feature_dims={EXPECTED_FEATURE_DIMS}, found={dims}")

    ordered_rows: list[dict[str, str]] = []
    for feature_dim in EXPECTED_FEATURE_DIMS:
        candidates = grouped[feature_dim]
        if len(candidates) != 10:
            raise ValueError(
                f"{step_name}: feature_dim={feature_dim} expected 10 scale rows, found {len(candidates)}"
            )

        def score(row: dict[str, str]) -> tuple[float, float, float]:
            context = f"{step_name} feature_dim={feature_dim} scale={row.get('scale', '')}"
            return (
                require_float(row, "best_verify_val_acc_mean", context),
                require_float(row, "best_verify_test_acc_mean", context),
                -require_float(row, "scale", context),
            )

        ordered_rows.append(max(candidates, key=score))
    return ordered_rows


def format_feature_dim_tick(value: float) -> str:
    return str(int(value))


def plot_accuracy(step_rows: dict[str, list[dict[str, str]]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.1))
    y_min = float("inf")
    y_max = float("-inf")

    for step_name in sorted(step_rows, key=step_sort_key):
        rows = step_rows[step_name]
        x_values = EXPECTED_FEATURE_DIMS
        means = [float(row["best_verify_test_acc_mean"]) for row in rows]
        y_min = min(y_min, min(means))
        y_max = max(y_max, max(means))
        ax.plot(
            x_values,
            means,
            color=STEP_COLORS[step_name],
            linewidth=1.8,
            label=step_name,
        )

    y_span = max(y_max - y_min, 1e-6)
    margin = y_span * 0.08
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Feature Dimension")
    ax.set_ylabel("Classification Accuracy (%)")
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.xaxis.set_major_locator(FixedLocator(EXPECTED_FEATURE_DIMS))
    ax.xaxis.set_major_formatter(FixedFormatter([format_feature_dim_tick(v) for v in EXPECTED_FEATURE_DIMS]))
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=8, handlelength=2.3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_relative_error(step_rows: dict[str, list[dict[str, str]]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.1))
    y_min = float("inf")
    y_max = float("-inf")

    for step_name in sorted(step_rows, key=step_sort_key):
        rows = step_rows[step_name]
        x_values = EXPECTED_FEATURE_DIMS
        values = [float(row["best_verify_sanity_e_pg_mean"]) for row in rows]
        y_min = min(y_min, min(values))
        y_max = max(y_max, max(values))
        ax.plot(
            x_values,
            values,
            color=STEP_COLORS[step_name],
            linewidth=1.8,
            label=step_name,
        )

    y_span = max(y_max - y_min, 1e-6)
    margin = y_span * 0.08
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Feature Dimension")
    ax.set_ylabel("Relative Error")
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.xaxis.set_major_locator(FixedLocator(EXPECTED_FEATURE_DIMS))
    ax.xaxis.set_major_formatter(FixedFormatter([format_feature_dim_tick(v) for v in EXPECTED_FEATURE_DIMS]))
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=8, handlelength=2.3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    manifest_paths = sorted(
        input_root.glob("x_steps=*.yaml/manifest.csv"),
        key=lambda path: step_sort_key(parse_step_name(path)),
    )
    if len(manifest_paths) != 6:
        raise ValueError(f"Expected 6 figure10 manifests under {input_root}, found {len(manifest_paths)}")

    step_rows: dict[str, list[dict[str, str]]] = {}
    for manifest_path in manifest_paths:
        step_name = parse_step_name(manifest_path)
        rows = load_manifest(manifest_path)
        best_rows = choose_best_rows(rows, step_name)
        step_rows[step_name] = best_rows
        selected_scales = [row["scale"] for row in best_rows]
        print(f"{step_name}: selected_scales={selected_scales}")

    plot_accuracy(step_rows, output_dir / "figure10_best_test_accuracy_all_steps.pdf")
    plot_relative_error(step_rows, output_dir / "figure10_best_relative_error_all_steps.pdf")
    print(f"Saved Figure 10 plots to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
