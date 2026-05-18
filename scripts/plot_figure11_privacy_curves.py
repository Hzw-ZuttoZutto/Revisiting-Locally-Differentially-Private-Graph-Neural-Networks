#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator


EXPECTED_X_EPS = [
    0.0078125,
    0.015625,
    0.03125,
    0.0625,
    0.125,
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
    16.0,
    32.0,
    64.0,
    128.0,
    256.0,
    512.0,
    1024.0,
    2048.0,
    4096.0,
    8192.0,
    16384.0,
]

VALID_SEARCH_STATUSES = {"completed", "skipped_existing_result"}
STEP_COLORS = {
    "x_steps=0": "#111111",
    "x_steps=2": "#234A6F",
    "x_steps=4": "#A7583B",
    "x_steps=8": "#5E7D4D",
    "x_steps=16": "#6E5A8A",
    "x_steps=32": "#8C6D31",
    "x_steps=64": "#3F6B6E",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate aggregated Figure 11 privacy-budget plots."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("paper_experiments/figure11"),
        help="Directory containing x_steps=*.yaml subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper_experiments/figure11/plots"),
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


def validate_rows(rows: list[dict[str, str]], step_name: str) -> list[dict[str, str]]:
    filtered = [row for row in rows if row.get("search_status") in VALID_SEARCH_STATUSES]
    if len(filtered) != len(EXPECTED_X_EPS):
        raise ValueError(
            f"{step_name}: expected {len(EXPECTED_X_EPS)} valid rows, found {len(filtered)}"
        )

    rows_by_eps: dict[float, dict[str, str]] = {}
    for row in filtered:
        context = f"{step_name} x_eps={row.get('x_eps', '')}"
        x_eps = require_float(row, "x_eps", context)
        require_float(row, "best_verify_test_acc_mean", context)
        require_float(row, "best_verify_sanity_e_pg_mean", context)
        if x_eps in rows_by_eps:
            raise ValueError(f"{step_name}: duplicate x_eps={x_eps}")
        rows_by_eps[x_eps] = row

    missing = [x for x in EXPECTED_X_EPS if x not in rows_by_eps]
    extra = [x for x in rows_by_eps if x not in EXPECTED_X_EPS]
    if missing or extra:
        raise ValueError(f"{step_name}: missing x_eps={missing}, extra x_eps={extra}")
    return [rows_by_eps[x_eps] for x_eps in EXPECTED_X_EPS]


def format_x_eps_tick(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def plot_accuracy(step_rows: dict[str, list[dict[str, str]]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    y_min = float("inf")
    y_max = float("-inf")

    for step_name in sorted(step_rows, key=step_sort_key):
        rows = step_rows[step_name]
        means = [float(row["best_verify_test_acc_mean"]) for row in rows]
        y_min = min(y_min, min(means))
        y_max = max(y_max, max(means))
        ax.plot(
            EXPECTED_X_EPS,
            means,
            color=STEP_COLORS[step_name],
            linewidth=1.8,
            label=step_name,
        )

    y_span = max(y_max - y_min, 1e-6)
    margin = y_span * 0.08
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Privacy Budget ε")
    ax.set_ylabel("Classification Accuracy (%)")
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.xaxis.set_major_locator(FixedLocator(EXPECTED_X_EPS))
    ax.xaxis.set_major_formatter(FixedFormatter([format_x_eps_tick(v) for v in EXPECTED_X_EPS]))
    ax.tick_params(axis="x", labelrotation=50, labelsize=7)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=8, handlelength=2.3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_relative_error(step_rows: dict[str, list[dict[str, str]]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    y_min = float("inf")
    y_max = float("-inf")

    for step_name in sorted(step_rows, key=step_sort_key):
        rows = step_rows[step_name]
        values = [float(row["best_verify_sanity_e_pg_mean"]) for row in rows]
        y_min = min(y_min, min(values))
        y_max = max(y_max, max(values))
        ax.plot(
            EXPECTED_X_EPS,
            values,
            color=STEP_COLORS[step_name],
            linewidth=1.8,
            label=step_name,
        )

    y_span = max(y_max - y_min, 1e-6)
    margin = y_span * 0.08
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Privacy Budget ε")
    ax.set_ylabel("Relative Error")
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.xaxis.set_major_locator(FixedLocator(EXPECTED_X_EPS))
    ax.xaxis.set_major_formatter(FixedFormatter([format_x_eps_tick(v) for v in EXPECTED_X_EPS]))
    ax.tick_params(axis="x", labelrotation=50, labelsize=7)
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
    if len(manifest_paths) != 7:
        raise ValueError(f"Expected 7 figure11 manifests under {input_root}, found {len(manifest_paths)}")

    step_rows: dict[str, list[dict[str, str]]] = {}
    for manifest_path in manifest_paths:
        step_name = parse_step_name(manifest_path)
        rows = load_manifest(manifest_path)
        ordered_rows = validate_rows(rows, step_name)
        step_rows[step_name] = ordered_rows
        print(f"{step_name}: points={len(ordered_rows)}")

    plot_accuracy(step_rows, output_dir / "figure11_best_test_accuracy_all_steps.pdf")
    plot_relative_error(step_rows, output_dir / "figure11_best_relative_error_all_steps.pdf")
    print(f"Saved Figure 11 plots to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
