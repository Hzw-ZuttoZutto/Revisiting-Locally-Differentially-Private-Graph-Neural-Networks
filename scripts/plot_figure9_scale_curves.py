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


VALID_SEARCH_STATUSES = {"completed", "skipped_existing_result"}
MODE_STYLES = {
    "direct": {"color": "#234A6F", "fill": "#B7C8D9", "label": "direct"},
    "learned_projected": {"color": "#A7583B", "fill": "#E7C9BD", "label": "learned_projected"},
    "random_projected": {"color": "#5E7D4D", "fill": "#C7D5BD", "label": "random_projected"},
}
DATASET_ORDER = ["books-history", "cora", "lastfm", "ogbn-arxiv"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate aggregated Figure 9 scale plots."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        nargs="+",
        default=[Path("paper_experiments/figure9")],
        help="One or more directories containing figure9-style experiment outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper_experiments/figure9/plots"),
        help="Directory where PDF plots will be written.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def require_float(row: dict[str, str], field_name: str, context: str) -> float:
    raw_value = row.get(field_name, "").strip()
    if raw_value == "":
        raise ValueError(f"{context}: missing {field_name}")
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{context}: invalid {field_name}={raw_value}") from exc


def format_scale_tick(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def infer_expected_scales(manifest_paths: list[Path]) -> list[float]:
    all_scales: set[float] = set()
    for manifest_path in manifest_paths:
        rows = load_manifest(manifest_path)
        all_scales.update(
            float(row["scale"])
            for row in rows
            if row.get("search_status") in VALID_SEARCH_STATUSES and row.get("scale", "").strip() != ""
        )
    if not all_scales:
        raise ValueError("Could not infer scale grid from manifests")
    return sorted(all_scales)


def validate_rows(rows: list[dict[str, str]], context: str, expected_scales: list[float]) -> list[dict[str, str]]:
    filtered = [row for row in rows if row.get("search_status") in VALID_SEARCH_STATUSES]
    if len(filtered) != len(expected_scales):
        raise ValueError(
            f"{context}: expected {len(expected_scales)} valid rows, found {len(filtered)}"
        )

    rows_by_scale: dict[float, dict[str, str]] = {}
    for row in filtered:
        scale = require_float(row, "scale", context)
        require_float(row, "best_verify_test_acc_mean", f"{context} scale={format_scale_tick(scale)}")
        require_float(row, "best_verify_test_acc_std", f"{context} scale={format_scale_tick(scale)}")
        if scale in rows_by_scale:
            raise ValueError(f"{context}: duplicate scale={scale}")
        rows_by_scale[scale] = row

    missing = [scale for scale in expected_scales if scale not in rows_by_scale]
    extra = [scale for scale in rows_by_scale if scale not in expected_scales]
    if missing or extra:
        raise ValueError(f"{context}: missing scales={missing}, extra scales={extra}")
    return [rows_by_scale[scale] for scale in expected_scales]


def plot_dataset(
    dataset_name: str,
    mode_rows: dict[str, list[dict[str, str]]],
    expected_scales: list[float],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    y_min = float("inf")
    y_max = float("-inf")

    for mode_name in ["direct", "learned_projected", "random_projected"]:
        rows = mode_rows[mode_name]
        style = MODE_STYLES[mode_name]
        means = [float(row["best_verify_test_acc_mean"]) for row in rows]
        stds = [float(row["best_verify_test_acc_std"]) for row in rows]
        lower = [mean - std for mean, std in zip(means, stds)]
        upper = [mean + std for mean, std in zip(means, stds)]
        y_min = min(y_min, min(lower))
        y_max = max(y_max, max(upper))

        ax.plot(
            expected_scales,
            means,
            color=style["color"],
            linewidth=1.8,
            label=style["label"],
        )
        ax.fill_between(
            expected_scales,
            lower,
            upper,
            color=style["fill"],
            alpha=0.20,
            linewidth=0,
        )

    y_span = max(y_max - y_min, 1e-6)
    margin = y_span * 0.08
    ax.set_xscale("log")
    ax.set_xlabel("Scale")
    ax.set_ylabel("Classification Accuracy (%)")
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.xaxis.set_major_locator(FixedLocator(expected_scales))
    ax.xaxis.set_major_formatter(FixedFormatter([format_scale_tick(v) for v in expected_scales]))
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
    input_roots = [path.resolve() for path in args.input_root]
    output_dir = args.output_dir.resolve()

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    manifest_paths: list[Path] = []
    for input_root in input_roots:
        manifest_paths.extend(sorted(input_root.glob("*/*/x_steps=*/manifest.csv")))
    if not manifest_paths:
        raise ValueError(f"No figure9 manifests found under: {input_roots}")
    expected_scales = infer_expected_scales(manifest_paths)
    print(f"inferred_scales={[format_scale_tick(v) for v in expected_scales]}")

    dataset_mode_rows: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(dict)
    dataset_mode_contexts: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for manifest_path in manifest_paths:
        mode_name = manifest_path.parents[2].name
        dataset_name = manifest_path.parents[1].name
        step_dir = manifest_path.parent.name
        context = f"{mode_name}/{dataset_name}/{step_dir}"
        rows = load_manifest(manifest_path)
        dataset_mode_rows[dataset_name].setdefault(mode_name, [])
        dataset_mode_rows[dataset_name][mode_name].extend(rows)
        dataset_mode_contexts[dataset_name][mode_name].append(context)
        print(f"{context}: loaded")

    datasets = sorted(dataset_mode_rows, key=lambda name: DATASET_ORDER.index(name))
    if datasets != DATASET_ORDER:
        raise ValueError(f"Unexpected datasets for figure9: {datasets}")

    for dataset_name in datasets:
        mode_rows = dataset_mode_rows[dataset_name]
        if sorted(mode_rows) != sorted(MODE_STYLES):
            raise ValueError(f"{dataset_name}: expected modes={sorted(MODE_STYLES)}, found={sorted(mode_rows)}")
        validated_mode_rows: dict[str, list[dict[str, str]]] = {}
        for mode_name, rows in mode_rows.items():
            contexts = ", ".join(dataset_mode_contexts[dataset_name][mode_name])
            validated_mode_rows[mode_name] = validate_rows(rows, f"{dataset_name}/{mode_name} [{contexts}]", expected_scales)
        output_path = output_dir / f"{dataset_name}.pdf"
        plot_dataset(dataset_name, validated_mode_rows, expected_scales, output_path)
        print(f"{dataset_name}: saved={output_path.name}")

    print(f"Saved Figure 9 plots to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
