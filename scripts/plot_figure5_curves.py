#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter


EXPECTED_SCALES = [
    0.125,
    0.25,
    0.5,
    1.0,
    4.0,
    16.0,
    64.0,
    256.0,
    1024.0,
    4096.0,
    16384.0,
    65536.0,
    262144.0,
    524288.0,
    1048576.0,
    4194304.0,
]

DATASETS = [
    ("cora", "cora"),
    ("lastfm", "lastfm"),
    ("ogbn-arxiv", "ogbn-arxiv"),
    ("Books-History", "books-history"),
]

FILTERS = {
    "feature": "raw",
    "mechanism": "mbm",
    "x_eps": "0.001",
    "m": "best",
    "norm": "true",
    "smoother": "kprop",
    "backbone": "sage",
    "use_nfr": "false",
    "search_status": "completed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Figure 5 per-dataset paper plots.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("paper_experiments/figure5/figure5.yaml/manifest.csv"),
        help="Path to the Figure 5 manifest CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper_experiments/figure5/plots"),
        help="Directory where the plot files will be written.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def filter_rows(rows: list[dict[str, str]], dataset_name: str) -> list[dict[str, str]]:
    filtered: list[dict[str, str]] = []
    for row in rows:
        if row.get("dataset") != dataset_name:
            continue
        if any(row.get(key) != value for key, value in FILTERS.items()):
            continue
        filtered.append(row)
    return filtered


def validate_dataset_rows(rows: list[dict[str, str]], dataset_name: str) -> list[dict[str, str]]:
    if len(rows) != len(EXPECTED_SCALES):
        raise ValueError(
            f"{dataset_name}: expected {len(EXPECTED_SCALES)} rows, found {len(rows)}"
        )

    rows_by_scale: dict[float, dict[str, str]] = {}
    for row in rows:
        try:
            scale = float(row["norm_scale"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{dataset_name}: invalid norm_scale in row {row}") from exc
        if scale in rows_by_scale:
            raise ValueError(f"{dataset_name}: duplicate norm_scale={scale}")
        rows_by_scale[scale] = row

    missing = [scale for scale in EXPECTED_SCALES if scale not in rows_by_scale]
    extra = [scale for scale in rows_by_scale if scale not in EXPECTED_SCALES]
    if missing or extra:
        raise ValueError(f"{dataset_name}: missing scales={missing}, extra scales={extra}")

    ordered_rows = [rows_by_scale[scale] for scale in EXPECTED_SCALES]
    for row in ordered_rows:
        if row.get("best_verify_test_acc_mean", "").strip() == "":
            raise ValueError(f"{dataset_name}: missing best_verify_test_acc_mean for scale={row['norm_scale']}")
        if row.get("best_verify_test_acc_std", "").strip() == "":
            raise ValueError(f"{dataset_name}: missing best_verify_test_acc_std for scale={row['norm_scale']}")
    return ordered_rows


def format_scale_tick(scale: float) -> str:
    return str(int(scale)) if float(scale).is_integer() else str(scale)


def plot_dataset(rows: list[dict[str, str]], dataset_name: str, output_prefix: Path) -> None:
    x_values = EXPECTED_SCALES
    means = [float(row["best_verify_test_acc_mean"]) for row in rows]
    stds = [float(row["best_verify_test_acc_std"]) for row in rows]
    lower = [mean - std for mean, std in zip(means, stds)]
    upper = [mean + std for mean, std in zip(means, stds)]

    y_min = min(lower)
    y_max = max(upper)
    y_span = max(y_max - y_min, 1e-6)
    margin = y_span * 0.08

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(x_values, means, color="black", linewidth=1.6)
    ax.fill_between(x_values, lower, upper, color="0.7", alpha=0.35, linewidth=0)

    ax.set_xscale("log")
    ax.set_xlabel("r")
    ax.set_ylabel("Classification Accuracy (%)")
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.xaxis.set_major_locator(FixedLocator(x_values))
    ax.xaxis.set_major_formatter(FixedFormatter([format_scale_tick(scale) for scale in x_values]))
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(
        f"{dataset_name}: points={len(rows)} "
        f"r_values={[format_scale_tick(scale) for scale in x_values]}"
    )


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    rows = load_manifest(manifest_path)
    for dataset_name, output_stem in DATASETS:
        dataset_rows = filter_rows(rows, dataset_name)
        ordered_rows = validate_dataset_rows(dataset_rows, dataset_name)
        plot_dataset(ordered_rows, dataset_name, output_dir / output_stem)

    print(f"Saved plots to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

