#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from decimal import Decimal
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_figure5_final_norm_scale_curves import (
    EPSILON_SPECS,
    EXPECTED_REPEATS,
    SCALE_LABELS,
    SCALE_SPECS,
    STYLE_CONFIGS,
    VerifyRecord,
    build_plot_rows,
    collect_verify_records,
    decimal_key,
    validate_plot_rows,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIGURE5_ROOT = REPO_ROOT / "rebuttal_experiments" / "figure5_heter" / "figure5.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "rebuttal_figure"
EXPECTED_MANIFEST_ROWS = 168

# These are the exact layout/style values embedded in the reference PDF.
FIGSIZE_X = 13.2
FIGSIZE_Y = 7.1
BOTTOM = 0.36
TOP = 0.99
LEFT = 0.08
RIGHT = 0.985
FONTSIZE = 31
X_LABEL_FONTSIZE = 30
LEGEND_FONTSIZE = 30
TICKLABEL_FONTSIZE = 22
LINEWIDTH = 2.5
MARKERSIZE = 11


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Figure 5 norm-scale curves on Flickr.")
    parser.add_argument(
        "--figure5-root",
        type=Path,
        default=DEFAULT_FIGURE5_ROOT,
        help="Root containing the Flickr figure5.yaml result directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where figure5_heter.pdf and figure5_heter.png are written.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Number of bootstrap resamples for 95%% confidence intervals.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=12345,
        help="Random seed for bootstrap resampling.",
    )
    return parser.parse_args()


def load_manifest_index(manifest_path: Path) -> dict[tuple[Decimal, Decimal], dict[str, str]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_MANIFEST_ROWS:
        raise ValueError(
            f"{manifest_path}: expected {EXPECTED_MANIFEST_ROWS} rows, found {len(rows)}"
        )

    expected = {
        "dataset": "attributedgraph-flickr",
        "feature": "raw",
        "mechanism": "mbm",
        "m": "best",
        "smoother": "hoa",
        "backbone": "sage",
    }
    index: dict[tuple[Decimal, Decimal], dict[str, str]] = {}
    for row in rows:
        for field, value in expected.items():
            if row.get(field) != value:
                raise ValueError(
                    f"{manifest_path}: expected {field}={value}, found {row.get(field)!r}"
                )
        if row.get("norm") not in {"True", "true", "1"}:
            raise ValueError(f"{manifest_path}: expected norm=true, found {row.get('norm')!r}")
        if row.get("use_nfr") not in {"False", "false", "0"}:
            raise ValueError(
                f"{manifest_path}: expected use_nfr=false, found {row.get('use_nfr')!r}"
            )

        key = (
            decimal_key(row.get("x_eps"), field="x_eps"),
            decimal_key(row.get("norm_scale"), field="norm_scale"),
        )
        if key in index:
            raise ValueError(f"Duplicate manifest key {key} in {manifest_path}")
        index[key] = row
    return index


def load_records(figure5_root: Path) -> list[VerifyRecord]:
    manifest_path = figure5_root / "manifest.csv"
    index = load_manifest_index(manifest_path)
    records: list[VerifyRecord] = []

    for epsilon, _ in EPSILON_SPECS:
        for scale_exponent, scale_value, _ in SCALE_SPECS:
            key = (epsilon, scale_value)
            if key not in index:
                raise ValueError(
                    f"{manifest_path}: missing row for eps={epsilon}, r=2^{scale_exponent}"
                )
            records.extend(
                collect_verify_records(
                    source="figure5_heter",
                    epsilon=epsilon,
                    scale_exponent=scale_exponent,
                    scale_value=scale_value,
                    job_dir=Path(index[key]["job_dir"]),
                )
            )

    expected_records = len(EPSILON_SPECS) * len(SCALE_SPECS) * EXPECTED_REPEATS
    if len(records) != expected_records:
        raise RuntimeError(f"Expected {expected_records} records, found {len(records)}")
    return records


def plot_curves(rows: list[dict[str, object]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(FIGSIZE_X, FIGSIZE_Y))
    fig.subplots_adjust(bottom=BOTTOM, top=TOP, left=LEFT, right=RIGHT)

    for epsilon, epsilon_label in EPSILON_SPECS:
        sub = sorted(
            [row for row in rows if Decimal(str(row["epsilon"])) == epsilon],
            key=lambda row: int(row["x_index"]),
        )
        if len(sub) != len(SCALE_SPECS):
            raise RuntimeError(
                f"{epsilon}: expected {len(SCALE_SPECS)} plot points, found {len(sub)}"
            )

        style = STYLE_CONFIGS[epsilon]
        x = np.asarray([float(row["norm_scale"]) for row in sub], dtype=float)
        y = np.asarray([float(row["test_acc_mean"]) for row in sub], dtype=float)
        low = np.asarray([float(row["test_acc_ci_low"]) for row in sub], dtype=float)
        high = np.asarray([float(row["test_acc_ci_high"]) for row in sub], dtype=float)
        ax.errorbar(
            x,
            y,
            yerr=np.vstack([y - low, high - y]),
            label=epsilon_label,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            markersize=MARKERSIZE,
            linewidth=LINEWIDTH,
            capsize=0,
            markerfacecolor="none",
            markeredgewidth=2,
        )

    scale_values = [float(scale_value) for _, scale_value, _ in SCALE_SPECS]
    ax.set_xscale("log", base=2)
    ax.set_xticks(scale_values)
    ax.set_xticklabels(SCALE_LABELS, rotation=35, ha="right")
    ax.axvline(1.0, color="black", linestyle="--", linewidth=2.4, alpha=0.95, zorder=0)
    ax.set_xlabel(r"$r$", fontsize=X_LABEL_FONTSIZE, fontweight="medium")
    ax.set_ylabel("Test Accuracy", fontsize=FONTSIZE, fontweight="medium")
    ax.grid(True, color="white", linestyle="-", linewidth=1, alpha=1.0)
    ax.tick_params(axis="both", which="major", labelsize=TICKLABEL_FONTSIZE)
    handles, labels = ax.get_legend_handles_labels()
    first_legend = ax.legend(
        handles[:3],
        labels[:3],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.46),
        ncol=3,
        fontsize=LEGEND_FONTSIZE,
        frameon=False,
        shadow=False,
        borderpad=0.2,
        handlelength=1.5,
        columnspacing=0.85,
    )
    ax.add_artist(first_legend)
    ax.legend(
        handles[3:],
        labels[3:],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.64),
        ncol=2,
        fontsize=LEGEND_FONTSIZE,
        frameon=False,
        shadow=False,
        borderpad=0.2,
        handlelength=1.5,
        columnspacing=0.85,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "figure5_heter.pdf"
    png_path = output_dir / "figure5_heter.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


def main() -> None:
    args = parse_args()
    records = load_records(args.figure5_root)
    rows = build_plot_rows(
        records,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    validate_plot_rows(rows)
    plot_curves(rows, args.output_dir)
    print(f"Long rows: {len(records)}")
    print(f"Plot rows: {len(rows)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
