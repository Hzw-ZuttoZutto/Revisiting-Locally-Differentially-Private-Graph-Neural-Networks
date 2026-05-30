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
LINE_COLOR = "#234A6F"
FILL_COLOR = "#8FB2D9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot mend_test manifest validation accuracy vs privacy budget."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/data/hzw/Rethinking_DP_GNN_runtime/paper_experiments/mend_test/manifest.csv"),
        help="Manifest CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/hzw/Rethinking_DP_GNN_runtime/paper_experiments/mend_test/plots"),
        help="Directory for output figures.",
    )
    return parser.parse_args()


def require_float(row: dict[str, str], field_name: str, context: str) -> float:
    raw_value = row.get(field_name, "").strip()
    if raw_value == "":
        raise ValueError(f"{context}: missing {field_name}")
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{context}: invalid {field_name}={raw_value}") from exc


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    filtered = [row for row in rows if row.get("search_status") in VALID_SEARCH_STATUSES]
    if len(filtered) != len(EXPECTED_X_EPS):
        raise ValueError(
            f"Expected {len(EXPECTED_X_EPS)} completed rows, found {len(filtered)} in {path}"
        )

    by_eps: dict[float, dict[str, str]] = {}
    for row in filtered:
        context = f"{path.name} x_eps={row.get('x_eps', '')}"
        x_eps = require_float(row, "x_eps", context)
        require_float(row, "best_verify_val_acc_mean", context)
        require_float(row, "best_verify_val_acc_std", context)
        if x_eps in by_eps:
            raise ValueError(f"Duplicate x_eps={x_eps} in {path}")
        by_eps[x_eps] = row

    missing = [x_eps for x_eps in EXPECTED_X_EPS if x_eps not in by_eps]
    extra = [x_eps for x_eps in by_eps if x_eps not in EXPECTED_X_EPS]
    if missing or extra:
        raise ValueError(f"Missing x_eps={missing}, extra x_eps={extra} in {path}")
    return [by_eps[x_eps] for x_eps in EXPECTED_X_EPS]


def format_x_eps_tick(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def plot_manifest(rows: list[dict[str, str]], output_dir: Path) -> tuple[Path, Path]:
    x_values = EXPECTED_X_EPS
    means = [float(row["best_verify_val_acc_mean"]) for row in rows]
    stds = [float(row["best_verify_val_acc_std"]) for row in rows]
    lower = [mean - std for mean, std in zip(means, stds)]
    upper = [mean + std for mean, std in zip(means, stds)]

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.plot(
        x_values,
        means,
        color=LINE_COLOR,
        linewidth=2.0,
        marker="o",
        markersize=3.8,
        label="Validation Accuracy",
    )
    ax.fill_between(
        x_values,
        lower,
        upper,
        color=FILL_COLOR,
        alpha=0.25,
        linewidth=0.0,
        label="?1 std",
    )

    y_min = min(lower)
    y_max = max(upper)
    y_span = max(y_max - y_min, 1e-6)
    margin = y_span * 0.08

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Privacy Budget ?")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.xaxis.set_major_locator(FixedLocator(EXPECTED_X_EPS))
    ax.xaxis.set_major_formatter(FixedFormatter([format_x_eps_tick(v) for v in EXPECTED_X_EPS]))
    ax.tick_params(axis="x", labelrotation=50, labelsize=7)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower left", frameon=False, fontsize=8, handlelength=2.4)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "mend_test_val_vs_privacy.png"
    pdf_path = output_dir / "mend_test_val_vs_privacy.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main() -> int:
    args = parse_args()
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    rows = load_rows(args.manifest.resolve())
    png_path, pdf_path = plot_manifest(rows, args.output_dir.resolve())
    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
