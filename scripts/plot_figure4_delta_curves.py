#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator


EXPECTED_EPSILONS = [
    0.0001,
    0.001,
    0.01,
    0.1,
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
]

DATASETS = [
    ("cora", "cora"),
    ("lastfm", "lastfm"),
]

MECHANISMS = [
    ("pm", "pm"),
    ("mbm", "mbm"),
    ("hds", "hds"),
]

ORI_FILTERS = {
    "feature": "raw",
    "m": "best",
    "norm": "false",
    "smoother": "kprop",
    "backbone": "sage",
    "use_nfr": "false",
    "search_status": "completed",
}

SIM_FILTERS = {
    "feature": "sim",
    "x_eps": "inf",
    "m": "best",
    "norm": "false",
    "smoother": "kprop",
    "backbone": "sage",
    "use_nfr": "false",
    "search_status": "completed",
}

LDP_COLOR = "#234A6F"
LDP_FILL = "#B7C8D9"
SIM_COLOR = "#A7583B"
SIM_FILL = "#E7C9BD"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Figure 4 per-dataset/per-mechanism accuracy plots."
    )
    parser.add_argument(
        "--ori-manifest",
        type=Path,
        default=Path("paper_experiments/figure4/figure4_ori.yaml/manifest.csv"),
        help="Path to the Figure 4 ori manifest CSV.",
    )
    parser.add_argument(
        "--sim-manifest",
        type=Path,
        default=Path("paper_experiments/figure4/figure4_sim.yaml/manifest.csv"),
        help="Path to the Figure 4 sim manifest CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper_experiments/figure4/plots"),
        help="Directory where the plot files will be written.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def format_epsilon(epsilon: float) -> str:
    return str(int(epsilon)) if float(epsilon).is_integer() else str(epsilon)


def filter_rows(
    rows: list[dict[str, str]],
    dataset_name: str,
    mechanism_name: str,
    filters: dict[str, str],
) -> list[dict[str, str]]:
    filtered: list[dict[str, str]] = []
    for row in rows:
        if row.get("dataset") != dataset_name:
            continue
        if row.get("mechanism") != mechanism_name:
            continue
        if any(row.get(key) != value for key, value in filters.items()):
            continue
        filtered.append(row)
    return filtered


def build_row_index(
    rows: list[dict[str, str]],
    epsilon_field: str,
    dataset_name: str,
    mechanism_name: str,
    source_name: str,
) -> dict[float, dict[str, str]]:
    row_index: dict[float, dict[str, str]] = {}
    for row in rows:
        raw_value = row.get(epsilon_field, "").strip()
        if raw_value == "":
            raise ValueError(
                f"{source_name} {dataset_name}/{mechanism_name}: missing {epsilon_field} in row {row}"
            )
        try:
            epsilon = float(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"{source_name} {dataset_name}/{mechanism_name}: invalid {epsilon_field}={raw_value}"
            ) from exc
        if epsilon in row_index:
            raise ValueError(
                f"{source_name} {dataset_name}/{mechanism_name}: duplicate {epsilon_field}={raw_value}"
            )
        row_index[epsilon] = row

    missing = [epsilon for epsilon in EXPECTED_EPSILONS if epsilon not in row_index]
    extra = [epsilon for epsilon in row_index if epsilon not in EXPECTED_EPSILONS]
    if missing or extra:
        raise ValueError(
            f"{source_name} {dataset_name}/{mechanism_name}: missing eps={missing}, extra eps={extra}"
        )
    return row_index


def require_metric(
    row: dict[str, str],
    field_name: str,
    dataset_name: str,
    mechanism_name: str,
    epsilon: float,
    source_name: str,
) -> float:
    raw_value = row.get(field_name, "").strip()
    if raw_value == "":
        raise ValueError(
            f"{source_name} {dataset_name}/{mechanism_name} epsilon={format_epsilon(epsilon)}: "
            f"missing {field_name}"
        )
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{source_name} {dataset_name}/{mechanism_name} epsilon={format_epsilon(epsilon)}: "
            f"invalid {field_name}={raw_value}"
        ) from exc


def build_series(
    ori_rows: list[dict[str, str]],
    sim_rows: list[dict[str, str]],
    dataset_name: str,
    mechanism_name: str,
) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
    if len(ori_rows) != len(EXPECTED_EPSILONS):
        raise ValueError(
            f"ori {dataset_name}/{mechanism_name}: expected {len(EXPECTED_EPSILONS)} rows, found {len(ori_rows)}"
        )
    if len(sim_rows) != len(EXPECTED_EPSILONS):
        raise ValueError(
            f"sim {dataset_name}/{mechanism_name}: expected {len(EXPECTED_EPSILONS)} rows, found {len(sim_rows)}"
        )

    ori_index = build_row_index(ori_rows, "x_eps", dataset_name, mechanism_name, "ori")
    sim_index = build_row_index(sim_rows, "sim_reference_eps", dataset_name, mechanism_name, "sim")

    ori_means: list[float] = []
    ori_stds: list[float] = []
    sim_means: list[float] = []
    sim_stds: list[float] = []
    for epsilon in EXPECTED_EPSILONS:
        ori_row = ori_index[epsilon]
        sim_row = sim_index[epsilon]

        ori_mean = require_metric(
            ori_row, "best_verify_test_acc_mean", dataset_name, mechanism_name, epsilon, "ori"
        )
        sim_mean = require_metric(
            sim_row, "best_verify_test_acc_mean", dataset_name, mechanism_name, epsilon, "sim"
        )
        ori_std = require_metric(
            ori_row, "best_verify_test_acc_std", dataset_name, mechanism_name, epsilon, "ori"
        )
        sim_std = require_metric(
            sim_row, "best_verify_test_acc_std", dataset_name, mechanism_name, epsilon, "sim"
        )

        ori_means.append(ori_mean)
        ori_stds.append(ori_std)
        sim_means.append(sim_mean)
        sim_stds.append(sim_std)

    return EXPECTED_EPSILONS, ori_means, ori_stds, sim_means, sim_stds


def plot_series(
    x_values: list[float],
    ori_means: list[float],
    ori_stds: list[float],
    sim_means: list[float],
    sim_stds: list[float],
    output_prefix: Path,
) -> None:
    x_positions = list(range(len(x_values)))
    ori_lower = [mean - std for mean, std in zip(ori_means, ori_stds)]
    ori_upper = [mean + std for mean, std in zip(ori_means, ori_stds)]
    sim_lower = [mean - std for mean, std in zip(sim_means, sim_stds)]
    sim_upper = [mean + std for mean, std in zip(sim_means, sim_stds)]

    y_min = min(ori_lower + sim_lower)
    y_max = max(ori_upper + sim_upper)
    y_span = max(y_max - y_min, 1e-6)
    margin = y_span * 0.08

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(x_positions, ori_means, color=LDP_COLOR, linewidth=1.9, label="LDP")
    ax.fill_between(x_positions, ori_lower, ori_upper, color=LDP_FILL, alpha=0.22, linewidth=0)
    ax.plot(x_positions, sim_means, color=SIM_COLOR, linewidth=1.9, label="Simulate")
    ax.fill_between(x_positions, sim_lower, sim_upper, color=SIM_FILL, alpha=0.18, linewidth=0)

    ax.set_xlabel("Privacy Budget ε")
    ax.set_ylabel("Classification Accuracy (%)")
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_xlim(-0.2, len(x_positions) - 0.8)
    ax.xaxis.set_major_locator(FixedLocator(x_positions))
    ax.xaxis.set_major_formatter(FixedFormatter([format_epsilon(value) for value in x_values]))
    ax.tick_params(axis="x", labelrotation=50, labelsize=7)
    ax.tick_params(axis="y", labelsize=9)
    plt.setp(ax.get_xticklabels(), ha="right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend_handles = [
        Line2D([0], [0], color=SIM_COLOR, linewidth=1.9, label="Simulate"),
        Line2D([0], [0], color=LDP_COLOR, linewidth=1.9, label="LDP"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.23),
        frameon=True,
        facecolor="white",
        edgecolor="0.8",
        framealpha=0.92,
        fontsize=9,
        handlelength=2.4,
    )
    fig.tight_layout()

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    ori_manifest = args.ori_manifest.resolve()
    sim_manifest = args.sim_manifest.resolve()
    output_dir = args.output_dir.resolve()

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    ori_all_rows = load_manifest(ori_manifest)
    sim_all_rows = load_manifest(sim_manifest)

    for dataset_name, dataset_stem in DATASETS:
        for mechanism_name, mechanism_stem in MECHANISMS:
            ori_rows = filter_rows(ori_all_rows, dataset_name, mechanism_name, ORI_FILTERS)
            sim_rows = filter_rows(sim_all_rows, dataset_name, mechanism_name, SIM_FILTERS)
            x_values, ori_means, ori_stds, sim_means, sim_stds = build_series(
                ori_rows, sim_rows, dataset_name, mechanism_name
            )
            plot_series(
                x_values,
                ori_means,
                ori_stds,
                sim_means,
                sim_stds,
                output_dir / f"{dataset_stem}-{mechanism_stem}",
            )
            print(
                f"{dataset_name}/{mechanism_name}: points={len(x_values)} "
                f"eps_values={[format_epsilon(value) for value in x_values]}"
            )

    print(f"Saved plots to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
