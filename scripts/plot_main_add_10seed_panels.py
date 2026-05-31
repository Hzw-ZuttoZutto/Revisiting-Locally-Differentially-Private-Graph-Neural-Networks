#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


PAPER_ROOT = Path("/data/hzw/Rethinking_DP_GNN_runtime/paper_experiments")
DEFAULT_MAIN_LONG_CSV = PAPER_ROOT / "main_add_10seed_backfill" / "test_acc_long.csv"
DEFAULT_MAIN2_ROOT = PAPER_ROOT / "main2_again"
DEFAULT_CLEAN_REFERENCE_MANIFEST = PAPER_ROOT / "clean_reference" / "manifest.csv"
DEFAULT_OUTPUT_DIR = PAPER_ROOT / "main_add_10seed_backfill" / "plots"

BACKBONES = ("gcn", "sage", "gat")
BACKBONE_LABELS = {"gcn": "GCN", "sage": "GraphSAGE", "gat": "GAT"}
DATASETS = ("cora", "lastfm", "citeseer", "facebook")
DATASET_LABELS = {
    "cora": "Cora",
    "lastfm": "Lastfm",
    "citeseer": "Citeseer",
    "facebook": "Facebook",
    "Books-History": "Books-History",
}
X_EPS_VALUES = ("0.001", "0.01", "0.1", "1.0", "2.0", "3.0", "4.0", "6.0", "8.0", "10.0")
PIPELINE_LABELS = {
    "figure3_pipeline1": "LPGNNv",
    "figure3_pipeline2": "PrivGEv",
    "figure3_pipeline3": "UPGNet-MBM",
    "figure3_pipeline4": "UPGNet-PM",
}
REFERENCE_LABELS = {
    "kprop": r"$F^2_{\mathrm{KProp}}$",
    "hoa": r"$F^2_{\mathrm{HOA}}$",
}
PLOTTED_REFERENCE_SMOOTHERS = ("kprop", "hoa")
NON_PRIVATE_LABEL = "Non private"
REFERENCE_DIMS = (800, 1600, 3200)
VERIFY_DIR_RE = re.compile(r"^rank=(\d+)__repeat=(\d+)__candidate=(\d+)__")

BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 12345

# --- Visual Style Settings, aligned with plot/*.py examples ---
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

TITLE_FONTSIZE = 18
FONTSIZE = 14
X_LABEL_FONTSIZE = 14
LEGEND_FONTSIZE = 16
TICKLABEL_FONTSIZE = 10
LINEWIDTH = 2
MARKERSIZE = 8
FIGSIZE_X = 20
FIGSIZE_Y = 11.5
BOTTOM = 0.14
TOP = 0.92
LEFT = 0.065
RIGHT = 0.99

STYLE_CONFIGS = {
    "LPGNNv": {"color": "#5372ab", "marker": "s", "linestyle": "-"},
    "PrivGEv": {"color": "#936bb9", "marker": "^", "linestyle": "-"},
    "UPGNet-MBM": {"color": "#6aa56e", "marker": "o", "linestyle": "-"},
    "UPGNet-PM": {"color": "#c9b97d", "marker": "x", "linestyle": "-"},
    REFERENCE_LABELS["kprop"]: {"color": "#f2a65a", "marker": "v", "linestyle": "--"},
    REFERENCE_LABELS["hoa"]: {"color": "#b75555", "marker": "D", "linestyle": "--"},
    NON_PRIVATE_LABEL: {"color": "#2f7fb8", "marker": "P", "linestyle": "--"},
}
LINE_ORDER = (
    "LPGNNv",
    "PrivGEv",
    "UPGNet-MBM",
    "UPGNet-PM",
    REFERENCE_LABELS["kprop"],
    REFERENCE_LABELS["hoa"],
    NON_PRIVATE_LABEL,
)


@dataclass(frozen=True)
class Candidate:
    candidate_id: int
    x_steps: int
    learning_rate: str
    weight_decay: str
    dropout: str
    tao2: str


@dataclass
class ReferenceChoice:
    backbone: str
    dataset: str
    smoother: str
    feature_dim: int
    candidate: Candidate
    val_accs: list[float]
    test_accs: list[float]
    source_job_dir: Path
    actual_verify_rank: int | None

    @property
    def val_mean(self) -> float:
        return float(np.mean(self.val_accs))

    @property
    def test_mean(self) -> float:
        return float(np.mean(self.test_accs))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot 3x5 main-add 10-seed curves with feature-free references."
    )
    parser.add_argument("--main-long-csv", type=Path, default=DEFAULT_MAIN_LONG_CSV)
    parser.add_argument("--main2-root", type=Path, default=DEFAULT_MAIN2_ROOT)
    parser.add_argument("--clean-reference-manifest", type=Path, default=DEFAULT_CLEAN_REFERENCE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


def canonical_x_eps(raw_value: Any) -> str:
    value = float(raw_value)
    for candidate in X_EPS_VALUES:
        if math.isclose(value, float(candidate), rel_tol=0.0, abs_tol=1e-12):
            return candidate
    raise ValueError(f"Unsupported x_eps value: {raw_value!r}")


def metric_stats(values: list[float], *, bootstrap_samples: int, bootstrap_seed: int, key: tuple[Any, ...]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size != 10:
        raise ValueError(f"Expected 10 values for {key}, found {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Non-finite values for {key}: {values}")

    mean = float(np.mean(arr))
    digest = hashlib.sha256("|".join(map(str, key)).encode("utf-8")).hexdigest()
    seed = (int(digest[:8], 16) ^ int(bootstrap_seed)) % (2**32)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, arr.size, size=(bootstrap_samples, arr.size))
    sample_means = arr[indices].mean(axis=1)
    ci_low, ci_high = np.percentile(sample_means, [2.5, 97.5])
    return {
        "mean": mean,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "std": float(np.std(arr, ddof=1)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": int(arr.size),
    }


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def candidate_from_best_config(path: Path) -> Candidate:
    data = read_yaml(path)
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid best_config.yaml: {path}")
    raw = data["best_candidate"]
    return Candidate(
        candidate_id=int(raw["candidate_id"]),
        x_steps=int(raw["x_steps"]),
        learning_rate=f"{float(raw['learning_rate']):.12g}",
        weight_decay=f"{float(raw['weight_decay']):.12g}",
        dropout=f"{float(raw['dropout']):.12g}",
        tao2="none" if str(raw.get("tao2", "none")).lower() == "none" else f"{float(raw['tao2']):.12g}",
    )


def row_float(row: dict[str, str], name: str, *, csv_path: Path) -> float:
    try:
        value = float(row[name])
    except (KeyError, ValueError) as exc:
        raise RuntimeError(f"Invalid {name!r} in {csv_path}: {row.get(name)!r}") from exc
    if not math.isfinite(value):
        raise RuntimeError(f"Non-finite {name!r} in {csv_path}: {value}")
    return value


def row_candidate_matches(row: dict[str, str], candidate: Candidate, *, csv_path: Path) -> bool:
    tao2_raw = row.get("tao2", "").strip()
    tao2 = "none" if not tao2_raw or tao2_raw.lower() == "none" else f"{float(tao2_raw):.12g}"
    row_candidate = Candidate(
        candidate_id=candidate.candidate_id,
        x_steps=int(row_float(row, "x_steps", csv_path=csv_path)),
        learning_rate=f"{row_float(row, 'learning_rate', csv_path=csv_path):.12g}",
        weight_decay=f"{row_float(row, 'weight_decay', csv_path=csv_path):.12g}",
        dropout=f"{row_float(row, 'dropout', csv_path=csv_path):.12g}",
        tao2=tao2,
    )
    return row_candidate == candidate


def collect_reference_repeats(job_dir: Path, candidate: Candidate) -> tuple[list[float], list[float], int | None]:
    verify_dir = job_dir / "verify_top5"
    if not verify_dir.is_dir():
        raise RuntimeError(f"Missing verify_top5 directory: {verify_dir}")

    repeat_dirs: dict[int, tuple[int, Path]] = {}
    for child in sorted(verify_dir.iterdir()):
        if not child.is_dir():
            continue
        match = VERIFY_DIR_RE.match(child.name)
        if match is None:
            continue
        rank = int(match.group(1))
        repeat_id = int(match.group(2))
        candidate_id = int(match.group(3))
        if candidate_id != candidate.candidate_id:
            continue
        if repeat_id in repeat_dirs:
            raise RuntimeError(f"Duplicate verify repeat {repeat_id} for candidate {candidate.candidate_id} in {verify_dir}")
        repeat_dirs[repeat_id] = (rank, child)

    if sorted(repeat_dirs) != list(range(1, 11)):
        raise RuntimeError(
            f"Expected repeats 1..10 for candidate {candidate.candidate_id} in {verify_dir}, got {sorted(repeat_dirs)}"
        )

    val_accs: list[float] = []
    test_accs: list[float] = []
    observed_ranks: set[int] = set()
    for repeat_id in range(1, 11):
        rank, repeat_dir = repeat_dirs[repeat_id]
        observed_ranks.add(rank)
        csv_files = sorted(repeat_dir.glob("*.csv"))
        if len(csv_files) != 1:
            raise RuntimeError(f"Expected exactly one CSV in {repeat_dir}, found {len(csv_files)}")
        csv_path = csv_files[0]
        rows = read_csv_rows(csv_path)
        if len(rows) != 1:
            raise RuntimeError(f"Expected exactly one row in {csv_path}, found {len(rows)}")
        row = rows[0]
        if not row_candidate_matches(row, candidate, csv_path=csv_path):
            raise RuntimeError(f"Candidate hyperparameter mismatch in {csv_path}")
        val_accs.append(row_float(row, "val/acc", csv_path=csv_path))
        test_accs.append(row_float(row, "test/acc", csv_path=csv_path))

    actual_rank = next(iter(observed_ranks)) if len(observed_ranks) == 1 else None
    return val_accs, test_accs, actual_rank


def load_main_curves(
    path: Path,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    df = pd.read_csv(path)
    if len(df) != 6000:
        raise RuntimeError(f"Expected 6000 rows in {path}, found {len(df)}")

    df = df.copy()
    df["x_eps"] = df["x_eps"].map(canonical_x_eps)
    df = df[
        df["backbone"].isin(BACKBONES)
        & df["dataset"].isin(DATASETS)
        & df["pipeline"].isin(PIPELINE_LABELS)
    ]
    expected_groups = len(BACKBONES) * len(DATASETS) * len(X_EPS_VALUES) * len(PIPELINE_LABELS)
    grouped = df.groupby(["backbone", "pipeline", "dataset", "x_eps"], sort=False)
    if len(grouped) != expected_groups:
        raise RuntimeError(f"Expected {expected_groups} main groups, found {len(grouped)}")

    rows: list[dict[str, Any]] = []
    for (backbone, pipeline, dataset, x_eps), group in grouped:
        if len(group) != 10:
            raise RuntimeError(f"Expected 10 rows for {(backbone, pipeline, dataset, x_eps)}, found {len(group)}")
        stats = metric_stats(
            [float(v) for v in group["test_acc"]],
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            key=("main", backbone, pipeline, dataset, x_eps),
        )
        val_stats = metric_stats(
            [float(v) for v in group["val_acc"]],
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            key=("main-val", backbone, pipeline, dataset, x_eps),
        )
        first = group.iloc[0]
        rows.append(
            {
                "source": "main_add_10seed",
                "backbone": backbone,
                "dataset": dataset,
                "x_eps": x_eps,
                "x_index": X_EPS_VALUES.index(x_eps),
                "line_label": PIPELINE_LABELS[pipeline],
                "pipeline": pipeline,
                "smoother": str(first["smoother"]),
                "feature_dim": "",
                "candidate_id": int(first["candidate_id"]),
                "x_steps": int(first["x_steps"]),
                "learning_rate": first["learning_rate"],
                "weight_decay": first["weight_decay"],
                "dropout": first["dropout"],
                "tao2": first["tao2"],
                "test_acc_mean": stats["mean"],
                "test_acc_ci_low": stats["ci_low"],
                "test_acc_ci_high": stats["ci_high"],
                "test_acc_std": stats["std"],
                "test_acc_min": stats["min"],
                "test_acc_max": stats["max"],
                "val_acc_mean": val_stats["mean"],
                "val_acc_ci_low": val_stats["ci_low"],
                "val_acc_ci_high": val_stats["ci_high"],
                "n": stats["n"],
            }
        )
    return rows


def load_reference_choices(main2_root: Path) -> dict[tuple[str, str, str], ReferenceChoice]:
    choices: dict[tuple[str, str, str], list[ReferenceChoice]] = {}

    for backbone in BACKBONES:
        manifest_path = main2_root / backbone / "random_projected.yaml" / "manifest.csv"
        manifest_rows = read_csv_rows(manifest_path)
        for row in manifest_rows:
            dataset = row["dataset"]
            if dataset not in DATASETS:
                continue
            smoother = row["smoother"].lower()
            if smoother not in REFERENCE_LABELS:
                continue
            feature_dim = int(float(row["feature_dim"]))
            if feature_dim not in REFERENCE_DIMS:
                continue
            if row["search_status"] not in {"completed", "skipped_existing_result"}:
                raise RuntimeError(f"Incomplete reference job: {manifest_path} {dataset} {feature_dim} {smoother}")
            job_dir = Path(row["job_dir"])
            best_config_path = Path(row["best_config_path"])
            if not job_dir.is_dir() or not best_config_path.is_file():
                raise RuntimeError(f"Missing reference artifacts for {backbone}/{dataset}/{feature_dim}/{smoother}")
            candidate = candidate_from_best_config(best_config_path)
            val_accs, test_accs, actual_rank = collect_reference_repeats(job_dir, candidate)
            choices.setdefault((backbone, dataset, smoother), []).append(
                ReferenceChoice(
                    backbone=backbone,
                    dataset=dataset,
                    smoother=smoother,
                    feature_dim=feature_dim,
                    candidate=candidate,
                    val_accs=val_accs,
                    test_accs=test_accs,
                    source_job_dir=job_dir,
                    actual_verify_rank=actual_rank,
                )
            )

    selected: dict[tuple[str, str, str], ReferenceChoice] = {}
    expected_keys = {
        (backbone, dataset, smoother)
        for backbone in BACKBONES
        for dataset in DATASETS
        for smoother in PLOTTED_REFERENCE_SMOOTHERS
    }
    for key in expected_keys:
        candidates = choices.get(key, [])
        if len(candidates) != len(REFERENCE_DIMS):
            raise RuntimeError(f"Expected 3 feature_dim candidates for {key}, found {len(candidates)}")
        selected[key] = sorted(
            candidates,
            key=lambda item: (-item.val_mean, -item.test_mean, item.feature_dim),
        )[0]
    return selected


def reference_plot_rows(
    choices: dict[tuple[str, str, str], ReferenceChoice],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (backbone, dataset, smoother), choice in sorted(choices.items()):
        test_stats = metric_stats(
            choice.test_accs,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            key=("reference", backbone, dataset, smoother, choice.feature_dim),
        )
        val_stats = metric_stats(
            choice.val_accs,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            key=("reference-val", backbone, dataset, smoother, choice.feature_dim),
        )
        for x_eps in X_EPS_VALUES:
            rows.append(
                {
                    "source": "main2_again_reference",
                    "backbone": backbone,
                    "dataset": dataset,
                    "x_eps": x_eps,
                    "x_index": X_EPS_VALUES.index(x_eps),
                    "line_label": REFERENCE_LABELS[smoother],
                    "pipeline": "reference",
                    "smoother": smoother,
                    "feature_dim": choice.feature_dim,
                    "candidate_id": choice.candidate.candidate_id,
                    "x_steps": choice.candidate.x_steps,
                    "learning_rate": choice.candidate.learning_rate,
                    "weight_decay": choice.candidate.weight_decay,
                    "dropout": choice.candidate.dropout,
                    "tao2": choice.candidate.tao2,
                    "test_acc_mean": test_stats["mean"],
                    "test_acc_ci_low": test_stats["ci_low"],
                    "test_acc_ci_high": test_stats["ci_high"],
                    "test_acc_std": test_stats["std"],
                    "test_acc_min": test_stats["min"],
                    "test_acc_max": test_stats["max"],
                    "val_acc_mean": val_stats["mean"],
                    "val_acc_ci_low": val_stats["ci_low"],
                    "val_acc_ci_high": val_stats["ci_high"],
                    "n": test_stats["n"],
                    "source_job_dir": str(choice.source_job_dir),
                    "actual_verify_rank": "" if choice.actual_verify_rank is None else choice.actual_verify_rank,
                }
            )
    return rows


def clean_reference_plot_rows(
    manifest_path: Path,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    rows = read_csv_rows(manifest_path)
    expected_count = len(BACKBONES) * len(DATASETS)
    if len(rows) != expected_count:
        raise RuntimeError(f"Expected {expected_count} clean-reference rows in {manifest_path}, found {len(rows)}")

    by_key = {(row["backbone"], row["dataset"]): row for row in rows}
    missing = [
        (backbone, dataset)
        for backbone in BACKBONES
        for dataset in DATASETS
        if (backbone, dataset) not in by_key
    ]
    if missing:
        raise RuntimeError(f"Missing clean-reference entries: {missing}")

    plot_rows: list[dict[str, Any]] = []
    for backbone in BACKBONES:
        for dataset in DATASETS:
            row = by_key[(backbone, dataset)]
            if row["search_status"] not in {"completed", "skipped_existing_result"}:
                raise RuntimeError(f"Incomplete clean-reference entry for {(backbone, dataset)}: {row['search_status']}")
            mean = float(row["best_verify_test_acc_mean"])
            std = float(row["best_verify_test_acc_std"])
            n = 10
            # The clean-reference manifest only stores aggregate statistics.
            # Approximate the requested mean CI from std and n.
            half_width = 1.96 * std / math.sqrt(n)
            for x_eps in X_EPS_VALUES:
                plot_rows.append(
                    {
                        "source": "clean_reference",
                        "backbone": backbone,
                        "dataset": dataset,
                        "x_eps": x_eps,
                        "x_index": X_EPS_VALUES.index(x_eps),
                        "line_label": NON_PRIVATE_LABEL,
                        "pipeline": "clean_reference",
                        "smoother": row["smoother"],
                        "feature_dim": "",
                        "candidate_id": row["best_candidate_id"],
                        "x_steps": row["best_x_steps"],
                        "learning_rate": row["best_learning_rate"],
                        "weight_decay": row["best_weight_decay"],
                        "dropout": row["best_dropout"],
                        "tao2": row["best_tao2"],
                        "test_acc_mean": mean,
                        "test_acc_ci_low": mean - half_width,
                        "test_acc_ci_high": mean + half_width,
                        "test_acc_std": std,
                        "test_acc_min": "",
                        "test_acc_max": "",
                        "val_acc_mean": row["best_verify_val_acc_mean"],
                        "val_acc_ci_low": "",
                        "val_acc_ci_high": "",
                        "n": n,
                        "source_job_dir": row["job_dir"],
                        "actual_verify_rank": row["best_rank"],
                        "ci_method": "normal_from_manifest_std",
                    }
                )
    return plot_rows


def write_plot_data(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source",
        "backbone",
        "dataset",
        "x_eps",
        "x_index",
        "line_label",
        "pipeline",
        "smoother",
        "feature_dim",
        "candidate_id",
        "x_steps",
        "learning_rate",
        "weight_decay",
        "dropout",
        "tao2",
        "test_acc_mean",
        "test_acc_ci_low",
        "test_acc_ci_high",
        "test_acc_std",
        "test_acc_min",
        "test_acc_max",
        "val_acc_mean",
        "val_acc_ci_low",
        "val_acc_ci_high",
        "n",
        "source_job_dir",
        "actual_verify_rank",
        "ci_method",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def validate_plot_rows(rows: list[dict[str, Any]]) -> None:
    expected = len(BACKBONES) * len(DATASETS) * len(X_EPS_VALUES) * len(LINE_ORDER)
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} plot rows, found {len(rows)}")
    seen = {(row["backbone"], row["dataset"], row["x_eps"], row["line_label"]) for row in rows}
    if len(seen) != expected:
        raise RuntimeError(f"Duplicate plot rows detected: unique={len(seen)} expected={expected}")
    for row in rows:
        mean = float(row["test_acc_mean"])
        low = float(row["test_acc_ci_low"])
        high = float(row["test_acc_ci_high"])
        if int(row["n"]) != 10:
            raise RuntimeError(f"Expected n=10 for {row}")
        if not (low <= mean <= high):
            raise RuntimeError(f"CI does not contain mean for {row}")


def plot_panels(rows: list[dict[str, Any]], output_dir: Path) -> None:
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(len(BACKBONES), len(DATASETS), figsize=(FIGSIZE_X, FIGSIZE_Y), sharex=False, sharey=False)
    fig.subplots_adjust(bottom=BOTTOM, top=TOP, left=LEFT, right=RIGHT, hspace=0.34, wspace=0.16)

    for row_idx, backbone in enumerate(BACKBONES):
        for col_idx, dataset in enumerate(DATASETS):
            ax = axes[row_idx, col_idx]
            if row_idx == 0:
                letter = chr(97 + col_idx)
                ax.set_title(f"({letter}) {DATASET_LABELS[dataset]}", fontsize=TITLE_FONTSIZE, fontweight="medium")

            panel = df[(df["backbone"] == backbone) & (df["dataset"] == dataset)]
            for label in LINE_ORDER:
                sub = panel[panel["line_label"] == label].sort_values("x_index")
                if sub.empty:
                    continue
                style = STYLE_CONFIGS[label]
                y = sub["test_acc_mean"].to_numpy(dtype=float)
                yerr = np.vstack(
                    [
                        y - sub["test_acc_ci_low"].to_numpy(dtype=float),
                        sub["test_acc_ci_high"].to_numpy(dtype=float) - y,
                    ]
                )
                ax.errorbar(
                    sub["x_index"].to_numpy(dtype=float),
                    y,
                    yerr=yerr,
                    label=label,
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    markersize=MARKERSIZE,
                    linewidth=LINEWIDTH,
                    capsize=0,
                    markerfacecolor="none",
                    markeredgewidth=2,
                )

            ax.set_xticks(range(len(X_EPS_VALUES)))
            ax.set_xticklabels(X_EPS_VALUES, rotation=30, ha="right")
            ax.set_xlabel(r"$\epsilon$", fontsize=X_LABEL_FONTSIZE, fontweight="medium")
            ax.grid(True, color="white", linestyle="-", linewidth=1, alpha=1.0)
            ax.tick_params(axis="both", which="major", labelsize=TICKLABEL_FONTSIZE)
            if col_idx == 0:
                ax.set_ylabel(f"{BACKBONE_LABELS[backbone]}\nAccuracy", fontsize=FONTSIZE, fontweight="medium")
            else:
                ax.set_ylabel("")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if len(labels) != len(LINE_ORDER):
        raise RuntimeError(f"Expected {len(LINE_ORDER)} legend labels, found {labels}")
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=len(LINE_ORDER),
        fontsize=LEGEND_FONTSIZE,
        frameon=False,
        shadow=False,
        borderpad=1,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "main_add_10seed_panels.pdf"
    png_path = output_dir / "main_add_10seed_panels.png"
    fig.savefig(pdf_path, dpi=300)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


def main() -> None:
    args = parse_args()
    main_rows = load_main_curves(
        args.main_long_csv,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    reference_choices = load_reference_choices(args.main2_root)
    reference_rows = reference_plot_rows(
        reference_choices,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    non_private_rows = clean_reference_plot_rows(
        args.clean_reference_manifest,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    plot_rows = main_rows + reference_rows + non_private_rows
    validate_plot_rows(plot_rows)
    plot_data_path = args.output_dir / "main_add_10seed_panel_plot_data.csv"
    write_plot_data(plot_data_path, plot_rows)
    plot_panels(plot_rows, args.output_dir)
    print(f"Saved {plot_data_path}")
    print(f"Plot rows: {len(plot_rows)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
