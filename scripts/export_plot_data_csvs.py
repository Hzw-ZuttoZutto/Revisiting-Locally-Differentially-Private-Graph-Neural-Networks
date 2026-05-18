#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from plot_figure10_feature_dim_curves import (
    EXPECTED_FEATURE_DIMS,
    choose_best_rows as choose_figure10_best_rows,
    format_feature_dim_tick,
    parse_step_name as parse_figure10_step_name,
    step_sort_key as figure10_step_sort_key,
)
from plot_figure11_privacy_curves import (
    EXPECTED_X_EPS,
    format_x_eps_tick,
    parse_step_name as parse_figure11_step_name,
    step_sort_key as figure11_step_sort_key,
    validate_rows as validate_figure11_rows,
)
from plot_figure4_delta_curves import (
    DATASETS as FIGURE4_DATASETS,
    EXPECTED_EPSILONS,
    MECHANISMS as FIGURE4_MECHANISMS,
    ORI_FILTERS,
    SIM_FILTERS,
    build_row_index,
    filter_rows as filter_figure4_rows,
    format_epsilon,
    require_metric as require_figure4_metric,
)
from plot_figure5_curves import (
    DATASETS as FIGURE5_DATASETS,
    EXPECTED_SCALES as FIGURE5_EXPECTED_SCALES,
    FILTERS as FIGURE5_FILTERS,
    format_scale_tick as format_figure5_scale_tick,
    validate_dataset_rows as validate_figure5_dataset_rows,
)
from plot_figure9_scale_curves import (
    DATASET_ORDER as FIGURE9_DATASET_ORDER,
    MODE_STYLES as FIGURE9_MODE_STYLES,
    format_scale_tick as format_figure9_scale_tick,
    infer_expected_scales as infer_figure9_expected_scales,
    validate_rows as validate_figure9_rows,
)


FIGURE5_VALID_SEARCH_STATUSES = {"completed", "skipped_existing_result"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the exact data points used by the plotting scripts as grouped CSV files."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("paper_experiments/plot_data_exports"),
        help="Root directory where grouped plot-data CSVs will be written.",
    )
    return parser.parse_args()


def repo_relative(path: Path) -> str:
    return path.as_posix()


def load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def filter_figure5_rows(rows: list[dict[str, str]], dataset_name: str) -> list[dict[str, str]]:
    filtered: list[dict[str, str]] = []
    for row in rows:
        if row.get("dataset") != dataset_name:
            continue
        if row.get("search_status") not in FIGURE5_VALID_SEARCH_STATUSES:
            continue
        mismatch = False
        for key, expected_value in FIGURE5_FILTERS.items():
            if key == "search_status":
                continue
            if row.get(key) != expected_value:
                mismatch = True
                break
        if mismatch:
            continue
        filtered.append(row)
    return filtered


def sort_step_name(step_name: str) -> int:
    return int(step_name.split("=", 1)[1])


def export_figure10(output_root: Path) -> list[tuple[str, int]]:
    input_root = Path("paper_experiments/figure10")
    output_dir = output_root / "figure10"
    manifest_paths = sorted(
        input_root.glob("x_steps=*.yaml/manifest.csv"),
        key=lambda path: figure10_step_sort_key(parse_figure10_step_name(path)),
    )
    if len(manifest_paths) != 6:
        raise ValueError(f"Expected 6 figure10 manifests under {input_root}, found {len(manifest_paths)}")

    accuracy_rows: list[dict[str, str]] = []
    relative_error_rows: list[dict[str, str]] = []
    feature_dim_order = {value: index for index, value in enumerate(EXPECTED_FEATURE_DIMS)}

    for manifest_path in manifest_paths:
        step_name = parse_figure10_step_name(manifest_path)
        best_rows = choose_figure10_best_rows(load_manifest(manifest_path), step_name)
        for row in best_rows:
            feature_dim = int(row["feature_dim"])
            base_row = {
                "plot_group": "figure10",
                "series_name": step_name,
                "source_manifest": repo_relative(manifest_path),
                "x_steps": step_name,
                "feature_dim": str(feature_dim),
                "selected_scale": row["scale"],
                "selection_best_verify_val_acc_mean": row["best_verify_val_acc_mean"],
                "x_value": str(feature_dim),
                "x_display": format_feature_dim_tick(feature_dim),
            }
            accuracy_rows.append(
                {
                    **base_row,
                    "plot_name": "best_test_accuracy_all_steps",
                    "y_value": row["best_verify_test_acc_mean"],
                }
            )
            relative_error_rows.append(
                {
                    **base_row,
                    "plot_name": "best_relative_error_all_steps",
                    "y_value": row["best_verify_sanity_e_pg_mean"],
                }
            )

    accuracy_rows.sort(key=lambda row: (sort_step_name(row["series_name"]), feature_dim_order[int(row["feature_dim"])]))
    relative_error_rows.sort(
        key=lambda row: (sort_step_name(row["series_name"]), feature_dim_order[int(row["feature_dim"])])
    )

    fieldnames = [
        "plot_group",
        "plot_name",
        "series_name",
        "source_manifest",
        "x_steps",
        "feature_dim",
        "selected_scale",
        "selection_best_verify_val_acc_mean",
        "x_value",
        "x_display",
        "y_value",
    ]
    outputs = [
        (output_dir / "best_test_accuracy_all_steps.csv", accuracy_rows),
        (output_dir / "best_relative_error_all_steps.csv", relative_error_rows),
    ]
    for path, rows in outputs:
        write_csv(path, fieldnames, rows)
    return [(repo_relative(path), len(rows)) for path, rows in outputs]


def export_figure11(output_root: Path) -> list[tuple[str, int]]:
    input_root = Path("paper_experiments/figure11")
    output_dir = output_root / "figure11"
    manifest_paths = sorted(
        input_root.glob("x_steps=*.yaml/manifest.csv"),
        key=lambda path: figure11_step_sort_key(parse_figure11_step_name(path)),
    )
    if len(manifest_paths) != 7:
        raise ValueError(f"Expected 7 figure11 manifests under {input_root}, found {len(manifest_paths)}")

    accuracy_rows: list[dict[str, str]] = []
    relative_error_rows: list[dict[str, str]] = []
    x_eps_order = {value: index for index, value in enumerate(EXPECTED_X_EPS)}

    for manifest_path in manifest_paths:
        step_name = parse_figure11_step_name(manifest_path)
        ordered_rows = validate_figure11_rows(load_manifest(manifest_path), step_name)
        for row in ordered_rows:
            x_eps = float(row["x_eps"])
            base_row = {
                "plot_group": "figure11",
                "series_name": step_name,
                "source_manifest": repo_relative(manifest_path),
                "x_steps": step_name,
                "x_eps": row["x_eps"],
                "x_value": row["x_eps"],
                "x_display": format_x_eps_tick(x_eps),
            }
            accuracy_rows.append(
                {
                    **base_row,
                    "plot_name": "best_test_accuracy_all_steps",
                    "y_value": row["best_verify_test_acc_mean"],
                }
            )
            relative_error_rows.append(
                {
                    **base_row,
                    "plot_name": "best_relative_error_all_steps",
                    "y_value": row["best_verify_sanity_e_pg_mean"],
                }
            )

    accuracy_rows.sort(key=lambda row: (sort_step_name(row["series_name"]), x_eps_order[float(row["x_eps"])]))
    relative_error_rows.sort(
        key=lambda row: (sort_step_name(row["series_name"]), x_eps_order[float(row["x_eps"])])
    )

    fieldnames = [
        "plot_group",
        "plot_name",
        "series_name",
        "source_manifest",
        "x_steps",
        "x_eps",
        "x_value",
        "x_display",
        "y_value",
    ]
    outputs = [
        (output_dir / "best_test_accuracy_all_steps.csv", accuracy_rows),
        (output_dir / "best_relative_error_all_steps.csv", relative_error_rows),
    ]
    for path, rows in outputs:
        write_csv(path, fieldnames, rows)
    return [(repo_relative(path), len(rows)) for path, rows in outputs]


def export_figure5(output_root: Path) -> list[tuple[str, int]]:
    manifest_path = Path("paper_experiments/figure5/figure5.yaml/manifest.csv")
    output_dir = output_root / "figure5"
    all_rows = load_manifest(manifest_path)
    scale_order = {value: index for index, value in enumerate(FIGURE5_EXPECTED_SCALES)}
    outputs: list[tuple[str, int]] = []

    fieldnames = [
        "plot_group",
        "plot_name",
        "series_name",
        "source_manifest",
        "dataset",
        "norm_scale",
        "x_value",
        "x_display",
        "y_value",
        "y_std",
    ]

    for dataset_name, output_stem in FIGURE5_DATASETS:
        filtered_rows = filter_figure5_rows(all_rows, dataset_name)
        ordered_rows = validate_figure5_dataset_rows(filtered_rows, dataset_name)
        exported_rows: list[dict[str, str]] = []
        for row in ordered_rows:
            scale = float(row["norm_scale"])
            exported_rows.append(
                {
                    "plot_group": "figure5",
                    "plot_name": output_stem,
                    "series_name": output_stem,
                    "source_manifest": repo_relative(manifest_path),
                    "dataset": dataset_name,
                    "norm_scale": row["norm_scale"],
                    "x_value": row["norm_scale"],
                    "x_display": format_figure5_scale_tick(scale),
                    "y_value": row["best_verify_test_acc_mean"],
                    "y_std": row["best_verify_test_acc_std"],
                }
            )
        exported_rows.sort(key=lambda row: (row["series_name"], scale_order[float(row["norm_scale"])]))
        output_path = output_dir / f"{output_stem}.csv"
        write_csv(output_path, fieldnames, exported_rows)
        outputs.append((repo_relative(output_path), len(exported_rows)))

    return outputs


def export_figure4(output_root: Path) -> list[tuple[str, int]]:
    ori_manifest = Path("paper_experiments/figure4/figure4_ori.yaml/manifest.csv")
    sim_manifest = Path("paper_experiments/figure4/figure4_sim.yaml/manifest.csv")
    output_dir = output_root / "figure4"
    ori_all_rows = load_manifest(ori_manifest)
    sim_all_rows = load_manifest(sim_manifest)
    epsilon_order = {value: index for index, value in enumerate(EXPECTED_EPSILONS)}
    outputs: list[tuple[str, int]] = []

    fieldnames = [
        "plot_group",
        "plot_name",
        "series_name",
        "source",
        "source_manifest",
        "dataset",
        "mechanism",
        "epsilon",
        "manifest_x_eps",
        "manifest_sim_reference_eps",
        "x_value",
        "x_display",
        "y_value",
        "y_std",
    ]

    for dataset_name, dataset_stem in FIGURE4_DATASETS:
        for mechanism_name, mechanism_stem in FIGURE4_MECHANISMS:
            ori_rows = filter_figure4_rows(ori_all_rows, dataset_name, mechanism_name, ORI_FILTERS)
            sim_rows = filter_figure4_rows(sim_all_rows, dataset_name, mechanism_name, SIM_FILTERS)
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
            exported_rows: list[dict[str, str]] = []

            for epsilon in EXPECTED_EPSILONS:
                epsilon_display = format_epsilon(epsilon)
                ori_row = ori_index[epsilon]
                sim_row = sim_index[epsilon]
                exported_rows.append(
                    {
                        "plot_group": "figure4",
                        "plot_name": f"{dataset_stem}-{mechanism_stem}",
                        "series_name": "LDP",
                        "source": "ori",
                        "source_manifest": repo_relative(ori_manifest),
                        "dataset": dataset_name,
                        "mechanism": mechanism_name,
                        "epsilon": epsilon_display,
                        "manifest_x_eps": ori_row.get("x_eps", ""),
                        "manifest_sim_reference_eps": ori_row.get("sim_reference_eps", ""),
                        "x_value": epsilon_display,
                        "x_display": epsilon_display,
                        "y_value": str(
                            require_figure4_metric(
                                ori_row,
                                "best_verify_test_acc_mean",
                                dataset_name,
                                mechanism_name,
                                epsilon,
                                "ori",
                            )
                        ),
                        "y_std": str(
                            require_figure4_metric(
                                ori_row,
                                "best_verify_test_acc_std",
                                dataset_name,
                                mechanism_name,
                                epsilon,
                                "ori",
                            )
                        ),
                    }
                )
                exported_rows.append(
                    {
                        "plot_group": "figure4",
                        "plot_name": f"{dataset_stem}-{mechanism_stem}",
                        "series_name": "Simulate",
                        "source": "sim",
                        "source_manifest": repo_relative(sim_manifest),
                        "dataset": dataset_name,
                        "mechanism": mechanism_name,
                        "epsilon": epsilon_display,
                        "manifest_x_eps": sim_row.get("x_eps", ""),
                        "manifest_sim_reference_eps": sim_row.get("sim_reference_eps", ""),
                        "x_value": epsilon_display,
                        "x_display": epsilon_display,
                        "y_value": str(
                            require_figure4_metric(
                                sim_row,
                                "best_verify_test_acc_mean",
                                dataset_name,
                                mechanism_name,
                                epsilon,
                                "sim",
                            )
                        ),
                        "y_std": str(
                            require_figure4_metric(
                                sim_row,
                                "best_verify_test_acc_std",
                                dataset_name,
                                mechanism_name,
                                epsilon,
                                "sim",
                            )
                        ),
                    }
                )

            exported_rows.sort(key=lambda row: (row["series_name"], epsilon_order[float(row["epsilon"])]))
            output_path = output_dir / f"{dataset_stem}-{mechanism_stem}.csv"
            write_csv(output_path, fieldnames, exported_rows)
            outputs.append((repo_relative(output_path), len(exported_rows)))

    return outputs


def export_figure9_merged(output_root: Path) -> list[tuple[str, int]]:
    input_roots = [
        Path("paper_experiments/figure9_newgrid"),
        Path("paper_experiments/figure9_newgrid_plus"),
    ]
    output_dir = output_root / "figure9_newgrid_merged"
    manifest_paths: list[Path] = []
    for input_root in input_roots:
        manifest_paths.extend(sorted(input_root.glob("*/*/x_steps=*/manifest.csv")))
    if not manifest_paths:
        raise ValueError(f"No figure9 manifests found under: {[repo_relative(path) for path in input_roots]}")

    expected_scales = infer_figure9_expected_scales(manifest_paths)
    scale_order = {value: index for index, value in enumerate(expected_scales)}
    dataset_mode_rows: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(dict)

    for manifest_path in manifest_paths:
        mode_name = manifest_path.parents[2].name
        dataset_name = manifest_path.parents[1].name
        source_root = repo_relative(manifest_path.parents[3])
        rows = load_manifest(manifest_path)
        annotated_rows: list[dict[str, str]] = []
        for row in rows:
            annotated_row = dict(row)
            annotated_row["__source_manifest"] = repo_relative(manifest_path)
            annotated_row["__source_root"] = source_root
            annotated_rows.append(annotated_row)
        dataset_mode_rows[dataset_name].setdefault(mode_name, [])
        dataset_mode_rows[dataset_name][mode_name].extend(annotated_rows)

    datasets = sorted(dataset_mode_rows, key=lambda name: FIGURE9_DATASET_ORDER.index(name))
    if datasets != FIGURE9_DATASET_ORDER:
        raise ValueError(f"Unexpected datasets for figure9 merged export: {datasets}")

    outputs: list[tuple[str, int]] = []
    fieldnames = [
        "plot_group",
        "plot_name",
        "series_name",
        "source_manifest",
        "source_root",
        "dataset",
        "mode",
        "scale",
        "x_value",
        "x_display",
        "y_value",
        "y_std",
    ]

    for dataset_name in datasets:
        mode_rows = dataset_mode_rows[dataset_name]
        if sorted(mode_rows) != sorted(FIGURE9_MODE_STYLES):
            raise ValueError(
                f"{dataset_name}: expected modes={sorted(FIGURE9_MODE_STYLES)}, found={sorted(mode_rows)}"
            )
        exported_rows: list[dict[str, str]] = []
        for mode_name in ["direct", "learned_projected", "random_projected"]:
            validated_rows = validate_figure9_rows(mode_rows[mode_name], f"{dataset_name}/{mode_name}", expected_scales)
            for row in validated_rows:
                scale = float(row["scale"])
                exported_rows.append(
                    {
                        "plot_group": "figure9_newgrid_merged",
                        "plot_name": dataset_name,
                        "series_name": mode_name,
                        "source_manifest": row["__source_manifest"],
                        "source_root": row["__source_root"],
                        "dataset": dataset_name,
                        "mode": mode_name,
                        "scale": row["scale"],
                        "x_value": row["scale"],
                        "x_display": format_figure9_scale_tick(scale),
                        "y_value": row["best_verify_test_acc_mean"],
                        "y_std": row["best_verify_test_acc_std"],
                    }
                )
        exported_rows.sort(key=lambda row: (row["series_name"], scale_order[float(row["scale"])]))
        output_path = output_dir / f"{dataset_name}.csv"
        write_csv(output_path, fieldnames, exported_rows)
        outputs.append((repo_relative(output_path), len(exported_rows)))

    return outputs


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()

    export_results: list[tuple[str, int]] = []
    export_results.extend(export_figure10(output_root))
    export_results.extend(export_figure11(output_root))
    export_results.extend(export_figure5(output_root))
    export_results.extend(export_figure4(output_root))
    export_results.extend(export_figure9_merged(output_root))

    for path, row_count in export_results:
        print(f"{path}: rows={row_count}")
    print(f"Saved plot-data CSV exports to: {repo_relative(output_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
