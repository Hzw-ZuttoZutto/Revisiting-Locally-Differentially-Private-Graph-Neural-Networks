from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import yaml

from hparams_search_scripts import run_mechanism_hparam_search as search_impl
from hparams_search_scripts import table_search_suite_core as core
from hparams_search_scripts import mechanism_stage_utils

from .paths import FIXED_ROOT, REFERENCE_ROOT, WORK_ROOT, max_parallel_per_gpu, parse_gpu_ids
from .search_runner import _aggregate_search_output

REPO_ROOT = Path(__file__).resolve().parents[2]


def fixed_points(figure_id: int | str) -> list[dict[str, Any]]:
    path = FIXED_ROOT / f"figure{figure_id}.yaml"
    if not path.is_file():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return list(data.get("points", []))


def _coverage(figure_id: int, points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for point in points:
        fixed = point.get("fixed_params", point)
        dataset = str(fixed.get("dataset", "")).lower()
        backbone = str(fixed.get("backbone", "")).lower()
        if figure_id == 1 and dataset not in {"cora", "facebook"}:
            continue
        if figure_id == 6 and (dataset not in {"actor", "flickr"} or backbone != "sage"):
            continue
        if not fixed.get("mechanism"):
            continue
        result.append(point)
    return result


def plan_fixed_jobs(
    figure_id: int,
    *,
    repeats: int = 3,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    points = _coverage(figure_id, fixed_points(figure_id))
    jobs = [
        {
            "figure_id": figure_id,
            "point_id": point.get("point_id"),
            "dataset": point.get("fixed_params", {}).get("dataset"),
            "backbone": point.get("fixed_params", {}).get("backbone"),
            "mechanism": point.get("fixed_params", {}).get("mechanism"),
            "x_eps": point.get("fixed_params", {}).get("x_eps"),
            "repeats": repeats,
        }
        for point in points
    ]
    return jobs if limit is None else jobs[:limit]


def reference_repeats(target: int | str) -> int:
    """Return the repeat count used by the corresponding paper reference."""
    if isinstance(target, str) and target.startswith("table"):
        path = REFERENCE_ROOT / f"{target}_seed_rows.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise RuntimeError(f"Reference table is empty: {path}")
        # Every fixed table point in the published reference uses the same
        # seed count; count one complete setting rather than guessing a value.
        first_key = tuple(rows[0].get(key, "") for key in ("setting", "dataset", "backbone", "feature_dim"))
        count = sum(
            tuple(row.get(key, "") for key in ("setting", "dataset", "backbone", "feature_dim")) == first_key
            for row in rows
        )
        return int(count)

    path = REFERENCE_ROOT / f"figure{target}_plot_data.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        values = {int(row["n"]) for row in csv.DictReader(handle) if row.get("n")}
    if len(values) != 1:
        raise RuntimeError(f"Reference Figure {target} has inconsistent repeat counts: {sorted(values)}")
    return values.pop()


def _one_candidate_config(
    point: dict[str, Any],
    repeats: int,
    gpu_ids: list[int] | None = None,
    max_parallel: int | None = None,
) -> dict[str, Any]:
    fixed = point.get("fixed_params", point)
    candidate = point.get("candidate", {})
    defaults = point.get("defaults", {}) or {}
    feature = str(fixed.get("feature", "raw"))

    feature_cfg: dict[str, list[Any]] = {
        key: []
        for key in (
            "sim_reference_eps",
            "feature_dim",
            "scale",
            "feature_preprojection",
            "preprojection_output_dim",
            "random_normal_mean",
            "random_normal_std",
            "shared_value",
            "degree_bucket_num_buckets",
            "degree_bucket_range_max",
            "deepwalk_walk_length",
            "deepwalk_number_walks",
            "deepwalk_window_size",
            "deepwalk_workers",
            "deepwalk_undirected",
        )
    }
    feature_cfg["features"] = [feature]
    scale = fixed.get("scale")
    feature_cfg["scale"] = [1 if scale is None else scale]
    for key in feature_cfg:
        if key not in {"features", "scale"} and fixed.get(key) is not None:
            feature_cfg[key] = [fixed[key]]
    if feature == "sim":
        feature_cfg["sim_reference_eps"] = [fixed.get("sim_reference_eps")]

    perturbation = {
        "mechanisms": [fixed.get("mechanism")],
        "x_eps": [fixed.get("x_eps")],
        "m": [fixed.get("m", "best")],
    }
    norm = bool(fixed.get("norm", False))
    calibrator = {
        "norm": [norm],
        "norm_scale": [fixed.get("norm_scale", "none")] if norm else [],
        "x_steps": [candidate.get("x_steps", 0)],
        "smoother": [
            fixed.get("smoother")
            if fixed.get("smoother") not in (None, "none", "")
            else "hoa"
        ],
    }
    use_nfr = bool(fixed.get("use_nfr", False))
    nfr = {
        "use_nfr": [use_nfr],
        "tao2": [candidate.get("tao2")] if use_nfr else [],
    }

    selected_gpu_ids = parse_gpu_ids() if gpu_ids is None else list(gpu_ids)
    selected_parallel = max_parallel_per_gpu() if max_parallel is None else int(max_parallel)
    device = {
        "device": "gpu",
        "cpu_worker_count": None,
        "gpu_ids": selected_gpu_ids,
        "max_parallel_per_gpu": selected_parallel,
        "gpu_launch_interval_sec": 0.1,
    }

    clean_defaults = dict(defaults)
    clean_stage = dict(clean_defaults.get("stage", {}))
    clean_grid = dict(clean_stage.get("grid", {}))
    clean_verify = dict(clean_stage.get("verify", {}))
    clean_grid.pop("max_epochs", None)
    clean_verify.pop("max_epochs", None)
    clean_grid["repeats"] = 1
    clean_verify["repeats"] = repeats
    clean_stage["grid"] = clean_grid
    clean_stage["verify"] = clean_verify
    clean_defaults["stage"] = clean_stage

    return {
        "seed": int(point.get("base_seed", 12345)),
        "device": device,
        "defaults": clean_defaults,
        "search_space": {
            "dataset": {"datasets": [fixed.get("dataset")]},
            "feature_transformation": feature_cfg,
            "feature_perturbation": perturbation,
            "calibrator": calibrator,
            "model": {
                "backbones": [fixed.get("backbone")],
                "dropout": [candidate.get("dropout", 0.5)],
            },
            "trainer": {
                "learning_rate": [candidate.get("learning_rate", 0.001)],
                "weight_decay": [candidate.get("weight_decay", 0.0)],
            },
            "nfr": nfr,
        },
    }


def _write_point_configs(
    points: list[dict[str, Any]],
    *,
    root: Path,
    repeats: int,
) -> list[Path]:
    config_root = root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    configs: list[Path] = []
    for index, point in enumerate(points):
        path = config_root / f"point_{index:05d}.yaml"
        path.write_text(
            yaml.safe_dump(_one_candidate_config(point, repeats), sort_keys=False),
            encoding="utf-8",
        )
        configs.append(path)
    return configs


def _build_fixed_batch(
    points: list[dict[str, Any]],
    *,
    root: Path,
    repeats: int,
) -> tuple[core.BatchSpec | None, list[Path]]:
    configs = _write_point_configs(points, root=root, repeats=repeats)
    if not configs:
        return None, configs

    jobs: list[core.BatchJob] = []
    execution = None
    for index, config in enumerate(configs):
        search_config = search_impl.load_search_config(config)
        point_root = root / f"result_{index:05d}"
        point_batch = search_impl.build_batch_spec(
            search_config=search_config,
            output_root=point_root,
            config_copy_source=config,
        )
        if len(point_batch.jobs) != 1:
            raise RuntimeError(
                f"Fixed point {index} expanded to {len(point_batch.jobs)} jobs; expected exactly one"
            )
        if execution is None:
            execution = point_batch.execution
        elif execution != point_batch.execution:
            raise RuntimeError("Fixed-point configurations disagree on execution settings")
        jobs.extend(point_batch.jobs)

    assert execution is not None
    return (
        core.BatchSpec(
            output_root=root,
            execution=execution,
            jobs=jobs,
            config_copy_source=configs[0],
        ),
        configs,
    )


def _run_fixed_batch(
    points: list[dict[str, Any]],
    *,
    root: Path,
    repeats: int,
    execute: bool,
) -> dict[str, Any]:
    batch, configs = _build_fixed_batch(points, root=root, repeats=repeats)
    if batch is None:
        print("planned 0 fixed jobs; execute=False")
        return {"jobs": [], "configs": [], "completed": 0, "skipped": 0, "failed": 0}

    if not execute:
        print(
            f"planned {len(batch.jobs)} fixed jobs in one global task pool; execute=False"
        )
        return {
            "jobs": batch.jobs,
            "configs": configs,
            "completed": 0,
            "skipped": 0,
            "failed": 0,
        }

    completed, skipped, failed, manifest = core.run_batch_search(
        batch,
        repo_root=REPO_ROOT,
    )
    summary = {
        "jobs": batch.jobs,
        "configs": configs,
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "manifest": manifest,
    }
    print(
        f"global fixed batch finished: jobs={len(batch.jobs)} "
        f"completed={completed} skipped={skipped} failed={failed}"
    )
    if failed:
        raise RuntimeError(f"Fixed batch failed for {failed} job(s); see {manifest}")
    return summary


def run_fixed(
    figure_id: int,
    *,
    repeats: int = 3,
    limit: int | None = None,
    execute: bool = False,
) -> list[dict[str, Any]]:
    points = _coverage(figure_id, fixed_points(figure_id))
    jobs = plan_fixed_jobs(figure_id, repeats=repeats, limit=limit)
    selected_ids = {job["point_id"] for job in jobs}
    selected_points = [point for point in points if point.get("point_id") in selected_ids]
    root = WORK_ROOT / "search" / f"figure{figure_id}" / "fixed"
    _run_fixed_batch(selected_points, root=root, repeats=repeats, execute=execute)
    if execute:
        _aggregate_search_output(figure_id, "fixed")
    return jobs


def _point_matches_manifest(point: dict[str, Any], manifest: dict[str, str]) -> bool:
    fixed = point.get("fixed_params", {})
    for key in (
        "dataset",
        "feature",
        "feature_dim",
        "mechanism",
        "x_eps",
        "m",
        "norm",
        "norm_scale",
        "smoother",
        "backbone",
        "use_nfr",
    ):
        expected = "" if fixed.get(key) is None else str(fixed.get(key))
        actual = str(manifest.get(key, ""))
        if expected.lower() != actual.lower():
            return False
    return True


def _collect_table_outputs(
    table: str,
    points: list[dict[str, Any]],
    root: Path,
) -> Path:
    manifest_path = root / mechanism_stage_utils.ROOT_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Batch manifest not found: {manifest_path}")

    rows: list[dict[str, str]] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        manifests = list(csv.DictReader(handle))

    for manifest in manifests:
        if manifest.get("search_status") not in {"completed", "skipped_existing_result"}:
            continue
        job = Path(manifest["job_dir"])
        best_path = job / "best_config.yaml"
        if not best_path.is_file():
            continue
        best = yaml.safe_load(best_path.read_text(encoding="utf-8")) or {}
        best_candidate = best.get("best_candidate", {})
        candidate_id = int(best_candidate.get("candidate_id", 0))
        point = next((item for item in points if _point_matches_manifest(item, manifest)), None)
        if point is None:
            continue

        fixed = point.get("fixed_params", {})
        setting = point.get("setting", "")
        for child in sorted((job / "verify_top5").glob("rank=*__repeat=*")):
            match = re.match(r"rank=(\d+)__repeat=(\d+)__candidate=(\d+)__", child.name)
            if not match or int(match.group(3)) != candidate_id:
                continue
            files = sorted(child.glob("*.csv"))
            if len(files) != 1:
                continue
            with files[0].open(newline="", encoding="utf-8") as handle:
                record = next(csv.DictReader(handle), None)
            if not record:
                continue
            rows.append(
                {
                    "table": table,
                    "setting": str(setting),
                    "dataset": str(fixed.get("dataset", "")),
                    "backbone": str(fixed.get("backbone", "")),
                    "feature_dim": str(fixed.get("feature_dim", "") or ""),
                    "seed": record.get("seed", ""),
                    "val_acc": record.get("val/acc", ""),
                    "test_acc": record.get("test/acc", ""),
                }
            )

    output = root / f"{table}_seed_rows.csv"
    if not rows:
        raise RuntimeError(f"No completed seed rows were found for {table}")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output


def run_fixed_table(
    table: str,
    *,
    repeats: int = 3,
    execute: bool = False,
) -> dict[str, Any]:
    path = FIXED_ROOT / f"{table}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
    points = list((data or {}).get("points", []))
    root = WORK_ROOT / "search" / table / "fixed"
    summary = _run_fixed_batch(points, root=root, repeats=repeats, execute=execute)
    if execute:
        summary["seed_rows"] = _collect_table_outputs(table, points, root)
    summary["table"] = table
    return summary
