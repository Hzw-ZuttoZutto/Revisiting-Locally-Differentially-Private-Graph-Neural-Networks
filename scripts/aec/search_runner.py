from __future__ import annotations

import csv, json, math, re, sys
from pathlib import Path
from typing import Any
import yaml
import numpy as np
from hparams_search_scripts import run_mechanism_hparam_search as search_impl
from hparams_search_scripts import table_search_suite_core as core
from hparams_search_scripts import mechanism_stage_utils
from .paths import FIXED_ROOT, REFERENCE_ROOT, parse_gpu_ids, max_parallel_per_gpu, WORK_ROOT

REPO_ROOT=Path(__file__).resolve().parents[2]

PIPELINE_AXES = {
    "figure3_pipeline1": {"mechanism": "mbm", "smoother": "kprop", "use_nfr": "false"},
    "figure3_pipeline2": {"mechanism": "hds", "smoother": "kprop", "use_nfr": "false"},
    "figure3_pipeline3": {"mechanism": "mbm", "smoother": "hoa", "use_nfr": "true"},
    "figure3_pipeline4": {"mechanism": "pm", "smoother": "hoa", "use_nfr": "true"},
}


def _fixed_point_id_by_result_index(figure_id: int) -> dict[int, str]:
    path = FIXED_ROOT / f"figure{figure_id}.yaml"
    if not path.is_file():
        return {}
    points = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("points", [])
    selected = []
    for point in points:
        fixed = point.get("fixed_params", point)
        dataset = str(fixed.get("dataset", "")).lower()
        backbone = str(fixed.get("backbone", "")).lower()
        if figure_id == 1 and dataset not in {"cora", "facebook"}:
            continue
        if figure_id == 6 and (dataset not in {"actor", "flickr"} or backbone != "sage"):
            continue
        if fixed.get("mechanism"):
            selected.append(str(point.get("point_id", "")))
    return {index: point_id for index, point_id in enumerate(selected) if point_id}


def _result_index(job_dir: str) -> int | None:
    matches = re.findall(r"(?:^|/)result_(\d+)(?:/|$)", str(job_dir).replace("\\", "/"))
    return int(matches[-1]) if matches else None


def _load_verify_values(manifest: dict[str, str]) -> tuple[list[float], list[float]]:
    job_dir = Path(manifest["job_dir"])
    candidate_id = int(manifest.get("best_candidate_id") or 0)
    records = []
    pattern = re.compile(r"rank=(\d+)__repeat=(\d+)__candidate=(\d+)__")
    for child in sorted((job_dir / "verify_top5").glob("rank=*__repeat=*")):
        match = pattern.match(child.name)
        if not match or int(match.group(3)) != candidate_id:
            continue
        files = sorted(child.glob("*.csv"))
        if len(files) != 1:
            continue
        with files[0].open(newline="", encoding="utf-8") as handle:
            row = next(csv.DictReader(handle), None)
        if row is None:
            continue
        records.append((int(match.group(2)), float(row["val/acc"]), float(row["test/acc"])))
    records.sort(key=lambda item: item[0])
    return [item[1] for item in records], [item[2] for item in records]


def _bootstrap_stats(figure_id: int, row: dict[str, str], manifest: dict[str, str]) -> dict[str, float]:
    val_values, test_values = _load_verify_values(manifest)
    if not test_values:
        raise RuntimeError(f"No raw verify records found under {manifest.get('job_dir')}")
    bootstrap_samples = 1000
    bootstrap_seed = 12345
    if figure_id in {1, 6}:
        from draw_figure import draw_figure1 as reference

        key = (
            row.get("source", ""),
            row.get("backbone", ""),
            row.get("dataset", ""),
            row.get("pipeline", ""),
            row.get("x_eps", ""),
        )
        stats = reference.metric_stats(
            test_values,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            key=("test", *key),
        )
        val_stats = reference.metric_stats(
            val_values,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            key=("val", *key),
        )
        return {
            "mean": stats["mean"],
            "std": stats["std"],
            "low": stats["ci_low"],
            "high": stats["ci_high"],
            "val_mean": val_stats["mean"],
            "n": len(test_values),
        }

    if figure_id == 3:
        from draw_figure.draw_figure3 import bootstrap_ci
        epsilon_key = row.get("sim_reference_eps", "") or row.get("x_eps", "")
        key = (row.get("source", ""), row.get("mechanism", ""), epsilon_key)
    elif figure_id == 4:
        from draw_figure.draw_figure4 import bootstrap_ci
        key = (row.get("epsilon", ""), row.get("scale_exponent", ""))
    elif figure_id == 5:
        from draw_figure.draw_figure5 import bootstrap_ci
        key = (row.get("epsilon", ""), row.get("tao2", ""))
    elif figure_id == 7:
        from draw_figure.draw_figure4 import bootstrap_ci
        key = (row.get("epsilon", ""), row.get("scale_exponent", ""))
    else:
        raise RuntimeError(f"Unsupported bootstrap figure: {figure_id}")

    mean, low, high = bootstrap_ci(
        test_values,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
        key=key,
    )
    return {
        "mean": mean,
        "std": float(np.std(test_values, ddof=1)),
        "low": low,
        "high": high,
        "val_mean": float(np.mean(val_values)),
        "n": len(test_values),
    }

def _configs(figure_id: int) -> list[Path]:
    roots={1:REPO_ROOT/"configs_AEC/figure1",3:REPO_ROOT/"configs_AEC/figure3",4:REPO_ROOT/"configs_AEC/figure4",5:REPO_ROOT/"configs_AEC/figure5",6:REPO_ROOT/"configs_AEC/figure6",7:REPO_ROOT/"configs_AEC/figure7","table4":REPO_ROOT/"configs_AEC/table4","table6":REPO_ROOT/"configs_AEC/table6"}
    if figure_id == 5:
        return sorted(p for base in REPO_ROOT.glob("configs_AEC/figure5/*") for p in base.rglob("*.yaml"))
    return sorted(roots[figure_id].rglob("*.yaml")) if figure_id in roots else []

def _claim_paths(figure_id):
    root=REPO_ROOT
    if figure_id==1:
        paths=[]
        paths += sorted((root/"configs_AEC/figure1/FeatFree/cora").rglob("*.yaml"))
        paths += sorted((root/"configs_AEC/figure1/FeatFree/facebook").rglob("*.yaml"))
        paths += sorted((root/"configs_AEC/figure1/LDPGNN").rglob("*.yaml"))
        paths += sorted((root/"configs_AEC/figure1/Non-private").rglob("*.yaml"))
        return paths
    if figure_id==6:
        paths=[]
        paths += sorted((root/"configs_AEC/figure6/FeatFree/actor/sage").rglob("*.yaml"))
        paths += sorted((root/"configs_AEC/figure6/FeatFree/flickr/sage").rglob("*.yaml"))
        paths += sorted((root/"configs_AEC/figure6/LDPGNN/sage").rglob("*.yaml"))
        paths += sorted((root/"configs_AEC/figure6/Non-private").rglob("*.yaml"))
        return paths
    if figure_id==5: return _configs(5)
    return _configs(figure_id)

def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle: return yaml.safe_load(handle)

def _config_token(path: Path) -> str:
    return path.relative_to(REPO_ROOT).with_suffix("").as_posix().replace("/", "__")


def _search_root_name(figure_id: int | str) -> str:
    if isinstance(figure_id, str) and figure_id.startswith("table"):
        return figure_id
    return f"figure{figure_id}"


def _runtime_config(path: Path, *, mode: str, output: Path) -> Path:
    """Materialize a runtime config controlled by the notebook configuration cell."""
    data = load_config(path)
    datasets = data.get("search_space", {}).get("dataset", {}).get("datasets", [])
    path_text = path.as_posix().lower()
    if mode == "scaled":
        if "configs_aec/figure1" in path_text:
            data["search_space"]["dataset"]["datasets"] = [
                value for value in datasets if str(value).lower() in {"cora", "facebook"}
            ]
        elif "configs_aec/figure6" in path_text:
            data["search_space"]["dataset"]["datasets"] = [
                value for value in datasets if str(value).lower() in {"actor", "flickr"}
            ]
        # Keep the repeat count declared by each YAML.  The reference plot
        # validators require Figure 3 to use 20 repeats and the other
        # statistical figures/tables to use 10; reducing every scaled run to
        # three would make the generated artifacts invalid and change their CI.

    device = data["device"]
    if device["device"] == "gpu":
        device["gpu_ids"] = parse_gpu_ids()
        device["max_parallel_per_gpu"] = max_parallel_per_gpu()
        device["gpu_launch_interval_sec"] = 0.01

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return output


def scaled_config(path: Path, *, output: Path) -> Path:
    return _runtime_config(path, mode="scaled", output=output)


def _build_runtime_batch(
    config: Path,
    *,
    mode: str,
    output_root: Path,
) -> tuple[Path, core.BatchSpec]:
    token = _config_token(config)
    actual = _runtime_config(
        config,
        mode=mode,
        output=WORK_ROOT / "runtime_configs" / mode / f"{token}.yaml",
    )
    search_config = search_impl.load_search_config(actual)
    batch = search_impl.build_batch_spec(
        search_config=search_config,
        output_root=output_root / token,
        config_copy_source=actual,
    )
    return actual, batch


def run_search(
    config: Path,
    *,
    mode: str = "scaled",
    output_root: Path | None = None,
    dry_run: bool = True,
) -> dict[str, object]:
    output_root = output_root or WORK_ROOT / "search" / config.stem / mode
    actual, batch = _build_runtime_batch(config, mode=mode, output_root=output_root.parent)
    result = {
        "mode": mode,
        "config": str(actual),
        "output_root": str(batch.output_root),
        "jobs_planned": len(batch.jobs),
        "training_tasks_planned": sum(
            len(job.job_spec["candidates"])
            + min(
                mechanism_stage_utils.VERIFY_TOPK,
                len(job.job_spec["candidates"]),
            )
            * int(job.job_spec["defaults"]["stage"]["verify"]["repeats"])
            for job in batch.jobs
        ),
        "dry_run": dry_run,
    }
    if not dry_run:
        completed, skipped, failed, manifest = core.run_batch_search(
            batch,
            repo_root=REPO_ROOT,
        )
        result.update(
            completed=completed,
            skipped=skipped,
            failed=failed,
            manifest=str(manifest),
        )
    return result

def _select_paths(figure_id, paths, mode):
    if mode == "full":
        return paths
    selected=[]
    for path in paths:
        text=path.as_posix().lower()
        if figure_id == 1:
            if "ldpgnn" in text or ("featfree" in text and any(f"/{d}/" in text for d in ("cora","facebook"))) or "non-private" in text:
                selected.append(path)
        elif figure_id == 3:
            if path.name in {"LDP.yaml","SIM.yaml"}: selected.append(path)
        elif figure_id == 5:
            selected.append(path)
        elif figure_id == 6:
            if "ldpgnn/sage" in text or ("featfree" in text and "/sage/" in text) or "non-private" in text:
                selected.append(path)
        elif figure_id in {4,7,"table4","table6"}:
            selected.append(path)
    return selected or paths


def _planned_task_count(batch: core.BatchSpec) -> int:
    total = 0
    for job in batch.jobs:
        candidate_count = len(job.job_spec["candidates"])
        repeats = int(job.job_spec["defaults"]["stage"]["verify"]["repeats"])
        total += candidate_count + min(mechanism_stage_utils.VERIFY_TOPK, candidate_count) * repeats
    return total


def _table_setting_from_config(path: Path) -> str:
    # Configs are stored as table4_FeatFree-P/... and
    # table6_FeatFree-HOA/...; the backbone directory is not the table setting.
    for part in path.parts:
        for prefix in ("table4_", "table6_"):
            if part.startswith(prefix):
                return part[len(prefix):]
    raise ValueError(f"Cannot derive table setting from config path: {path}")


def _manifest_rows(root: Path) -> list[dict[str, str]]:
    path = root / mechanism_stage_utils.ROOT_MANIFEST_FILENAME
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _aggregate_table_seed_rows(
    table: str,
    mode: str,
    *,
    setting_by_job_id: dict[str, str],
) -> Path:
    root = WORK_ROOT / "search" / table / mode
    rows: list[dict[str, str]] = []
    for manifest in _manifest_rows(root):
        if manifest.get("search_status") not in {"completed", "skipped_existing_result"}:
            continue
        setting = setting_by_job_id.get(manifest.get("job_id", ""))
        if not setting:
            continue
        job = Path(manifest["job_dir"])
        best_path = job / "best_config.yaml"
        if not best_path.is_file():
            continue
        best = yaml.safe_load(best_path.read_text(encoding="utf-8")) or {}
        candidate_id = int(best.get("best_candidate", {}).get("candidate_id", 0))
        fixed = {
            key: manifest.get(key, "")
            for key in ("dataset", "backbone", "feature_dim", "smoother")
        }
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
                    "setting": setting,
                    "dataset": fixed["dataset"],
                    "backbone": fixed["backbone"],
                    "feature_dim": fixed["feature_dim"],
                    "smoother": fixed["smoother"],
                    "seed": record.get("seed", ""),
                    "val_acc": record.get("val/acc", ""),
                    "test_acc": record.get("test/acc", ""),
                }
            )

    if not rows:
        raise RuntimeError(f"No completed seed rows were found for {table} mode={mode}")
    output = root / f"{table}_seed_rows.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output


def run_search_for_figure(
    figure_id: int | str,
    *,
    mode: str = "scaled",
    execute: bool = False,
) -> dict[str, object]:
    if figure_id in (2, 8):
        return {"figure_id": figure_id, "mode": "analytic", "configs": 0}

    paths = _configs(figure_id)
    selected = _select_paths(figure_id, paths, mode)
    root = WORK_ROOT / "search" / _search_root_name(figure_id) / mode
    jobs: list[core.BatchJob] = []
    execution = None
    config_results: list[dict[str, object]] = []
    setting_by_job_id: dict[str, str] = {}

    for config in selected:
        actual, batch = _build_runtime_batch(config, mode=mode, output_root=root)
        if execution is None:
            execution = batch.execution
        elif execution != batch.execution:
            raise RuntimeError(
                f"Selected configs do not share one execution pool: {config}"
            )
        # stable_job_id is based only on fixed parameters. Separate YAMLs
        # such as Figure 5's tao2 files can therefore produce the same ID
        # while intentionally having different candidate spaces. Namespace the
        # ID with the source config before merging batches so the global
        # manifest and scheduler cannot overwrite one another.
        namespaced_jobs = []
        config_token = _config_token(config)
        for job in batch.jobs:
            namespaced_id = f"{config_token}__{job.job_id}"
            job_spec = dict(job.job_spec)
            job_spec["job_id"] = namespaced_id
            namespaced_jobs.append(
                core.BatchJob(
                    job_id=namespaced_id,
                    job_dir=job.job_dir,
                    display_name=job.display_name,
                    job_spec=job_spec,
                )
            )
        jobs.extend(namespaced_jobs)
        if isinstance(figure_id, str) and figure_id.startswith("table"):
            setting = _table_setting_from_config(config)
            for job in namespaced_jobs:
                setting_by_job_id[job.job_id] = setting
        config_results.append(
            {
                "source_config": str(config),
                "runtime_config": str(actual),
                "jobs_planned": len(batch.jobs),
                "training_tasks_planned": _planned_task_count(batch),
            }
        )

    if execution is None:
        return {
            "figure_id": figure_id,
            "mode": mode,
            "configs_total": len(paths),
            "configs_selected": 0,
            "jobs_planned": 0,
            "training_tasks_planned": 0,
            "configs": [],
        }

    batch = core.BatchSpec(
        output_root=root,
        execution=execution,
        jobs=jobs,
        config_copy_source=Path(config_results[0]["runtime_config"]),
    )
    result: dict[str, object] = {
        "figure_id": figure_id,
        "mode": mode,
        "configs_total": len(paths),
        "configs_selected": len(selected),
        "jobs_planned": len(jobs),
        "training_tasks_planned": _planned_task_count(batch),
        "configs": config_results,
        "execute": execute,
    }
    if execute:
        completed, skipped, failed, manifest = core.run_batch_search(
            batch,
            repo_root=REPO_ROOT,
        )
        result.update(
            completed=completed,
            skipped=skipped,
            failed=failed,
            manifest=str(manifest),
        )
        if isinstance(figure_id, str) and figure_id.startswith("table"):
            result["seed_rows"] = str(
                _aggregate_table_seed_rows(
                    figure_id,
                    mode,
                    setting_by_job_id=setting_by_job_id,
                )
            )
        else:
            _aggregate_search_output(figure_id, mode)
    return result

def _aggregate_search_output(figure_id, mode):
    from .paths import REFERENCE_ROOT
    table= {1:"figure1_plot_data.csv",3:"figure3_plot_data.csv",4:"figure4_plot_data.csv",5:"figure5_plot_data.csv",6:"figure6_plot_data.csv",7:"figure7_plot_data.csv"}.get(figure_id)
    if not table: return
    ref=REFERENCE_ROOT/table
    if not ref.is_file(): return
    with ref.open(newline="",encoding="utf-8") as handle: rows=list(csv.DictReader(handle))
    root=WORK_ROOT/"search"/f"figure{figure_id}"/mode
    batch_manifest = root/"manifest.csv"
    manifests=[]
    if batch_manifest.is_file():
        with batch_manifest.open(newline="",encoding="utf-8") as handle:
            manifests.extend(csv.DictReader(handle))
    else:
        for path in root.rglob("manifest.csv"):
            with path.open(newline="",encoding="utf-8") as handle:
                manifests.extend(csv.DictReader(handle))
    def same_value(a: object, b: object) -> bool:
        left = "" if a is None else str(a).strip().lower()
        right = "" if b is None else str(b).strip().lower()
        if left == right:
            return True
        try:
            return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False

    def manifest_matches_row(out: dict[str, str], manifest: dict[str, str]) -> bool:
        if manifest.get("search_status") not in {"completed", "skipped_existing_result"}:
            return False
        for field in (
            "dataset", "backbone", "mechanism", "smoother", "feature_dim",
            "norm", "norm_scale", "use_nfr",
        ):
            expected = out.get(field, "")
            if expected and not same_value(manifest.get(field, ""), expected):
                return False

        figure_source = out.get("source", "").strip().lower()
        if figure_id == 3:
            # SIM stores the plotted epsilon in sim_reference_eps and uses
            # x_eps=inf for the perturbation stage. Matching x_eps here would
            # merge SIM rows with the wrong LDP rows.
            expected_feature = {"ldp": "raw", "sim": "sim"}.get(figure_source)
            if expected_feature and not same_value(manifest.get("feature", ""), expected_feature):
                return False
            epsilon = out.get("x_eps", "")
            manifest_field = "sim_reference_eps" if figure_source == "sim" else "x_eps"
            if epsilon and not same_value(manifest.get(manifest_field, ""), epsilon):
                return False
        elif out.get("x_eps") and not same_value(manifest.get("x_eps", ""), out["x_eps"]):
            return False

        if out.get("epsilon") and not same_value(manifest.get("x_eps", ""), out["epsilon"]):
            return False
        if out.get("tao2"):
            actual_tao2 = manifest.get("best_tao2", manifest.get("tao2", ""))
            if not same_value(actual_tao2, out["tao2"]):
                return False
        return True

    def apply_manifest(out, manifest):
        stats = _bootstrap_stats(figure_id, out, manifest)
        out["test_acc_mean"] = f"{stats['mean']:.12g}"
        out["test_acc_std"] = f"{stats['std']:.12g}"
        out["test_acc_ci_low"] = f"{stats['low']:.12g}"
        out["test_acc_ci_high"] = f"{stats['high']:.12g}"
        out["val_acc_mean"] = f"{stats['val_mean']:.12g}"
        out["n"] = str(int(stats["n"]))

    fixed_point_map = _fixed_point_id_by_result_index(figure_id) if mode == "fixed" else {}
    fixed_manifests = {}
    if fixed_point_map:
        for manifest in manifests:
            point_id = fixed_point_map.get(_result_index(manifest.get("job_dir", "")))
            if point_id:
                fixed_manifests[point_id] = manifest

    missing = []
    for out in rows:
        pipeline = out.get("pipeline", "")
        expected_axes = PIPELINE_AXES.get(pipeline)

        if fixed_point_map:
            # Fixed runs are bound to the point index used to create result_N,
            # never to display axes such as epsilon or norm_scale. Baseline
            # lines intentionally remain the frozen reference values.
            if figure_id in {1, 6} and expected_axes is None:
                continue
            point_id = out.get("point_id", "")
            if point_id not in fixed_point_map.values():
                continue
            manifest = fixed_manifests.get(point_id)
            if not manifest or manifest.get("best_verify_test_acc_mean", "") == "":
                missing.append({"point_id": point_id, "pipeline": pipeline})
                continue
            apply_manifest(out, manifest)
            continue

        # Figure 1 and Figure 6 contain fixed baseline lines whose x-axis is
        # privacy budget for presentation only. They must remain frozen and
        # must never be replaced by a privacy-method manifest.
        if figure_id in {1, 6} and expected_axes is None:
            continue
        candidates = []
        for manifest in manifests:
            if expected_axes and any(
                not same_value(manifest.get(key, ""), value)
                for key, value in expected_axes.items()
            ):
                continue
            if manifest_matches_row(out, manifest):
                candidates.append(manifest)
        if len(candidates) != 1:
            missing.append({
                "source": out.get("source", ""),
                "pipeline": out.get("pipeline", ""),
                "dataset": out.get("dataset", ""),
                "backbone": out.get("backbone", ""),
                "x_eps": out.get("x_eps", ""),
                "epsilon": out.get("epsilon", ""),
                "norm_scale": out.get("norm_scale", ""),
                "tao2": out.get("tao2", ""),
                "candidate_manifests": len(candidates),
            })
            continue
        apply_manifest(out, candidates[0])
    if missing:
        sample = "; ".join(str(item) for item in missing[:5])
        raise RuntimeError(
            f"Missing completed fixed/search results for {len(missing)} plot rows; sample: {sample}"
        )
    root.mkdir(parents=True,exist_ok=True); output=root/"plot_data.csv"
    with output.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
