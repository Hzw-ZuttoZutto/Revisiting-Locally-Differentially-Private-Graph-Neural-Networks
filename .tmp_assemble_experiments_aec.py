#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import os
import shutil
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from hparams_search_scripts import mechanism_stage_utils as stage_utils
from hparams_search_scripts import table_search_suite_core as suite_core
from hparams_search_scripts.mechanism_stage_context import (
    build_recommended_command_parts,
    resolve_job_context,
)
from hparams_search_scripts.run_mechanism_hparam_search import (
    build_batch_spec,
    load_search_config,
)


REPO_ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = REPO_ROOT / "configs_AEC"
RUNTIME_ROOT = Path("/data/hzw/Rethinking_DP_GNN_runtime")
PAPER_ROOT = RUNTIME_ROOT / "paper_experiments"
REBUTTAL_ROOT = RUNTIME_ROOT / "rebuttal_experiments"
FINAL_ROOT = RUNTIME_ROOT / "experiments_AEC"
STAGING_ROOT = RUNTIME_ROOT / "experiments_AEC.staging"
BACKFILL_ROOT = PAPER_ROOT / "main_add_10seed_backfill"

PARTIAL_CONFIG_PREFIX = ("figure1", "LDPGNN")
PIPELINE_NAMES = {
    "LPGNN.yaml": "figure3_pipeline1",
    "PrivGE.yaml": "figure3_pipeline2",
    "UPGNET-MBM.yaml": "figure3_pipeline3",
    "UPGNET-PM.yaml": "figure3_pipeline4",
}
COMPLETE_STATUSES = {"completed", "skipped_existing_result"}
ADD_SCALE_EXPONENTS = {-6, -4}
ADD_AGAIN_SCALE_EXPONENTS = {-5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15, 17}


@dataclass(frozen=True)
class SourceChoice:
    job_dir: Path
    manifest_path: Path
    manifest_row: dict[str, str]
    backfill_row: dict[str, str] | None = None


@dataclass(frozen=True)
class AssemblyItem:
    config_path: Path
    config_relative: Path
    job: suite_core.BatchJob
    actual_job_dir: Path
    source: SourceChoice
    partial: bool


def scalar_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).strip()


def fixed_key(mapping: dict[str, Any]) -> tuple[str, ...]:
    values = {
        axis: scalar_text(mapping.get(axis, ""))
        for axis in stage_utils.OUTER_AXIS_NAMES
    }
    if values["dataset"] == "attributedgraph-flickr":
        values["dataset"] = "flickr"
    if values["scale"] in {"1", "1.0"}:
        values["scale"] = ""
    if values["feature_preprojection"].lower() in {"", "false", "0"}:
        values["feature_preprojection"] = ""
        values["preprojection_output_dim"] = ""
    if values["sanity_check"].lower() in {"", "false", "0"}:
        values["sanity_check"] = ""
        values["node_ratio"] = ""
    if values["norm"].lower() in {"", "false", "0"}:
        values["norm_scale"] = ""
    return tuple(values[axis] for axis in stage_utils.OUTER_AXIS_NAMES)


def canonical_x_eps(value: Any) -> str:
    return stage_utils.canonical_float_text(value)


def actual_path(logical_path: Path) -> Path:
    return STAGING_ROOT / logical_path.relative_to(FINAL_ROOT)


def path_after_anchor(raw_path: Path, anchor: str) -> tuple[str, ...]:
    matching = [index for index, part in enumerate(raw_path.parts) if part == anchor]
    if not matching:
        raise RuntimeError(f"Could not locate source root {anchor!r} in {raw_path}")
    return raw_path.parts[matching[-1] + 1 :]


class ManifestIndex:
    def __init__(self) -> None:
        self._cache: dict[Path, dict[tuple[str, ...], list[dict[str, str]]]] = {}

    def load(self, config_root: Path) -> dict[tuple[str, ...], list[dict[str, str]]]:
        config_root = config_root.resolve()
        cached = self._cache.get(config_root)
        if cached is not None:
            return cached
        manifest_path = config_root / stage_utils.ROOT_MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Source manifest is missing: {manifest_path}")
        index: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("search_status") not in COMPLETE_STATUSES:
                    continue
                index[fixed_key(row)].append(dict(row))
        self._cache[config_root] = index
        return index

    def select(self, config_root: Path, fixed_params: dict[str, Any]) -> SourceChoice:
        index = self.load(config_root)
        rows = index.get(fixed_key(fixed_params), [])
        if len(rows) != 1:
            raise RuntimeError(
                f"Expected exactly one completed source row in {config_root}, "
                f"found {len(rows)} for {fixed_params}"
            )
        row = rows[0]
        raw_job_dir = Path(row["job_dir"])
        if raw_job_dir.is_dir():
            job_dir = raw_job_dir
        else:
            tail = path_after_anchor(raw_job_dir, config_root.name)
            job_dir = config_root.joinpath(*tail)
        if not job_dir.is_dir():
            raise FileNotFoundError(f"Resolved source job directory is missing: {job_dir}")
        return SourceChoice(
            job_dir=job_dir.resolve(),
            manifest_path=(config_root / stage_utils.ROOT_MANIFEST_FILENAME).resolve(),
            manifest_row=row,
        )


def nearest_manifest(job_dir: Path) -> Path:
    for parent in (job_dir, *job_dir.parents):
        candidate = parent / stage_utils.ROOT_MANIFEST_FILENAME
        if candidate.is_file():
            return candidate.resolve()
        if parent == RUNTIME_ROOT:
            break
    raise FileNotFoundError(f"Could not find source manifest above {job_dir}")


def read_backfill_index() -> dict[tuple[str, str, str, str], dict[str, str]]:
    path = BACKFILL_ROOT / "backfill_manifest.csv"
    index: dict[tuple[str, str, str, str], dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (
                row["backbone"],
                row["pipeline"],
                row["dataset"],
                canonical_x_eps(row["x_eps"]),
            )
            if key in index:
                raise RuntimeError(f"Duplicate backfill key: {key}")
            index[key] = dict(row)
    return index


def source_config_root(config_relative: Path, fixed_params: dict[str, Any]) -> Path:
    parts = config_relative.parts
    if parts[0] == "figure1" and parts[1] == "FeatFree":
        dataset, backbone = parts[2], parts[3]
        return REBUTTAL_ROOT / "featfree_homo_rerun" / "HOA" / dataset / backbone / "random_projected.yaml"
    if parts[0] == "figure1" and parts[1] == "Non-private":
        return PAPER_ROOT / "clean_reference"
    if parts[0] == "figure3":
        source_name = "figure4_ori.yaml" if config_relative.name == "LDP.yaml" else "figure4_sim.yaml"
        return PAPER_ROOT / "figure4_final" / source_name
    if parts[0] == "figure4":
        norm_scale = float(fixed_params["norm_scale"])
        exponent = round(math.log2(norm_scale))
        if not math.isclose(norm_scale, 2.0**exponent, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"Figure 4 norm_scale is not a power of two: {norm_scale}")
        if exponent in ADD_SCALE_EXPONENTS:
            root = PAPER_ROOT / "figure5_final_add"
        elif exponent in ADD_AGAIN_SCALE_EXPONENTS:
            root = PAPER_ROOT / "figure5_final_add_again"
        else:
            root = PAPER_ROOT / "figure5_final"
        return root / "figure5.yaml"
    if parts[0] == "figure6" and parts[1] == "FeatFree":
        dataset = "attributedgraph-flickr" if parts[2] == "flickr" else parts[2]
        backbone = parts[3]
        return REBUTTAL_ROOT / "featfree_heter" / "HOA" / dataset / backbone / "random_projected.yaml"
    if parts[0] == "figure6" and parts[1] == "LDPGNN":
        dataset = "attributedgraph-flickr" if fixed_params["dataset"] == "flickr" else fixed_params["dataset"]
        backbone = parts[2]
        pipeline = PIPELINE_NAMES[config_relative.name]
        return REBUTTAL_ROOT / "figure1_heter" / dataset / backbone / f"{pipeline}.yaml"
    if parts[0] == "figure6" and parts[1] == "Non-private":
        dataset = "attributedgraph-flickr" if fixed_params["dataset"] == "flickr" else fixed_params["dataset"]
        return REBUTTAL_ROOT / "clean_heter" / dataset / "figure1_clean.yaml"
    if parts[0] == "figure7":
        return REBUTTAL_ROOT / "figure5_heter" / "figure5.yaml"
    if parts[0] == "table4":
        backbone = parts[-2]
        return PAPER_ROOT / "main2_again" / backbone / "direct.yaml"
    if parts[0] == "table6":
        backbone = parts[-2]
        return PAPER_ROOT / "main2_again" / backbone / "random_projected.yaml"
    raise RuntimeError(f"No source routing rule for {config_relative}")


def choose_source(
    manifest_index: ManifestIndex,
    backfill_index: dict[tuple[str, str, str, str], dict[str, str]],
    config_relative: Path,
    job: suite_core.BatchJob,
) -> SourceChoice:
    parts = config_relative.parts
    fixed_params = job.job_spec["fixed_params"]
    if parts[:2] == PARTIAL_CONFIG_PREFIX:
        pipeline = PIPELINE_NAMES[config_relative.name]
        key = (
            parts[2],
            pipeline,
            str(fixed_params["dataset"]),
            canonical_x_eps(fixed_params["x_eps"]),
        )
        row = backfill_index.get(key)
        if row is None:
            raise RuntimeError(f"Backfill mapping is missing: {key}")
        if row.get("merge_status") != "complete":
            raise RuntimeError(f"Backfill row is not complete: {key}: {row}")
        job_dir = Path(row["source_job_dir"]).resolve()
        extra_csv = Path(row["extra_csv_path"])
        if not job_dir.is_dir() or not extra_csv.is_file():
            raise FileNotFoundError(f"Backfill source is missing for {key}")
        return SourceChoice(
            job_dir=job_dir,
            manifest_path=nearest_manifest(job_dir),
            manifest_row={},
            backfill_row=row,
        )
    return manifest_index.select(source_config_root(config_relative, fixed_params), fixed_params)


def candidate_signature(job_spec: dict[str, Any]) -> list[tuple[int, tuple[str, ...]]]:
    return [
        (candidate.candidate_id, candidate.key)
        for candidate in stage_utils.job_candidates(job_spec)
    ]


def validate_source_spec(item: AssemblyItem) -> None:
    expected = item.job.job_spec
    source = stage_utils.load_job_spec(item.source.job_dir)
    if fixed_key(source.get("fixed_params", {})) != fixed_key(expected["fixed_params"]):
        raise RuntimeError(f"Fixed parameters do not match for {item.config_relative}: {item.source.job_dir}")
    if candidate_signature(source) != candidate_signature(expected):
        raise RuntimeError(f"Candidate space does not match for {item.config_relative}: {item.source.job_dir}")
    if int(source.get("base_seed", -1)) != int(expected["base_seed"]):
        raise RuntimeError(f"Base seed does not match for {item.source.job_dir}")
    source_repeats = int(source["defaults"]["stage"]["verify"]["repeats"])
    expected_repeats = int(expected["defaults"]["stage"]["verify"]["repeats"])
    required_source_repeats = 5 if item.partial else expected_repeats
    if source_repeats != required_source_repeats:
        raise RuntimeError(
            f"Verify repeats mismatch for {item.source.job_dir}: "
            f"source={source_repeats}, required={required_source_repeats}"
        )
    if not stage_utils.verify_topk_path(item.source.job_dir).is_file():
        raise FileNotFoundError(f"verify_topk.csv is missing: {item.source.job_dir}")


def build_items() -> list[AssemblyItem]:
    manifest_index = ManifestIndex()
    backfill_index = read_backfill_index()
    items: list[AssemblyItem] = []
    config_paths = sorted(CONFIG_ROOT.rglob("*.yaml"))
    if len(config_paths) != 57:
        raise RuntimeError(f"Expected 57 AEC configs, found {len(config_paths)}")
    for config_number, config_path in enumerate(config_paths, start=1):
        config_relative = config_path.relative_to(CONFIG_ROOT)
        search_config = load_search_config(config_path)
        batch = build_batch_spec(
            search_config=search_config,
            output_root=FINAL_ROOT / config_relative,
            config_copy_source=config_path,
        )
        partial = config_relative.parts[:2] == PARTIAL_CONFIG_PREFIX
        for job in batch.jobs:
            source = choose_source(manifest_index, backfill_index, config_relative, job)
            item = AssemblyItem(
                config_path=config_path,
                config_relative=config_relative,
                job=job,
                actual_job_dir=actual_path(job.job_dir),
                source=source,
                partial=partial,
            )
            validate_source_spec(item)
            items.append(item)
        print(
            f"Mapped config {config_number:02d}/{len(config_paths)}: "
            f"{config_relative} ({len(batch.jobs)} jobs)",
            flush=True,
        )
    counts = Counter("partial" if item.partial else "complete" for item in items)
    if len(items) != 1266 or counts != Counter({"complete": 786, "partial": 480}):
        raise RuntimeError(f"Unexpected assembly item counts: total={len(items)}, {counts}")
    return items


def copy_job_tree(item: AssemblyItem) -> None:
    item.actual_job_dir.parent.mkdir(parents=True, exist_ok=True)
    if item.actual_job_dir.exists():
        raise FileExistsError(f"Target job already exists: {item.actual_job_dir}")
    shutil.copytree(
        item.source.job_dir,
        item.actual_job_dir,
        symlinks=False,
        copy_function=shutil.copy2,
    )
    stage_utils.write_yaml_file(
        stage_utils.job_spec_path(item.actual_job_dir),
        item.job.job_spec,
    )


def validate_grid(job_spec: dict[str, Any], job_dir: Path) -> int:
    base_seed = int(job_spec["base_seed"])
    missing: list[int] = []
    candidates = stage_utils.job_candidates(job_spec)
    for candidate in candidates:
        output_dir = stage_utils.grid_candidate_output_dir(job_dir, candidate)
        if not stage_utils.candidate_result_exists(
            output_dir,
            candidate,
            expected_seed=base_seed,
        ):
            missing.append(candidate.candidate_id)
    if missing:
        raise RuntimeError(f"Missing grid results in {job_dir}: {missing[:20]}")
    return len(candidates)


def verify_inventory(
    job_spec: dict[str, Any],
    job_dir: Path,
) -> tuple[list[stage_utils.RankedCandidate], list[tuple[int, int, int]]]:
    ranked = stage_utils.load_ranked_candidates(stage_utils.verify_topk_path(job_dir))
    expected_topk = min(int(job_spec["verify_topk"]), len(stage_utils.job_candidates(job_spec)))
    if len(ranked) != expected_topk:
        raise RuntimeError(f"Unexpected top-k size in {job_dir}: {len(ranked)} != {expected_topk}")
    base_seed = int(job_spec["base_seed"])
    repeats = int(job_spec["defaults"]["stage"]["verify"]["repeats"])
    missing: list[tuple[int, int, int]] = []
    for ranked_candidate in ranked:
        for repeat_id in range(1, repeats + 1):
            expected_seed = base_seed + repeat_id - 1
            output_dir = stage_utils.verify_candidate_output_dir(job_dir, ranked_candidate, repeat_id)
            if not stage_utils.candidate_result_exists(
                output_dir,
                ranked_candidate.candidate,
                expected_seed=expected_seed,
            ):
                missing.append((ranked_candidate.rank, repeat_id, expected_seed))
    return ranked, missing


def write_best_config(
    actual_job_dir: Path,
    logical_job_dir: Path,
    job_spec: dict[str, Any],
    summary_rows: list[dict[str, Any]],
) -> None:
    ctx = resolve_job_context(actual_job_dir)
    winner = summary_rows[0]
    candidate = stage_utils.job_candidate_by_id(job_spec, int(winner["candidate_id"]))
    recommended_parts = build_recommended_command_parts(ctx, candidate)
    recommended_parts = [
        part.replace(str(actual_job_dir), str(logical_job_dir))
        for part in recommended_parts
    ]
    recommended_command = stage_utils.shell_join(recommended_parts)
    stage_utils.recommended_command_path(actual_job_dir).write_text(
        recommended_command + "\n",
        encoding="utf-8",
    )
    best_config: dict[str, Any] = {
        "schema_version": stage_utils.SCHEMA_VERSION,
        "job_id": job_spec["job_id"],
        "best_rank": 1,
        "best_candidate": candidate.to_dict(),
        "fixed_params": ctx.fixed_params,
        "defaults": ctx.defaults,
        "verify_metrics": {
            "val_acc": {
                "mean": float(winner["val_acc_mean"]),
                "std": float(winner["val_acc_std"]),
                "min": float(winner["val_acc_min"]),
                "max": float(winner["val_acc_max"]),
                "n": int(winner["n"]),
            },
            "test_acc": {
                "mean": float(winner["test_acc_mean"]),
                "std": float(winner["test_acc_std"]),
                "min": float(winner["test_acc_min"]),
                "max": float(winner["test_acc_max"]),
                "n": int(winner["n"]),
            },
        },
        "artifacts": {
            "verify_summary_csv": str(logical_job_dir / stage_utils.VERIFY_SUMMARY_FILENAME),
            "recommended_command_txt": str(logical_job_dir / stage_utils.RECOMMENDED_COMMAND_FILENAME),
        },
        "recommended_command": recommended_command,
    }
    if winner.get("sanity_e_pg_mean") not in (None, ""):
        best_config["verify_metrics"]["sanity_e_pg"] = {
            "mean": float(winner["sanity_e_pg_mean"]),
            "std": float(winner["sanity_e_pg_std"]),
            "min": float(winner["sanity_e_pg_min"]),
            "max": float(winner["sanity_e_pg_max"]),
            "n": int(winner["sanity_e_pg_n"]),
        }
    stage_utils.write_yaml_file(stage_utils.best_config_path(actual_job_dir), best_config)


def finalize_complete_job(item: AssemblyItem) -> tuple[int, int]:
    grid_done = validate_grid(item.job.job_spec, item.actual_job_dir)
    ranked, missing = verify_inventory(item.job.job_spec, item.actual_job_dir)
    if missing:
        raise RuntimeError(f"Complete source has missing verify results in {item.actual_job_dir}: {missing[:20]}")
    summary_rows = stage_utils.aggregate_verify_results(item.job.job_spec, item.actual_job_dir)
    repeats = int(item.job.job_spec["defaults"]["stage"]["verify"]["repeats"])
    if any(int(row["n"]) != repeats for row in summary_rows):
        raise RuntimeError(f"Verify summary count mismatch in {item.actual_job_dir}")
    stage_utils.write_verify_summary(item.actual_job_dir, summary_rows)
    write_best_config(item.actual_job_dir, item.job.job_dir, item.job.job_spec, summary_rows)
    if not stage_utils.job_outputs_complete(item.actual_job_dir):
        raise RuntimeError(f"Finalized job is not complete: {item.actual_job_dir}")
    return grid_done, len(ranked) * repeats


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        return list(reader.fieldnames or []), rows


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def split_backfill_rows(
    item: AssemblyItem,
    ranked_candidate: stage_utils.RankedCandidate,
    extra_csv: Path,
    provenance_dir: Path,
) -> list[Path]:
    fieldnames, rows = read_csv_rows(extra_csv)
    if len(rows) != 5:
        raise RuntimeError(f"Expected five backfill rows in {extra_csv}, found {len(rows)}")
    normalized: list[tuple[int, dict[str, str]]] = []
    for row in rows:
        effective_seed = int(float(row["seed"])) + int(row["version"])
        if stage_utils.candidate_key_from_row(row) != ranked_candidate.candidate.key:
            raise RuntimeError(f"Backfill candidate mismatch in {extra_csv}")
        normalized.append((effective_seed, row))
    normalized.sort(key=lambda item_pair: item_pair[0])
    expected_seeds = list(range(int(item.job.job_spec["base_seed"]) + 5, int(item.job.job_spec["base_seed"]) + 10))
    if [seed for seed, _ in normalized] != expected_seeds:
        raise RuntimeError(f"Unexpected backfill seeds in {extra_csv}: {[seed for seed, _ in normalized]}")
    shutil.copy2(extra_csv, provenance_dir / "extra_backfill.original.csv")
    written: list[Path] = []
    for offset, (effective_seed, row) in enumerate(normalized, start=6):
        output_dir = stage_utils.verify_candidate_output_dir(
            item.actual_job_dir,
            ranked_candidate,
            offset,
        )
        if output_dir.exists():
            raise FileExistsError(f"Backfill target already exists: {output_dir}")
        output_dir.mkdir(parents=True)
        logical_output_dir = item.job.job_dir / output_dir.relative_to(item.actual_job_dir)
        normalized_row = dict(row)
        normalized_row["version"] = "0"
        normalized_row["seed"] = str(effective_seed)
        normalized_row["repeats"] = "1"
        normalized_row["output_dir"] = str(logical_output_dir)
        output_csv = output_dir / f"aec_backfill_repeat_{offset:02d}.csv"
        write_csv_rows(output_csv, fieldnames, [normalized_row])
        if not stage_utils.candidate_result_exists(
            output_dir,
            ranked_candidate.candidate,
            expected_seed=effective_seed,
        ):
            raise RuntimeError(f"Normalized backfill result did not validate: {output_csv}")
        written.append(output_csv)
    return written


def summary_stats(values: list[float]) -> dict[str, Any]:
    return {
        "mean": float(statistics.mean(values)),
        "std": 0.0 if len(values) == 1 else float(statistics.stdev(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "n": len(values),
    }


def partial_best_rows(
    item: AssemblyItem,
    ranked_candidate: stage_utils.RankedCandidate,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    base_seed = int(item.job.job_spec["base_seed"])
    for repeat_id in range(1, 11):
        output_dir = stage_utils.verify_candidate_output_dir(
            item.actual_job_dir,
            ranked_candidate,
            repeat_id,
        )
        csv_path = stage_utils.find_matching_result_csv(
            output_dir,
            ranked_candidate.candidate,
            expected_seed=base_seed + repeat_id - 1,
        )
        if csv_path is None:
            raise RuntimeError(f"Best candidate repeat is missing: {output_dir}")
        _, rows = read_csv_rows(csv_path)
        row = rows[0]
        result.append(
            {
                "repeat": repeat_id,
                "seed": base_seed + repeat_id - 1,
                "seed_origin": "existing_verify" if repeat_id <= 5 else "extra_backfill",
                "val_acc": float(row["val/acc"]),
                "test_acc": float(row["test/acc"]),
                "result_csv_path": str(
                    item.job.job_dir / csv_path.relative_to(item.actual_job_dir)
                ),
            }
        )
    return result


def finalize_partial_job(item: AssemblyItem) -> tuple[int, int, int]:
    row = item.source.backfill_row
    if row is None:
        raise RuntimeError(f"Partial job lacks a backfill row: {item.source.job_dir}")
    grid_done = validate_grid(item.job.job_spec, item.actual_job_dir)
    ranked = stage_utils.load_ranked_candidates(stage_utils.verify_topk_path(item.actual_job_dir))
    if len(ranked) != 5:
        raise RuntimeError(f"Expected five ranked candidates in {item.actual_job_dir}")
    actual_rank = int(row["actual_verify_rank"])
    selected = next((entry for entry in ranked if entry.rank == actual_rank), None)
    if selected is None:
        raise RuntimeError(f"Backfill rank {actual_rank} is absent in {item.actual_job_dir}")
    if selected.candidate_id != int(row["candidate_id"]):
        raise RuntimeError(f"Backfill candidate id mismatch in {item.actual_job_dir}")

    provenance_dir = item.actual_job_dir / "aec_provenance"
    provenance_dir.mkdir(exist_ok=False)
    legacy_files = {
        stage_utils.BEST_CONFIG_FILENAME: "best_config.legacy_5repeat.yaml",
        stage_utils.VERIFY_SUMMARY_FILENAME: "verify_top5_summary.legacy_5repeat.csv",
        stage_utils.RECOMMENDED_COMMAND_FILENAME: "recommended_command.legacy_5repeat.txt",
    }
    for source_name, target_name in legacy_files.items():
        source_path = item.actual_job_dir / source_name
        if not source_path.is_file():
            raise FileNotFoundError(f"Legacy final artifact is missing: {source_path}")
        shutil.move(str(source_path), provenance_dir / target_name)

    extra_csv = Path(row["extra_csv_path"])
    split_backfill_rows(item, selected, extra_csv, provenance_dir)
    ranked_after, missing = verify_inventory(item.job.job_spec, item.actual_job_dir)
    if len(ranked_after) != 5 or len(missing) != 20:
        raise RuntimeError(f"Unexpected partial verify inventory in {item.actual_job_dir}: missing={len(missing)}")
    expected_missing = {
        (entry.rank, repeat_id, int(item.job.job_spec["base_seed"]) + repeat_id - 1)
        for entry in ranked_after
        if entry.rank != actual_rank
        for repeat_id in range(6, 11)
    }
    if set(missing) != expected_missing:
        raise RuntimeError(f"Partial verify missing pattern is wrong in {item.actual_job_dir}")

    missing_rows = [
        {
            "rank": rank,
            "repeat": repeat_id,
            "expected_seed": seed,
            "candidate_id": next(entry.candidate_id for entry in ranked_after if entry.rank == rank),
        }
        for rank, repeat_id, seed in missing
    ]
    write_csv_rows(
        item.actual_job_dir / "AEC_MISSING_VERIFY.csv",
        ["rank", "repeat", "expected_seed", "candidate_id"],
        missing_rows,
    )

    best_rows = partial_best_rows(item, selected)
    write_csv_rows(
        item.actual_job_dir / "AEC_BEST_CANDIDATE_10SEED.csv",
        ["repeat", "seed", "seed_origin", "val_acc", "test_acc", "result_csv_path"],
        best_rows,
    )
    best_summary = {
        "candidate": selected.candidate.to_dict(),
        "actual_verify_rank_from_5repeat_search": actual_rank,
        "val_acc": summary_stats([float(value["val_acc"]) for value in best_rows]),
        "test_acc": summary_stats([float(value["test_acc"]) for value in best_rows]),
    }
    stage_utils.write_yaml_file(
        item.actual_job_dir / "AEC_BEST_CANDIDATE_10SEED_SUMMARY.yaml",
        best_summary,
    )
    marker = {
        "status": "incomplete_verify",
        "reason": "Only the selected best candidate received five additional backfill repeats.",
        "expected_verify_total": 50,
        "present_verify_total": 30,
        "missing_verify_total": 20,
        "selected_best_rank": actual_rank,
        "selected_best_candidate_id": selected.candidate_id,
        "source_job_dir": str(item.source.job_dir),
        "backfill_source_csv": str(extra_csv),
        "missing_verify_csv": str(item.job.job_dir / "AEC_MISSING_VERIFY.csv"),
        "resume_note": "Run the current config against this output root to fill only the missing verify slots.",
    }
    stage_utils.write_yaml_file(item.actual_job_dir / "AEC_INCOMPLETE.yaml", marker)
    if stage_utils.job_outputs_complete(item.actual_job_dir):
        raise RuntimeError(f"Partial job was incorrectly marked complete: {item.actual_job_dir}")
    return grid_done, 30, 20


def populate_complete_manifest_row(
    row: dict[str, str],
    item: AssemblyItem,
    grid_done: int,
    verify_done: int,
) -> None:
    best_config = stage_utils.load_best_config(item.actual_job_dir)
    best_candidate = best_config["best_candidate"]
    verify_metrics = best_config["verify_metrics"]
    row["search_status"] = "completed"
    row["grid_total"] = str(grid_done)
    row["grid_done"] = str(grid_done)
    row["verify_total"] = str(verify_done)
    row["verify_done"] = str(verify_done)
    row["best_rank"] = str(best_config["best_rank"])
    row["best_candidate_id"] = str(best_candidate["candidate_id"])
    row["best_x_steps"] = stage_utils.canonical_search_value(best_candidate.get("x_steps"))
    row["best_learning_rate"] = stage_utils.canonical_search_value(best_candidate.get("learning_rate"))
    row["best_weight_decay"] = stage_utils.canonical_search_value(best_candidate.get("weight_decay"))
    row["best_dropout"] = stage_utils.canonical_search_value(best_candidate.get("dropout"))
    row["best_tao2"] = stage_utils.canonical_search_value(best_candidate.get("tao2"))
    row["best_verify_val_acc_mean"] = stage_utils.canonical_search_value(verify_metrics["val_acc"]["mean"])
    row["best_verify_val_acc_std"] = stage_utils.canonical_search_value(verify_metrics["val_acc"]["std"])
    row["best_verify_test_acc_mean"] = stage_utils.canonical_search_value(verify_metrics["test_acc"]["mean"])
    row["best_verify_test_acc_std"] = stage_utils.canonical_search_value(verify_metrics["test_acc"]["std"])
    if "sanity_e_pg" in verify_metrics:
        row["best_verify_sanity_e_pg_mean"] = stage_utils.canonical_search_value(
            verify_metrics["sanity_e_pg"]["mean"]
        )
    row["best_config_path"] = str(item.job.job_dir / stage_utils.BEST_CONFIG_FILENAME)
    row["recommended_command_path"] = str(item.job.job_dir / stage_utils.RECOMMENDED_COMMAND_FILENAME)


def assemble(items: list[AssemblyItem]) -> None:
    if FINAL_ROOT.exists() or STAGING_ROOT.exists():
        raise FileExistsError(f"Final or staging target already exists: {FINAL_ROOT}, {STAGING_ROOT}")
    STAGING_ROOT.mkdir(parents=True)
    grouped: dict[Path, list[AssemblyItem]] = defaultdict(list)
    for item in items:
        grouped[item.config_relative].append(item)

    source_rows: list[dict[str, Any]] = []
    completed_count = 0
    partial_count = 0
    for config_number, config_relative in enumerate(sorted(grouped), start=1):
        config_items = grouped[config_relative]
        actual_config_root = actual_path(FINAL_ROOT / config_relative)
        actual_config_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_items[0].config_path, actual_config_root / stage_utils.INPUT_CONFIG_COPY_FILENAME)
        manifest_rows: list[dict[str, str]] = []
        for item_number, item in enumerate(config_items, start=1):
            copy_job_tree(item)
            row = suite_core._default_row(item.job)
            if item.partial:
                grid_done, verify_done, verify_missing = finalize_partial_job(item)
                row["search_status"] = "in_progress"
                row["grid_total"] = str(grid_done)
                row["grid_done"] = str(grid_done)
                row["verify_total"] = "50"
                row["verify_done"] = str(verify_done)
                row["error_message"] = "AEC assembly: 20 verify slots remain; see AEC_INCOMPLETE.yaml"
                partial_count += 1
                assembly_status = "partial_30_of_50"
            else:
                grid_done, verify_done = finalize_complete_job(item)
                verify_missing = 0
                populate_complete_manifest_row(row, item, grid_done, verify_done)
                completed_count += 1
                assembly_status = "complete"
            manifest_rows.append(row)
            source_rows.append(
                {
                    "config": str(item.config_relative),
                    "job_id": item.job.job_id,
                    "target_job_dir": str(item.job.job_dir),
                    "assembly_status": assembly_status,
                    "source_manifest": str(item.source.manifest_path),
                    "source_job_dir": str(item.source.job_dir),
                    "backfill_csv": "" if item.source.backfill_row is None else item.source.backfill_row["extra_csv_path"],
                    "grid_done": grid_done,
                    "verify_done": verify_done,
                    "verify_missing": verify_missing,
                }
            )
            if item_number % 10 == 0 or item_number == len(config_items):
                print(
                    f"Assembled {config_relative}: {item_number}/{len(config_items)} jobs "
                    f"(global complete={completed_count}, partial={partial_count})",
                    flush=True,
                )
        suite_core._write_manifest(
            actual_config_root / stage_utils.ROOT_MANIFEST_FILENAME,
            manifest_rows,
        )
        print(f"Finished config {config_number:02d}/{len(grouped)}: {config_relative}", flush=True)

    write_csv_rows(
        STAGING_ROOT / "assembly_sources.csv",
        [
            "config",
            "job_id",
            "target_job_dir",
            "assembly_status",
            "source_manifest",
            "source_job_dir",
            "backfill_csv",
            "grid_done",
            "verify_done",
            "verify_missing",
        ],
        source_rows,
    )
    stage_utils.write_yaml_file(
        STAGING_ROOT / "assembly_summary.yaml",
        {
            "config_count": len(grouped),
            "job_count": len(items),
            "complete_job_count": completed_count,
            "partial_job_count": partial_count,
            "partial_verify_present_per_job": 30,
            "partial_verify_missing_per_job": 20,
            "physical_root": str(FINAL_ROOT),
            "source_roots": [str(PAPER_ROOT), str(REBUTTAL_ROOT), str(BACKFILL_ROOT)],
        },
    )
    if completed_count != 786 or partial_count != 480:
        raise RuntimeError(f"Unexpected assembled counts: complete={completed_count}, partial={partial_count}")


def validate_staging(items: list[AssemblyItem]) -> None:
    symlinks: list[Path] = []
    for root, dirs, files in os.walk(STAGING_ROOT):
        root_path = Path(root)
        for name in [*dirs, *files]:
            path = root_path / name
            if path.is_symlink():
                symlinks.append(path)
    if symlinks:
        raise RuntimeError(f"Internal symlinks found in staging: {symlinks[:20]}")

    config_roots = [path.parent for path in STAGING_ROOT.rglob(stage_utils.INPUT_CONFIG_COPY_FILENAME)]
    if len(config_roots) != 57:
        raise RuntimeError(f"Expected 57 assembled config roots, found {len(config_roots)}")
    actual_job_dirs = [actual_path(item.job.job_dir) for item in items]
    if len(actual_job_dirs) != 1266 or len(set(actual_job_dirs)) != 1266:
        raise RuntimeError("Target job directory count is invalid")
    for item in items:
        target = actual_path(item.job.job_dir)
        if not stage_utils.job_spec_path(target).is_file():
            raise RuntimeError(f"job_spec.yaml is missing: {target}")
        if item.partial:
            if not (target / "AEC_INCOMPLETE.yaml").is_file():
                raise RuntimeError(f"Partial marker is missing: {target}")
            if stage_utils.job_outputs_complete(target):
                raise RuntimeError(f"Partial job is marked complete: {target}")
        elif not stage_utils.job_outputs_complete(target):
            raise RuntimeError(f"Complete job is missing final artifacts: {target}")

    _, source_rows = read_csv_rows(STAGING_ROOT / "assembly_sources.csv")
    statuses = Counter(row["assembly_status"] for row in source_rows)
    if statuses != Counter({"complete": 786, "partial_30_of_50": 480}):
        raise RuntimeError(f"Source manifest status counts are invalid: {statuses}")

    staging_text = str(STAGING_ROOT)
    generated_paths = [
        STAGING_ROOT / "assembly_sources.csv",
        STAGING_ROOT / "assembly_summary.yaml",
        *STAGING_ROOT.rglob(stage_utils.ROOT_MANIFEST_FILENAME),
        *STAGING_ROOT.rglob(stage_utils.BEST_CONFIG_FILENAME),
        *STAGING_ROOT.rglob(stage_utils.RECOMMENDED_COMMAND_FILENAME),
        *STAGING_ROOT.rglob("AEC_*.yaml"),
        *STAGING_ROOT.rglob("AEC_*.csv"),
    ]
    for path in generated_paths:
        if path.is_file() and staging_text in path.read_text(encoding="utf-8", errors="ignore"):
            raise RuntimeError(f"Generated metadata contains a staging path: {path}")
    print(
        "Staging validation passed: configs=57 jobs=1266 complete=786 partial=480 internal_symlinks=0",
        flush=True,
    )


def main() -> None:
    items = build_items()
    print("Source mapping validation passed for 1,266 jobs", flush=True)
    assemble(items)
    validate_staging(items)
    STAGING_ROOT.rename(FINAL_ROOT)
    print(f"Published physical result tree: {FINAL_ROOT}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"AEC assembly failed: {exc}", file=sys.stderr, flush=True)
        raise
