#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
import shlex
import statistics
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
HSEARCH_ROOT = REPO_ROOT / "hparams_search_scripts"
if str(HSEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(HSEARCH_ROOT))

try:
    from hparams_search_scripts.mechanism_stage_context import resolve_job_context
    from hparams_search_scripts.mechanism_stage_utils import (
        CandidateSpec,
        shell_join,
        verify_stage_dir,
    )
    from hparams_search_scripts.multi_pool_scheduler import (
        MultiPoolScheduler,
        SchedulerTask,
        TaskResult,
    )
except ModuleNotFoundError:
    from mechanism_stage_context import resolve_job_context  # type: ignore
    from mechanism_stage_utils import CandidateSpec, shell_join, verify_stage_dir  # type: ignore
    from multi_pool_scheduler import MultiPoolScheduler, SchedulerTask, TaskResult  # type: ignore

DEFAULT_OUTPUT_ROOT = Path(
    "/data/hzw/Rethinking_DP_GNN_runtime/paper_experiments/main_add_10seed_backfill"
)
PAPER_EXPERIMENTS_ROOT = Path("/data/hzw/Rethinking_DP_GNN_runtime/paper_experiments")
SOURCE_ROOTS = {
    "figure3": PAPER_EXPERIMENTS_ROOT / "figure3",
    "figure30": PAPER_EXPERIMENTS_ROOT / "figure30",
    "figure32": PAPER_EXPERIMENTS_ROOT / "figure32",
    "main_add": PAPER_EXPERIMENTS_ROOT / "main_add",
}
BACKBONES = ("sage", "gcn", "gat")
PIPELINES = (
    "figure3_pipeline1",
    "figure3_pipeline2",
    "figure3_pipeline3",
    "figure3_pipeline4",
)
TARGET_DATASETS = ("cora", "lastfm", "citeseer", "facebook", "Books-History")
TARGET_X_EPS = (
    "0.001",
    "0.01",
    "0.1",
    "1.0",
    "2.0",
    "3.0",
    "4.0",
    "6.0",
    "8.0",
    "10.0",
)
LOW_BUDGETS = set(TARGET_X_EPS[:6])
HIGH_BUDGETS = set(TARGET_X_EPS[6:])
COMPLETE_STATUSES = {"completed", "skipped_existing_result"}
EXPECTED_EXISTING_SEED_START = 12345
VERIFY_DIR_RE = re.compile(r"^rank=(\d+)__repeat=(\d+)__candidate=(\d+)__")
PROGRESS_BAR_WIDTH = 30


@dataclass(frozen=True)
class LogicalJobKey:
    backbone: str
    pipeline: str
    dataset: str
    x_eps: str


@dataclass
class SourceJob:
    key: LogicalJobKey
    source_root_name: str
    source_root_dir: Path
    manifest_path: Path
    row: dict[str, str]


@dataclass
class SeedResultRow:
    seed: int
    seed_origin: str
    val_acc: float
    test_acc: float
    result_csv_path: str


@dataclass
class BackfillRecord:
    key: LogicalJobKey
    source_root_name: str
    source_job_dir: Path
    recommended_command_path: Path
    best_config_path: Path
    best_rank: int
    best_candidate: CandidateSpec
    mechanism: str
    smoother: str
    use_nfr: bool
    existing_rows: list[SeedResultRow]
    actual_verify_rank: int | None
    extra_output_dir: Path
    existing_verify_status: str
    extra_backfill_status: str
    merge_status: str
    error_message: str = ""
    extra_csv_path: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill suite_main_add best configs to 10 total seeds."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for backfill artifacts.",
    )
    parser.add_argument(
        "--extra-seed-start",
        type=int,
        default=12350,
        help="Starting seed for the extra rerun block.",
    )
    parser.add_argument(
        "--extra-seed-count",
        type=int,
        default=5,
        help="Number of extra seeds to run per logical job.",
    )
    parser.add_argument(
        "--worker-ids",
        type=str,
        default="0,1,2,3,4,5",
        help="Comma-separated worker GPU ids for the scheduler.",
    )
    parser.add_argument(
        "--max-parallel-per-worker",
        type=int,
        default=10,
        help="Maximum concurrent tasks per worker/GPU.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only resolve sources and enumerate rerun work; do not launch tasks.",
    )
    return parser.parse_args()


def canonical_x_eps_text(raw_value: Any) -> str:
    try:
        value = Decimal(str(raw_value).strip())
    except (InvalidOperation, AttributeError) as exc:
        raise ValueError(f"Invalid x_eps value: {raw_value!r}") from exc
    if value == Decimal("0.001"):
        return "0.001"
    if value == Decimal("0.01"):
        return "0.01"
    if value == Decimal("0.1"):
        return "0.1"
    if value == Decimal("1"):
        return "1.0"
    if value == Decimal("2"):
        return "2.0"
    if value == Decimal("3"):
        return "3.0"
    if value == Decimal("4"):
        return "4.0"
    if value == Decimal("6"):
        return "6.0"
    if value == Decimal("8"):
        return "8.0"
    if value == Decimal("10"):
        return "10.0"
    raise ValueError(f"Unsupported x_eps value: {raw_value!r}")


def parse_worker_ids(raw: str) -> list[int]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError("--worker-ids must be non-empty")
    return [int(part) for part in parts]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def progress_bar(prefix: str, completed: int, total: int, *, extra: str = "") -> str:
    total = max(total, 1)
    completed = max(0, min(completed, total))
    ratio = completed / total
    filled = int(round(PROGRESS_BAR_WIDTH * ratio))
    bar = "#" * filled + "-" * (PROGRESS_BAR_WIDTH - filled)
    suffix = f" {extra}" if extra else ""
    return (
        f"{prefix} [{bar}] {completed}/{total} "
        f"({ratio * 100:5.1f}%)" + suffix
    )


def print_progress(
    prefix: str,
    completed: int,
    total: int,
    *,
    extra: str = "",
    final: bool = False,
) -> None:
    line = progress_bar(prefix, completed, total, extra=extra)
    end = "\n" if final else "\r"
    print(line, end=end, flush=True)


def read_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def manifest_path_for(source_root_name: str, backbone: str, pipeline: str) -> Path:
    return SOURCE_ROOTS[source_root_name] / backbone / f"{pipeline}.yaml" / "manifest.csv"


def resolve_actual_job_dir(source: SourceJob) -> Path | None:
    raw_job_dir = source.row.get("job_dir", "")
    if raw_job_dir:
        candidate = Path(raw_job_dir)
        if candidate.is_dir():
            return candidate

        pipeline_dirname = f"{source.key.pipeline}.yaml"
        parts = list(candidate.parts)
        if pipeline_dirname in parts:
            pipeline_index = parts.index(pipeline_dirname)
            suffix_parts = parts[pipeline_index + 1 :]
            rebuilt = source.manifest_path.parent
            for part in suffix_parts:
                rebuilt = rebuilt / part
            if rebuilt.is_dir():
                return rebuilt

    dataset = source.row.get("dataset", source.key.dataset)
    mechanism = source.row.get("mechanism", "")
    x_eps_raw = source.row.get("x_eps", "")
    smoother = source.row.get("smoother", "")
    backbone = source.row.get("backbone", source.key.backbone)
    use_nfr = source.row.get("use_nfr", "")
    fallback = (
        source.manifest_path.parent
        / f"dataset={dataset}"
        / "feature=raw"
        / f"mechanism={mechanism}"
        / f"x_eps={x_eps_raw}"
        / "m=best"
        / "norm=false"
        / f"smoother={smoother}"
        / f"backbone={backbone}"
        / f"use_nfr={use_nfr}"
    )
    if fallback.is_dir():
        return fallback
    return None


def expected_source_root_name(key: LogicalJobKey) -> str:
    if key.x_eps in HIGH_BUDGETS:
        return "main_add"
    if key.dataset == "facebook":
        return "figure32"
    if (
        key.dataset == "Books-History"
        and key.backbone == "gat"
        and key.pipeline == "figure3_pipeline4"
    ):
        return "figure30"
    return "figure3"


def scan_source_rows() -> dict[tuple[str, str, str, str, str], SourceJob]:
    scanned: dict[tuple[str, str, str, str, str], SourceJob] = {}
    for source_root_name, source_root_dir in SOURCE_ROOTS.items():
        for backbone in BACKBONES:
            for pipeline in PIPELINES:
                manifest_path = manifest_path_for(source_root_name, backbone, pipeline)
                if not manifest_path.is_file():
                    continue
                for row in read_csv_rows(manifest_path):
                    dataset = row.get("dataset", "")
                    if dataset not in TARGET_DATASETS:
                        continue
                    try:
                        x_eps = canonical_x_eps_text(row.get("x_eps", ""))
                    except ValueError:
                        continue
                    if x_eps not in TARGET_X_EPS:
                        continue
                    key = LogicalJobKey(
                        backbone=backbone,
                        pipeline=pipeline,
                        dataset=dataset,
                        x_eps=x_eps,
                    )
                    scanned[(source_root_name, backbone, pipeline, dataset, x_eps)] = SourceJob(
                        key=key,
                        source_root_name=source_root_name,
                        source_root_dir=source_root_dir,
                        manifest_path=manifest_path,
                        row=row,
                    )
    return scanned


def resolve_sources(
    progress_callback: callable | None = None,
) -> tuple[list[SourceJob], list[dict[str, Any]]]:
    scanned = scan_source_rows()
    resolved_jobs: list[SourceJob] = []
    resolution_rows: list[dict[str, Any]] = []
    total_keys = len(BACKBONES) * len(PIPELINES) * len(TARGET_DATASETS) * len(TARGET_X_EPS)
    processed = 0
    for backbone in BACKBONES:
        for pipeline in PIPELINES:
            for dataset in TARGET_DATASETS:
                for x_eps in TARGET_X_EPS:
                    key = LogicalJobKey(
                        backbone=backbone,
                        pipeline=pipeline,
                        dataset=dataset,
                        x_eps=x_eps,
                    )
                    source_root_name = expected_source_root_name(key)
                    scanned_key = (source_root_name, backbone, pipeline, dataset, x_eps)
                    source = scanned.get(scanned_key)
                    if source is None:
                        resolution_rows.append(
                            {
                                "backbone": backbone,
                                "pipeline": pipeline,
                                "dataset": dataset,
                                "x_eps": x_eps,
                                "expected_source_root": source_root_name,
                                "manifest_path": str(
                                    manifest_path_for(source_root_name, backbone, pipeline)
                                ),
                                "job_dir": "",
                                "search_status": "",
                                "resolution_status": "missing",
                                "reason": "source row not found",
                            }
                        )
                        continue

                    search_status = source.row.get("search_status", "")
                    actual_job_dir = resolve_actual_job_dir(source)
                    job_dir = "" if actual_job_dir is None else str(actual_job_dir)
                    best_config_path = "" if actual_job_dir is None else str(actual_job_dir / "best_config.yaml")
                    recommended_command_path = "" if actual_job_dir is None else str(actual_job_dir / "recommended_command.txt")
                    reason = ""
                    resolution_status = "resolved"
                    if search_status not in COMPLETE_STATUSES:
                        resolution_status = "incomplete"
                        reason = f"search_status={search_status!r}"
                    elif not job_dir:
                        resolution_status = "incomplete"
                        reason = f"job_dir not found: {source.row.get('job_dir', '')}"
                    elif not Path(best_config_path).is_file():
                        resolution_status = "incomplete"
                        reason = f"best_config missing: {best_config_path}"
                    elif not Path(recommended_command_path).is_file():
                        resolution_status = "incomplete"
                        reason = f"recommended_command missing: {recommended_command_path}"

                    source.row["job_dir"] = job_dir
                    source.row["best_config_path"] = best_config_path
                    source.row["recommended_command_path"] = recommended_command_path

                    resolution_rows.append(
                        {
                            "backbone": backbone,
                            "pipeline": pipeline,
                            "dataset": dataset,
                            "x_eps": x_eps,
                            "expected_source_root": source_root_name,
                            "manifest_path": str(source.manifest_path),
                            "job_dir": job_dir,
                            "search_status": search_status,
                            "resolution_status": resolution_status,
                            "reason": reason,
                        }
                    )
                    if resolution_status == "resolved":
                        resolved_jobs.append(source)
                    processed += 1
                    if progress_callback is not None:
                        progress_callback(
                            processed,
                            total_keys,
                            len(resolved_jobs),
                            len(resolution_rows) - len(resolved_jobs),
                        )

    return resolved_jobs, resolution_rows


def read_single_result_row(csv_path: Path) -> dict[str, str]:
    rows = read_csv_rows(csv_path)
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly one row in {csv_path}, found {len(rows)}")
    return rows[0]


def read_result_block(csv_path: Path) -> list[dict[str, str]]:
    rows = read_csv_rows(csv_path)
    if len(rows) == 0:
        raise RuntimeError(f"Expected at least one row in {csv_path}")
    return rows


def replace_arg(parts: list[str], short_flag: str, long_flag: str, value: str) -> list[str]:
    replaced = list(parts)
    for flag in (short_flag, long_flag):
        if flag in replaced:
            index = replaced.index(flag)
            if index == len(replaced) - 1:
                raise RuntimeError(f"Missing value for {flag} in command: {shell_join(parts)}")
            replaced[index + 1] = value
            return replaced
    replaced.extend([short_flag, value])
    return replaced


def parse_recommended_command(path: Path, *, new_seed: int, new_repeats: int, new_output_dir: Path) -> list[str]:
    command_text = path.read_text(encoding="utf-8").strip()
    if not command_text:
        raise RuntimeError(f"Empty recommended command: {path}")
    parts = shlex.split(command_text)
    parts = replace_arg(parts, "-s", "--seed", str(new_seed))
    parts = replace_arg(parts, "-r", "--repeats", str(new_repeats))
    parts = replace_arg(parts, "-o", "--output_dir", str(new_output_dir))
    return parts


def float_from_row(row: dict[str, str], key: str, *, csv_path: Path) -> float:
    raw_value = row.get(key, "")
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"Invalid float {key}={raw_value!r} in {csv_path}") from exc
    if not math.isfinite(value):
        raise RuntimeError(f"Non-finite float {key}={raw_value!r} in {csv_path}")
    return value


def int_from_row(row: dict[str, str], key: str, *, csv_path: Path) -> int:
    raw_value = row.get(key, "")
    try:
        return int(float(raw_value))
    except ValueError as exc:
        raise RuntimeError(f"Invalid integer {key}={raw_value!r} in {csv_path}") from exc


def effective_seed_from_row(row: dict[str, str], *, csv_path: Path) -> int:
    base_seed = int_from_row(row, "seed", csv_path=csv_path)
    version_raw = row.get("version", "").strip()
    if not version_raw:
        return base_seed
    version = int_from_row(row, "version", csv_path=csv_path)
    return base_seed + version


def validate_result_row_against_candidate(
    row: dict[str, str],
    candidate: CandidateSpec,
    *,
    expected_seed: int,
    csv_path: Path,
) -> None:
    row_candidate = CandidateSpec(
        candidate_id=int_from_row(row, "candidate_id", csv_path=csv_path)
        if "candidate_id" in row and row.get("candidate_id", "").strip()
        else candidate.candidate_id,
        x_steps=int_from_row(row, "x_steps", csv_path=csv_path),
        learning_rate=f"{float_from_row(row, 'learning_rate', csv_path=csv_path):.12g}",
        weight_decay=f"{float_from_row(row, 'weight_decay', csv_path=csv_path):.12g}",
        dropout=f"{float_from_row(row, 'dropout', csv_path=csv_path):.12g}",
        tao2="none"
        if not row.get("tao2", "").strip() or row.get("tao2", "").strip().lower() == "none"
        else f"{float_from_row(row, 'tao2', csv_path=csv_path):.12g}",
    )
    if row_candidate.key != candidate.key:
        raise RuntimeError(
            f"Candidate mismatch in {csv_path}: expected {candidate.key}, got {row_candidate.key}"
        )
    effective_seed = effective_seed_from_row(row, csv_path=csv_path)
    if effective_seed != expected_seed:
        raise RuntimeError(
            f"Seed mismatch in {csv_path}: expected {expected_seed}, got {effective_seed}"
        )


def collect_existing_verify_rows(source_job_dir: Path, candidate: CandidateSpec) -> tuple[list[SeedResultRow], int | None]:
    ctx = resolve_job_context(source_job_dir)
    verify_dir = verify_stage_dir(source_job_dir)
    if not verify_dir.is_dir():
        raise RuntimeError(f"verify_top5 directory not found: {verify_dir}")

    matches: dict[int, tuple[int, Path]] = {}
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
        if repeat_id in matches:
            raise RuntimeError(
                f"Duplicate verify directory for candidate {candidate.candidate_id} repeat {repeat_id}: "
                f"{matches[repeat_id][1]} and {child}"
            )
        matches[repeat_id] = (rank, child)

    expected_repeat_ids = range(1, int(ctx.defaults["stage"]["verify"]["repeats"]) + 1)
    rows: list[SeedResultRow] = []
    observed_ranks: set[int] = set()

    for repeat_id in expected_repeat_ids:
        if repeat_id not in matches:
            raise RuntimeError(
                f"Missing verify directory for candidate {candidate.candidate_id} repeat {repeat_id} in {verify_dir}"
            )
        rank, repeat_dir = matches[repeat_id]
        observed_ranks.add(rank)
        csv_files = sorted(repeat_dir.glob("*.csv"))
        if len(csv_files) != 1:
            raise RuntimeError(
                f"Expected exactly one result CSV in {repeat_dir}, found {len(csv_files)}"
            )
        csv_path = csv_files[0]
        row = read_single_result_row(csv_path)
        expected_seed = ctx.base_seed + repeat_id - 1
        validate_result_row_against_candidate(
            row,
            candidate,
            expected_seed=expected_seed,
            csv_path=csv_path,
        )
        rows.append(
            SeedResultRow(
                seed=expected_seed,
                seed_origin="existing_verify",
                val_acc=float_from_row(row, "val/acc", csv_path=csv_path),
                test_acc=float_from_row(row, "test/acc", csv_path=csv_path),
                result_csv_path=str(csv_path),
            )
        )

    actual_rank = next(iter(observed_ranks)) if len(observed_ranks) == 1 else None
    if len(rows) != 5:
        raise RuntimeError(
            f"Expected exactly 5 existing verify rows for {source_job_dir}, found {len(rows)}"
        )
    return rows, actual_rank


def collect_complete_extra_csv(
    output_dir: Path,
    candidate: CandidateSpec,
    *,
    extra_seed_start: int,
    extra_seed_count: int,
) -> Path | None:
    if not output_dir.is_dir():
        return None

    matching_csvs: list[Path] = []
    expected_seeds = {
        extra_seed_start + offset
        for offset in range(extra_seed_count)
    }

    for csv_path in sorted(output_dir.glob("*.csv")):
        rows = read_result_block(csv_path)
        if len(rows) != extra_seed_count:
            continue
        seen_seeds: set[int] = set()
        seen_versions: set[int] = set()
        valid = True
        for row in rows:
            try:
                seed = effective_seed_from_row(row, csv_path=csv_path)
            except ValueError:
                valid = False
                break
            if seed in seen_seeds:
                valid = False
                break
            seen_seeds.add(seed)
            version = int_from_row(row, "version", csv_path=csv_path)
            if version in seen_versions:
                valid = False
                break
            seen_versions.add(version)
            try:
                validate_result_row_against_candidate(
                    row,
                    candidate,
                    expected_seed=seed,
                    csv_path=csv_path,
                )
            except RuntimeError:
                valid = False
                break
        if valid and seen_seeds == expected_seeds and seen_versions == set(range(extra_seed_count)):
            matching_csvs.append(csv_path)

    if len(matching_csvs) > 1:
        raise RuntimeError(
            f"Multiple complete extra result CSVs found in {output_dir}: {matching_csvs}"
        )
    return matching_csvs[0] if matching_csvs else None


def collect_extra_rows_from_csv(
    csv_path: Path,
    candidate: CandidateSpec,
    *,
    extra_seed_start: int,
    extra_seed_count: int,
) -> list[SeedResultRow]:
    rows = read_result_block(csv_path)
    if len(rows) != extra_seed_count:
        raise RuntimeError(
            f"Expected {extra_seed_count} rows in {csv_path}, found {len(rows)}"
        )

    seed_rows: list[SeedResultRow] = []
    expected_seeds = [extra_seed_start + offset for offset in range(extra_seed_count)]
    seen_seeds: set[int] = set()
    for row in rows:
        seed = effective_seed_from_row(row, csv_path=csv_path)
        validate_result_row_against_candidate(
            row,
            candidate,
            expected_seed=seed,
            csv_path=csv_path,
        )
        if seed in seen_seeds:
            raise RuntimeError(f"Duplicate seed {seed} in {csv_path}")
        seen_seeds.add(seed)
        seed_rows.append(
            SeedResultRow(
                seed=seed,
                seed_origin="extra_backfill",
                val_acc=float_from_row(row, "val/acc", csv_path=csv_path),
                test_acc=float_from_row(row, "test/acc", csv_path=csv_path),
                result_csv_path=str(csv_path),
            )
        )
    if seen_seeds != set(expected_seeds):
        raise RuntimeError(
            f"Unexpected seed set in {csv_path}: expected {expected_seeds}, got {sorted(seen_seeds)}"
        )
    seed_rows.sort(key=lambda item: item.seed)
    return seed_rows


def build_backfill_record(
    source: SourceJob,
    *,
    output_root: Path,
    extra_seed_start: int,
    extra_seed_count: int,
) -> BackfillRecord:
    best_config_path = Path(source.row["best_config_path"])
    recommended_command_path = Path(source.row["recommended_command_path"])
    best_config = read_yaml(best_config_path)
    if not isinstance(best_config, dict):
        raise RuntimeError(f"best_config must be a mapping: {best_config_path}")
    best_candidate = CandidateSpec.from_dict(best_config["best_candidate"])
    existing_rows, actual_rank = collect_existing_verify_rows(Path(source.row["job_dir"]), best_candidate)
    extra_output_dir = (
        output_root
        / "runs"
        / source.key.backbone
        / source.key.pipeline
        / source.key.dataset
        / f"x_eps={source.key.x_eps}"
    )
    existing_verify_status = "complete"
    extra_backfill_status = "pending"
    merge_status = "pending"
    error_message = ""
    extra_csv_path = collect_complete_extra_csv(
        extra_output_dir,
        best_candidate,
        extra_seed_start=extra_seed_start,
        extra_seed_count=extra_seed_count,
    )
    if extra_csv_path is not None:
        extra_backfill_status = "existing_complete"

    return BackfillRecord(
        key=source.key,
        source_root_name=source.source_root_name,
        source_job_dir=Path(source.row["job_dir"]),
        recommended_command_path=recommended_command_path,
        best_config_path=best_config_path,
        best_rank=int(best_config.get("best_rank", 1)),
        best_candidate=best_candidate,
        mechanism=str(best_config["fixed_params"]["mechanism"]),
        smoother=str(best_config["fixed_params"]["smoother"]),
        use_nfr=bool(best_config["fixed_params"]["use_nfr"]),
        existing_rows=existing_rows,
        actual_verify_rank=actual_rank,
        extra_output_dir=extra_output_dir,
        existing_verify_status=existing_verify_status,
        extra_backfill_status=extra_backfill_status,
        merge_status=merge_status,
        error_message=error_message,
        extra_csv_path=extra_csv_path,
    )


def build_manifest_rows(records: list[BackfillRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "backbone": record.key.backbone,
                "pipeline": record.key.pipeline,
                "dataset": record.key.dataset,
                "x_eps": record.key.x_eps,
                "source_root": record.source_root_name,
                "source_job_dir": str(record.source_job_dir),
                "best_config_path": str(record.best_config_path),
                "recommended_command_path": str(record.recommended_command_path),
                "best_rank": record.best_rank,
                "actual_verify_rank": "" if record.actual_verify_rank is None else record.actual_verify_rank,
                "candidate_id": record.best_candidate.candidate_id,
                "x_steps": record.best_candidate.x_steps,
                "learning_rate": record.best_candidate.learning_rate,
                "weight_decay": record.best_candidate.weight_decay,
                "dropout": record.best_candidate.dropout,
                "tao2": record.best_candidate.tao2,
                "mechanism": record.mechanism,
                "smoother": record.smoother,
                "use_nfr": "true" if record.use_nfr else "false",
                "existing_verify_status": record.existing_verify_status,
                "existing_verify_count": len(record.existing_rows),
                "extra_output_dir": str(record.extra_output_dir),
                "extra_backfill_status": record.extra_backfill_status,
                "extra_csv_path": "" if record.extra_csv_path is None else str(record.extra_csv_path),
                "merge_status": record.merge_status,
                "error_message": record.error_message,
            }
        )
    return rows


def write_backfill_manifest(output_root: Path, records: list[BackfillRecord]) -> None:
    manifest_path = output_root / "backfill_manifest.csv"
    rows = build_manifest_rows(records)
    fieldnames = [
        "backbone",
        "pipeline",
        "dataset",
        "x_eps",
        "source_root",
        "source_job_dir",
        "best_config_path",
        "recommended_command_path",
        "best_rank",
        "actual_verify_rank",
        "candidate_id",
        "x_steps",
        "learning_rate",
        "weight_decay",
        "dropout",
        "tao2",
        "mechanism",
        "smoother",
        "use_nfr",
        "existing_verify_status",
        "existing_verify_count",
        "extra_output_dir",
        "extra_backfill_status",
        "extra_csv_path",
        "merge_status",
        "error_message",
    ]
    write_csv(manifest_path, rows, fieldnames)


def write_prepare_cache(output_root: Path, records: list[BackfillRecord]) -> None:
    prepared_rows: list[dict[str, Any]] = []
    existing_verify_rows: list[dict[str, Any]] = []

    for record in records:
        prepared_rows.append(
            {
                "backbone": record.key.backbone,
                "pipeline": record.key.pipeline,
                "dataset": record.key.dataset,
                "x_eps": record.key.x_eps,
                "source_root": record.source_root_name,
                "source_job_dir": str(record.source_job_dir),
                "best_config_path": str(record.best_config_path),
                "recommended_command_path": str(record.recommended_command_path),
                "best_rank": record.best_rank,
                "actual_verify_rank": "" if record.actual_verify_rank is None else record.actual_verify_rank,
                "candidate_id": record.best_candidate.candidate_id,
                "x_steps": record.best_candidate.x_steps,
                "learning_rate": record.best_candidate.learning_rate,
                "weight_decay": record.best_candidate.weight_decay,
                "dropout": record.best_candidate.dropout,
                "tao2": record.best_candidate.tao2,
                "mechanism": record.mechanism,
                "smoother": record.smoother,
                "use_nfr": "true" if record.use_nfr else "false",
            }
        )
        for seed_row in record.existing_rows:
            existing_verify_rows.append(
                {
                    "backbone": record.key.backbone,
                    "pipeline": record.key.pipeline,
                    "dataset": record.key.dataset,
                    "x_eps": record.key.x_eps,
                    "seed": seed_row.seed,
                    "seed_origin": seed_row.seed_origin,
                    "val_acc": seed_row.val_acc,
                    "test_acc": seed_row.test_acc,
                    "result_csv_path": seed_row.result_csv_path,
                }
            )

    write_csv(
        output_root / "prepared_records.csv",
        prepared_rows,
        [
            "backbone",
            "pipeline",
            "dataset",
            "x_eps",
            "source_root",
            "source_job_dir",
            "best_config_path",
            "recommended_command_path",
            "best_rank",
            "actual_verify_rank",
            "candidate_id",
            "x_steps",
            "learning_rate",
            "weight_decay",
            "dropout",
            "tao2",
            "mechanism",
            "smoother",
            "use_nfr",
        ],
    )
    write_csv(
        output_root / "existing_verify_rows.csv",
        existing_verify_rows,
        [
            "backbone",
            "pipeline",
            "dataset",
            "x_eps",
            "seed",
            "seed_origin",
            "val_acc",
            "test_acc",
            "result_csv_path",
        ],
    )


def load_prepare_cache(
    output_root: Path,
    *,
    extra_seed_start: int,
    extra_seed_count: int,
) -> list[BackfillRecord] | None:
    prepared_path = output_root / "prepared_records.csv"
    existing_rows_path = output_root / "existing_verify_rows.csv"
    if not prepared_path.is_file() or not existing_rows_path.is_file():
        return None

    prepared_rows = read_csv_rows(prepared_path)
    existing_rows = read_csv_rows(existing_rows_path)
    if len(prepared_rows) != 600 or len(existing_rows) != 3000:
        raise RuntimeError(
            f"Invalid prepare cache size: prepared={len(prepared_rows)} existing_rows={len(existing_rows)}"
        )

    existing_by_key: dict[LogicalJobKey, list[SeedResultRow]] = {}
    for row in existing_rows:
        key = LogicalJobKey(
            backbone=row["backbone"],
            pipeline=row["pipeline"],
            dataset=row["dataset"],
            x_eps=row["x_eps"],
        )
        existing_by_key.setdefault(key, []).append(
            SeedResultRow(
                seed=int(row["seed"]),
                seed_origin=row["seed_origin"],
                val_acc=float(row["val_acc"]),
                test_acc=float(row["test_acc"]),
                result_csv_path=row["result_csv_path"],
            )
        )

    records: list[BackfillRecord] = []
    for row in prepared_rows:
        key = LogicalJobKey(
            backbone=row["backbone"],
            pipeline=row["pipeline"],
            dataset=row["dataset"],
            x_eps=row["x_eps"],
        )
        candidate = CandidateSpec(
            candidate_id=int(row["candidate_id"]),
            x_steps=int(row["x_steps"]),
            learning_rate=row["learning_rate"],
            weight_decay=row["weight_decay"],
            dropout=row["dropout"],
            tao2=row["tao2"],
        )
        existing = sorted(existing_by_key.get(key, []), key=lambda item: item.seed)
        if len(existing) != 5:
            raise RuntimeError(f"Prepare cache missing 5 existing rows for {key}")

        extra_output_dir = (
            output_root
            / "runs"
            / key.backbone
            / key.pipeline
            / key.dataset
            / f"x_eps={key.x_eps}"
        )
        extra_csv_path = collect_complete_extra_csv(
            extra_output_dir,
            candidate,
            extra_seed_start=extra_seed_start,
            extra_seed_count=extra_seed_count,
        )

        records.append(
            BackfillRecord(
                key=key,
                source_root_name=row["source_root"],
                source_job_dir=Path(row["source_job_dir"]),
                recommended_command_path=Path(row["recommended_command_path"]),
                best_config_path=Path(row["best_config_path"]),
                best_rank=int(row["best_rank"]),
                best_candidate=candidate,
                mechanism=row["mechanism"],
                smoother=row["smoother"],
                use_nfr=row["use_nfr"].strip().lower() == "true",
                existing_rows=existing,
                actual_verify_rank=None
                if not row.get("actual_verify_rank", "").strip()
                else int(row["actual_verify_rank"]),
                extra_output_dir=extra_output_dir,
                existing_verify_status="complete",
                extra_backfill_status="existing_complete" if extra_csv_path is not None else "pending",
                merge_status="pending",
                error_message="",
                extra_csv_path=extra_csv_path,
            )
        )

    return sorted(
        records,
        key=lambda item: (
            item.key.backbone,
            item.key.pipeline,
            item.key.dataset,
            item.key.x_eps,
        ),
    )


def rerun_task_for_record(
    record: BackfillRecord,
    *,
    output_root: Path,
    extra_seed_start: int,
    extra_seed_count: int,
) -> SchedulerTask:
    command = parse_recommended_command(
        record.recommended_command_path,
        new_seed=extra_seed_start,
        new_repeats=extra_seed_count,
        new_output_dir=record.extra_output_dir,
    )
    label = (
        f"{record.key.backbone}/{record.key.pipeline}/{record.key.dataset}/x_eps={record.key.x_eps}"
    )
    log_path = (
        output_root
        / "logs"
        / f"{record.key.backbone}__{record.key.pipeline}__{record.key.dataset}__x_eps={record.key.x_eps}.log"
    )
    return SchedulerTask(
        task_id=label,
        combo_key=label,
        pool="main_add_backfill",
        command=command,
        env={"PYTHONUNBUFFERED": "1"},
        cwd=REPO_ROOT,
        log_path=log_path,
        retry_count=0,
        label=label,
    )


def build_long_rows(
    record: BackfillRecord,
    seed_rows: list[SeedResultRow],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed_row in seed_rows:
        rows.append(
            {
                "backbone": record.key.backbone,
                "pipeline": record.key.pipeline,
                "dataset": record.key.dataset,
                "x_eps": record.key.x_eps,
                "mechanism": record.mechanism,
                "smoother": record.smoother,
                "use_nfr": "true" if record.use_nfr else "false",
                "best_rank": record.best_rank,
                "candidate_id": record.best_candidate.candidate_id,
                "x_steps": record.best_candidate.x_steps,
                "learning_rate": record.best_candidate.learning_rate,
                "weight_decay": record.best_candidate.weight_decay,
                "dropout": record.best_candidate.dropout,
                "tao2": record.best_candidate.tao2,
                "seed": seed_row.seed,
                "seed_origin": seed_row.seed_origin,
                "val_acc": seed_row.val_acc,
                "test_acc": seed_row.test_acc,
                "source_root": record.source_root_name,
                "source_job_dir": str(record.source_job_dir),
                "result_csv_path": seed_row.result_csv_path,
            }
        )
    return rows


def write_final_csvs(
    output_root: Path,
    records: list[BackfillRecord],
    *,
    extra_seed_start: int,
    extra_seed_count: int,
) -> None:
    long_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    expected_total_seeds = 5 + extra_seed_count

    for record in records:
        if record.extra_csv_path is None:
            raise RuntimeError(
                f"Missing extra backfill CSV for {record.key.backbone}/{record.key.pipeline}/"
                f"{record.key.dataset}/x_eps={record.key.x_eps}"
            )
        extra_rows = collect_extra_rows_from_csv(
            record.extra_csv_path,
            record.best_candidate,
            extra_seed_start=extra_seed_start,
            extra_seed_count=extra_seed_count,
        )
        combined = sorted(record.existing_rows + extra_rows, key=lambda item: item.seed)
        if len(combined) != expected_total_seeds:
            raise RuntimeError(
                f"Expected {expected_total_seeds} total seed rows for {record.key}, found {len(combined)}"
            )
        seeds = [item.seed for item in combined]
        if seeds != list(range(EXPECTED_EXISTING_SEED_START, EXPECTED_EXISTING_SEED_START + 5)) + list(
            range(extra_seed_start, extra_seed_start + extra_seed_count)
        ):
            raise RuntimeError(
                f"Unexpected seed ordering for {record.key}: {seeds}"
            )

        long_rows.extend(build_long_rows(record, combined))

        val_accs = [item.val_acc for item in combined]
        test_accs = [item.test_acc for item in combined]
        summary_rows.append(
            {
                "backbone": record.key.backbone,
                "pipeline": record.key.pipeline,
                "dataset": record.key.dataset,
                "x_eps": record.key.x_eps,
                "mechanism": record.mechanism,
                "smoother": record.smoother,
                "use_nfr": "true" if record.use_nfr else "false",
                "best_rank": record.best_rank,
                "candidate_id": record.best_candidate.candidate_id,
                "x_steps": record.best_candidate.x_steps,
                "learning_rate": record.best_candidate.learning_rate,
                "weight_decay": record.best_candidate.weight_decay,
                "dropout": record.best_candidate.dropout,
                "tao2": record.best_candidate.tao2,
                "n": expected_total_seeds,
                "val_acc_mean": statistics.mean(val_accs),
                "val_acc_std": statistics.stdev(val_accs),
                "val_acc_min": min(val_accs),
                "val_acc_max": max(val_accs),
                "test_acc_mean": statistics.mean(test_accs),
                "test_acc_std": statistics.stdev(test_accs),
                "test_acc_min": min(test_accs),
                "test_acc_max": max(test_accs),
                "source_root": record.source_root_name,
                "source_job_dir": str(record.source_job_dir),
            }
        )
        record.merge_status = "complete"

    expected_long_rows = 600 * expected_total_seeds
    if len(long_rows) != expected_long_rows:
        raise RuntimeError(
            f"Expected {expected_long_rows} long rows, found {len(long_rows)}"
        )
    if len(summary_rows) != 600:
        raise RuntimeError(f"Expected 600 summary rows, found {len(summary_rows)}")

    long_fieldnames = [
        "backbone",
        "pipeline",
        "dataset",
        "x_eps",
        "mechanism",
        "smoother",
        "use_nfr",
        "best_rank",
        "candidate_id",
        "x_steps",
        "learning_rate",
        "weight_decay",
        "dropout",
        "tao2",
        "seed",
        "seed_origin",
        "val_acc",
        "test_acc",
        "source_root",
        "source_job_dir",
        "result_csv_path",
    ]
    summary_fieldnames = [
        "backbone",
        "pipeline",
        "dataset",
        "x_eps",
        "mechanism",
        "smoother",
        "use_nfr",
        "best_rank",
        "candidate_id",
        "x_steps",
        "learning_rate",
        "weight_decay",
        "dropout",
        "tao2",
        "n",
        "val_acc_mean",
        "val_acc_std",
        "val_acc_min",
        "val_acc_max",
        "test_acc_mean",
        "test_acc_std",
        "test_acc_min",
        "test_acc_max",
        "source_root",
        "source_job_dir",
    ]

    write_csv(output_root / "test_acc_long.csv", long_rows, long_fieldnames)
    write_csv(output_root / "test_acc_summary.csv", summary_rows, summary_fieldnames)


def main() -> None:
    args = parse_args()
    worker_ids = parse_worker_ids(args.worker_ids)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "logs").mkdir(parents=True, exist_ok=True)
    (args.output_root / "runs").mkdir(parents=True, exist_ok=True)

    def on_resolve_progress(processed: int, total: int, resolved: int, unresolved: int) -> None:
        print_progress(
            "Resolve",
            processed,
            total,
            extra=f"resolved={resolved} unresolved={unresolved}",
        )

    resolved_jobs, resolution_rows = resolve_sources(progress_callback=on_resolve_progress)
    print_progress(
        "Resolve",
        len(resolution_rows),
        len(BACKBONES) * len(PIPELINES) * len(TARGET_DATASETS) * len(TARGET_X_EPS),
        extra=(
            f"resolved={sum(1 for row in resolution_rows if row['resolution_status'] == 'resolved')} "
            f"unresolved={sum(1 for row in resolution_rows if row['resolution_status'] != 'resolved')}"
        ),
        final=True,
    )
    write_csv(
        args.output_root / "resolved_sources.csv",
        resolution_rows,
        [
            "backbone",
            "pipeline",
            "dataset",
            "x_eps",
            "expected_source_root",
            "manifest_path",
            "job_dir",
            "search_status",
            "resolution_status",
            "reason",
        ],
    )

    unresolved = [row for row in resolution_rows if row["resolution_status"] != "resolved"]
    if unresolved:
        sample = "\n".join(
            f"{row['backbone']}/{row['pipeline']}/{row['dataset']}/x_eps={row['x_eps']}: {row['reason']}"
            for row in unresolved[:20]
        )
        raise SystemExit(
            "Source results are incomplete; see resolved_sources.csv.\n"
            f"First unresolved entries:\n{sample}"
        )

    if len(resolved_jobs) != 600:
        raise SystemExit(f"Expected 600 resolved jobs, found {len(resolved_jobs)}")

    prepare_cache_error = ""
    try:
        records = load_prepare_cache(
            args.output_root,
            extra_seed_start=args.extra_seed_start,
            extra_seed_count=args.extra_seed_count,
        )
    except Exception as exc:  # noqa: BLE001
        records = None
        prepare_cache_error = str(exc)
    if records is not None:
        print_progress(
            "Prepare",
            len(records),
            len(records),
            extra=(
                f"cache_hit existing_complete={sum(1 for record in records if record.extra_backfill_status == 'existing_complete')} "
                f"pending={sum(1 for record in records if record.extra_backfill_status == 'pending')}"
            ),
            final=True,
        )
    else:
        if prepare_cache_error:
            print(f"Prepare cache invalid, rebuilding: {prepare_cache_error}", flush=True)
        sorted_sources = sorted(
            resolved_jobs,
            key=lambda item: (
                item.key.backbone,
                item.key.pipeline,
                item.key.dataset,
                item.key.x_eps,
            ),
        )
        records = []
        total_records = len(sorted_sources)
        for index, source in enumerate(sorted_sources, start=1):
            records.append(
                build_backfill_record(
                    source,
                    output_root=args.output_root,
                    extra_seed_start=args.extra_seed_start,
                    extra_seed_count=args.extra_seed_count,
                )
            )
            if index == 1 or index % 10 == 0 or index == total_records:
                write_backfill_manifest(args.output_root, records)
            existing_complete = sum(
                1 for record in records if record.extra_backfill_status == "existing_complete"
            )
            print_progress(
                "Prepare",
                index,
                total_records,
                extra=f"existing_complete={existing_complete} pending={index - existing_complete}",
            )
        print_progress(
            "Prepare",
            total_records,
            total_records,
            extra=(
                f"existing_complete={sum(1 for record in records if record.extra_backfill_status == 'existing_complete')} "
                f"pending={sum(1 for record in records if record.extra_backfill_status == 'pending')}"
            ),
            final=True,
        )
        write_prepare_cache(args.output_root, records)
    write_backfill_manifest(args.output_root, records)

    pending_records = [
        record for record in records if record.extra_backfill_status == "pending"
    ]

    if args.dry_run:
        print(
            f"DRY RUN OK: resolved={len(records)} pending_extra={len(pending_records)} "
            f"already_complete={len(records) - len(pending_records)}"
        )
        return

    record_by_task_id = {
        f"{record.key.backbone}/{record.key.pipeline}/{record.key.dataset}/x_eps={record.key.x_eps}": record
        for record in pending_records
    }

    if pending_records:
        existing_complete = len(records) - len(pending_records)
        progress_state = {"completed": 0, "failed": 0}

        def print_scheduler_progress(*, final: bool = False) -> None:
            done = existing_complete + progress_state["completed"] + progress_state["failed"]
            print_progress(
                "Backfill",
                done,
                len(records),
                extra=(
                    f"existing={existing_complete} completed={progress_state['completed']} "
                    f"failed={progress_state['failed']} pending={len(records) - done}"
                ),
                final=final,
            )

        scheduler = MultiPoolScheduler(
            worker_ids=worker_ids,
            max_parallel_per_worker=int(args.max_parallel_per_worker),
            poll_interval_sec=0.1,
            launch_interval_sec=0.1,
            device="gpu",
        )
        for record in pending_records:
            scheduler.enqueue(
                rerun_task_for_record(
                    record,
                    output_root=args.output_root,
                    extra_seed_start=args.extra_seed_start,
                    extra_seed_count=args.extra_seed_count,
                )
            )
        print_scheduler_progress()

        def on_result(result: TaskResult) -> list[SchedulerTask] | None:
            record = record_by_task_id[result.task.task_id]
            if not result.success:
                record.extra_backfill_status = "failed"
                record.error_message = result.error_message or f"exit code {result.return_code}"
                progress_state["failed"] += 1
                write_backfill_manifest(args.output_root, records)
                print_scheduler_progress()
                return None

            try:
                complete_csv = collect_complete_extra_csv(
                    record.extra_output_dir,
                    record.best_candidate,
                    extra_seed_start=args.extra_seed_start,
                    extra_seed_count=args.extra_seed_count,
                )
            except Exception as exc:  # noqa: BLE001
                record.extra_backfill_status = "failed"
                record.error_message = str(exc)
                progress_state["failed"] += 1
                write_backfill_manifest(args.output_root, records)
                print_scheduler_progress()
                return None

            if complete_csv is None:
                record.extra_backfill_status = "failed"
                record.error_message = (
                    "scheduler task succeeded but no complete extra result CSV was found"
                )
                progress_state["failed"] += 1
            else:
                record.extra_backfill_status = "completed"
                record.extra_csv_path = complete_csv
                record.error_message = ""
                progress_state["completed"] += 1
            write_backfill_manifest(args.output_root, records)
            print_scheduler_progress()
            return None

        scheduler.run(on_result=on_result)
        print_scheduler_progress(final=True)

    failed_records = [
        record for record in records if record.extra_backfill_status not in {"completed", "existing_complete"}
    ]
    if failed_records:
        write_backfill_manifest(args.output_root, records)
        sample = "\n".join(
            f"{record.key.backbone}/{record.key.pipeline}/{record.key.dataset}/x_eps={record.key.x_eps}: "
            f"{record.error_message or record.extra_backfill_status}"
            for record in failed_records[:20]
        )
        raise SystemExit(
            "Some backfill reruns did not complete; see backfill_manifest.csv.\n"
            f"First failures:\n{sample}"
        )

    write_final_csvs(
        args.output_root,
        records,
        extra_seed_start=args.extra_seed_start,
        extra_seed_count=args.extra_seed_count,
    )
    write_backfill_manifest(args.output_root, records)
    print(
        f"BACKFILL COMPLETE: logical_jobs={len(records)} long_rows={600 * (5 + args.extra_seed_count)} "
        f"summary_rows=600 output_root={args.output_root}"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
