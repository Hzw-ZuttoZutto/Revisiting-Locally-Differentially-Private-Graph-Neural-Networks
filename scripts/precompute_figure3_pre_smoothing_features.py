#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import multiprocessing as mp
import os
import random
import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from datasets import load_dataset, resolve_dataset_name
    from hparams_search_scripts import run_mechanism_hparam_search as search_runner
    from pre_smoothing_feature_cache import (
        GRAPH_FINGERPRINT_ATTR,
        RAW_FINGERPRINT_ATTR,
        build_cache_key_payload,
        cache_path_for_payload,
        graph_fingerprint,
        prepare_pre_smoothing_input,
        raw_feature_fingerprint,
        resolve_cache_root,
    )
except ModuleNotFoundError:
    from datasets import load_dataset, resolve_dataset_name  # type: ignore
    from hparams_search_scripts import run_mechanism_hparam_search as search_runner  # type: ignore
    from pre_smoothing_feature_cache import (  # type: ignore
        GRAPH_FINGERPRINT_ATTR,
        RAW_FINGERPRINT_ATTR,
        build_cache_key_payload,
        cache_path_for_payload,
        graph_fingerprint,
        prepare_pre_smoothing_input,
        raw_feature_fingerprint,
        resolve_cache_root,
    )


DEFAULT_CACHE_ROOT = "/data/hzw/Rethinking_DP_GNN_runtime/cache/figure3_pre_smoothing_suite_figure8"


@dataclass(frozen=True)
class CacheTask:
    dataset: str
    data_range: tuple[float, float]
    val_ratio: float
    test_ratio: float
    mechanism: str
    x_eps: str
    m: str
    norm: bool
    norm_scale: str
    use_nfr: bool
    tao2: str
    seed: int
    source_configs: tuple[str, ...]

    def cache_identity(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "data_range": list(self.data_range),
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "mechanism": self.mechanism,
            "x_eps": self.x_eps,
            "m": self.m,
            "norm": self.norm,
            "norm_scale": self.norm_scale,
            "use_nfr": self.use_nfr,
            "tao2": self.tao2,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class TaskResult:
    task: CacheTask
    status: str
    cache_path: str | None
    elapsed_sec: float
    bytes_written: int
    worker_gpu: int | None
    error_message: str | None = None


def _parse_csv_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    values = [item.strip() for item in raw.split(",") if item.strip() != ""]
    return values or None


def _parse_int_csv_list(raw: str | None) -> list[int] | None:
    values = _parse_csv_list(raw)
    if values is None:
        return None
    return [int(item) for item in values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute figure3 pre-smoothing feature caches for the current suite_figure8.sh subset."
    )
    parser.add_argument(
        "--suite-script",
        type=str,
        default=str(REPO_ROOT / "suite_figure8.sh"),
        help="suite script to inspect for figure3 config lines",
    )
    parser.add_argument(
        "--cache-root",
        type=str,
        default=DEFAULT_CACHE_ROOT,
        help="target cache root for pre-smoothing feature tensors",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="optional manifest CSV path; defaults to <cache-root>/manifest.csv",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="0",
        help="comma-separated physical GPU ids for precompute workers",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=1,
        help="number of parallel precompute worker processes to launch per listed GPU",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild cache entries even when the target file already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="enumerate tasks and estimate storage without building caches",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="optional comma-separated dataset filter",
    )
    parser.add_argument(
        "--mechanisms",
        type=str,
        default=None,
        help="optional comma-separated mechanism filter",
    )
    parser.add_argument(
        "--x-eps",
        dest="x_eps",
        type=str,
        default=None,
        help="optional comma-separated x_eps filter",
    )
    parser.add_argument(
        "--tao2",
        type=str,
        default=None,
        help="optional comma-separated tao2 filter; use 'none' for no-NFR tasks",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="optional comma-separated seed filter",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="optional limit on the number of tasks after filtering",
    )
    return parser.parse_args()


def _read_suite_script_config_paths(path: Path) -> list[Path]:
    config_paths: list[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "" or stripped.startswith("#"):
            continue
        if "configs_final/figure3/" not in stripped:
            continue
        parts = shlex.split(stripped)
        for index, part in enumerate(parts):
            if part == "--config" and index + 1 < len(parts):
                config_paths.append((path.parent / parts[index + 1]).resolve())
                break
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in config_paths:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return unique


def _task_key(task: CacheTask) -> str:
    return json.dumps(task.cache_identity(), sort_keys=True, separators=(",", ":"))


def _collect_cache_tasks(config_paths: list[Path]) -> list[CacheTask]:
    task_map: dict[str, CacheTask] = {}
    for config_path in config_paths:
        search_config = search_runner.load_search_config(config_path)
        batch_spec = search_runner.build_batch_spec(
            search_config=search_config,
            output_root=REPO_ROOT / ".cache_pre_smoothing_probe" / config_path.stem,
            config_copy_source=config_path,
            pre_smoothing_feature_cache_root=None,
        )
        for job in batch_spec.jobs:
            fixed_params = job.job_spec["fixed_params"]
            defaults = job.job_spec["defaults"]
            if not isinstance(fixed_params, dict) or not isinstance(defaults, dict):
                raise RuntimeError("job_spec fixed_params/defaults must be mappings")
            if str(fixed_params.get("feature")) != "raw":
                continue
            mechanism = str(fixed_params.get("mechanism"))
            if mechanism not in {"mbm", "pm", "hds"}:
                continue
            x_eps = str(fixed_params.get("x_eps"))
            if x_eps == "inf":
                continue

            verify_repeats = int(defaults["stage"]["verify"]["repeats"])  # type: ignore[index]
            base_seed = int(job.job_spec["base_seed"])
            seeds = [base_seed + offset for offset in range(verify_repeats)]
            candidate_tao2_values = {
                str(candidate["tao2"])
                for candidate in job.job_spec["candidates"]  # type: ignore[index]
            }
            for seed in seeds:
                for tao2 in sorted(candidate_tao2_values):
                    task = CacheTask(
                        dataset=resolve_dataset_name(str(fixed_params["dataset"])),
                        data_range=(
                            float(defaults["dataset"]["data_range"][0]),  # type: ignore[index]
                            float(defaults["dataset"]["data_range"][1]),  # type: ignore[index]
                        ),
                        val_ratio=float(defaults["dataset"]["val_ratio"]),  # type: ignore[index]
                        test_ratio=float(defaults["dataset"]["test_ratio"]),  # type: ignore[index]
                        mechanism=mechanism,
                        x_eps=x_eps,
                        m=str(fixed_params["m"]),
                        norm=bool(fixed_params["norm"]),
                        norm_scale=str(fixed_params["norm_scale"]),
                        use_nfr=bool(fixed_params["use_nfr"]),
                        tao2=tao2,
                        seed=seed,
                        source_configs=(str(config_path),),
                    )
                    key = _task_key(task)
                    if key in task_map:
                        merged_sources = tuple(sorted(set(task_map[key].source_configs) | {str(config_path)}))
                        task_map[key] = CacheTask(
                            dataset=task.dataset,
                            data_range=task.data_range,
                            val_ratio=task.val_ratio,
                            test_ratio=task.test_ratio,
                            mechanism=task.mechanism,
                            x_eps=task.x_eps,
                            m=task.m,
                            norm=task.norm,
                            norm_scale=task.norm_scale,
                            use_nfr=task.use_nfr,
                            tao2=task.tao2,
                            seed=task.seed,
                            source_configs=merged_sources,
                        )
                    else:
                        task_map[key] = task

    tasks = list(task_map.values())
    tasks.sort(
        key=lambda task: (
            task.dataset.casefold(),
            task.mechanism,
            float(task.x_eps),
            task.use_nfr,
            task.tao2,
            task.seed,
        )
    )
    return tasks


def _filter_tasks(tasks: list[CacheTask], args: argparse.Namespace) -> list[CacheTask]:
    dataset_filter = set(_parse_csv_list(args.datasets) or [])
    mechanism_filter = set(_parse_csv_list(args.mechanisms) or [])
    x_eps_filter = set(_parse_csv_list(args.x_eps) or [])
    tao2_filter = set(_parse_csv_list(args.tao2) or [])
    seed_filter = set(_parse_int_csv_list(args.seeds) or [])

    filtered: list[CacheTask] = []
    for task in tasks:
        if dataset_filter and task.dataset not in dataset_filter:
            continue
        if mechanism_filter and task.mechanism not in mechanism_filter:
            continue
        if x_eps_filter and task.x_eps not in x_eps_filter:
            continue
        if tao2_filter and task.tao2 not in tao2_filter:
            continue
        if seed_filter and task.seed not in seed_filter:
            continue
        filtered.append(task)

    if args.limit is not None:
        filtered = filtered[: int(args.limit)]
    return filtered


def _estimate_storage(tasks: list[CacheTask]) -> tuple[int, dict[str, int]]:
    dataset_bytes: dict[str, int] = {}
    per_dataset_seen: set[str] = set()
    total_bytes = 0
    for task in tasks:
        if task.dataset not in dataset_bytes:
            data = load_dataset(
                task.dataset,
                data_range=task.data_range,
                val_ratio=task.val_ratio,
                test_ratio=task.test_ratio,
            )
            dataset_bytes[task.dataset] = int(data.x.numel() * data.x.element_size())
        total_bytes += dataset_bytes[task.dataset]
        per_dataset_seen.add(task.dataset)
    return total_bytes, dataset_bytes


def _configure_determinism() -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state(state["torch_cuda"])


def _task_args_namespace(task: CacheTask, cache_root: Path) -> SimpleNamespace:
    tao2_value = None if task.tao2 == "none" else float(task.tao2)
    return SimpleNamespace(
        dataset=task.dataset,
        feature="raw",
        sim_reference_eps=None,
        feature_dim=None,
        scale=1.0,
        feature_preprojection=False,
        preprojection_output_dim=None,
        random_normal_mean=None,
        random_normal_std=None,
        shared_value=None,
        degree_bucket_num_buckets=None,
        degree_bucket_range_max=None,
        deepwalk_walk_length=None,
        deepwalk_number_walks=None,
        deepwalk_window_size=None,
        deepwalk_workers=None,
        deepwalk_undirected=None,
        mechanism=task.mechanism,
        x_eps=float(task.x_eps),
        m=task.m,
        norm=task.norm,
        norm_scale=task.norm_scale,
        inf_eps_unit_map=False,
        use_nfr=task.use_nfr,
        tao2=tao2_value,
        data_range=list(task.data_range),
        pre_smoothing_feature_cache_root=str(cache_root),
    )


def _task_cache_path(task: CacheTask, cache_root: Path, base_data) -> Path:
    args_ns = _task_args_namespace(task, cache_root)
    payload = build_cache_key_payload(base_data, args_ns, rewrite_seed=task.seed)
    return cache_path_for_payload(cache_root, payload)


def _run_worker(gpu_id: int, tasks: list[CacheTask], cache_root: str, force: bool) -> list[TaskResult]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    _configure_determinism()
    if not torch.cuda.is_available():
        raise RuntimeError(f"GPU worker {gpu_id} cannot see CUDA after setting CUDA_VISIBLE_DEVICES")

    device = torch.device("cuda")
    cache_root_path = resolve_cache_root(cache_root)
    if cache_root_path is None:
        raise RuntimeError("cache root must be provided")

    base_datasets: dict[str, Any] = {}
    post_load_rng_states: dict[tuple[str, int], dict[str, Any]] = {}
    results: list[TaskResult] = []
    for task in tasks:
        started = time.time()
        try:
            if task.dataset not in base_datasets:
                _seed_everything(task.seed)
                base = load_dataset(
                    task.dataset,
                    data_range=task.data_range,
                    val_ratio=task.val_ratio,
                    test_ratio=task.test_ratio,
                )
                setattr(base, GRAPH_FINGERPRINT_ATTR, graph_fingerprint(base))
                setattr(base, RAW_FINGERPRINT_ATTR, raw_feature_fingerprint(base))
                base_datasets[task.dataset] = base
                post_load_rng_states[(task.dataset, task.seed)] = _capture_rng_state()
            elif (task.dataset, task.seed) not in post_load_rng_states:
                _seed_everything(task.seed)
                _ = load_dataset(
                    task.dataset,
                    data_range=task.data_range,
                    val_ratio=task.val_ratio,
                    test_ratio=task.test_ratio,
                )
                post_load_rng_states[(task.dataset, task.seed)] = _capture_rng_state()
            base = base_datasets[task.dataset]
            _restore_rng_state(post_load_rng_states[(task.dataset, task.seed)])
            data = base.clone().to(device)
            setattr(data, GRAPH_FINGERPRINT_ATTR, getattr(base, GRAPH_FINGERPRINT_ATTR))
            setattr(data, RAW_FINGERPRINT_ATTR, getattr(base, RAW_FINGERPRINT_ATTR))
            args_ns = _task_args_namespace(task, cache_root_path)
            _, cache_result = prepare_pre_smoothing_input(
                data,
                args_ns,
                rewrite_seed=task.seed,
                force_rebuild=force,
            )
            results.append(
                TaskResult(
                    task=task,
                    status=cache_result.status,
                    cache_path=cache_result.cache_path,
                    elapsed_sec=time.time() - started,
                    bytes_written=cache_result.bytes_written,
                    worker_gpu=gpu_id,
                    error_message=None,
                )
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                TaskResult(
                    task=task,
                    status="failed",
                    cache_path=None,
                    elapsed_sec=time.time() - started,
                    bytes_written=0,
                    worker_gpu=gpu_id,
                    error_message=str(exc),
                )
            )
    return results


def _chunk_tasks(tasks: list[CacheTask], worker_count: int) -> list[list[CacheTask]]:
    chunks: list[list[CacheTask]] = [[] for _ in range(worker_count)]
    for index, task in enumerate(tasks):
        chunks[index % worker_count].append(task)
    return [chunk for chunk in chunks if chunk]


def _expand_worker_gpu_ids(gpu_ids: list[int], workers_per_gpu: int) -> list[int]:
    if workers_per_gpu <= 0:
        raise RuntimeError("--workers-per-gpu must be > 0")
    expanded: list[int] = []
    for gpu_id in gpu_ids:
        expanded.extend([gpu_id] * workers_per_gpu)
    return expanded


def _write_manifest(path: Path, results: list[TaskResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset",
                "mechanism",
                "x_eps",
                "m",
                "norm",
                "norm_scale",
                "use_nfr",
                "tao2",
                "seed",
                "status",
                "bytes_written",
                "elapsed_sec",
                "worker_gpu",
                "cache_path",
                "source_configs",
                "error_message",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.task.dataset,
                    result.task.mechanism,
                    result.task.x_eps,
                    result.task.m,
                    "true" if result.task.norm else "false",
                    result.task.norm_scale,
                    "true" if result.task.use_nfr else "false",
                    result.task.tao2,
                    result.task.seed,
                    result.status,
                    result.bytes_written,
                    f"{result.elapsed_sec:.6f}",
                    "" if result.worker_gpu is None else result.worker_gpu,
                    "" if result.cache_path is None else result.cache_path,
                    ";".join(result.task.source_configs),
                    "" if result.error_message is None else result.error_message,
                ]
            )


def main() -> int:
    args = parse_args()
    suite_script = Path(args.suite_script).resolve()
    if not suite_script.is_file():
        raise RuntimeError(f"suite script not found: {suite_script}")

    cache_root = resolve_cache_root(args.cache_root)
    if cache_root is None:
        raise RuntimeError("--cache-root must not be empty")
    manifest_path = Path(args.manifest).resolve() if args.manifest else cache_root / "manifest.csv"

    config_paths = _read_suite_script_config_paths(suite_script)
    tasks = _collect_cache_tasks(config_paths)
    tasks = _filter_tasks(tasks, args)

    total_bytes, dataset_bytes = _estimate_storage(tasks)
    total_gib = total_bytes / (1024 ** 3)
    print(f"Suite script: {suite_script}")
    print(f"Figure3 configs: {len(config_paths)}")
    for config_path in config_paths:
        print(f"  - {config_path}")
    print(f"Unique tasks: {len(tasks)}")
    print(f"Estimated payload GiB: {total_gib:.6f}")
    for dataset_name in sorted(dataset_bytes, key=str.casefold):
        dataset_count = sum(1 for task in tasks if task.dataset == dataset_name)
        dataset_total_gib = (dataset_bytes[dataset_name] * dataset_count) / (1024 ** 3)
        print(
            f"  - {dataset_name}: task_count={dataset_count}, "
            f"tensor_bytes={dataset_bytes[dataset_name]}, total_GiB={dataset_total_gib:.6f}"
        )

    if args.dry_run:
        return 0

    gpu_ids = _parse_int_csv_list(args.gpu_ids)
    if gpu_ids is None or len(gpu_ids) == 0:
        raise RuntimeError("--gpu-ids must contain at least one GPU id")
    if int(args.workers_per_gpu) <= 0:
        raise RuntimeError("--workers-per-gpu must be > 0")

    cache_root.mkdir(parents=True, exist_ok=True)
    worker_gpu_ids = _expand_worker_gpu_ids(gpu_ids, int(args.workers_per_gpu))
    worker_chunks = _chunk_tasks(tasks, len(worker_gpu_ids))
    if len(worker_chunks) == 0:
        _write_manifest(manifest_path, [])
        return 0

    print(
        f"Precompute workers: {len(worker_chunks)} "
        f"({int(args.workers_per_gpu)} per GPU across ids {gpu_ids})"
    )

    results: list[TaskResult] = []
    if len(worker_chunks) == 1:
        results.extend(_run_worker(worker_gpu_ids[0], worker_chunks[0], str(cache_root), bool(args.force)))
    else:
        ctx = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=len(worker_chunks),
            mp_context=ctx,
        ) as executor:
            futures = []
            for gpu_id, chunk in zip(worker_gpu_ids, worker_chunks):
                futures.append(
                    executor.submit(
                        _run_worker,
                        gpu_id,
                        chunk,
                        str(cache_root),
                        bool(args.force),
                    )
                )
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())

    results.sort(
        key=lambda result: (
            result.task.dataset.casefold(),
            result.task.mechanism,
            float(result.task.x_eps),
            result.task.use_nfr,
            result.task.tao2,
            result.task.seed,
        )
    )
    _write_manifest(manifest_path, results)

    status_counts: dict[str, int] = {}
    for result in results:
        status_counts[result.status] = status_counts.get(result.status, 0) + 1
    print(f"Manifest: {manifest_path}")
    for status_name in sorted(status_counts):
        print(f"  - {status_name}: {status_counts[status_name]}")

    failures = [result for result in results if result.status == "failed"]
    if failures:
        for failed in failures[:20]:
            print(
                "FAILED",
                failed.task.dataset,
                failed.task.mechanism,
                failed.task.x_eps,
                failed.task.tao2,
                failed.task.seed,
                failed.error_message,
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
