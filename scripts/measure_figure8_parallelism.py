#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_ARTIFACT_ROOT = Path('/tmp/rethinking_dp_gnn_figure8_parallel_probe')
SUMMARY_PATH = REPO_ROOT / 'paper_experiments' / 'figure8_parallel_probe' / 'measurement_summary.csv'
PROBE_GPU = 0
MAX_PARALLEL = 10
OOM_PATTERNS = (
    'out of memory',
    'torch.cuda.outofmemoryerror',
    'cuda error: out of memory',
    'cuda out of memory',
    'cublas_status_alloc_failed',
)
SUMMARY_COLUMNS = [
    'backbone',
    'family',
    'dataset',
    'preprojection_output_dim',
    'safe_max_parallel_per_gpu',
    'probe_gpu',
    'feature',
    'feature_dim',
    'x_steps',
    'dropout',
    'learning_rate',
    'weight_decay',
    'hidden_dim',
    'command_template',
]
BASE_CONFIGS = [
    REPO_ROOT / 'configs_final' / 'figure8' / 'gat' / 'direct.yaml',
    REPO_ROOT / 'configs_final' / 'figure8' / 'gat' / 'random_projected.yaml',
    REPO_ROOT / 'configs_final' / 'figure8' / 'gat' / 'learned_projected.yaml',
    REPO_ROOT / 'configs_final' / 'figure8' / 'gcn' / 'direct.yaml',
    REPO_ROOT / 'configs_final' / 'figure8' / 'gcn' / 'random_projected.yaml',
    REPO_ROOT / 'configs_final' / 'figure8' / 'gcn' / 'learned_projected.yaml',
]


@dataclass(frozen=True)
class ProbeCombo:
    backbone: str
    family: str
    dataset: str
    feature: str
    smoother: str
    x_steps: int
    hidden_dim: int
    learning_rate: str
    weight_decay: str
    dropout: str
    feature_dim: int | None
    random_normal_mean: str | None
    random_normal_std: str | None
    preprojection_output_dim: int | None
    base_config_path: Path

    @property
    def key(self) -> tuple[str, str, str, str]:
        dim = '' if self.preprojection_output_dim is None else str(self.preprojection_output_dim)
        return (self.backbone, self.family, self.dataset, dim)

    @property
    def slug(self) -> str:
        parts = [
            f'backbone={self.backbone}',
            f'family={self.family}',
            f'dataset={self.dataset}',
        ]
        if self.preprojection_output_dim is not None:
            parts.append(f'preprojection_output_dim={self.preprojection_output_dim}')
        return '/'.join(parts)


@dataclass(frozen=True)
class ProbeAttemptResult:
    success: bool
    parallelism: int
    seconds: float
    failure_reason: str | None = None


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(raw, dict):
        raise TypeError(f'Expected mapping YAML in {path}')
    return raw


def _first(values: list[Any], *, name: str, path: Path) -> Any:
    if not isinstance(values, list) or len(values) == 0:
        raise ValueError(f'Expected non-empty list for {name} in {path}')
    return values[0]


def _discover_combos() -> list[ProbeCombo]:
    combos: list[ProbeCombo] = []
    for path in BASE_CONFIGS:
        raw = _load_yaml(path)
        search_space = raw['search_space']
        feature_cfg = search_space['feature_transformation']
        trainer_cfg = search_space['trainer']
        model_cfg = search_space['model']
        datasets = list(search_space['dataset']['datasets'])
        backbone = str(_first(model_cfg['backbones'], name='backbones', path=path))
        feature = str(_first(feature_cfg['features'], name='features', path=path))
        smoother = str(_first(search_space['calibrator']['smoother'], name='smoother', path=path))
        x_steps = int(max(search_space['calibrator']['x_steps']))
        hidden_dim = int(raw['defaults']['model']['hidden_dim'])
        learning_rate = str(_first(trainer_cfg['learning_rate'], name='learning_rate', path=path))
        weight_decay = str(_first(trainer_cfg['weight_decay'], name='weight_decay', path=path))
        dropout = str(max(float(value) for value in model_cfg['dropout']))
        family = path.stem
        feature_dim = None
        random_normal_mean = None
        random_normal_std = None
        if feature == 'random_normal':
            feature_dim = int(_first(feature_cfg['feature_dim'], name='feature_dim', path=path))
            random_normal_mean = str(_first(feature_cfg['random_normal_mean'], name='random_normal_mean', path=path))
            random_normal_std = str(_first(feature_cfg['random_normal_std'], name='random_normal_std', path=path))
        if family == 'learned_projected':
            dims = [int(value) for value in feature_cfg['preprojection_output_dim']]
            for dataset in datasets:
                for dim in dims:
                    combos.append(
                        ProbeCombo(
                            backbone=backbone,
                            family=family,
                            dataset=str(dataset),
                            feature=feature,
                            smoother=smoother,
                            x_steps=x_steps,
                            hidden_dim=hidden_dim,
                            learning_rate=learning_rate,
                            weight_decay=weight_decay,
                            dropout=dropout,
                            feature_dim=feature_dim,
                            random_normal_mean=random_normal_mean,
                            random_normal_std=random_normal_std,
                            preprojection_output_dim=dim,
                            base_config_path=path,
                        )
                    )
            continue
        for dataset in datasets:
            combos.append(
                ProbeCombo(
                    backbone=backbone,
                    family=family,
                    dataset=str(dataset),
                    feature=feature,
                    smoother=smoother,
                    x_steps=x_steps,
                    hidden_dim=hidden_dim,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    dropout=dropout,
                    feature_dim=feature_dim,
                    random_normal_mean=random_normal_mean,
                    random_normal_std=random_normal_std,
                    preprojection_output_dim=None,
                    base_config_path=path,
                )
            )
    return combos


def _command_for_combo(combo: ProbeCombo, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / 'main.py'),
        '--dataset',
        combo.dataset,
        '--feature',
        combo.feature,
        '--model',
        combo.backbone,
        '--smoother',
        combo.smoother,
        '--x-steps',
        str(combo.x_steps),
        '--max-epochs',
        '1',
        '--patience',
        '1',
        '--repeats',
        '1',
        '--device',
        'cuda',
        '--log',
        'false',
        '--show-progress',
        'false',
        '--output-dir',
        str(output_dir),
        '--dropout',
        combo.dropout,
        '--learning-rate',
        combo.learning_rate,
        '--weight-decay',
        combo.weight_decay,
        '--hidden-dim',
        str(combo.hidden_dim),
    ]
    if combo.feature == 'random_normal':
        command.extend(
            [
                '--feature-dim',
                str(combo.feature_dim),
                '--random-normal-mean',
                str(combo.random_normal_mean),
                '--random-normal-std',
                str(combo.random_normal_std),
            ]
        )
    elif combo.preprojection_output_dim is not None:
        command.extend(
            [
                '--feature-preprojection',
                'true',
                '--preprojection-output-dim',
                str(combo.preprojection_output_dim),
            ]
        )
    return command


def _command_template(combo: ProbeCombo) -> str:
    template_dir = Path('__OUTPUT_DIR__')
    return shlex.join(_command_for_combo(combo, template_dir))


def _log_contains_oom(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    text = log_path.read_text(encoding='utf-8', errors='replace').lower()
    return any(pattern in text for pattern in OOM_PATTERNS)


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _run_probe_attempt(combo: ProbeCombo, parallelism: int) -> ProbeAttemptResult:
    attempt_root = PROBE_ARTIFACT_ROOT / combo.slug / f'parallelism={parallelism}'
    if attempt_root.exists():
        shutil.rmtree(attempt_root)
    attempt_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(PROBE_GPU)
    env['PYTHONUNBUFFERED'] = '1'
    env.setdefault('OMP_NUM_THREADS', '1')
    env.setdefault('MKL_NUM_THREADS', '1')

    running: list[tuple[subprocess.Popen[bytes], Any, Path, Path]] = []
    start = time.monotonic()
    try:
        for run_index in range(parallelism):
            run_dir = attempt_root / f'run_{run_index:02d}'
            log_path = attempt_root / f'run_{run_index:02d}.log'
            run_dir.mkdir(parents=True, exist_ok=True)
            handle = log_path.open('wb')
            process = subprocess.Popen(
                _command_for_combo(combo, run_dir),
                cwd=REPO_ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=env,
            )
            running.append((process, handle, log_path, run_dir))
            time.sleep(0.2)

        failed_reason: str | None = None
        while len(running) > 0:
            next_running: list[tuple[subprocess.Popen[bytes], Any, Path, Path]] = []
            for process, handle, log_path, run_dir in running:
                return_code = process.poll()
                if return_code is None:
                    next_running.append((process, handle, log_path, run_dir))
                    continue
                handle.flush()
                handle.close()
                if return_code != 0:
                    failed_reason = f'nonzero_exit:{return_code}'
                elif _log_contains_oom(log_path):
                    failed_reason = 'oom_in_log'
                elif len(list(run_dir.glob('*.csv'))) == 0:
                    failed_reason = 'missing_output_csv'
                if failed_reason is not None:
                    break
            if failed_reason is not None:
                for process, handle, _, _ in next_running:
                    _terminate_process(process)
                    if not handle.closed:
                        handle.flush()
                        handle.close()
                elapsed = time.monotonic() - start
                return ProbeAttemptResult(False, parallelism, elapsed, failed_reason)
            running = next_running
            if len(running) > 0:
                time.sleep(0.5)
    finally:
        for process, handle, _, _ in running:
            _terminate_process(process)
            if not handle.closed:
                handle.flush()
                handle.close()
    elapsed = time.monotonic() - start
    return ProbeAttemptResult(True, parallelism, elapsed)


def _measure_safe_max_parallel(combo: ProbeCombo) -> int:
    print(f'[Probe] start backbone={combo.backbone} family={combo.family} dataset={combo.dataset} dim={combo.preprojection_output_dim}')
    top_attempt = _run_probe_attempt(combo, MAX_PARALLEL)
    print(f'[Probe] parallel={MAX_PARALLEL} success={top_attempt.success} seconds={top_attempt.seconds:.2f}')
    if top_attempt.success:
        return MAX_PARALLEL

    low = 1
    high = MAX_PARALLEL - 1
    safe = 0
    while low <= high:
        mid = (low + high) // 2
        attempt = _run_probe_attempt(combo, mid)
        print(f'[Probe] parallel={mid} success={attempt.success} seconds={attempt.seconds:.2f}')
        if attempt.success:
            safe = mid
            low = mid + 1
        else:
            high = mid - 1
    if safe < 1:
        raise RuntimeError(
            f'Could not find any safe parallelism for combo {combo.key}; even parallelism=1 failed.'
        )
    return safe


def _load_existing_summary() -> dict[tuple[str, str, str, str], dict[str, str]]:
    if not SUMMARY_PATH.exists():
        return {}
    with SUMMARY_PATH.open('r', encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    return {
        (
            row['backbone'],
            row['family'],
            row['dataset'],
            row['preprojection_output_dim'],
        ): row
        for row in rows
    }


def _write_summary(rows: list[dict[str, str]]) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    combos = _discover_combos()
    if len(combos) != 72:
        raise RuntimeError(f'Expected 72 probe combos, got {len(combos)}')

    existing = _load_existing_summary()
    rows_by_key = dict(existing)
    ordered_rows: list[dict[str, str]] = []

    for combo in combos:
        key = combo.key
        dim_text = '' if combo.preprojection_output_dim is None else str(combo.preprojection_output_dim)
        if key in rows_by_key:
            print(f'[Probe] skip existing backbone={combo.backbone} family={combo.family} dataset={combo.dataset} dim={combo.preprojection_output_dim}')
            continue
        safe_max_parallel = _measure_safe_max_parallel(combo)
        rows_by_key[key] = {
            'backbone': combo.backbone,
            'family': combo.family,
            'dataset': combo.dataset,
            'preprojection_output_dim': dim_text,
            'safe_max_parallel_per_gpu': str(safe_max_parallel),
            'probe_gpu': str(PROBE_GPU),
            'feature': combo.feature,
            'feature_dim': '' if combo.feature_dim is None else str(combo.feature_dim),
            'x_steps': str(combo.x_steps),
            'dropout': combo.dropout,
            'learning_rate': combo.learning_rate,
            'weight_decay': combo.weight_decay,
            'hidden_dim': str(combo.hidden_dim),
            'command_template': _command_template(combo),
        }
        ordered_rows = [rows_by_key[c.key] for c in combos if c.key in rows_by_key]
        _write_summary(ordered_rows)
        print(f'[Probe] recorded safe_max_parallel_per_gpu={safe_max_parallel}')

    ordered_rows = [rows_by_key[c.key] for c in combos if c.key in rows_by_key]
    _write_summary(ordered_rows)
    print(f'[Probe] summary written to {SUMMARY_PATH}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
