#!/usr/bin/env python3
from __future__ import annotations

import csv
import shlex
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = REPO_ROOT / 'paper_experiments' / 'figure8_parallel_probe' / 'measurement_summary.csv'
SPLIT_ROOT = REPO_ROOT / 'configs_final' / 'figure8_split'
SUITE_PATH = REPO_ROOT / 'suite_figure8.sh'
BASE_SUITE_PREFIX = [
    'python -m hparams_search_scripts.run_mechanism_hparam_search --config configs_final/figure8/sage/direct.yaml --output_root_dir paper_experiments/figure8/sage/direct.yaml',
    'python -m hparams_search_scripts.run_mechanism_hparam_search --config configs_final/figure8/sage/learned_projected.yaml --output_root_dir paper_experiments/figure8/sage/learned_projected.yaml',
    'python -m hparams_search_scripts.run_mechanism_hparam_search --config configs_final/figure8/sage/random_projected.yaml --output_root_dir paper_experiments/figure8/sage/random_projected.yaml',
]
BASE_CONFIGS = {
    ('gat', 'direct'): REPO_ROOT / 'configs_final' / 'figure8' / 'gat' / 'direct.yaml',
    ('gat', 'random_projected'): REPO_ROOT / 'configs_final' / 'figure8' / 'gat' / 'random_projected.yaml',
    ('gat', 'learned_projected'): REPO_ROOT / 'configs_final' / 'figure8' / 'gat' / 'learned_projected.yaml',
    ('gcn', 'direct'): REPO_ROOT / 'configs_final' / 'figure8' / 'gcn' / 'direct.yaml',
    ('gcn', 'random_projected'): REPO_ROOT / 'configs_final' / 'figure8' / 'gcn' / 'random_projected.yaml',
    ('gcn', 'learned_projected'): REPO_ROOT / 'configs_final' / 'figure8' / 'gcn' / 'learned_projected.yaml',
}
BACKBONE_ORDER = ('gat', 'gcn')
DATASET_ORDER = ('cora', 'lastfm', 'citeseer', 'Books-Children', 'Books-History', 'ogbn-arxiv')
LEARNED_DIM_ORDER = (64, 128, 256, 512)
FAMILY_ORDER = ('direct', 'learned_projected', 'random_projected')


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(raw, dict):
        raise TypeError(f'Expected mapping YAML in {path}')
    return raw


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


def _load_summary() -> dict[tuple[str, str, str, str], dict[str, str]]:
    with SUMMARY_PATH.open('r', encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    summary = {
        (
            row['backbone'],
            row['family'],
            row['dataset'],
            row['preprojection_output_dim'],
        ): row
        for row in rows
    }
    if len(summary) != 72:
        raise RuntimeError(f'Expected 72 summary rows, got {len(summary)}')
    return summary


def _update_base_gpu_ids() -> None:
    for path in BASE_CONFIGS.values():
        raw = _load_yaml(path)
        raw['device']['gpu_ids'] = [0, 1, 2, 3, 4, 5]
        _write_yaml(path, raw)


def _relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _split_config_path(backbone: str, family: str, dataset: str, preprojection_output_dim: int | None) -> Path:
    if family == 'learned_projected':
        assert preprojection_output_dim is not None
        return SPLIT_ROOT / backbone / family / dataset / f'preprojection_output_dim={preprojection_output_dim}.yaml'
    return SPLIT_ROOT / backbone / family / f'{dataset}.yaml'


def _split_output_root(backbone: str, family: str, dataset: str, preprojection_output_dim: int | None) -> str:
    base = Path('paper_experiments') / 'figure8' / backbone / family / dataset
    if family == 'learned_projected':
        assert preprojection_output_dim is not None
        base = base / f'preprojection_output_dim={preprojection_output_dim}'
    return base.as_posix()


def _generate_split_configs(summary: dict[tuple[str, str, str, str], dict[str, str]]) -> list[tuple[Path, str]]:
    generated: list[tuple[Path, str]] = []
    for backbone in BACKBONE_ORDER:
        for family in FAMILY_ORDER:
            base_raw = _load_yaml(BASE_CONFIGS[(backbone, family)])
            for dataset in DATASET_ORDER:
                dims = LEARNED_DIM_ORDER if family == 'learned_projected' else (None,)
                for dim in dims:
                    dim_text = '' if dim is None else str(dim)
                    row = summary[(backbone, family, dataset, dim_text)]
                    split_raw = deepcopy(base_raw)
                    split_raw['device']['gpu_ids'] = [0, 1, 2, 3, 4, 5]
                    split_raw['device']['max_parallel_per_gpu'] = int(row['safe_max_parallel_per_gpu'])
                    split_raw['search_space']['dataset']['datasets'] = [dataset]
                    if family == 'learned_projected':
                        split_raw['search_space']['feature_transformation']['preprojection_output_dim'] = [int(dim)]
                    config_path = _split_config_path(backbone, family, dataset, dim)
                    _write_yaml(config_path, split_raw)
                    generated.append((config_path, _split_output_root(backbone, family, dataset, dim)))
    if len(generated) != 72:
        raise RuntimeError(f'Expected 72 generated configs, got {len(generated)}')
    return generated


def _rewrite_suite(generated: list[tuple[Path, str]]) -> None:
    lines = list(BASE_SUITE_PREFIX)
    for config_path, output_root in generated:
        lines.append(
            'python -m hparams_search_scripts.run_mechanism_hparam_search '
            f'--config {_relative(config_path)} '
            f'--output_root_dir {output_root}'
        )
    if len(lines) != 75:
        raise RuntimeError(f'Expected 75 suite lines, got {len(lines)}')
    SUITE_PATH.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> int:
    summary = _load_summary()
    _update_base_gpu_ids()
    generated = _generate_split_configs(summary)
    _rewrite_suite(generated)
    print(f'generated_configs={len(generated)}')
    print(f'suite_path={SUITE_PATH}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
