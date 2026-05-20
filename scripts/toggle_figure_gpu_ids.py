#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = [
    Path('configs_final/figure3'),
    Path('configs_final/figure8'),
    Path('configs_final/figure8_split'),
]
GPU_OPTIONS = {
    '0-5': [0, 1, 2, 3, 4, 5],
    '2-5': [2, 3, 4, 5],
}
BLOCK_PATTERN = re.compile(r'(?m)^(\s*gpu_ids:)\n((?:\s*-\s*\d+\n)+)')
INLINE_PATTERN = re.compile(r'(?m)^(\s*gpu_ids:)\s*\[[^\]]*\]\s*$')


def build_block(indent: str, values: list[int]) -> str:
    items = ''.join(f'{indent}  - {value}\n' for value in values)
    return f'{indent}gpu_ids:\n{items}'


def build_inline(indent: str, values: list[int]) -> str:
    joined = ','.join(str(v) for v in values)
    return f'{indent}gpu_ids: [{joined}]'


def update_text(text: str, values: list[int]) -> tuple[str, bool]:
    changed = False

    def replace_block(match: re.Match[str]) -> str:
        nonlocal changed
        indent = match.group(1)[:-len('gpu_ids:')]
        replacement = build_block(indent, values)
        if match.group(0) != replacement:
            changed = True
        return replacement

    def replace_inline(match: re.Match[str]) -> str:
        nonlocal changed
        indent = match.group(1)[:-len('gpu_ids:')]
        replacement = build_inline(indent, values)
        if match.group(0) != replacement:
            changed = True
        return replacement

    updated = BLOCK_PATTERN.sub(replace_block, text)
    updated = INLINE_PATTERN.sub(replace_inline, updated)
    return updated, changed


def iter_yaml_files(targets: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for target in targets:
        root = REPO_ROOT / target
        if not root.exists():
            raise FileNotFoundError(f'Missing target directory: {root}')
        paths.extend(sorted(root.rglob('*.yaml')))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Toggle gpu_ids for figure3 / figure8 / figure8_split configs.'
    )
    parser.add_argument('mode', choices=sorted(GPU_OPTIONS), help='Target gpu_ids set.')
    parser.add_argument(
        '--targets',
        nargs='*',
        choices=['figure3', 'figure8', 'figure8_split'],
        default=['figure3', 'figure8', 'figure8_split'],
        help='Subset of config groups to update. Defaults to all.',
    )
    parser.add_argument('--dry-run', action='store_true', help='Show what would change without writing files.')
    args = parser.parse_args()

    target_dirs = [Path('configs_final') / name for name in args.targets]
    values = GPU_OPTIONS[args.mode]
    yaml_files = iter_yaml_files(target_dirs)

    changed_paths: list[Path] = []
    for path in yaml_files:
        original = path.read_text(encoding='utf-8')
        updated, changed = update_text(original, values)
        if changed:
            changed_paths.append(path)
            if not args.dry_run:
                path.write_text(updated, encoding='utf-8')

    joined_values = ','.join(str(v) for v in values)
    print(f'mode={args.mode} gpu_ids=[{joined_values}]')
    print('targets=' + ','.join(args.targets))
    print(f'files_scanned={len(yaml_files)}')
    print(f'files_changed={len(changed_paths)}')
    for path in changed_paths[:20]:
        print(path.relative_to(REPO_ROOT))
    if len(changed_paths) > 20:
        print(f'... and {len(changed_paths) - 20} more')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
