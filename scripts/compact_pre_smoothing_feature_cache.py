#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pre_smoothing_feature_cache import compact_legacy_cache_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Losslessly compact existing legacy dense pre-smoothing cache files."
    )
    parser.add_argument("--cache-root", required=True, help="pre-smoothing cache root to migrate")
    parser.add_argument("--limit", type=int, default=None, help="optional maximum number of files to inspect")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="print cumulative progress after this many inspected files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache_root = Path(args.cache_root).expanduser().resolve()
    if not cache_root.is_dir():
        raise RuntimeError(f"cache root is not a directory: {cache_root}")
    if args.limit is not None and args.limit < 0:
        raise RuntimeError("--limit must be >= 0")
    if args.progress_every <= 0:
        raise RuntimeError("--progress-every must be > 0")

    cache_files = sorted(cache_root.rglob("*.pt"))
    if args.limit is not None:
        cache_files = cache_files[: args.limit]

    started = time.time()
    status_counts: dict[str, int] = {}
    bytes_before = 0
    bytes_after = 0
    print(f"Cache root: {cache_root}", flush=True)
    print(f"Cache files: {len(cache_files)}", flush=True)
    for index, cache_file in enumerate(cache_files, start=1):
        status, before, after = compact_legacy_cache_file(cache_file)
        status_counts[status] = status_counts.get(status, 0) + 1
        bytes_before += before
        bytes_after += after
        if index % args.progress_every == 0 or index == len(cache_files):
            elapsed = time.time() - started
            saved_gib = (bytes_before - bytes_after) / (1024 ** 3)
            print(
                f"Progress: {index}/{len(cache_files)} "
                f"saved_GiB={saved_gib:.3f} elapsed_sec={elapsed:.1f}",
                flush=True,
            )

    print(f"Elapsed sec: {time.time() - started:.3f}")
    print(f"Input GiB: {bytes_before / (1024 ** 3):.6f}")
    print(f"Output GiB: {bytes_after / (1024 ** 3):.6f}")
    print(f"Saved GiB: {(bytes_before - bytes_after) / (1024 ** 3):.6f}")
    for status in sorted(status_counts):
        print(f"  - {status}: {status_counts[status]}")
    return 1 if status_counts.get("invalid", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
