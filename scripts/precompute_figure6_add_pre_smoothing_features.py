#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET = REPO_ROOT / "scripts" / "precompute_figure3_pre_smoothing_features.py"


def main() -> None:
    sys.argv = [
        str(TARGET),
        "--suite-script", str(REPO_ROOT / "suite_figure6_add.sh"),
        "--cache-root", "/data/hzw/Rethinking_DP_GNN_runtime/cache/figure6_add_pre_smoothing_suite_figure6_add",
        "--config-substring", "configs_final/figure6_add/",
        *sys.argv[1:],
    ]
    runpy.run_path(str(TARGET), run_name="__main__")


if __name__ == "__main__":
    main()
