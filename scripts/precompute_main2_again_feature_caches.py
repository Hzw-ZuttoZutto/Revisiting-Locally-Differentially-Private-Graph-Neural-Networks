#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET = REPO_ROOT / "hparams_search_scripts" / "precompute_feature_caches.py"


def main() -> None:
    sys.argv = [
        str(TARGET),
        "--suite-script", str(REPO_ROOT / "suite_main2_again.sh"),
        *sys.argv[1:],
    ]
    runpy.run_path(str(TARGET), run_name="__main__")


if __name__ == "__main__":
    main()
