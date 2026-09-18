from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
AEC_ROOT = REPO_ROOT / "aec"
REFERENCE_ROOT = AEC_ROOT / "reference"
FIXED_ROOT = AEC_ROOT / "fixed_hparams"
SCALED_ROOT = AEC_ROOT / "scaled_search"
WORK_ROOT = REPO_ROOT / "work"
OUTPUT_ROOT = REPO_ROOT / "outputs"

def ensure_runtime_dirs() -> None:
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

def parse_gpu_ids(value: str | None = None) -> list[int]:
    text = os.environ.get("AEC_GPU_IDS", "0") if value is None else value
    ids = [int(part.strip()) for part in text.split(",") if part.strip()]
    return ids or [0]

def max_parallel_per_gpu() -> int:
    return max(1, int(os.environ.get("AEC_MAX_PARALLEL_PER_GPU", "1")))

def output_dir(mode: str, figure_id: int) -> Path:
    suffix = "" if mode == "reference" else f"_{mode}"
    path = OUTPUT_ROOT / f"figure{figure_id}{suffix}"
    path.mkdir(parents=True, exist_ok=True)
    return path
