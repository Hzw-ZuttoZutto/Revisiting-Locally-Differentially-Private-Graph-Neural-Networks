from __future__ import annotations

import importlib.metadata
import platform
import sys
from pathlib import Path
from .paths import ensure_runtime_dirs, parse_gpu_ids, max_parallel_per_gpu

def check_environment(*, require_gpu: bool = False) -> dict[str, object]:
    ensure_runtime_dirs()
    versions = {}
    for package in ("torch", "torch-geometric", "numpy", "pandas", "matplotlib", "pyyaml"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    try:
        import torch
        cuda_available = bool(torch.cuda.is_available())
        cuda_count = int(torch.cuda.device_count())
    except Exception:
        cuda_available, cuda_count = False, 0
    if require_gpu and not cuda_available:
        raise RuntimeError("GPU execution requested but torch.cuda.is_available() is false")
    info = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "versions": versions,
        "cuda_available": cuda_available,
        "cuda_count": cuda_count,
        "gpu_ids": parse_gpu_ids(),
        "max_parallel_per_gpu": max_parallel_per_gpu(),
    }
    print(info)
    return info
