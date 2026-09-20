from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset

DATASET_NAMES = ("cora", "citeseer", "facebook", "lastfm", "actor", "flickr")


def ensure_datasets(repo_root: Path, names: Iterable[str] = DATASET_NAMES) -> list[dict[str, Any]]:
    """Sequentially download/process all datasets before batch scheduling."""
    data_root = Path(repo_root) / "datasets"
    data_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for name in names:
        print(f"[dataset-preflight] {name}: downloading/checking", flush=True)
        data = load_dataset(
            dataset=name,
            data_dir=str(data_root),
            data_range=(0.0, 1.0),
            val_ratio=0.25,
            test_ratio=0.25,
        )
        record = {
            "dataset": name,
            "nodes": int(getattr(data, "num_nodes", data.x.size(0))),
            "features": int(getattr(data, "num_features", data.x.size(1))),
            "classes": int(getattr(data, "num_classes", int(data.y.max().item()) + 1)),
        }
        results.append(record)
        print(f"[dataset-preflight] {name}: ready {record}", flush=True)
    return results
