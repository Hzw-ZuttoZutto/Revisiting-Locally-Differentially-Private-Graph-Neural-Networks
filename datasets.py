import csv
import json
import os
import sys
import time
import types
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import quote

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import ToSparseTensor
from torch_geometric.utils import coalesce, to_undirected

from transforms import FilterTopClass, Normalize

try:
    from torch_geometric.transforms import AddTrainValTestMask
except ImportError:
    AddTrainValTestMask = None

try:
    from torch_geometric.transforms import RandomNodeSplit
except ImportError:
    RandomNodeSplit = None


HF_DATASET_URL_TEMPLATE = "https://huggingface.co/datasets/{repo_id}/resolve/main/{path}?download=true"
HTTP_USER_AGENT = "HZW-DP/1.0"
HTTP_TIMEOUT_SECONDS = 120
DOWNLOAD_CHUNK_SIZE_BYTES = 1024 * 1024
DOWNLOAD_ATTEMPTS = 3

LOADER_BUILTIN_PLANETOID = "builtin_planetoid"
LOADER_BUILTIN_KARATECLUB = "builtin_karateclub"
LOADER_OGB_NODEPROP = "ogb_nodeprop"
LOADER_HF_OGB_FEATURE_SWAP = "hf_ogb_feature_swap"
LOADER_HF_CSV_NPY_GRAPH = "hf_csv_npy_graph"


@dataclass(frozen=True)
class DatasetSpec:
    """Declarative specification for a benchmark dataset."""

    canonical_name: str
    loader_kind: str
    loader_target: str | None = None
    loader_kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class RawDatasetBundle:
    """Raw graph object emitted by a dataset-specific acquisition routine."""

    data: Data


def _builtin_benchmark_spec(
    canonical_name: str,
    loader_kind: str,
    loader_target: str,
) -> DatasetSpec:
    return DatasetSpec(
        canonical_name=canonical_name,
        loader_kind=loader_kind,
        loader_target=loader_target,
    )


def _text_attributed_graph_loader_kwargs(csv_path: str, feature_path: str) -> dict[str, str]:
    return {
        "csv_path": csv_path,
        "feature_path": feature_path,
        "node_id_column": "node_id",
        "label_column": "label",
        "neighbor_column": "neighbour",
        "cached_csv_name": "graph.csv",
        "cached_feature_name": "features.npy",
    }


class KarateClub(InMemoryDataset):
    """Local wrapper around the KarateClub node-classification benchmark format."""

    url = "https://raw.githubusercontent.com/benedekrozemberczki/karateclub/master/dataset/node_level"
    available_datasets = {
        "twitch",
        "facebook",
        "github",
        "deezer",
        "lastfm",
        "wikipedia",
    }
    raw_parts = ("edges", "features", "target")

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in self.available_datasets

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = self._load_processed_data()

    def _load_processed_data(self):
        try:
            return torch.load(self.processed_paths[0], weights_only=False)
        except TypeError:
            return torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, "raw")

    @property
    def raw_file_names(self):
        return ["edges.csv", "features.csv", "target.csv"]

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, "processed")

    @property
    def processed_file_names(self):
        return "data.pt"

    def download(self):
        for part in self.raw_parts:
            download_url(f"{self.url}/{self.name}/{part}.csv", self.raw_dir)

    def process(self):
        target_file = os.path.join(self.raw_dir, self.raw_file_names[2])
        y = pd.read_csv(target_file)["target"]
        y = torch.from_numpy(y.to_numpy(dtype=int))
        num_nodes = len(y)

        edge_file = os.path.join(self.raw_dir, self.raw_file_names[0])
        edge_frame = pd.read_csv(edge_file)
        edge_index = torch.from_numpy(edge_frame.to_numpy()).t().contiguous()
        edge_index = to_undirected(edge_index, num_nodes)

        feature_file = os.path.join(self.raw_dir, self.raw_file_names[1])
        feature_frame = pd.read_csv(feature_file).drop_duplicates()
        feature_frame = feature_frame.pivot(index="node_id", columns="feature_id", values="value").fillna(0)
        feature_frame = feature_frame.reindex(range(num_nodes), fill_value=0)
        x = torch.from_numpy(feature_frame.to_numpy()).float()

        data = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return f"KarateClub-{self.name}()"


#
# Benchmark registry
#
DATASET_SPECS: dict[str, DatasetSpec] = {
    "cora": _builtin_benchmark_spec("cora", LOADER_BUILTIN_PLANETOID, "cora"),
    "citeseer": _builtin_benchmark_spec("citeseer", LOADER_BUILTIN_PLANETOID, "citeseer"),
    "pubmed": _builtin_benchmark_spec("pubmed", LOADER_BUILTIN_PLANETOID, "pubmed"),
    "facebook": _builtin_benchmark_spec("facebook", LOADER_BUILTIN_KARATECLUB, "facebook"),
    "lastfm": _builtin_benchmark_spec("lastfm", LOADER_BUILTIN_KARATECLUB, "lastfm"),
    "ogbn-arxiv": _builtin_benchmark_spec("ogbn-arxiv", LOADER_OGB_NODEPROP, "ogbn-arxiv"),
    "ogbn-arxiv-TA": DatasetSpec(
        canonical_name="ogbn-arxiv-TA",
        loader_kind=LOADER_HF_OGB_FEATURE_SWAP,
        loader_target="ogbn-arxiv",
        loader_kwargs={
            "repo_id": "Sherirto/CSTAG",
            "feature_path": "Arxiv/Feature/Arxiv_roberta_base_512_cls.npy",
            "cached_feature_name": "features.npy",
        },
    ),
    "Books-Children": DatasetSpec(
        canonical_name="Books-Children",
        loader_kind=LOADER_HF_CSV_NPY_GRAPH,
        loader_target="Sherirto/CSTAG",
        loader_kwargs=_text_attributed_graph_loader_kwargs(
            csv_path="Children/Children.csv",
            feature_path="Children/Feature/Children_roberta_base_512_cls.npy",
        ),
    ),
    "Books-History": DatasetSpec(
        canonical_name="Books-History",
        loader_kind=LOADER_HF_CSV_NPY_GRAPH,
        loader_target="Sherirto/CSTAG",
        loader_kwargs=_text_attributed_graph_loader_kwargs(
            csv_path="History/History.csv",
            feature_path="History/Feature/History_roberta_base_512_cls.npy",
        ),
    ),
}


BUILTIN_DATASET_BUILDERS = {
    "cora": partial(Planetoid, name="cora"),
    "citeseer": partial(Planetoid, name="citeseer"),
    "pubmed": partial(Planetoid, name="pubmed"),
    "facebook": partial(KarateClub, name="facebook"),
    "lastfm": partial(KarateClub, name="lastfm", transform=FilterTopClass(10)),
}


_LOOKUP_TO_CANONICAL = {
    "".join(ch for ch in canonical_name.lower() if ch.isalnum()): canonical_name
    for canonical_name in DATASET_SPECS
}


#
# Dataset-name resolution
#
def _require_dataset_input(dataset: Any) -> str:
    if not isinstance(dataset, str):
        raise ValueError(f"dataset must be a string, got {type(dataset).__name__}")

    normalized = dataset.strip()
    if normalized == "":
        raise ValueError("dataset must be non-empty")
    return normalized


def _normalize_lookup_key(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


#
# Split transform construction
#
def build_split_transform(val_ratio, test_ratio):
    split_kwargs = dict(split="train_rest", num_val=val_ratio, num_test=test_ratio)
    if RandomNodeSplit is not None:
        return RandomNodeSplit(**split_kwargs)
    if AddTrainValTestMask is not None:
        return AddTrainValTestMask(**split_kwargs)
    raise ImportError(
        "Could not find a train/val/test split transform in torch_geometric.transforms. "
        "Expected AddTrainValTestMask (old versions) or RandomNodeSplit (new versions)."
    )


def list_supported_datasets() -> list[str]:
    return sorted(DATASET_SPECS, key=str.casefold)


def resolve_dataset_name(dataset: str) -> str:
    requested = _require_dataset_input(dataset)
    try:
        return _LOOKUP_TO_CANONICAL[_normalize_lookup_key(requested)]
    except KeyError as exc:
        raise ValueError(
            f'Unknown dataset "{requested}". Available datasets: {list_supported_datasets()}'
        ) from exc


def is_dataset_supported(dataset: str) -> bool:
    try:
        resolve_dataset_name(dataset)
    except ValueError:
        return False
    return True


def _load_builtin_raw_dataset(spec: DatasetSpec, data_dir: str | Path) -> RawDatasetBundle:
    dataset_name = spec.canonical_name
    dataset = BUILTIN_DATASET_BUILDERS[dataset_name](root=os.path.join(str(data_dir), dataset_name))
    return RawDatasetBundle(data=dataset[0])


@contextmanager
def _torch_load_compat_context():
    original_torch_load = torch.load

    def compat_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = compat_torch_load
    try:
        yield
    finally:
        torch.load = original_torch_load


def _dataset_root(data_dir: str | Path, canonical_name: str) -> Path:
    return Path(data_dir) / canonical_name


def _hf_file_url(repo_id: str, repo_relative_path: str) -> str:
    normalized_path = str(repo_relative_path).lstrip("/")
    return HF_DATASET_URL_TEMPLATE.format(
        repo_id=repo_id,
        path=quote(normalized_path, safe="/"),
    )


def _download_to_cache(url: str, destination: Path) -> Path:
    if destination.exists():
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    request = urllib.request.Request(url, headers={"User-Agent": HTTP_USER_AGENT})

    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response, open(
                tmp_path, "wb"
            ) as handle:
                while True:
                    chunk = response.read(DOWNLOAD_CHUNK_SIZE_BYTES)
                    if not chunk:
                        break
                    handle.write(chunk)
            os.replace(tmp_path, destination)
            return destination
        except urllib.error.HTTPError:
            if tmp_path.exists():
                tmp_path.unlink()
            raise
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt >= DOWNLOAD_ATTEMPTS:
                raise
            time.sleep(float(attempt))

    return destination


def _ensure_hf_cached_file(
    root: Path,
    *,
    repo_id: str,
    repo_relative_path: str,
    cached_name: str,
) -> Path:
    cache_path = root / "raw" / cached_name
    if cache_path.exists():
        return cache_path
    return _download_to_cache(_hf_file_url(repo_id, repo_relative_path), cache_path)


def _load_npy_feature_matrix(path: Path, *, dataset_name: str) -> np.ndarray:
    feature_matrix = np.load(path)
    if feature_matrix.ndim != 2:
        raise ValueError(
            f"{dataset_name} feature matrix must be 2D, got shape {tuple(feature_matrix.shape)}"
        )
    return np.asarray(feature_matrix)


def _set_max_csv_field_size_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _parse_neighbor_list(
    raw_value: str,
    *,
    dataset_name: str,
    row_index: int,
) -> list[int]:
    if raw_value is None:
        raise ValueError(f"{dataset_name} row {row_index} is missing neighbour data")

    text = raw_value.strip()
    if text == "":
        return []

    try:
        values = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{dataset_name} row {row_index} has invalid neighbour JSON: {text!r}"
        ) from exc

    if not isinstance(values, list):
        raise ValueError(f"{dataset_name} row {row_index} neighbour field must decode to a list")

    neighbors: list[int] = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError(f"{dataset_name} row {row_index} neighbour entries must be integers")
        try:
            neighbors.append(int(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{dataset_name} row {row_index} neighbour entry {value!r} is not an integer"
            ) from exc
    return neighbors


def _validate_required_csv_fields(
    fieldnames: set[str],
    *,
    dataset_name: str,
    required_fields: set[str],
) -> None:
    missing_fields = required_fields - fieldnames
    if missing_fields:
        raise ValueError(
            f"{dataset_name} CSV is missing required columns: {sorted(missing_fields)}"
        )


def _build_edge_index_array(edge_sources: list[int], edge_targets: list[int]) -> np.ndarray:
    if not edge_sources:
        return np.empty((2, 0), dtype=np.int64)
    return np.vstack(
        (
            np.asarray(edge_sources, dtype=np.int64),
            np.asarray(edge_targets, dtype=np.int64),
        )
    )


def _read_text_attributed_graph_table(
    *,
    dataset_name: str,
    csv_path: Path,
    num_nodes: int,
    node_id_column: str,
    label_column: str,
    neighbor_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    label_vector = np.empty(num_nodes, dtype=np.int64)
    seen_node_ids = np.zeros(num_nodes, dtype=bool)
    source_nodes: list[int] = []
    target_nodes: list[int] = []
    row_count = 0

    _set_max_csv_field_size_limit()

    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _validate_required_csv_fields(
            set(reader.fieldnames or ()),
            dataset_name=dataset_name,
            required_fields={node_id_column, label_column, neighbor_column},
        )

        for row_count, row in enumerate(reader, start=1):
            try:
                node_id = int(row[node_id_column])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{dataset_name} row {row_count} has invalid node id {row[node_id_column]!r}"
                ) from exc

            if node_id < 0 or node_id >= num_nodes:
                raise ValueError(
                    f"{dataset_name} row {row_count} has node id {node_id} outside feature matrix range [0, {num_nodes - 1}]"
                )
            if seen_node_ids[node_id]:
                raise ValueError(f"{dataset_name} CSV contains duplicate node id {node_id}")
            seen_node_ids[node_id] = True

            try:
                label_vector[node_id] = int(row[label_column])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{dataset_name} row {row_count} has invalid label {row[label_column]!r}"
                ) from exc

            neighbors = _parse_neighbor_list(
                row[neighbor_column],
                dataset_name=dataset_name,
                row_index=row_count,
            )
            for neighbor in neighbors:
                if neighbor < 0 or neighbor >= num_nodes:
                    raise ValueError(
                        f"{dataset_name} row {row_count} contains neighbour {neighbor} outside feature matrix range [0, {num_nodes - 1}]"
                    )
            if neighbors:
                source_nodes.extend([node_id] * len(neighbors))
                target_nodes.extend(neighbors)

    if row_count != num_nodes:
        raise ValueError(
            f"{dataset_name} feature matrix has {num_nodes} rows but CSV contains {row_count} rows"
        )
    if not bool(seen_node_ids.all()):
        missing_node_ids = np.flatnonzero(~seen_node_ids)
        preview = missing_node_ids[:10].tolist()
        raise ValueError(
            f"{dataset_name} CSV node ids must cover 0..{num_nodes - 1}; missing ids begin with {preview}"
        )

    return label_vector, _build_edge_index_array(source_nodes, target_nodes)


def _load_hf_csv_npy_graph_dataset(spec: DatasetSpec, data_dir: str | Path) -> RawDatasetBundle:
    root = _dataset_root(data_dir, spec.canonical_name)
    repo_id = str(spec.loader_target)
    loader_kwargs = spec.loader_kwargs

    csv_path = _ensure_hf_cached_file(
        root,
        repo_id=repo_id,
        repo_relative_path=str(loader_kwargs["csv_path"]),
        cached_name=str(loader_kwargs["cached_csv_name"]),
    )
    feature_path = _ensure_hf_cached_file(
        root,
        repo_id=repo_id,
        repo_relative_path=str(loader_kwargs["feature_path"]),
        cached_name=str(loader_kwargs["cached_feature_name"]),
    )

    feature_matrix = _load_npy_feature_matrix(feature_path, dataset_name=spec.canonical_name)
    num_nodes = int(feature_matrix.shape[0])
    label_vector, graph_topology = _read_text_attributed_graph_table(
        dataset_name=spec.canonical_name,
        csv_path=csv_path,
        num_nodes=num_nodes,
        node_id_column=str(loader_kwargs["node_id_column"]),
        label_column=str(loader_kwargs["label_column"]),
        neighbor_column=str(loader_kwargs["neighbor_column"]),
    )

    data = Data(
        x=torch.from_numpy(feature_matrix),
        y=torch.from_numpy(label_vector),
        edge_index=torch.from_numpy(graph_topology),
    )
    data.num_nodes = num_nodes
    return RawDatasetBundle(data=data)


def _resolve_ogb_nodeprop_dataset_class():
    os.environ.setdefault("OUTDATED_IGNORE", "1")
    from ogb.nodeproppred import PygNodePropPredDataset

    return PygNodePropPredDataset


def _ensure_ogb_download_is_non_interactive() -> None:
    os.environ.setdefault("OUTDATED_IGNORE", "1")
    if "outdated" not in sys.modules:
        outdated_stub = types.ModuleType("outdated")
        outdated_stub.check_outdated = lambda *args, **kwargs: (False, None)
        sys.modules["outdated"] = outdated_stub

    import ogb.nodeproppred.dataset_pyg as dataset_pyg_module

    dataset_pyg_module.decide_download = lambda url: True


def _load_ogb_nodeprop_dataset(spec: DatasetSpec, data_dir: str | Path) -> RawDatasetBundle:
    _ensure_ogb_download_is_non_interactive()
    dataset_cls = _resolve_ogb_nodeprop_dataset_class()

    with _torch_load_compat_context():
        dataset = dataset_cls(
            name=str(spec.loader_target),
            root=str(_dataset_root(data_dir, spec.canonical_name)),
        )
    return RawDatasetBundle(data=dataset[0])


def _load_hf_ogb_feature_swap_dataset(spec: DatasetSpec, data_dir: str | Path) -> RawDatasetBundle:
    root = _dataset_root(data_dir, spec.canonical_name)
    loader_kwargs = spec.loader_kwargs
    feature_path = _ensure_hf_cached_file(
        root,
        repo_id=str(loader_kwargs["repo_id"]),
        repo_relative_path=str(loader_kwargs["feature_path"]),
        cached_name=str(loader_kwargs["cached_feature_name"]),
    )
    feature_matrix = _load_npy_feature_matrix(feature_path, dataset_name=spec.canonical_name)

    base_bundle = _load_ogb_nodeprop_dataset(spec, data_dir)
    num_nodes = int(base_bundle.data.x.size(0))
    if int(feature_matrix.shape[0]) != num_nodes:
        raise ValueError(
            f"{spec.canonical_name} feature matrix row count {feature_matrix.shape[0]} does not match OGB graph node count {num_nodes}"
        )

    base_bundle.data.x = torch.from_numpy(feature_matrix)
    return base_bundle


_RAW_DATASET_LOADERS: dict[str, Callable[[DatasetSpec, str | Path], RawDatasetBundle]] = {
    LOADER_BUILTIN_PLANETOID: _load_builtin_raw_dataset,
    LOADER_BUILTIN_KARATECLUB: _load_builtin_raw_dataset,
    LOADER_OGB_NODEPROP: _load_ogb_nodeprop_dataset,
    LOADER_HF_OGB_FEATURE_SWAP: _load_hf_ogb_feature_swap_dataset,
    LOADER_HF_CSV_NPY_GRAPH: _load_hf_csv_npy_graph_dataset,
}


def _load_raw_dataset(spec: DatasetSpec, data_dir: str | Path) -> RawDatasetBundle:
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    try:
        loader = _RAW_DATASET_LOADERS[spec.loader_kind]
    except KeyError as exc:
        raise ValueError(f"Unsupported raw loader kind: {spec.loader_kind}") from exc
    return loader(spec, data_dir)


def _validate_random_split_ratios(val_ratio: float, test_ratio: float) -> None:
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError(f"val_ratio must satisfy 0 <= val_ratio < 1, got {val_ratio}")
    if not (0.0 <= test_ratio < 1.0):
        raise ValueError(f"test_ratio must satisfy 0 <= test_ratio < 1, got {test_ratio}")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be strictly smaller than 1 for random splits")


# graph standardization
def _coerce_feature_matrix(x: torch.Tensor) -> torch.Tensor:
    sparse_layouts = {
        layout
        for layout in (
            getattr(torch, "sparse_coo", None),
            getattr(torch, "sparse_csr", None),
            getattr(torch, "sparse_csc", None),
            getattr(torch, "sparse_bsr", None),
            getattr(torch, "sparse_bsc", None),
        )
        if layout is not None
    }
    if x.is_sparse or x.layout in sparse_layouts:
        x = x.to_dense()
    return x.to(torch.float)


def _looks_like_one_hot(y: torch.Tensor) -> bool:
    if y.ndim != 2 or y.numel() == 0:
        return False
    if y.is_floating_point():
        if not (bool((y >= 0).all().item()) and bool((y <= 1).all().item())):
            return False
        row_sum = y.sum(dim=1)
        return bool(torch.isclose(row_sum, torch.ones_like(row_sum)).all().item())
    if y.dtype == torch.bool:
        row_sum = y.to(torch.int).sum(dim=1)
    else:
        if not bool(((y == 0) | (y == 1)).all().item()):
            return False
        row_sum = y.sum(dim=1)
    return bool((row_sum == 1).all().item())


def _remap_labels_to_contiguous_range(y: torch.Tensor) -> torch.Tensor:
    unique_labels = torch.unique(y, sorted=True)
    expected = torch.arange(unique_labels.numel(), device=unique_labels.device)
    if torch.equal(unique_labels, expected):
        return y

    remapped = torch.empty_like(y)
    for new_label, old_label in enumerate(unique_labels.tolist()):
        remapped[y == old_label] = new_label
    return remapped


def _coerce_label_vector(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    unlabeled_mask = torch.zeros(0, dtype=torch.bool)

    if y.ndim == 2 and y.size(-1) == 1:
        y = y.view(-1)
    elif y.ndim == 2 and y.size(-1) > 1:
        if _looks_like_one_hot(y):
            y = y.argmax(dim=1)
        else:
            raise ValueError(
                f"Expected one-dimensional class index labels, got shape {tuple(y.shape)}"
            )

    if y.ndim != 1:
        raise ValueError(f"Expected one-dimensional class index labels, got shape {tuple(y.shape)}")
    if y.numel() == 0:
        raise ValueError("y must not be empty")

    if y.is_floating_point():
        unlabeled_mask = torch.isnan(y)
        if bool(unlabeled_mask.all().item()):
            raise ValueError("y contains no finite class labels")
        if bool(unlabeled_mask.any().item()):
            finite_y = y[~unlabeled_mask]
            if not torch.allclose(finite_y, finite_y.round()):
                raise ValueError("y must contain integer-like class indices when finite")
            y = finite_y.round().to(torch.long)
        else:
            if not torch.allclose(y, y.round()):
                raise ValueError("y must contain integer-like class indices")
            y = y.round().to(torch.long)
    else:
        y = y.to(torch.long)

    if int(y.min().item()) < 0:
        raise ValueError("y must contain non-negative class indices")

    y = _remap_labels_to_contiguous_range(y)

    if bool(unlabeled_mask.any().item()):
        y_with_unlabeled = torch.zeros(unlabeled_mask.size(0), dtype=torch.long, device=y.device)
        y_with_unlabeled[~unlabeled_mask] = y
        return y_with_unlabeled, unlabeled_mask.to(torch.bool)

    return y, torch.zeros(y.size(0), dtype=torch.bool, device=y.device)


def _extract_edge_index(data: Data) -> torch.Tensor:
    edge_index = getattr(data, "edge_index", None)
    if edge_index is None:
        raise ValueError("Raw dataset must provide edge_index before standardization")
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")
    return edge_index.to(torch.long)


def _canonicalize_graph_topology(data: Data, *, num_nodes: int) -> torch.Tensor:
    graph_topology = _extract_edge_index(data)
    graph_topology = to_undirected(graph_topology, num_nodes=num_nodes)
    return coalesce(graph_topology, num_nodes=num_nodes)


def _sample_transductive_split(data: Data, *, val_ratio: float, test_ratio: float) -> Data:
    return build_split_transform(val_ratio, test_ratio)(data)


def _validate_transductive_split_against_unlabeled_targets(
    unlabeled_mask: torch.Tensor,
    data: Data,
) -> None:
    if not bool(unlabeled_mask.any().item()):
        return

    supervised_partition = data.train_mask | data.val_mask | data.test_mask
    if bool((unlabeled_mask.to(supervised_partition.device) & supervised_partition).any().item()):
        raise ValueError("Split masks include unlabeled (NaN) targets, which is unsupported")


def _finalize_benchmark_graph(
    data: Data,
    *,
    spec: DatasetSpec,
    num_nodes: int,
    num_classes: int,
    data_range: tuple[float, float] | None,
) -> Data:
    data = ToSparseTensor()(data)
    if data_range is not None:
        data = Normalize(*data_range)(data)

    data.train_mask = data.train_mask.to(torch.bool)
    data.val_mask = data.val_mask.to(torch.bool)
    data.test_mask = data.test_mask.to(torch.bool)
    data.name = spec.canonical_name
    data.num_classes = num_classes
    data.num_nodes = num_nodes
    return data


def _build_standardized_benchmark_graph(
    spec: DatasetSpec,
    bundle: RawDatasetBundle,
    *,
    data_range: tuple[float, float] | None,
    val_ratio: float,
    test_ratio: float,
) -> Data:
    _validate_random_split_ratios(val_ratio, test_ratio)

    raw_data = bundle.data
    feature_matrix = _coerce_feature_matrix(raw_data.x)
    label_vector, unlabeled_mask = _coerce_label_vector(raw_data.y)

    if feature_matrix.ndim != 2:
        raise ValueError(f"x must have shape [N, F], got {tuple(feature_matrix.shape)}")

    num_nodes = int(feature_matrix.size(0))
    if label_vector.size(0) != num_nodes:
        raise ValueError("x and y must agree on the first dimension")

    graph_topology = _canonicalize_graph_topology(raw_data, num_nodes=num_nodes)
    data = Data(x=feature_matrix, y=label_vector, edge_index=graph_topology, num_nodes=num_nodes)
    data = _sample_transductive_split(data, val_ratio=val_ratio, test_ratio=test_ratio)
    _validate_transductive_split_against_unlabeled_targets(unlabeled_mask, data)

    return _finalize_benchmark_graph(
        data,
        spec=spec,
        num_nodes=num_nodes,
        num_classes=int(label_vector.max().item()) + 1,
        data_range=data_range,
    )


def _standardize_loaded_data(
    spec: DatasetSpec,
    bundle: RawDatasetBundle,
    *,
    data_range: tuple[float, float] | None,
    val_ratio: float,
    test_ratio: float,
) -> Data:
    return _build_standardized_benchmark_graph(
        spec,
        bundle,
        data_range=data_range,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )


def load_dataset(
    dataset: dict(help="name of the dataset", option="-d") = "cora",
    data_dir: dict(help="directory to store the dataset") = "./datasets",
    data_range: dict(help="min and max feature value", nargs=2, type=float) = (0, 1),
    val_ratio: dict(help="fraction of nodes used for validation") = 0.25,
    test_ratio: dict(help="fraction of nodes used for test") = 0.25,
):
    canonical_name = resolve_dataset_name(dataset)
    spec = DATASET_SPECS[canonical_name]
    raw_bundle = _load_raw_dataset(spec, data_dir)
    return _standardize_loaded_data(
        spec,
        raw_bundle,
        data_range=data_range,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )


__all__ = [
    "is_dataset_supported",
    "list_supported_datasets",
    "load_dataset",
    "resolve_dataset_name",
]
