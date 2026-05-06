import csv
import gc
import json
import os
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

import datasets as datasets_module


CANONICAL_DATASETS = [
    "Books-Children",
    "Books-History",
    "cora",
    "citeseer",
    "facebook",
    "lastfm",
    "ogbn-arxiv",
    "ogbn-arxiv-TA",
    "pubmed",
]


def make_data(x, y, directed_edges, **kwargs):
    edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
    return Data(x=x, y=y, edge_index=edge_index, **kwargs)


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_npy(path: Path, array: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def test_supported_dataset_helpers_match_expected_nine_datasets():
    assert datasets_module.list_supported_datasets() == CANONICAL_DATASETS
    for name in CANONICAL_DATASETS:
        assert datasets_module.is_dataset_supported(name)


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("CORA", "cora"),
        ("CiteSeer", "citeseer"),
        ("PubMed", "pubmed"),
        ("FACE_BOOK", "facebook"),
        ("Last-FM", "lastfm"),
        ("ogbn_arxiv", "ogbn-arxiv"),
        ("OGBN-ARXIV-TA", "ogbn-arxiv-TA"),
        ("books_children", "Books-Children"),
        ("BOOKS-HISTORY", "Books-History"),
    ],
)
def test_resolve_dataset_name_accepts_form_compatible_inputs(raw_name, expected):
    assert datasets_module.resolve_dataset_name(raw_name) == expected


@pytest.mark.parametrize("raw_name", ["Arxiv", "Children", "History"])
def test_resolve_dataset_name_rejects_semantic_aliases(raw_name):
    with pytest.raises(ValueError, match="Unknown dataset"):
        datasets_module.resolve_dataset_name(raw_name)
    assert not datasets_module.is_dataset_supported(raw_name)


def test_load_dataset_routes_all_supported_datasets_through_unified_pipeline():
    seen_raw = set()
    seen_standardize = set()

    def fake_raw_dataset(spec, data_dir):
        del data_dir
        seen_raw.add(spec.canonical_name)
        return datasets_module.RawDatasetBundle(data=Data())

    def fake_standardize(spec, bundle, *, data_range, val_ratio, test_ratio):
        del bundle, data_range, val_ratio, test_ratio
        seen_standardize.add(spec.canonical_name)
        return {"source": "unified", "dataset_name": spec.canonical_name}

    with mock.patch.object(
        datasets_module,
        "_load_raw_dataset",
        side_effect=fake_raw_dataset,
    ), mock.patch.object(
        datasets_module,
        "_standardize_loaded_data",
        side_effect=fake_standardize,
    ):
        cases = {
            "cora": "cora",
            "CiteSeer": "citeseer",
            "PubMed": "pubmed",
            "FACE_BOOK": "facebook",
            "Last-FM": "lastfm",
            "ogbn_arxiv": "ogbn-arxiv",
            "OGBN-ARXIV-TA": "ogbn-arxiv-TA",
            "books_children": "Books-Children",
            "BOOKS-HISTORY": "Books-History",
        }
        for raw_name, expected in cases.items():
            loaded = datasets_module.load_dataset(
                raw_name,
                data_dir="./datasets",
                data_range=(0, 1),
                val_ratio=0.25,
                test_ratio=0.25,
            )
            assert loaded == {"source": "unified", "dataset_name": expected}

    assert seen_raw == set(CANONICAL_DATASETS)
    assert seen_standardize == set(CANONICAL_DATASETS)


@pytest.mark.skipif(
    os.environ.get("RUN_FULL_DATASET_LOAD_TEST") != "1",
    reason="Set RUN_FULL_DATASET_LOAD_TEST=1 to run real loading for every supported dataset.",
)
def test_real_load_all_supported_datasets():
    failures = []
    for name in CANONICAL_DATASETS:
        data = None
        try:
            data = datasets_module.load_dataset(name, data_dir=str(Path("datasets")))
            for attr in (
                "x",
                "y",
                "adj_t",
                "train_mask",
                "val_mask",
                "test_mask",
                "num_classes",
            ):
                assert hasattr(data, attr), f"Loaded data for {name} is missing `{attr}`."
        except Exception as exc:  # pragma: no cover
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
        finally:
            del data
            gc.collect()

    if failures:
        preview = "\n".join(failures[:20])
        suffix = ""
        if len(failures) > 20:
            suffix = f"\n... {len(failures) - 20} more failures omitted."
        pytest.fail(f"{len(failures)} dataset(s) failed real loading:\n{preview}{suffix}")


def test_lastfm_builder_retains_top_class_filter():
    builder = datasets_module.BUILTIN_DATASET_BUILDERS["lastfm"]
    transform = builder.keywords["transform"]

    assert isinstance(transform, datasets_module.FilterTopClass)
    assert transform.num_classes == 10


def test_hf_csv_npy_graph_loader_reorders_nodes_and_parses_neighbors(monkeypatch, tmp_path):
    csv_path = tmp_path / "Children" / "Children.csv"
    feature_path = tmp_path / "Children" / "Feature" / "Children_roberta_base_512_cls.npy"
    write_csv_rows(
        csv_path,
        ["category", "text", "label", "node_id", "neighbour"],
        [
            {
                "category": "c2",
                "text": "node 2",
                "label": 8,
                "node_id": 2,
                "neighbour": json.dumps([1]),
            },
            {
                "category": "c0",
                "text": "node 0",
                "label": 5,
                "node_id": 0,
                "neighbour": json.dumps([2, 1]),
            },
            {
                "category": "c1",
                "text": "node 1",
                "label": 5,
                "node_id": 1,
                "neighbour": json.dumps([]),
            },
        ],
    )
    write_npy(
        feature_path,
        np.asarray(
            [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]],
            dtype=np.float16,
        ),
    )

    monkeypatch.setattr(
        datasets_module,
        "_ensure_hf_cached_file",
        lambda root, *, repo_id, repo_relative_path, cached_name: (
            csv_path if repo_relative_path.endswith(".csv") else feature_path
        ),
    )

    bundle = datasets_module._load_hf_csv_npy_graph_dataset(
        datasets_module.DATASET_SPECS["Books-Children"],
        str(tmp_path / "cache"),
    )

    assert bundle.data.num_nodes == 3
    assert bundle.data.y.tolist() == [5, 5, 8]
    assert tuple(bundle.data.x.shape) == (3, 2)
    assert bundle.data.x.dtype == torch.float16
    assert bundle.data.edge_index.t().tolist() == [[2, 1], [0, 2], [0, 1]]


def test_hf_csv_npy_graph_loader_rejects_feature_csv_size_mismatch(monkeypatch, tmp_path):
    csv_path = tmp_path / "History" / "History.csv"
    feature_path = tmp_path / "History" / "Feature" / "History_roberta_base_512_cls.npy"
    write_csv_rows(
        csv_path,
        ["category", "price", "text", "label", "node_id", "neighbour"],
        [
            {
                "category": "history",
                "price": "$1",
                "text": "n0",
                "label": 0,
                "node_id": 0,
                "neighbour": json.dumps([1]),
            },
            {
                "category": "history",
                "price": "$1",
                "text": "n1",
                "label": 1,
                "node_id": 1,
                "neighbour": json.dumps([]),
            },
            {
                "category": "history",
                "price": "$1",
                "text": "n2",
                "label": 0,
                "node_id": 2,
                "neighbour": json.dumps([0]),
            },
        ],
    )
    write_npy(
        feature_path,
        np.asarray(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            dtype=np.float16,
        ),
    )

    monkeypatch.setattr(
        datasets_module,
        "_ensure_hf_cached_file",
        lambda root, *, repo_id, repo_relative_path, cached_name: (
            csv_path if repo_relative_path.endswith(".csv") else feature_path
        ),
    )

    with pytest.raises(ValueError, match="feature matrix has 4 rows but CSV contains 3 rows"):
        datasets_module._load_hf_csv_npy_graph_dataset(
            datasets_module.DATASET_SPECS["Books-History"],
            str(tmp_path / "cache"),
        )


def test_hf_ogb_feature_swap_loader_replaces_features_and_preserves_graph(
    monkeypatch,
    tmp_path,
):
    feature_path = tmp_path / "Arxiv" / "Feature" / "Arxiv_roberta_base_512_cls.npy"
    write_npy(feature_path, np.arange(4 * 768, dtype=np.float16).reshape(4, 768))

    raw_data = make_data(
        x=torch.randn(4, 3),
        y=torch.tensor([[0], [1], [0], [2]], dtype=torch.long),
        directed_edges=[(0, 1), (1, 2), (2, 3)],
    )

    monkeypatch.setattr(
        datasets_module,
        "_ensure_hf_cached_file",
        lambda root, *, repo_id, repo_relative_path, cached_name: feature_path,
    )
    monkeypatch.setattr(
        datasets_module,
        "_load_ogb_nodeprop_dataset",
        lambda spec, data_dir: datasets_module.RawDatasetBundle(data=raw_data),
    )

    bundle = datasets_module._load_hf_ogb_feature_swap_dataset(
        datasets_module.DATASET_SPECS["ogbn-arxiv-TA"],
        str(tmp_path),
    )

    assert tuple(bundle.data.x.shape) == (4, 768)
    assert bundle.data.y.tolist() == [[0], [1], [0], [2]]
    assert bundle.data.edge_index.t().tolist() == [[0, 1], [1, 2], [2, 3]]


def test_standardize_loaded_data_unifies_features_labels_edges_and_random_masks():
    raw_data = make_data(
        x=torch.tensor(
            [
                [1, 10, 5],
                [3, 10, 1],
                [2, 10, 1],
                [5, 10, 5],
                [4, 10, 3],
            ]
        ),
        y=torch.tensor([10, 30, 10, 20, 30], dtype=torch.long),
        directed_edges=[(0, 1), (1, 0), (1, 2), (2, 2), (3, 4)],
        train_mask=torch.ones(5, dtype=torch.bool),
        val_mask=torch.zeros(5, dtype=torch.bool),
        test_mask=torch.zeros(5, dtype=torch.bool),
    )
    original_train_mask = raw_data.train_mask.clone()

    torch.manual_seed(7)
    data = datasets_module._standardize_loaded_data(
        datasets_module.DATASET_SPECS["ogbn-arxiv"],
        datasets_module.RawDatasetBundle(data=raw_data),
        data_range=(0.0, 1.0),
        val_ratio=0.3,
        test_ratio=0.3,
    )

    expected_x = torch.tensor(
        [
            [0.00, 0.00, 1.00],
            [0.50, 0.00, 0.00],
            [0.25, 0.00, 0.00],
            [1.00, 0.00, 1.00],
            [0.75, 0.00, 0.50],
        ]
    )
    assert data.x.dtype == torch.float32
    assert torch.allclose(data.x, expected_x)
    assert data.y.tolist() == [0, 2, 0, 1, 2]
    assert data.name == "ogbn-arxiv"
    assert int(data.num_classes) == 3
    assert data.train_mask.dtype == torch.bool
    assert data.val_mask.dtype == torch.bool
    assert data.test_mask.dtype == torch.bool
    assert int(data.train_mask.sum()) == 1
    assert int(data.val_mask.sum()) == 2
    assert int(data.test_mask.sum()) == 2
    assert not torch.equal(data.train_mask, original_train_mask)

    row, col, _ = data.adj_t.coo()
    assert sorted(zip(row.tolist(), col.tolist())) == [
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 1),
        (2, 2),
        (3, 4),
        (4, 3),
    ]


def test_standardize_loaded_data_random_split_is_seed_reproducible():
    raw_data = make_data(
        x=torch.arange(40, dtype=torch.float32).reshape(20, 2),
        y=torch.tensor([idx % 3 for idx in range(20)], dtype=torch.long),
        directed_edges=[(idx, (idx + 1) % 20) for idx in range(20)],
    )
    bundle = datasets_module.RawDatasetBundle(data=raw_data)
    spec = datasets_module.DATASET_SPECS["Books-Children"]

    torch.manual_seed(123)
    data_a = datasets_module._standardize_loaded_data(
        spec,
        bundle,
        data_range=None,
        val_ratio=0.2,
        test_ratio=0.2,
    )
    torch.manual_seed(123)
    data_b = datasets_module._standardize_loaded_data(
        spec,
        bundle,
        data_range=None,
        val_ratio=0.2,
        test_ratio=0.2,
    )
    torch.manual_seed(456)
    data_c = datasets_module._standardize_loaded_data(
        spec,
        bundle,
        data_range=None,
        val_ratio=0.2,
        test_ratio=0.2,
    )

    assert torch.equal(data_a.train_mask, data_b.train_mask)
    assert torch.equal(data_a.val_mask, data_b.val_mask)
    assert torch.equal(data_a.test_mask, data_b.test_mask)
    assert int(data_a.train_mask.sum()) == 12
    assert int(data_a.val_mask.sum()) == 4
    assert int(data_a.test_mask.sum()) == 4
    assert (
        not torch.equal(data_a.train_mask, data_c.train_mask)
        or not torch.equal(data_a.val_mask, data_c.val_mask)
        or not torch.equal(data_a.test_mask, data_c.test_mask)
    )


def test_normalize_preserves_constant_columns_and_fills_with_lower_bound():
    data = Data(
        x=torch.tensor(
            [
                [1.0, 5.0, 2.0],
                [3.0, 5.0, 4.0],
                [2.0, 5.0, 6.0],
            ]
        )
    )

    normalized = datasets_module.Normalize(-1.0, 1.0)(data)

    expected_x = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, 0.0],
            [0.0, -1.0, 1.0],
        ]
    )
    assert tuple(normalized.x.shape) == (3, 3)
    assert torch.allclose(normalized.x, expected_x)
