from __future__ import annotations

import multiprocessing as mp
import os
from queue import Empty
import shutil
import time
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import ToSparseTensor

from artificial_node_feature_generator import get_provider, register_provider, rewrite_features
from artificial_node_feature_generator.cache import cache_root, feature_cache_dir, feature_cache_path
from artificial_node_feature_generator.graph import graph_fingerprint
from artificial_node_feature_generator.providers import BaseFeatureProvider
from artificial_node_feature_generator.types import ProviderOutput


def make_data():
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 0],
            [1, 0, 2, 1, 3, 2, 0, 3],
        ],
        dtype=torch.long,
    )
    x = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    return Data(x=x, edge_index=edge_index, num_nodes=4)


def make_variant_data():
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 0, 2],
            [1, 2, 3, 0, 2, 0],
        ],
        dtype=torch.long,
    )
    x = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    return Data(x=x, edge_index=edge_index, num_nodes=4)


class DuckData:
    def __init__(self, x, edge_index, num_nodes):
        self.x = x
        self.edge_index = edge_index
        self.num_nodes = num_nodes


def clear_cache():
    shutil.rmtree(cache_root(), ignore_errors=True)


class SlowCountingProvider(BaseFeatureProvider):
    name = "slow_counting"
    source = "generated"
    cacheable = True

    def build(self, data, *, params: dict, seed: int | None = None) -> ProviderOutput:
        marker_path = Path(params["marker_path"])
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        with marker_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{os.getpid()}\n")
            handle.flush()
            os.fsync(handle.fileno())
        time.sleep(float(params.get("delay_seconds", 0.3)))
        feature_dim = int(params["feature_dim"])
        return ProviderOutput(
            features=torch.full((data.num_nodes, feature_dim), float(seed or 0), dtype=data.x.dtype),
            source=self.source,
            cacheable=self.cacheable,
        )


def _rewrite_features_in_child(params: dict, seed: int, result_queue) -> None:
    try:
        rewritten = rewrite_features(make_data(), "slow_counting", params=params, seed=seed)
        result_queue.put(
            {
                "ok": True,
                "shape": tuple(rewritten.x.shape),
                "sum": float(rewritten.x.sum().item()),
            }
        )
    except Exception as exc:  # pragma: no cover - exercised through subprocess failure path
        result_queue.put({"ok": False, "error": repr(exc)})


def test_raw_returns_copy_and_preserves_original():
    data = make_data()
    rewritten = rewrite_features(data, "raw")
    assert rewritten is not data
    assert torch.equal(rewritten.x, data.x)
    rewritten.x[0, 0] = 99.0
    assert data.x[0, 0].item() == 1.0


def test_onehot_is_unsupported():
    with pytest.raises(ValueError, match='Unsupported feature "onehot"'):
        rewrite_features(make_data(), "onehot", params={"feature_dim": 4})


def test_random_normal_uses_seed_and_feature_dim():
    data = make_data()
    first = rewrite_features(data, "random_normal", params={"feature_dim": 5}, seed=7)
    second = rewrite_features(data, "random_normal", params={"feature_dim": 5}, seed=7)
    third = rewrite_features(data, "random_normal", params={"feature_dim": 5}, seed=8)
    assert first.x.shape == (4, 5)
    assert torch.allclose(first.x, second.x)
    assert not torch.allclose(first.x, third.x)


def test_random_normal_scale_one_is_noop():
    data = make_data()
    base = rewrite_features(data, "random_normal", params={"feature_dim": 5}, seed=7)
    scaled = rewrite_features(data, "random_normal", params={"feature_dim": 5}, seed=7, scale=1.0)
    assert torch.allclose(base.x, scaled.x)


def test_random_normal_scale_multiplies_generated_features():
    data = make_data()
    base = rewrite_features(data, "random_normal", params={"feature_dim": 5}, seed=7)
    scaled = rewrite_features(data, "random_normal", params={"feature_dim": 5}, seed=7, scale=2.5)
    assert torch.allclose(scaled.x, base.x * 2.5)


def test_random_signed_onehot_uses_seed_and_signed_onehot_structure():
    data = make_data()
    first = rewrite_features(data, "random_signed_onehot", params={"feature_dim": 5}, seed=7)
    second = rewrite_features(data, "random_signed_onehot", params={"feature_dim": 5}, seed=7)
    third = rewrite_features(data, "random_signed_onehot", params={"feature_dim": 5}, seed=8)
    assert first.x.shape == (4, 5)
    assert torch.allclose(first.x, second.x)
    assert not torch.allclose(first.x, third.x)
    assert torch.equal(torch.count_nonzero(first.x, dim=1), torch.ones(4, dtype=torch.long))
    assert torch.allclose(first.x.abs().sum(dim=1), torch.ones(4, dtype=first.x.dtype))
    assert torch.all((first.x == 0) | (first.x == 1) | (first.x == -1))


def test_node_degree_matches_random_frozen_embedding_semantics():
    data = make_data()
    first = rewrite_features(data, "node_degree", params={"feature_dim": 6}, seed=13)
    second = rewrite_features(data, "node_degree", params={"feature_dim": 6}, seed=13)
    third = rewrite_features(data, "node_degree", params={"feature_dim": 6}, seed=14)
    assert first.x.shape == (4, 6)
    assert torch.allclose(first.x, second.x)
    assert not torch.allclose(first.x, third.x)


def test_degree_buckets_map_to_feature_dim_and_support_adj_t():
    data_edge = make_data()
    data_adj = ToSparseTensor()(make_data())
    range_edge = rewrite_features(
        data_edge,
        "degree_bucket_range",
        params={"feature_dim": 4, "num_buckets": 3, "range_max": 6},
        seed=5,
    )
    range_adj = rewrite_features(
        data_adj,
        "degree_bucket_range",
        params={"feature_dim": 4, "num_buckets": 3, "range_max": 6},
        seed=5,
    )
    dist_first = rewrite_features(
        data_edge,
        "degree_bucket_distribution",
        params={"feature_dim": 4, "num_buckets": 4},
        seed=5,
    )
    dist_second = rewrite_features(
        data_edge,
        "degree_bucket_distribution",
        params={"feature_dim": 4, "num_buckets": 4},
        seed=5,
    )
    assert range_edge.x.shape == (4, 4)
    assert range_adj.x.shape == (4, 4)
    assert torch.allclose(range_edge.x, range_adj.x)
    assert dist_first.x.shape == (4, 4)
    assert torch.allclose(dist_first.x, dist_second.x)


def test_pagerank_repeats_same_scalar_across_all_columns():
    data = make_data()
    rewritten = rewrite_features(data, "pagerank", params={"feature_dim": 3}, seed=9)
    assert rewritten.x.shape == (4, 3)
    assert torch.allclose(rewritten.x[:, 0], rewritten.x[:, 1])
    assert torch.allclose(rewritten.x[:, 1], rewritten.x[:, 2])


def test_eigen_and_eigen_norm_still_work():
    data = make_data()
    eigen = rewrite_features(data, "eigen", params={"feature_dim": 2}, seed=1)
    eigen_norm = rewrite_features(data, "eigen_norm", params={"feature_dim": 2}, seed=1)
    assert eigen.x.shape == (4, 2)
    assert eigen_norm.x.shape == (4, 2)


def test_eigen_cache_reuses_payload_across_seeds():
    clear_cache()
    data = make_data()
    graph_key = graph_fingerprint(data)
    params = {"feature_dim": 2}
    shared_path = feature_cache_path("eigen", None, params, graph_key)
    seed_path_a = feature_cache_path("eigen", 11, params, graph_key)
    seed_path_b = feature_cache_path("eigen", 12, params, graph_key)
    first = rewrite_features(data, "eigen", params=params, seed=11)
    first_mtime = shared_path.stat().st_mtime_ns
    second = rewrite_features(data, "eigen", params=params, seed=12)
    second_mtime = shared_path.stat().st_mtime_ns
    assert shared_path.exists()
    assert not seed_path_a.exists()
    assert not seed_path_b.exists()
    assert torch.allclose(first.x, second.x)
    assert first_mtime == second_mtime


def test_eigen_norm_cache_reuses_payload_across_seeds():
    clear_cache()
    data = make_data()
    graph_key = graph_fingerprint(data)
    params = {"feature_dim": 2}
    shared_path = feature_cache_path("eigen_norm", None, params, graph_key)
    seed_path_a = feature_cache_path("eigen_norm", 11, params, graph_key)
    seed_path_b = feature_cache_path("eigen_norm", 12, params, graph_key)
    first = rewrite_features(data, "eigen_norm", params=params, seed=11)
    first_mtime = shared_path.stat().st_mtime_ns
    second = rewrite_features(data, "eigen_norm", params=params, seed=12)
    second_mtime = shared_path.stat().st_mtime_ns
    assert shared_path.exists()
    assert not seed_path_a.exists()
    assert not seed_path_b.exists()
    assert torch.allclose(first.x, second.x)
    assert first_mtime == second_mtime


def test_internal_cache_hits_for_same_graph_feature_params_and_seed():
    clear_cache()
    data = make_data()
    graph_key = graph_fingerprint(data)
    params = {"feature_dim": 3}
    cache_file = feature_cache_path("pagerank", 11, params, graph_key)
    assert not cache_file.exists()
    first = rewrite_features(data, "pagerank", params=params, seed=11)
    first_mtime = cache_file.stat().st_mtime_ns
    second = rewrite_features(data, "pagerank", params=params, seed=11)
    second_mtime = cache_file.stat().st_mtime_ns
    assert torch.allclose(first.x, second.x)
    assert first_mtime == second_mtime


def test_internal_cache_serializes_concurrent_cache_publication(tmp_path):
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("This test requires the fork start method.")

    clear_cache()
    register_provider("slow_counting", SlowCountingProvider)
    marker_path = tmp_path / "slow_counting_builds.txt"
    params = {
        "feature_dim": 3,
        "marker_path": marker_path,
        "delay_seconds": 0.4,
    }
    seed = 77
    graph_key = graph_fingerprint(make_data())
    cache_file = feature_cache_path("slow_counting", seed, params, graph_key)
    ctx = mp.get_context("fork")
    result_queue = ctx.Queue()
    processes = [
        ctx.Process(target=_rewrite_features_in_child, args=(params, seed, result_queue)),
        ctx.Process(target=_rewrite_features_in_child, args=(params, seed, result_queue)),
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)

    results = []
    for _ in processes:
        try:
            results.append(result_queue.get(timeout=5))
        except Empty as exc:  # pragma: no cover - indicates subprocess failed to report back
            pytest.fail(f"Timed out waiting for subprocess result: {exc}")

    for process in processes:
        assert process.exitcode == 0
    assert all(result["ok"] for result in results), results
    assert cache_file.exists()
    assert all(result["shape"] == (4, 3) for result in results)
    marker_lines = marker_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(marker_lines) == 1


def test_internal_cache_reuses_unscaled_payload_across_scales():
    clear_cache()
    data = make_data()
    graph_key = graph_fingerprint(data)
    params = {"feature_dim": 3}
    cache_file = feature_cache_path("pagerank", 11, params, graph_key)
    base = rewrite_features(data, "pagerank", params=params, seed=11)
    base_mtime = cache_file.stat().st_mtime_ns
    scaled = rewrite_features(data, "pagerank", params=params, seed=11, scale=3.0)
    scaled_mtime = cache_file.stat().st_mtime_ns
    assert torch.allclose(scaled.x, base.x * 3.0)
    assert base_mtime == scaled_mtime


def test_internal_cache_miss_when_seed_changes():
    clear_cache()
    data = make_data()
    graph_key = graph_fingerprint(data)
    params = {"feature_dim": 6}
    first_path = feature_cache_path("node_degree", 11, params, graph_key)
    second_path = feature_cache_path("node_degree", 12, params, graph_key)
    rewrite_features(data, "node_degree", params=params, seed=11)
    rewrite_features(data, "node_degree", params=params, seed=12)
    assert first_path.exists()
    assert second_path.exists()
    assert first_path != second_path


def test_internal_cache_miss_when_graph_changes():
    clear_cache()
    params = {"feature_dim": 3}
    data_a = make_data()
    data_b = make_variant_data()
    key_a = graph_fingerprint(data_a)
    key_b = graph_fingerprint(data_b)
    path_a = feature_cache_path("pagerank", 11, params, key_a)
    path_b = feature_cache_path("pagerank", 11, params, key_b)
    rewrite_features(data_a, "pagerank", params=params, seed=11)
    rewrite_features(data_b, "pagerank", params=params, seed=11)
    assert path_a.exists()
    assert path_b.exists()
    assert path_a != path_b


def test_deepwalk_generates_and_reuses_internal_cache():
    clear_cache()
    data = make_data()
    params = {
        "feature_dim": 4,
        "walk_length": 4,
        "number_walks": 2,
        "window_size": 2,
        "workers": 1,
        "undirected": True,
    }
    graph_key = graph_fingerprint(data)
    run_dir = feature_cache_dir("deepwalk", 7, params, graph_key)
    rewritten = rewrite_features(data, "deepwalk", params=params, seed=7)
    assert rewritten.x.shape == (4, 4)
    assert (run_dir / "graph.edgelist").exists()
    assert (run_dir / "embeddings.word2vec").exists()
    assert (run_dir / "features.pt").exists()
    first_mtime = (run_dir / "embeddings.word2vec").stat().st_mtime_ns
    cached = rewrite_features(data, "deepwalk", params=params, seed=7)
    second_mtime = (run_dir / "embeddings.word2vec").stat().st_mtime_ns
    assert torch.allclose(rewritten.x, cached.x)
    assert first_mtime == second_mtime


def test_deepwalk_regenerates_when_feature_dim_changes():
    clear_cache()
    data = make_data()
    params_small = {
        "feature_dim": 3,
        "walk_length": 4,
        "number_walks": 2,
        "window_size": 2,
        "workers": 1,
        "undirected": True,
    }
    params_large = {
        "feature_dim": 5,
        "walk_length": 4,
        "number_walks": 2,
        "window_size": 2,
        "workers": 1,
        "undirected": True,
    }
    graph_key = graph_fingerprint(data)
    dir_small = feature_cache_dir("deepwalk", 4, params_small, graph_key)
    dir_large = feature_cache_dir("deepwalk", 4, params_large, graph_key)
    small = rewrite_features(data, "deepwalk", params=params_small, seed=4)
    large = rewrite_features(data, "deepwalk", params=params_large, seed=4)
    assert small.x.shape == (4, 3)
    assert large.x.shape == (4, 5)
    assert dir_small.exists()
    assert dir_large.exists()
    assert dir_small != dir_large


def test_deepwalk_regenerates_when_parameters_change():
    clear_cache()
    data = make_data()
    params_short = {
        "feature_dim": 4,
        "walk_length": 4,
        "number_walks": 2,
        "window_size": 2,
        "workers": 1,
        "undirected": True,
    }
    params_long = {
        "feature_dim": 4,
        "walk_length": 6,
        "number_walks": 2,
        "window_size": 2,
        "workers": 1,
        "undirected": True,
    }
    graph_key = graph_fingerprint(data)
    dir_short = feature_cache_dir("deepwalk", 4, params_short, graph_key)
    dir_long = feature_cache_dir("deepwalk", 4, params_long, graph_key)
    short = rewrite_features(data, "deepwalk", params=params_short, seed=4)
    long = rewrite_features(data, "deepwalk", params=params_long, seed=4)
    assert short.x.shape == (4, 4)
    assert long.x.shape == (4, 4)
    assert dir_short.exists()
    assert dir_long.exists()
    assert dir_short != dir_long


def test_deepwalk_recovers_from_incomplete_embeddings_cache():
    clear_cache()
    data = make_data()
    params = {
        "feature_dim": 4,
        "walk_length": 4,
        "number_walks": 2,
        "window_size": 2,
        "workers": 1,
        "undirected": True,
    }
    graph_key = graph_fingerprint(data)
    run_dir = feature_cache_dir("deepwalk", 7, params, graph_key)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "embeddings.word2vec").write_text(
        "4 4\n1 0.1 0.2 0.3 0.4\n",
        encoding="utf-8",
    )

    rewritten = rewrite_features(data, "deepwalk", params=params, seed=7)

    assert rewritten.x.shape == (4, 4)
    assert (run_dir / "features.pt").exists()
    with (run_dir / "embeddings.word2vec").open("r", encoding="utf-8") as handle:
        header = handle.readline().strip().split()
        node_ids = {line.split()[0] for line in handle if line.strip()}
    assert header == ["4", "4"]
    assert node_ids == {"0", "1", "2", "3"}


def test_duck_typed_object_supported():
    data = make_data()
    duck = DuckData(x=data.x.clone(), edge_index=data.edge_index.clone(), num_nodes=data.num_nodes)
    rewritten = rewrite_features(duck, "shared", params={"feature_dim": 2, "value": 2.0}, seed=3)
    assert rewritten is not duck
    assert rewritten.x.shape == (4, 2)


def test_custom_provider_registration():
    class ZeroProvider(BaseFeatureProvider):
        name = "all_zero"

        def build(self, data, *, params: dict, seed: int | None = None) -> ProviderOutput:
            return ProviderOutput(features=torch.zeros((data.num_nodes, 2), dtype=data.x.dtype), source="generated")

    register_provider("all_zero", ZeroProvider)
    provider = get_provider("all_zero")
    assert isinstance(provider, BaseFeatureProvider)
    rewritten = rewrite_features(make_data(), "all_zero")
    assert rewritten.x.shape == (4, 2)
    assert torch.count_nonzero(rewritten.x) == 0
