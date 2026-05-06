from __future__ import annotations

import math

import networkx as nx
import numpy as np
import torch

from artificial_node_feature_generator.graph import build_nx_graph, get_feature_device, get_feature_dtype
from artificial_node_feature_generator.providers.basic import require_feature_dim
from artificial_node_feature_generator.providers.base import BaseFeatureProvider
from artificial_node_feature_generator.types import ProviderOutput


def _degree_values(data) -> torch.Tensor:
    graph = build_nx_graph(data)
    return torch.tensor([graph.degree(node) for node in range(graph.number_of_nodes())], dtype=torch.long)


def _build_random_frozen_embedding(
    indices: torch.Tensor,
    *,
    num_embeddings: int,
    feature_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int | None,
) -> torch.Tensor:
    embedding = torch.nn.Embedding(num_embeddings, feature_dim, device=torch.device("cpu"))
    with torch.no_grad():
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed))
        embedding.weight.normal_(mean=0.0, std=1.0, generator=generator)
        embedding.weight.requires_grad_(False)
    features = embedding(indices.cpu()).to(dtype=dtype, device=device)
    return features


def _bucket_indices(degrees: torch.Tensor, boundaries: list[float], num_buckets: int) -> torch.Tensor:
    if not boundaries:
        return torch.zeros_like(degrees, dtype=torch.long)
    idx = torch.bucketize(degrees.float(), torch.tensor(boundaries, dtype=torch.float32), right=False)
    return idx.clamp(max=num_buckets - 1).long()


class NodeDegreeFeatureProvider(BaseFeatureProvider):
    name = "node_degree"
    cacheable = True

    def build(self, data, *, params: dict, seed: int | None = None) -> ProviderOutput:
        degrees = _degree_values(data)
        feature_dim = require_feature_dim(self.name, params)
        num_embeddings = int(degrees.max().item()) + 1 if degrees.numel() else 1
        features = _build_random_frozen_embedding(
            degrees,
            num_embeddings=num_embeddings,
            feature_dim=feature_dim,
            dtype=get_feature_dtype(data),
            device=get_feature_device(data),
            seed=seed,
        )
        return ProviderOutput(features=features, source=self.source, cacheable=True)


class DegreeBucketRangeFeatureProvider(BaseFeatureProvider):
    name = "degree_bucket_range"
    cacheable = True

    def build(self, data, *, params: dict, seed: int | None = None) -> ProviderOutput:
        degrees = _degree_values(data)
        feature_dim = require_feature_dim(self.name, params)
        num_buckets = int(params.get("num_buckets", 10))
        range_max = int(params.get("range_max", int(degrees.max().item()) if degrees.numel() else num_buckets))
        bucket_size = max(1, math.ceil(range_max / num_buckets))
        boundaries = [bucket_size * (idx + 1) for idx in range(num_buckets - 1)]
        indices = _bucket_indices(degrees, boundaries, num_buckets)
        features = _build_random_frozen_embedding(
            indices,
            num_embeddings=num_buckets,
            feature_dim=feature_dim,
            dtype=get_feature_dtype(data),
            device=get_feature_device(data),
            seed=seed,
        )
        return ProviderOutput(features=features, source=self.source, cacheable=True)


class DegreeBucketDistributionFeatureProvider(BaseFeatureProvider):
    name = "degree_bucket_distribution"
    cacheable = True

    def build(self, data, *, params: dict, seed: int | None = None) -> ProviderOutput:
        degrees = _degree_values(data)
        feature_dim = require_feature_dim(self.name, params)
        num_buckets = int(params.get("num_buckets", 10))
        if degrees.numel() == 0:
            boundaries = []
        else:
            values = degrees.cpu().numpy()
            boundaries = []
            step = max(1, int(100 / num_buckets))
            for percentile in range(step, 100, step):
                try:
                    boundary = np.percentile(values, percentile, method="higher")
                except TypeError:
                    boundary = np.percentile(values, percentile, interpolation="higher")
                boundaries.append(float(boundary))
        indices = _bucket_indices(degrees, boundaries, num_buckets)
        features = _build_random_frozen_embedding(
            indices,
            num_embeddings=num_buckets,
            feature_dim=feature_dim,
            dtype=get_feature_dtype(data),
            device=get_feature_device(data),
            seed=seed,
        )
        return ProviderOutput(features=features, source=self.source, cacheable=True)


class PageRankFeatureProvider(BaseFeatureProvider):
    name = "pagerank"
    cacheable = True

    def build(self, data, *, params: dict, seed: int | None = None) -> ProviderOutput:
        dim = require_feature_dim(self.name, params)
        graph = build_nx_graph(data)
        pagerank = nx.pagerank(graph)
        scores = torch.tensor([float(pagerank[idx]) for idx in range(graph.number_of_nodes())], dtype=get_feature_dtype(data))
        features = scores.unsqueeze(1).repeat(1, dim).to(device=get_feature_device(data))
        return ProviderOutput(features=features, source=self.source, cacheable=True)


class EigenFeatureProvider(BaseFeatureProvider):
    name = "eigen"
    cacheable = True

    def cache_seed(self, *, seed: int | None, params: dict) -> int | None:
        return None

    def _build_matrix(self, data) -> np.ndarray:
        graph = build_nx_graph(data)
        return nx.to_numpy_array(graph, nodelist=range(graph.number_of_nodes()), dtype=float)

    def _transform_matrix(self, matrix: np.ndarray) -> np.ndarray:
        return matrix

    def build(self, data, *, params: dict, seed: int | None = None) -> ProviderOutput:
        dim = require_feature_dim(self.name, params)
        matrix = self._transform_matrix(self._build_matrix(data))
        if dim > matrix.shape[0]:
            raise ValueError(f"eigen feature dim ({dim}) cannot exceed num_nodes ({matrix.shape[0]})")
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        order = np.argsort(eigenvalues.real)[::-1]
        selected = np.asarray(eigenvectors[:, order[:dim]].real, dtype=np.float32)
        features = torch.as_tensor(selected, dtype=get_feature_dtype(data), device=get_feature_device(data))
        return ProviderOutput(features=features, source=self.source, cacheable=True)


class EigenNormFeatureProvider(EigenFeatureProvider):
    name = "eigen_norm"

    def _transform_matrix(self, matrix: np.ndarray) -> np.ndarray:
        row_sums = matrix.sum(axis=1, keepdims=True)
        safe_row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
        return matrix / safe_row_sums
