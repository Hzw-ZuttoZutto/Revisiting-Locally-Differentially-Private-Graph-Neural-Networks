from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor, matmul

from mechanisms import _multibit_kappa


_TARGET_BLOCK_ELEMENTS = 1_000_000


def _sparse_sizes(adj_t: SparseTensor) -> tuple[int, int]:
    sizes = adj_t.sparse_sizes()
    return int(sizes[0]), int(sizes[1])


def _to_float64_sparse_tensor(adj_t: SparseTensor) -> SparseTensor:
    row, col, value = adj_t.coo()
    if value is None:
        value = torch.ones(row.numel(), dtype=torch.float64, device=row.device)
    else:
        value = value.to(dtype=torch.float64)
    return SparseTensor(
        row=row,
        col=col,
        value=value,
        sparse_sizes=_sparse_sizes(adj_t),
    ).coalesce()


def _choose_block_width(num_rows: int, width: int) -> int:
    if num_rows <= 0 or width <= 0:
        raise ValueError("block dimensions must be positive")
    return max(1, min(int(width), max(1, _TARGET_BLOCK_ELEMENTS // int(num_rows))))


def _center_matrix(matrix: torch.Tensor) -> torch.Tensor:
    row_mean = matrix.mean(dim=1, keepdim=True)
    col_mean = matrix.mean(dim=0, keepdim=True)
    total_mean = matrix.mean()
    return matrix - row_mean - col_mean + total_mean


@dataclass(frozen=True)
class PropagationOperator:
    normalized_adj_t: SparseTensor | None
    smoother: str
    x_steps: int

    @classmethod
    def from_adj_t(cls, adj_t: SparseTensor, *, smoother: str, x_steps: int) -> "PropagationOperator":
        smoother_name = str(smoother).strip().lower()
        if smoother_name not in {"kprop", "hoa"}:
            raise ValueError(f"Unsupported smoother {smoother!r}; expected one of ['hoa', 'kprop'].")

        steps = int(x_steps)
        if steps <= 0:
            return cls(normalized_adj_t=None, smoother=smoother_name, x_steps=steps)

        normalized_adj_t = gcn_norm(adj_t, add_self_loops=False).coalesce()
        normalized_adj_t = _to_float64_sparse_tensor(normalized_adj_t)
        return cls(normalized_adj_t=normalized_adj_t, smoother=smoother_name, x_steps=steps)

    def apply(self, matrix: torch.Tensor) -> torch.Tensor:
        if self.x_steps <= 0 or self.normalized_adj_t is None:
            return matrix

        current = matrix
        if self.smoother == "kprop":
            for _ in range(self.x_steps):
                current = matmul(self.normalized_adj_t, current, reduce="add")
            return current

        accumulated = torch.zeros_like(matrix)
        for _ in range(self.x_steps):
            current = matmul(self.normalized_adj_t, current, reduce="add")
            accumulated = accumulated + current
        return accumulated / float(self.x_steps)

    def apply_squared(self, matrix: torch.Tensor) -> torch.Tensor:
        return self.apply(self.apply(matrix))


def _sample_subset_indices(
    num_nodes: int,
    *,
    node_ratio: float,
    device: torch.device,
    sampling_seed: int | None,
) -> torch.Tensor:
    if not math.isfinite(node_ratio) or node_ratio <= 0.0 or node_ratio > 1.0:
        raise ValueError(f"node_ratio must satisfy 0 < node_ratio <= 1, got {node_ratio!r}")

    sample_size = int(math.ceil(float(node_ratio) * int(num_nodes)))
    if sample_size < 2:
        raise ValueError(
            f"node_ratio={node_ratio!r} sampled only {sample_size} node(s); increase node_ratio so the subset has at least 2 nodes."
        )

    generator = None
    if sampling_seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(sampling_seed))
    permutation = torch.randperm(int(num_nodes), generator=generator, device=torch.device("cpu"))
    return permutation[:sample_size].to(device=device)


def _resolve_beta(
    *,
    data,
    feature: str,
    mechanism: str,
    x_eps: float,
    original_input_dim: int,
) -> float:
    feature_name = str(feature).strip().lower()
    if feature_name == "random_normal":
        return 1.0

    mechanism_name = str(mechanism).strip().lower()
    if feature_name != "raw" or mechanism_name != "mbm":
        raise ValueError(
            "sanity_check currently supports only feature=random_normal or feature=raw with mechanism=mbm"
        )
    if not math.isfinite(float(x_eps)) or float(x_eps) <= 0.0:
        raise ValueError("feature=raw sanity_check requires finite x_eps > 0")

    d = int(original_input_dim)
    resolved_m = getattr(data, "feature_mechanism_resolved_m", None)
    if resolved_m is None:
        raise ValueError("raw sanity_check expected data.feature_mechanism_resolved_m to be available")
    resolved_m = int(resolved_m)
    if resolved_m < 1 or resolved_m > d:
        raise ValueError(f"resolved MBM sample dimension must satisfy 1 <= m <= d, got m={resolved_m}, d={d}")

    c_eps_m = _multibit_kappa(float(x_eps) / float(resolved_m))
    if not math.isfinite(c_eps_m) or c_eps_m <= 0.0:
        raise ValueError("MBM beta is undefined because C_{eps,m} is non-finite or non-positive")
    pi_value = float(resolved_m) / float(d)
    return float(d) / (pi_value * (float(c_eps_m) ** 2))


def _compute_kz_subset(
    *,
    operator: PropagationOperator,
    x: torch.Tensor,
    subset_idx: torch.Tensor,
) -> torch.Tensor:
    num_nodes, feature_dim = x.size()
    sample_size = int(subset_idx.numel())
    block_width = _choose_block_width(num_nodes, feature_dim)
    kernel = torch.zeros((sample_size, sample_size), dtype=torch.float64, device=x.device)

    for start in range(0, feature_dim, block_width):
        end = min(feature_dim, start + block_width)
        x_block = x[:, start:end].to(dtype=torch.float64)
        z_block = operator.apply(x_block)
        z_subset = z_block.index_select(0, subset_idx)
        kernel = kernel + (z_subset @ z_subset.transpose(0, 1))

    return 0.5 * (kernel + kernel.transpose(0, 1))


def _compute_kq_subset(
    *,
    operator: PropagationOperator,
    num_nodes: int,
    subset_idx: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    sample_size = int(subset_idx.numel())
    block_width = _choose_block_width(num_nodes, sample_size)
    kernel = torch.zeros((sample_size, sample_size), dtype=torch.float64, device=device)

    for start in range(0, sample_size, block_width):
        end = min(sample_size, start + block_width)
        block_size = end - start
        basis = torch.zeros((num_nodes, block_size), dtype=torch.float64, device=device)
        local_positions = torch.arange(block_size, device=device)
        basis[subset_idx[start:end], local_positions] = 1.0
        propagated = operator.apply_squared(basis)
        kernel[:, start:end] = propagated.index_select(0, subset_idx)

    return 0.5 * (kernel + kernel.transpose(0, 1))


@torch.no_grad()
def compute_sanity_e_pg(
    data,
    *,
    feature: str,
    smoother: str,
    x_steps: int,
    node_ratio: float,
    mechanism: str,
    x_eps: float,
    original_input_dim: int,
    sampling_seed: int | None,
) -> float:
    if not hasattr(data, "adj_t") or not isinstance(data.adj_t, SparseTensor):
        raise ValueError("sanity_check requires data.adj_t to be a SparseTensor")
    if not hasattr(data, "x") or not isinstance(data.x, torch.Tensor):
        raise ValueError("sanity_check requires data.x to be a dense feature tensor")

    device = data.x.device
    num_nodes = int(data.x.size(0))
    if num_nodes < 2:
        raise ValueError("sanity_check requires at least 2 nodes")

    subset_idx = _sample_subset_indices(
        num_nodes,
        node_ratio=float(node_ratio),
        device=device,
        sampling_seed=sampling_seed,
    )
    operator = PropagationOperator.from_adj_t(
        data.adj_t,
        smoother=smoother,
        x_steps=int(x_steps),
    )

    k_z = _compute_kz_subset(
        operator=operator,
        x=data.x,
        subset_idx=subset_idx,
    )
    k_q = _compute_kq_subset(
        operator=operator,
        num_nodes=num_nodes,
        subset_idx=subset_idx,
        device=device,
    )
    beta = _resolve_beta(
        data=data,
        feature=feature,
        mechanism=mechanism,
        x_eps=float(x_eps),
        original_input_dim=int(original_input_dim),
    )
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError(f"sanity_check beta must be finite and > 0, got {beta!r}")

    centered_k_q = _center_matrix(k_q)
    denominator = torch.linalg.matrix_norm(centered_k_q, ord="fro")
    if not torch.isfinite(denominator) or float(denominator.item()) <= 0.0:
        raise ValueError("sanity_check denominator is non-finite or zero")

    centered_diff = _center_matrix((k_z / float(beta)) - k_q)
    numerator = torch.linalg.matrix_norm(centered_diff, ord="fro")
    if not torch.isfinite(numerator):
        raise ValueError("sanity_check numerator is non-finite")

    return float((numerator / denominator).item())
