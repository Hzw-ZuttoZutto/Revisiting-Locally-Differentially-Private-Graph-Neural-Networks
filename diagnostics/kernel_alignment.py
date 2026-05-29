from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor, matmul


@dataclass(frozen=True)
class CenteredKernel:
    matrix: torch.Tensor
    fro_norm: float


def sparse_sizes(adj_t: SparseTensor) -> tuple[int, int]:
    sizes = adj_t.sparse_sizes()
    return int(sizes[0]), int(sizes[1])


def move_sparse_tensor(
    adj_t: SparseTensor,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> SparseTensor:
    row, col, value = adj_t.coo()
    if value is None:
        value_dtype = torch.float32 if dtype is None else dtype
        value = torch.ones(row.numel(), dtype=value_dtype, device=row.device)
    elif dtype is not None:
        value = value.to(dtype=dtype)

    if row.device == device and col.device == device and value.device == device:
        return adj_t
    return SparseTensor(
        row=row.to(device),
        col=col.to(device),
        value=value.to(device=device),
        sparse_sizes=sparse_sizes(adj_t),
    ).coalesce()


def row_normalize(matrix: torch.Tensor) -> torch.Tensor:
    """L2-normalize rows, preserving exactly zero rows as zero."""
    if matrix.dim() != 2:
        raise ValueError(f"row_normalize expects a 2D tensor, got shape={tuple(matrix.shape)}")
    norms = torch.linalg.vector_norm(matrix, ord=2, dim=1, keepdim=True)
    return torch.where(norms > 0, matrix / norms.clamp_min(torch.finfo(matrix.dtype).tiny), torch.zeros_like(matrix))


def center_kernel(kernel: torch.Tensor) -> torch.Tensor:
    if kernel.dim() != 2 or kernel.size(0) != kernel.size(1):
        raise ValueError(f"center_kernel expects a square matrix, got shape={tuple(kernel.shape)}")
    row_mean = kernel.mean(dim=1, keepdim=True)
    col_mean = kernel.mean(dim=0, keepdim=True)
    total_mean = kernel.mean()
    return kernel - row_mean - col_mean + total_mean


def centered_kernel_from_kernel(kernel: torch.Tensor, *, name: str = "kernel") -> CenteredKernel:
    centered = center_kernel(kernel.to(dtype=torch.float64))
    centered = 0.5 * (centered + centered.transpose(0, 1))
    norm = float(torch.linalg.matrix_norm(centered, ord="fro").item())
    if not math.isfinite(norm) or norm <= 0.0:
        raise RuntimeError(f"Centered {name} has zero or invalid Frobenius norm.")
    return CenteredKernel(matrix=centered, fro_norm=norm)


def representation_kernel(features: torch.Tensor) -> torch.Tensor:
    normalized = row_normalize(features)
    kernel = normalized @ normalized.transpose(0, 1)
    return 0.5 * (kernel + kernel.transpose(0, 1))


def centered_representation_kernel(features: torch.Tensor, *, name: str = "representation kernel") -> CenteredKernel:
    return centered_kernel_from_kernel(representation_kernel(features), name=name)


def label_kernel(labels: torch.Tensor, *, device: torch.device | None = None) -> torch.Tensor:
    labels_cpu = labels.detach().to(device="cpu")
    if labels_cpu.dim() != 1:
        if labels_cpu.dim() == 2:
            labels_cpu = labels_cpu.argmax(dim=1)
        else:
            raise ValueError(f"labels must be a 1D class-index tensor or 2D one-hot tensor, got {tuple(labels.shape)}")
    labels_cpu = labels_cpu.to(dtype=torch.long)
    unique_labels, inverse = torch.unique(labels_cpu, sorted=True, return_inverse=True)
    if unique_labels.numel() == 0:
        raise ValueError("labels must contain at least one node")
    counts = torch.bincount(inverse, minlength=int(unique_labels.numel())).to(torch.float64)
    scaled = torch.zeros((int(labels_cpu.numel()), int(unique_labels.numel())), dtype=torch.float64)
    scaled[torch.arange(int(labels_cpu.numel())), inverse] = counts[inverse].rsqrt()
    if device is not None:
        scaled = scaled.to(device=device)
    kernel = scaled @ scaled.transpose(0, 1)
    return 0.5 * (kernel + kernel.transpose(0, 1))


def centered_label_kernel(labels: torch.Tensor, *, device: torch.device | None = None) -> CenteredKernel:
    return centered_kernel_from_kernel(label_kernel(labels, device=device), name="label kernel")


def centered_kernel_alignment(left: CenteredKernel, right: CenteredKernel) -> float:
    if left.matrix.shape != right.matrix.shape:
        raise ValueError(f"kernel shapes must match, got {tuple(left.matrix.shape)} and {tuple(right.matrix.shape)}")
    value = float(torch.sum(left.matrix * right.matrix).item() / (left.fro_norm * right.fro_norm))
    if not math.isfinite(value):
        raise RuntimeError("Centered kernel alignment became non-finite.")
    if value < -1.000001 or value > 1.000001:
        raise RuntimeError(f"Centered kernel alignment out of expected range [-1, 1]: {value}")
    return max(-1.0, min(1.0, value))


def normalized_adjacency(
    adj_t: SparseTensor,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> SparseTensor:
    normalized = gcn_norm(adj_t.cpu(), add_self_loops=False).coalesce()
    return move_sparse_tensor(normalized, device=device, dtype=dtype)


def apply_propagation(
    normalized_adj_t: SparseTensor | None,
    matrix: torch.Tensor,
    *,
    smoother: str,
    x_steps: int,
) -> torch.Tensor:
    smoother_name = str(smoother).strip().lower()
    if smoother_name not in {"hoa", "kprop"}:
        raise ValueError(f"Unsupported smoother {smoother!r}; expected one of ['hoa', 'kprop'].")
    steps = int(x_steps)
    if steps <= 0:
        return matrix
    if normalized_adj_t is None:
        raise ValueError("normalized_adj_t is required when x_steps > 0")

    current = matrix
    if smoother_name == "kprop":
        for _ in range(steps):
            current = matmul(normalized_adj_t, current, reduce="add")
        return current

    accumulated = torch.zeros_like(matrix)
    for _ in range(steps):
        current = matmul(normalized_adj_t, current, reduce="add")
        accumulated = accumulated + current
    return accumulated / float(steps)


def anchor_operator_matrix(
    adj_t: SparseTensor,
    *,
    smoother: str,
    x_steps: int,
    num_nodes: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if num_nodes is None:
        num_nodes = sparse_sizes(adj_t)[0]
    if device is None:
        row, _, value = adj_t.coo()
        device = value.device if value is not None else row.device

    identity = torch.eye(int(num_nodes), dtype=dtype, device=device)
    steps = int(x_steps)
    if steps <= 0:
        return identity

    normalized_adj_t = normalized_adjacency(adj_t, device=device, dtype=dtype)
    return apply_propagation(
        normalized_adj_t,
        identity,
        smoother=smoother,
        x_steps=steps,
    )


def centered_anchor_kernel(
    adj_t: SparseTensor,
    *,
    smoother: str,
    x_steps: int,
    num_nodes: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> CenteredKernel:
    operator_matrix = anchor_operator_matrix(
        adj_t,
        smoother=smoother,
        x_steps=x_steps,
        num_nodes=num_nodes,
        device=device,
        dtype=dtype,
    )
    return centered_representation_kernel(operator_matrix, name=f"anchor kernel x_steps={int(x_steps)}")


def kla(kernel: CenteredKernel, labels_or_label_kernel: torch.Tensor | CenteredKernel) -> float:
    if isinstance(labels_or_label_kernel, CenteredKernel):
        label = labels_or_label_kernel
    else:
        label = centered_label_kernel(labels_or_label_kernel, device=kernel.matrix.device)
    return centered_kernel_alignment(kernel, label)


def rs(left: CenteredKernel, right: CenteredKernel) -> float:
    return centered_kernel_alignment(left, right)
