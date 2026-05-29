import math

import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor, matmul

from diagnostics.kernel_alignment import (
    anchor_operator_matrix,
    centered_kernel_alignment,
    centered_kernel_from_kernel,
    label_kernel,
    row_normalize,
)


def test_row_normalize_preserves_zero_rows():
    matrix = torch.tensor([[3.0, 4.0], [0.0, 0.0], [5.0, 0.0]])
    normalized = row_normalize(matrix)
    expected = torch.tensor([[0.6, 0.8], [0.0, 0.0], [1.0, 0.0]])
    assert torch.allclose(normalized, expected)


def test_centered_kernel_alignment_is_symmetric_and_scale_invariant():
    x = torch.tensor(
        [
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 1.0],
            [2.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    y = torch.tensor(
        [
            [0.5, 1.0],
            [1.5, 0.0],
            [0.0, 2.0],
            [1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    kx = centered_kernel_from_kernel(x @ x.T)
    ky = centered_kernel_from_kernel(y @ y.T)
    left = centered_kernel_alignment(kx, ky)
    right = centered_kernel_alignment(ky, kx)
    assert math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
    assert -1.0 <= left <= 1.0

    scaled_x = centered_kernel_from_kernel(7.0 * (x @ x.T))
    scaled_y = centered_kernel_from_kernel(0.25 * (y @ y.T))
    assert math.isclose(left, centered_kernel_alignment(scaled_x, scaled_y), rel_tol=1e-12, abs_tol=1e-12)


def test_class_balanced_label_kernel():
    labels = torch.tensor([0, 0, 1, 2, 2, 2])
    kernel = label_kernel(labels)
    assert math.isclose(float(kernel[0, 1]), 0.5, rel_tol=1e-12)
    assert math.isclose(float(kernel[3, 5]), 1.0 / 3.0, rel_tol=1e-12)
    assert float(kernel[0, 2]) == 0.0


def make_test_adj() -> SparseTensor:
    row = torch.tensor([0, 1, 1, 2], dtype=torch.long)
    col = torch.tensor([1, 0, 2, 1], dtype=torch.long)
    return SparseTensor(
        row=row,
        col=col,
        value=torch.ones(row.numel(), dtype=torch.float32),
        sparse_sizes=(3, 3),
    ).coalesce()


def test_anchor_operator_k_zero_is_identity():
    adj_t = make_test_adj()
    operator = anchor_operator_matrix(adj_t, smoother="hoa", x_steps=0, num_nodes=3, device=torch.device("cpu"))
    assert torch.allclose(operator, torch.eye(3))


def test_anchor_operator_hoa_two_matches_model_definition():
    adj_t = make_test_adj()
    normalized = gcn_norm(adj_t, add_self_loops=False).coalesce()
    identity = torch.eye(3)
    current = identity
    accumulated = torch.zeros_like(identity)
    for _ in range(2):
        current = matmul(normalized, current, reduce="add")
        accumulated = accumulated + current
    expected = accumulated / 2.0

    actual = anchor_operator_matrix(adj_t, smoother="hoa", x_steps=2, num_nodes=3, device=torch.device("cpu"))
    assert torch.allclose(actual, expected)
