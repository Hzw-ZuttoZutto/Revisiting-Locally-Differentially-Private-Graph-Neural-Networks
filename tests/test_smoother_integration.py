import re
import subprocess
import sys
import unittest
from pathlib import Path

import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor

from models import HOA, KProp, NodeClassifier


def build_dense_operator_matrix(adj_t, *, x_steps):
    normalized_adj_t = gcn_norm(adj_t, add_self_loops=False)
    num_nodes = normalized_adj_t.sparse_sizes()[0]
    row, col, value = normalized_adj_t.coo()
    dense = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    if value is not None and row.numel() > 0:
        dense[row.long(), col.long()] = value.to(dtype=dense.dtype)

    if x_steps <= 0:
        return torch.eye(num_nodes, dtype=dense.dtype)

    power = dense.clone()
    accumulated = torch.zeros_like(dense)
    for _ in range(int(x_steps)):
        accumulated = accumulated + power
        power = power @ dense
    return accumulated / float(x_steps)


class SmootherCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]

    def _run_main(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "main.py", *args],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

    def _skip_if_runtime_missing(self, completed: subprocess.CompletedProcess[str]) -> None:
        combined = f"{completed.stdout}\n{completed.stderr}"
        if "ModuleNotFoundError" in combined:
            self.skipTest("runtime dependencies for main.py are unavailable in this environment")

    def test_main_help_exposes_smoother_choices_and_default(self):
        completed = self._run_main("-h")
        self._skip_if_runtime_missing(completed)
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)

        help_text = f"{completed.stdout}\n{completed.stderr}"
        compact = re.sub(r"\s+", " ", help_text)
        self.assertIn("--smoother", help_text)
        self.assertIn("kprop", help_text)
        self.assertIn("hoa", help_text)
        self.assertRegex(compact, r"--smoother.*default: kprop")

    def test_main_accepts_explicit_hoa_flag(self):
        completed = self._run_main("--smoother", "hoa", "-h")
        self._skip_if_runtime_missing(completed)
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)


class SmootherRoutingTests(unittest.TestCase):
    def test_node_classifier_default_uses_kprop(self):
        model = NodeClassifier(input_dim=4, num_classes=3)
        self.assertIsInstance(model.smoother, KProp)

    def test_node_classifier_hoa_route(self):
        model = NodeClassifier(input_dim=4, num_classes=3, smoother="hoa")
        self.assertIsInstance(model.smoother, HOA)

    def test_node_classifier_rejects_invalid_smoother(self):
        with self.assertRaises(ValueError):
            NodeClassifier(input_dim=4, num_classes=3, smoother="invalid")

    def test_operator_feature_bypasses_smoother(self):
        x = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]],
            dtype=torch.float32,
        )
        adj_t = SparseTensor.from_dense(torch.eye(2, dtype=torch.float32))
        model = NodeClassifier(input_dim=2, num_classes=2, feature="operator", x_steps=2)

        def _unexpected(*args, **kwargs):
            raise AssertionError("smoother should not run for feature=operator")

        model.smoother.forward = _unexpected
        data = type("Data", (), {"x": x, "adj_t": adj_t})()
        observed = model._build_feature_representation(data, adj_t)
        self.assertTrue(torch.equal(observed, x))

    def test_non_operator_scale_is_applied_after_smoother(self):
        x = torch.tensor(
            [[1.0], [2.0], [3.0]],
            dtype=torch.float32,
        )
        dense_adj = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        adj_t = SparseTensor.from_dense(dense_adj)
        model = NodeClassifier(
            input_dim=1,
            num_classes=2,
            feature="raw",
            smoother="hoa",
            x_steps=2,
            scale=2.5,
        )
        data = type("Data", (), {"x": x, "adj_t": adj_t})()

        observed = model._build_feature_representation(data, adj_t)
        expected = model.smoother.neighborhood_aggregation(x, adj_t) * 2.5
        self.assertTrue(torch.allclose(observed, expected, atol=1e-6, rtol=0.0))

    def test_operator_preprojection_changes_feature_dimension_before_gnn(self):
        x = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
        adj_t = SparseTensor.from_dense(torch.eye(2, dtype=torch.float32))
        model = NodeClassifier(
            input_dim=3,
            num_classes=2,
            feature="operator",
            feature_preprojection=True,
            preprojection_output_dim=2,
            dropout=0.0,
        )
        data = type("Data", (), {"x": x, "adj_t": adj_t})()

        observed = model._build_feature_representation(data, adj_t)
        self.assertEqual(tuple(observed.shape), (2, 2))
        self.assertEqual(model.feature_preprojection_layer.in_features, 3)
        self.assertEqual(model.feature_preprojection_layer.out_features, 2)

    def test_lazy_sparse_operator_direct_matches_dense_operator_for_sage(self):
        dense_adj = torch.tensor(
            [
                [0.0, 1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        adj_t = SparseTensor.from_dense(dense_adj)
        x_steps = 3
        operator_dense = build_dense_operator_matrix(adj_t, x_steps=x_steps)
        operator_sparse = gcn_norm(adj_t, add_self_loops=False)
        model = NodeClassifier(
            input_dim=operator_dense.size(1),
            num_classes=3,
            feature="operator",
            model="sage",
            hidden_dim=5,
            dropout=0.0,
            x_steps=x_steps,
        )
        model.eval()

        dense_data = type("Data", (), {"x": operator_dense, "adj_t": adj_t})()
        lazy_data = type(
            "Data",
            (),
            {
                "x": torch.zeros((operator_dense.size(0), 1), dtype=torch.float32),
                "adj_t": adj_t,
                "operator_feature_mode": "lazy_sparse",
                "operator_x_steps": x_steps,
                "operator_num_features": operator_dense.size(1),
                "operator_normalized_adj_t": operator_sparse,
            },
        )()

        observed_dense = model._forward_logits(dense_data, gnn_adj_t=adj_t, smoother_adj_t=adj_t)
        observed_lazy = model._forward_logits(lazy_data, gnn_adj_t=adj_t, smoother_adj_t=adj_t)
        self.assertTrue(torch.allclose(observed_lazy, observed_dense, atol=1e-5, rtol=1e-5))

    def test_lazy_sparse_operator_preprojection_matches_dense_operator(self):
        dense_adj = torch.tensor(
            [
                [0.0, 1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        adj_t = SparseTensor.from_dense(dense_adj)
        x_steps = 2
        operator_dense = build_dense_operator_matrix(adj_t, x_steps=x_steps)
        operator_sparse = gcn_norm(adj_t, add_self_loops=False)
        model = NodeClassifier(
            input_dim=operator_dense.size(1),
            num_classes=2,
            feature="operator",
            model="sage",
            hidden_dim=4,
            feature_preprojection=True,
            preprojection_output_dim=3,
            dropout=0.0,
            x_steps=x_steps,
        )
        model.eval()

        dense_data = type("Data", (), {"x": operator_dense, "adj_t": adj_t})()
        lazy_data = type(
            "Data",
            (),
            {
                "x": torch.zeros((operator_dense.size(0), 1), dtype=torch.float32),
                "adj_t": adj_t,
                "operator_feature_mode": "lazy_sparse",
                "operator_x_steps": x_steps,
                "operator_num_features": operator_dense.size(1),
                "operator_normalized_adj_t": operator_sparse,
            },
        )()

        observed_dense = model._forward_logits(dense_data, gnn_adj_t=adj_t, smoother_adj_t=adj_t)
        observed_lazy = model._forward_logits(lazy_data, gnn_adj_t=adj_t, smoother_adj_t=adj_t)
        self.assertTrue(torch.allclose(observed_lazy, observed_dense, atol=1e-5, rtol=1e-5))


class HoaMathTests(unittest.TestCase):
    def test_hoa_matches_manual_average_without_x0(self):
        x = torch.tensor(
            [[1.0], [2.0], [3.0]],
            dtype=torch.float32,
        )
        dense_adj = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        adj_t = SparseTensor.from_dense(dense_adj)

        smoother = HOA(
            steps=2,
            aggregator="add",
            add_self_loops=False,
            normalize=False,
            cached=False,
        )
        observed = smoother.neighborhood_aggregation(x, adj_t)

        x_1 = dense_adj @ x
        x_2 = dense_adj @ x_1
        expected = (x_1 + x_2) / 2.0
        self.assertTrue(torch.allclose(observed, expected, atol=1e-6, rtol=0.0))

    def test_hoa_with_non_positive_k_is_identity(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        adj_t = SparseTensor.from_dense(torch.eye(2, dtype=torch.float32))
        smoother = HOA(
            steps=0,
            aggregator="add",
            add_self_loops=False,
            normalize=False,
            cached=False,
        )
        observed = smoother.neighborhood_aggregation(x, adj_t)
        self.assertTrue(torch.equal(observed, x))


if __name__ == "__main__":
    unittest.main()
