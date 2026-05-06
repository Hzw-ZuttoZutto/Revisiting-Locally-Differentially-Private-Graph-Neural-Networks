import re
import subprocess
import sys
import unittest
from pathlib import Path

import torch
from torch_sparse import SparseTensor

from models import HOA, KProp, NodeClassifier


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
