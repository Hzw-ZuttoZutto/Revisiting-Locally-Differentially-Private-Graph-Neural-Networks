import os
import unittest


class MainDeterminismTests(unittest.TestCase):
    def test_configure_determinism_sets_strict_flags_and_is_idempotent(self):
        try:
            import torch
            import main as main_module
        except ModuleNotFoundError as exc:
            self.skipTest(f"runtime dependencies for main.py are unavailable in this environment: {exc}")

        original_cublas_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        original_cudnn_deterministic = torch.backends.cudnn.deterministic
        original_cudnn_benchmark = torch.backends.cudnn.benchmark
        original_deterministic_algorithms = torch.are_deterministic_algorithms_enabled()

        try:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            torch.use_deterministic_algorithms(False)

            main_module.configure_determinism()
            self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
            self.assertTrue(torch.backends.cudnn.deterministic)
            self.assertFalse(torch.backends.cudnn.benchmark)
            self.assertTrue(torch.are_deterministic_algorithms_enabled())

            main_module.configure_determinism()
            self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
            self.assertTrue(torch.backends.cudnn.deterministic)
            self.assertFalse(torch.backends.cudnn.benchmark)
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
        finally:
            if original_cublas_workspace is None:
                os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            else:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = original_cublas_workspace
            torch.backends.cudnn.deterministic = original_cudnn_deterministic
            torch.backends.cudnn.benchmark = original_cudnn_benchmark
            torch.use_deterministic_algorithms(original_deterministic_algorithms)


if __name__ == "__main__":
    unittest.main()
