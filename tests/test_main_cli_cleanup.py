import subprocess
import sys
import unittest
from pathlib import Path


class MainCliCleanupTests(unittest.TestCase):
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

    def _help_sections(self, help_text: str) -> dict[str, str]:
        sections: dict[str, list[str]] = {}
        current_section = None
        for line in help_text.splitlines():
            if line and not line.startswith(" ") and line.endswith(":"):
                current_section = line[:-1]
                sections[current_section] = []
                continue
            if current_section is not None:
                sections[current_section].append(line)
        return {name: "\n".join(lines) for name, lines in sections.items()}

    def test_main_help_excludes_removed_cli_flags(self):
        completed = self._run_main("-h")
        self._skip_if_runtime_missing(completed)
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)

        help_text = f"{completed.stdout}\n{completed.stderr}"
        removed_flags = [
            "--ensemble",
            "--ensemble-size",
            "--idea1",
            "--idea2",
            "--idea3",
            "--x_steps_ensemble_list",
            "--y_steps_ensemble_list",
            "--temperature_list",
            "--dropedge-prob",
            "--structural-prior",
            "--structural-prior-dim",
            "--transition-mode",
            "--reliability-mode",
            "--y-eps",
            "--y-steps",
            "--forward-correction",
            "--forward",
        ]
        for flag in removed_flags:
            self.assertNotIn(flag, help_text)

    def test_main_help_includes_performance_flags(self):
        completed = self._run_main("-h")
        self._skip_if_runtime_missing(completed)
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)

        help_text = f"{completed.stdout}\n{completed.stderr}"
        self.assertIn("--show-progress", help_text)
        self.assertIn("--log-every-epoch", help_text)

    def test_main_help_regroups_cli_sections(self):
        completed = self._run_main("-h")
        self._skip_if_runtime_missing(completed)
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)

        help_text = f"{completed.stdout}\n{completed.stderr}"
        sections = self._help_sections(help_text)
        expected_order = [
            "dataset arguments",
            "feature transformation arguments",
            "feature perturbation arguments",
            "calibrator arguments",
            "model arguments",
            "nfr arguments",
            "trainer arguments",
            "experiment arguments",
        ]

        last_index = -1
        for title in expected_order:
            marker = f"{title}:"
            self.assertIn(marker, help_text)
            current_index = help_text.index(marker)
            self.assertGreater(current_index, last_index)
            last_index = current_index

        self.assertIn("--dataset", sections["dataset arguments"])
        self.assertIn("--inf-eps-unit-map", sections["dataset arguments"])
        self.assertEqual(help_text.count("--inf-eps-unit-map"), 1)

        self.assertIn("--feature", sections["feature transformation arguments"])
        self.assertIn("--scale", sections["feature transformation arguments"])
        self.assertIn("--deepwalk-undirected", sections["feature transformation arguments"])
        self.assertNotIn("--inf-eps-unit-map", sections["feature transformation arguments"])

        self.assertIn("--mechanism", sections["feature perturbation arguments"])
        self.assertIn("--x-eps", sections["feature perturbation arguments"])
        self.assertIn("--m M", sections["feature perturbation arguments"])

        self.assertIn("--norm", sections["calibrator arguments"])
        self.assertIn("--norm-scale", sections["calibrator arguments"])
        self.assertIn("--x-steps", sections["calibrator arguments"])
        self.assertIn("--smoother", sections["calibrator arguments"])

        self.assertIn("--model", sections["model arguments"])
        self.assertIn("--hidden-dim", sections["model arguments"])
        self.assertIn("--dropout", sections["model arguments"])
        self.assertNotIn("--norm", sections["model arguments"])
        self.assertNotIn("--x-steps", sections["model arguments"])
        self.assertNotIn("--smoother", sections["model arguments"])

        self.assertIn("--use-nfr", sections["nfr arguments"])
        self.assertIn("--tao2", sections["nfr arguments"])

    def test_removed_flag_is_unrecognized(self):
        completed = self._run_main("--ensemble")
        self._skip_if_runtime_missing(completed)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("unrecognized arguments: --ensemble", completed.stderr)

    def test_structural_prior_flags_are_unrecognized(self):
        removed_flags = [
            "--structural_prior",
            "--structural_prior_dim",
            "--transition_mode",
            "--reliability_mode",
        ]
        for flag in removed_flags:
            with self.subTest(flag=flag):
                completed = self._run_main(flag, "dummy")
                self._skip_if_runtime_missing(completed)
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("unrecognized arguments", completed.stderr)

    def test_legacy_label_flags_are_unrecognized(self):
        flag_with_value = [
            ("--y_eps", "inf"),
            ("--y_steps", "0"),
            ("--forward_correction", "True"),
            ("--forward", "True"),
        ]
        for flag, value in flag_with_value:
            with self.subTest(flag=flag):
                completed = self._run_main(flag, value)
                self._skip_if_runtime_missing(completed)
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("unrecognized arguments", completed.stderr)

    def test_deleted_training_entries_do_not_exist(self):
        removed_paths = [
            "experiments.py",
            "scheduler.py",
            "scheduler_config.py",
            "configs/scheduler.yaml",
        ]
        for rel_path in removed_paths:
            self.assertFalse((self.repo_root / rel_path).exists(), msg=rel_path)

    def test_main_source_has_no_deleted_scheduler_imports(self):
        source = (self.repo_root / "main.py").read_text(encoding="utf-8")
        self.assertNotIn("from scheduler import", source)
        self.assertNotIn("from scheduler_config import", source)
        self.assertNotIn("load_scheduler_config", source)

    def test_main_source_has_no_removed_argument_tokens(self):
        source = (self.repo_root / "main.py").read_text(encoding="utf-8")
        removed_tokens = [
            "--ensemble",
            "--ensemble-size",
            "--idea1",
            "--idea2",
            "--idea3",
            "--x_steps_ensemble_list",
            "--y_steps_ensemble_list",
            "--temperature_list",
            "--dropedge-prob",
            "LabelPerturbation",
        ]
        for token in removed_tokens:
            self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
