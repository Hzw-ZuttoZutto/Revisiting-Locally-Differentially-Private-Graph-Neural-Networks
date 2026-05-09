from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hparams_search_scripts import mechanism_stage_utils
from hparams_search_scripts import run_mechanism_grid_task as grid_task_script
from hparams_search_scripts import run_mechanism_grid_rank as grid_rank_script
from hparams_search_scripts import run_mechanism_verify_finalize as verify_finalize_script
from hparams_search_scripts import run_mechanism_verify_task as verify_task_script
from hparams_search_scripts.mechanism_stage_context import build_recommended_command_parts, resolve_job_context


def _build_job_spec() -> dict[str, object]:
    candidates = mechanism_stage_utils.build_candidate_specs(
        x_steps_values=[0, 2],
        learning_rate_values=["0.001"],
        weight_decay_values=["0.0001"],
        dropout_values=["0.5"],
        tao2_values=["none"],
    )
    return {
        "schema_version": mechanism_stage_utils.SCHEMA_VERSION,
        "job_id": "job123",
        "display_name": "dataset=cora, feature=raw",
        "python_bin": "python",
        "training_device": "cpu",
        "base_seed": 12345,
        "rank_metric": "val/acc",
        "verify_topk": mechanism_stage_utils.VERIFY_TOPK,
        "defaults": {
            "dataset": {"data_range": [0.0, 1.0], "val_ratio": 0.25, "test_ratio": 0.25},
            "model": {"hidden_dim": 16},
            "trainer": {
                "optimizer": "adam",
                "collect_grad_stats": False,
                "gradient_clip": False,
                "gradient_clip_max_norm": 1.0,
                "sim_epoch_refresh": False,
                "show_progress": False,
                "log_every_epoch": False,
            },
            "stage": {
                "grid": {"patience": 80, "repeats": 1, "max_epochs": 300},
                "verify": {"patience": 150, "repeats": 5, "max_epochs": 500},
            },
        },
        "fixed_params": {
            "dataset": "cora",
            "feature": "raw",
            "sim_reference_eps": None,
            "feature_dim": None,
            "scale": None,
            "feature_preprojection": None,
            "preprojection_output_dim": None,
            "random_normal_mean": None,
            "random_normal_std": None,
            "shared_value": None,
            "degree_bucket_num_buckets": None,
            "degree_bucket_range_max": None,
            "deepwalk_walk_length": None,
            "deepwalk_number_walks": None,
            "deepwalk_window_size": None,
            "deepwalk_workers": None,
            "deepwalk_undirected": None,
            "mechanism": "hds",
            "x_eps": "inf",
            "m": "best",
            "norm": False,
            "norm_scale": "none",
            "smoother": "kprop",
            "backbone": "sage",
            "use_nfr": False,
        },
        "candidate_space": {
            "x_steps": [0, 2],
            "learning_rate": ["0.001"],
            "weight_decay": ["0.0001"],
            "dropout": ["0.5"],
            "tao2": ["none"],
        },
        "candidates": [candidate.to_dict() for candidate in candidates],
    }


def _build_rewrite_job_spec(*, scale: str) -> dict[str, object]:
    job_spec = _build_job_spec()
    fixed_params = dict(job_spec["fixed_params"])  # type: ignore[arg-type]
    fixed_params.update(
        {
            "feature": "shared",
            "feature_dim": 8,
            "scale": scale,
            "shared_value": "1",
        }
    )
    job_spec["display_name"] = "dataset=cora, feature=shared"
    job_spec["fixed_params"] = fixed_params
    return job_spec


def _write_first_row(csv_path: Path, row: dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


class MechanismStageScriptTests(unittest.TestCase):
    def test_recommended_command_omits_default_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp)
            job_spec = _build_rewrite_job_spec(scale="1")
            mechanism_stage_utils.write_yaml_file(
                mechanism_stage_utils.job_spec_path(job_dir),
                job_spec,
            )
            ctx = resolve_job_context(job_dir)
            candidate = mechanism_stage_utils.job_candidates(job_spec)[0]

            command_parts = build_recommended_command_parts(ctx, candidate)

        self.assertNotIn("--scale", command_parts)

    def test_recommended_command_includes_nondefault_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp)
            job_spec = _build_rewrite_job_spec(scale="2")
            mechanism_stage_utils.write_yaml_file(
                mechanism_stage_utils.job_spec_path(job_dir),
                job_spec,
            )
            ctx = resolve_job_context(job_dir)
            candidate = mechanism_stage_utils.job_candidates(job_spec)[0]

            command_parts = build_recommended_command_parts(ctx, candidate)

        self.assertIn("--scale", command_parts)
        scale_index = command_parts.index("--scale") + 1
        self.assertEqual(command_parts[scale_index], "2")

    def test_operator_recommended_command_includes_preprojection_and_omits_smoother(self):
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp)
            job_spec = _build_job_spec()
            fixed_params = dict(job_spec["fixed_params"])  # type: ignore[arg-type]
            fixed_params.update(
                {
                    "feature": "operator",
                    "scale": "2",
                    "feature_preprojection": True,
                    "preprojection_output_dim": 32,
                    "smoother": None,
                }
            )
            job_spec["fixed_params"] = fixed_params
            mechanism_stage_utils.write_yaml_file(
                mechanism_stage_utils.job_spec_path(job_dir),
                job_spec,
            )
            ctx = resolve_job_context(job_dir)
            candidate = mechanism_stage_utils.job_candidates(job_spec)[0]

            command_parts = build_recommended_command_parts(ctx, candidate)

        self.assertIn("--feature_preprojection", command_parts)
        self.assertIn("--preprojection_output_dim", command_parts)
        self.assertNotIn("--smoother", command_parts)

    def test_grid_rank_writes_topk_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp)
            mechanism_stage_utils.write_yaml_file(
                mechanism_stage_utils.job_spec_path(job_dir),
                _build_job_spec(),
            )
            candidates = mechanism_stage_utils.job_candidates(_build_job_spec())

            lower = candidates[0]
            better = candidates[1]
            _write_first_row(
                mechanism_stage_utils.grid_candidate_output_dir(job_dir, lower) / "grid.csv",
                {
                    "x_steps": lower.x_steps,
                    "learning_rate": lower.learning_rate,
                    "weight_decay": lower.weight_decay,
                    "dropout": lower.dropout,
                    "tao2": lower.tao2,
                    "seed": 12345,
                    "val/acc": 71.0,
                    "test/acc": 70.0,
                },
            )
            _write_first_row(
                mechanism_stage_utils.grid_candidate_output_dir(job_dir, better) / "grid.csv",
                {
                    "x_steps": better.x_steps,
                    "learning_rate": better.learning_rate,
                    "weight_decay": better.weight_decay,
                    "dropout": better.dropout,
                    "tao2": better.tao2,
                    "seed": 12345,
                    "val/acc": 81.0,
                    "test/acc": 80.0,
                },
            )

            with mock.patch("sys.argv", ["run_mechanism_grid_rank.py", str(job_dir)]):
                grid_rank_script.main()

            topk_path = mechanism_stage_utils.verify_topk_path(job_dir)
            with topk_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["rank"], "1")
            self.assertEqual(rows[0]["candidate_id"], str(better.candidate_id))

    def test_verify_finalize_writes_best_config_and_recommended_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp)
            job_spec = _build_job_spec()
            mechanism_stage_utils.write_yaml_file(
                mechanism_stage_utils.job_spec_path(job_dir),
                job_spec,
            )
            candidates = mechanism_stage_utils.job_candidates(job_spec)

            best = candidates[1]
            other = candidates[0]
            for repeat_id, val_acc, test_acc, candidate in (
                (1, 80.0, 79.0, other),
                (2, 81.0, 80.0, other),
                (1, 90.0, 88.0, best),
                (2, 92.0, 89.0, best),
            ):
                ranked = mechanism_stage_utils.RankedCandidate(
                    rank=1 if candidate.candidate_id == best.candidate_id else 2,
                    candidate=candidate,
                )
                output_dir = mechanism_stage_utils.verify_candidate_output_dir(job_dir, ranked, repeat_id)
                _write_first_row(
                    output_dir / "verify.csv",
                    {
                        "x_steps": candidate.x_steps,
                        "learning_rate": candidate.learning_rate,
                        "weight_decay": candidate.weight_decay,
                        "dropout": candidate.dropout,
                        "tao2": candidate.tao2,
                        "seed": 12344 + repeat_id,
                        "val/acc": val_acc,
                        "test/acc": test_acc,
                    },
                )

            with mock.patch("sys.argv", ["run_mechanism_verify_finalize.py", str(job_dir)]):
                verify_finalize_script.main()

            best_config = mechanism_stage_utils.load_best_config(job_dir)
            self.assertEqual(best_config["best_candidate"]["candidate_id"], best.candidate_id)
            self.assertIn("python", best_config["recommended_command"])
            recommended = mechanism_stage_utils.recommended_command_path(job_dir).read_text(encoding="utf-8")
            self.assertIn("--x_steps 2", recommended)
            self.assertIn("--dropout 0.5", recommended)

    def test_grid_task_uses_job_base_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp)
            job_spec = _build_job_spec()
            job_spec["base_seed"] = 2024
            mechanism_stage_utils.write_yaml_file(
                mechanism_stage_utils.job_spec_path(job_dir),
                job_spec,
            )
            candidate = mechanism_stage_utils.job_candidates(job_spec)[0]

            captured_command: list[str] = []

            def fake_run(command, cwd, check):  # noqa: ANN001
                del cwd, check
                captured_command[:] = command
                output_dir = mechanism_stage_utils.grid_candidate_output_dir(job_dir, candidate)
                _write_first_row(
                    output_dir / "grid.csv",
                    {
                        "x_steps": candidate.x_steps,
                        "learning_rate": candidate.learning_rate,
                        "weight_decay": candidate.weight_decay,
                        "dropout": candidate.dropout,
                        "tao2": candidate.tao2,
                        "seed": 2024,
                        "val/acc": 71.0,
                        "test/acc": 70.0,
                    },
                )

                class Result:
                    returncode = 0

                return Result()

            with mock.patch.object(grid_task_script.subprocess, "run", side_effect=fake_run):
                with mock.patch(
                    "sys.argv",
                    ["run_mechanism_grid_task.py", str(job_dir), "--candidate_id", str(candidate.candidate_id)],
                ):
                    grid_task_script.main()

            self.assertIn("--x_steps", captured_command)
            self.assertIn(str(candidate.x_steps), captured_command)
            seed_index = captured_command.index("-s") + 1
            self.assertEqual(captured_command[seed_index], "2024")

    def test_verify_task_uses_ranked_candidate_and_repeat_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp)
            job_spec = _build_job_spec()
            mechanism_stage_utils.write_yaml_file(
                mechanism_stage_utils.job_spec_path(job_dir),
                job_spec,
            )
            best = mechanism_stage_utils.job_candidates(job_spec)[1]
            mechanism_stage_utils.write_verify_topk(
                job_dir,
                [
                    {
                        "candidate_id": best.candidate_id,
                        "x_steps": best.x_steps,
                        "learning_rate": best.learning_rate,
                        "weight_decay": best.weight_decay,
                        "dropout": best.dropout,
                        "tao2": best.tao2,
                        "val_acc_mean": 90.0,
                        "val_acc_std": 0.0,
                        "val_acc_min": 90.0,
                        "val_acc_max": 90.0,
                        "test_acc_mean": 88.0,
                        "test_acc_std": 0.0,
                        "test_acc_min": 88.0,
                        "test_acc_max": 88.0,
                        "n": 1,
                    }
                ],
                topk=1,
            )

            captured_command: list[str] = []

            def fake_run(command, cwd, check):  # noqa: ANN001
                del cwd, check
                captured_command[:] = command
                output_dir = mechanism_stage_utils.verify_candidate_output_dir(
                    job_dir,
                    mechanism_stage_utils.RankedCandidate(rank=1, candidate=best),
                    3,
                )
                _write_first_row(
                    output_dir / "verify.csv",
                    {
                        "x_steps": best.x_steps,
                        "learning_rate": best.learning_rate,
                        "weight_decay": best.weight_decay,
                        "dropout": best.dropout,
                        "tao2": best.tao2,
                        "seed": 12347,
                        "val/acc": 90.0,
                        "test/acc": 88.0,
                    },
                )

                class Result:
                    returncode = 0

                return Result()

            with mock.patch.object(verify_task_script.subprocess, "run", side_effect=fake_run):
                with mock.patch("sys.argv", ["run_mechanism_verify_task.py", str(job_dir), "--rank", "1", "--repeat_id", "3"]):
                    verify_task_script.main()

            self.assertIn("--x_steps", captured_command)
            self.assertIn("2", captured_command)
            seed_index = captured_command.index("-s") + 1
            self.assertEqual(captured_command[seed_index], "12347")


if __name__ == "__main__":
    unittest.main()
