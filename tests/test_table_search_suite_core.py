from __future__ import annotations

import csv
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from hparams_search_scripts import mechanism_stage_utils
from hparams_search_scripts import table_search_suite_core as core


def _job_spec() -> dict[str, object]:
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
        "display_name": "dataset=cora, feature=raw, mechanism=hds",
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


def _write_first_row(csv_path: Path, row: dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


class TableSearchSuiteCoreTests(unittest.TestCase):
    def test_run_batch_search_skips_existing_outputs_and_updates_manifest(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir)
            job_spec = _job_spec()
            job_dir = output_root / "dataset=cora" / "feature=raw"
            mechanism_stage_utils.ensure_job_directories(job_dir)
            mechanism_stage_utils.write_yaml_file(mechanism_stage_utils.job_spec_path(job_dir), job_spec)
            mechanism_stage_utils.write_yaml_file(
                mechanism_stage_utils.best_config_path(job_dir),
                {
                    "best_rank": 1,
                    "best_candidate": {
                        "candidate_id": 2,
                        "x_steps": 2,
                        "learning_rate": "0.001",
                        "weight_decay": "0.0001",
                        "dropout": "0.5",
                        "tao2": "none",
                    },
                    "verify_metrics": {
                        "val_acc": {"mean": 88.0, "std": 1.0},
                        "test_acc": {"mean": 87.0, "std": 2.0},
                    },
                    "artifacts": {
                        "verify_summary_csv": str(mechanism_stage_utils.verify_summary_path(job_dir)),
                        "recommended_command_txt": str(mechanism_stage_utils.recommended_command_path(job_dir)),
                    },
                },
            )
            mechanism_stage_utils.verify_summary_path(job_dir).write_text("summary\n", encoding="utf-8")
            mechanism_stage_utils.recommended_command_path(job_dir).write_text("python main.py\n", encoding="utf-8")

            spec = core.BatchSpec(
                output_root=output_root,
                execution=mechanism_stage_utils.ExecutionSettings(
                    device="cpu",
                    worker_ids=[0],
                    max_parallel_per_worker=1,
                    launch_interval_sec=0.0,
                ),
                jobs=[
                    core.BatchJob(
                        job_id="job123",
                        job_dir=job_dir,
                        display_name="dataset=cora, feature=raw",
                        job_spec=job_spec,
                    )
                ],
                config_copy_source=repo_root / "requirements.txt",
            )

            completed, skipped, failed, manifest_path = core.run_batch_search(spec, repo_root=repo_root)
            self.assertEqual((completed, skipped, failed), (1, 1, 0))

            with manifest_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["search_status"], "skipped_existing_result")
            self.assertEqual(rows[0]["best_candidate_id"], "2")

    def test_run_batch_search_resumes_from_partial_grid_outputs(self):
        actual_repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            fake_repo_root = temp_root / "fake_repo"
            fake_repo_root.mkdir(parents=True, exist_ok=True)
            (fake_repo_root / "main.py").write_text("print('placeholder')\n", encoding="utf-8")

            helpers = fake_repo_root / "helpers"
            helpers.mkdir(parents=True, exist_ok=True)
            fake_grid_task = helpers / "fake_grid_task.py"
            fake_verify_task = helpers / "fake_verify_task.py"
            helper_prefix = textwrap.dedent(
                f"""
                import csv
                import sys
                from pathlib import Path
                REPO_ROOT = Path({str(actual_repo_root)!r})
                if str(REPO_ROOT) not in sys.path:
                    sys.path.insert(0, str(REPO_ROOT))
                from hparams_search_scripts import mechanism_stage_utils
                """
            )
            fake_grid_task.write_text(
                helper_prefix
                + textwrap.dedent(
                    """
                    job_dir = Path(sys.argv[1]).resolve()
                    candidate_id = int(sys.argv[sys.argv.index("--candidate_id") + 1])
                    job_spec = mechanism_stage_utils.load_job_spec(job_dir)
                    candidate = mechanism_stage_utils.job_candidate_by_id(job_spec, candidate_id)
                    output_dir = mechanism_stage_utils.grid_candidate_output_dir(job_dir, candidate)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    path = output_dir / "grid.csv"
                    with path.open("w", encoding="utf-8", newline="") as handle:
                        writer = csv.DictWriter(
                            handle,
                            fieldnames=["x_steps", "learning_rate", "weight_decay", "dropout", "tao2", "seed", "val/acc", "test/acc"],
                        )
                        writer.writeheader()
                        writer.writerow(
                            {
                                "x_steps": candidate.x_steps,
                                "learning_rate": candidate.learning_rate,
                                "weight_decay": candidate.weight_decay,
                                "dropout": candidate.dropout,
                                "tao2": candidate.tao2,
                                "seed": job_spec["base_seed"],
                                "val/acc": 70.0 + candidate.candidate_id,
                                "test/acc": 69.0 + candidate.candidate_id,
                            }
                        )
                    """
                ),
                encoding="utf-8",
            )
            fake_verify_task.write_text(
                helper_prefix
                + textwrap.dedent(
                    """
                    job_dir = Path(sys.argv[1]).resolve()
                    rank = int(sys.argv[sys.argv.index("--rank") + 1])
                    repeat_id = int(sys.argv[sys.argv.index("--repeat_id") + 1])
                    ranked = mechanism_stage_utils.ranked_candidate_by_rank(
                        mechanism_stage_utils.verify_topk_path(job_dir),
                        rank,
                    )
                    output_dir = mechanism_stage_utils.verify_candidate_output_dir(job_dir, ranked, repeat_id)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    path = output_dir / "verify.csv"
                    with path.open("w", encoding="utf-8", newline="") as handle:
                        writer = csv.DictWriter(
                            handle,
                            fieldnames=["x_steps", "learning_rate", "weight_decay", "dropout", "tao2", "seed", "val/acc", "test/acc"],
                        )
                        writer.writeheader()
                        writer.writerow(
                            {
                                "x_steps": ranked.x_steps,
                                "learning_rate": ranked.learning_rate,
                                "weight_decay": ranked.weight_decay,
                                "dropout": ranked.dropout,
                                "tao2": ranked.tao2,
                                "seed": 12345 + repeat_id - 1,
                                "val/acc": 80.0 + ranked.candidate_id,
                                "test/acc": 79.0 + ranked.candidate_id,
                            }
                        )
                    """
                ),
                encoding="utf-8",
            )

            output_root = temp_root / "out"
            job_spec = _job_spec()
            job_dir = output_root / "dataset=cora" / "feature=raw"
            mechanism_stage_utils.ensure_job_directories(job_dir)
            mechanism_stage_utils.write_yaml_file(mechanism_stage_utils.job_spec_path(job_dir), job_spec)
            candidates = mechanism_stage_utils.job_candidates(job_spec)
            existing = candidates[0]
            _write_first_row(
                mechanism_stage_utils.grid_candidate_output_dir(job_dir, existing) / "grid.csv",
                {
                    "x_steps": existing.x_steps,
                    "learning_rate": existing.learning_rate,
                    "weight_decay": existing.weight_decay,
                    "dropout": existing.dropout,
                    "tao2": existing.tao2,
                    "seed": 12345,
                    "val/acc": 71.0,
                    "test/acc": 70.0,
                },
            )

            spec = core.BatchSpec(
                output_root=output_root,
                execution=mechanism_stage_utils.ExecutionSettings(
                    device="cpu",
                    worker_ids=[0],
                    max_parallel_per_worker=1,
                    launch_interval_sec=0.0,
                ),
                jobs=[
                    core.BatchJob(
                        job_id="job123",
                        job_dir=job_dir,
                        display_name="dataset=cora, feature=raw",
                        job_spec=job_spec,
                    )
                ],
                config_copy_source=actual_repo_root / "requirements.txt",
            )

            with mock.patch.object(
                core,
                "_stage_scripts",
                return_value={
                    "grid_task": fake_grid_task,
                    "grid_rank": actual_repo_root / "hparams_search_scripts" / "run_mechanism_grid_rank.py",
                    "verify_task": fake_verify_task,
                    "verify_finalize": actual_repo_root / "hparams_search_scripts" / "run_mechanism_verify_finalize.py",
                },
            ):
                completed, skipped, failed, manifest_path = core.run_batch_search(spec, repo_root=fake_repo_root)

            self.assertEqual((completed, skipped, failed), (1, 0, 0))
            self.assertTrue(mechanism_stage_utils.job_outputs_complete(job_dir))
            grid_logs = list(mechanism_stage_utils.logs_dir(job_dir).glob("grid_candidate_*.log"))
            self.assertEqual(len(grid_logs), 1)

            with manifest_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["search_status"], "completed")
            self.assertEqual(rows[0]["grid_done"], "2")


if __name__ == "__main__":
    unittest.main()
