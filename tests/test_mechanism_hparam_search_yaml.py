from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from hparams_search_scripts import run_mechanism_hparam_search as search_script


def _base_config() -> dict[str, object]:
    return {
        "device": {
            "device": "cpu",
            "cpu_worker_count": 1,
            "gpu_ids": None,
            "max_parallel_per_gpu": None,
            "gpu_launch_interval_sec": None,
        },
        "defaults": {
            "dataset": {
                "data_range": [0.0, 1.0],
                "val_ratio": 0.25,
                "test_ratio": 0.25,
            },
            "model": {
                "hidden_dim": 16,
            },
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
                "grid": {"patience": 80, "repeats": 1},
                "verify": {"patience": 150, "repeats": 5},
            },
        },
        "search_space": {
            "dataset": {"datasets": ["cora"]},
            "feature_transformation": {
                "features": ["raw"],
                "sim_reference_eps": [],
                "feature_dim": [],
                "scale": [],
                "feature_preprojection": [],
                "preprojection_output_dim": [],
                "random_normal_mean": [],
                "random_normal_std": [],
                "shared_value": [],
                "degree_bucket_num_buckets": [],
                "degree_bucket_range_max": [],
                "deepwalk_walk_length": [],
                "deepwalk_number_walks": [],
                "deepwalk_window_size": [],
                "deepwalk_workers": [],
                "deepwalk_undirected": [],
            },
            "feature_perturbation": {
                "mechanisms": ["hds"],
                "x_eps": ["inf"],
                "m": ["best"],
            },
            "calibrator": {
                "norm": [False],
                "norm_scale": [],
                "x_steps": [0],
                "smoother": ["kprop"],
            },
            "model": {
                "backbones": ["sage"],
                "dropout": ["0.5"],
            },
            "trainer": {
                "learning_rate": ["0.001"],
                "weight_decay": ["0"],
            },
            "nfr": {
                "use_nfr": [False],
                "tao2": [],
            },
        },
    }


def _write_config(tmp_dir: str, config: dict[str, object]) -> Path:
    path = Path(tmp_dir) / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


class MechanismHparamSearchYamlTests(unittest.TestCase):
    def test_example_configs_load(self):
        repo_root = Path(__file__).resolve().parents[1]
        candidate_paths = [
            repo_root / "configs" / "mechanism_hparam_search.full_feature_example.yaml",
            repo_root / "configs" / "mechanism_hparam_search.nfr_example.yaml",
            repo_root / "configs" / "mechanism_hparam_search.yaml",
        ]
        example_paths = [path for path in candidate_paths if path.is_file()]
        self.assertTrue(example_paths, "Expected at least one example search config to exist.")

        for example_path in example_paths:
            with self.subTest(example_path=example_path.name):
                parsed = search_script.load_search_config(example_path)
                batch = search_script.build_batch_spec(
                    search_config=parsed,
                    output_root=repo_root / "tmp_example_output",
                    config_copy_source=example_path,
                )
                self.assertGreater(len(batch.jobs), 0)

    def test_load_search_config_rejects_unknown_top_level_field(self):
        config = _base_config()
        config["unexpected"] = {}

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_config(tmp_dir, config)
            with self.assertRaises(search_script.SearchError):
                search_script.load_search_config(path)

    def test_load_search_config_uses_explicit_top_level_seed(self):
        config = _base_config()
        config["seed"] = 2024

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_config(tmp_dir, config)
            parsed = search_script.load_search_config(path)

        self.assertEqual(parsed["meta"]["base_seed"], 2024)

    def test_load_search_config_defaults_top_level_seed_when_omitted(self):
        config = _base_config()

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_config(tmp_dir, config)
            parsed = search_script.load_search_config(path)

        self.assertEqual(parsed["meta"]["base_seed"], 12345)

    def test_load_search_config_rejects_invalid_top_level_seed(self):
        for invalid_seed in (-1, "oops"):
            with self.subTest(seed=invalid_seed):
                config = _base_config()
                config["seed"] = invalid_seed

                with tempfile.TemporaryDirectory() as tmp_dir:
                    path = _write_config(tmp_dir, config)
                    with self.assertRaisesRegex(search_script.SearchError, "seed"):
                        search_script.load_search_config(path)

    def test_load_search_config_requires_sim_reference_eps_for_sim(self):
        config = _base_config()
        config["search_space"]["feature_transformation"]["features"] = ["sim"]  # type: ignore[index]

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_config(tmp_dir, config)
            with self.assertRaisesRegex(search_script.SearchError, "sim_reference_eps"):
                search_script.load_search_config(path)

    def test_load_search_config_defaults_missing_x_eps_for_sim(self):
        config = _base_config()
        config["search_space"]["feature_transformation"]["features"] = ["sim"]  # type: ignore[index]
        config["search_space"]["feature_transformation"]["sim_reference_eps"] = ["1"]  # type: ignore[index]
        del config["search_space"]["feature_perturbation"]["x_eps"]  # type: ignore[index]

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_config(tmp_dir, config)
            parsed = search_script.load_search_config(path)
            batch = search_script.build_batch_spec(
                search_config=parsed,
                output_root=Path(tmp_dir) / "out",
                config_copy_source=path,
            )

        self.assertEqual(parsed["search_space"]["feature_perturbation"]["x_eps"], ["inf"])
        self.assertEqual(len(batch.jobs), 1)
        self.assertEqual(batch.jobs[0].job_spec["fixed_params"]["feature"], "sim")
        self.assertEqual(batch.jobs[0].job_spec["fixed_params"]["x_eps"], "inf")

    def test_load_search_config_defaults_empty_x_eps_for_sim(self):
        config = _base_config()
        config["search_space"]["feature_transformation"]["features"] = ["sim"]  # type: ignore[index]
        config["search_space"]["feature_transformation"]["sim_reference_eps"] = ["1"]  # type: ignore[index]
        config["search_space"]["feature_perturbation"]["x_eps"] = []  # type: ignore[index]

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_config(tmp_dir, config)
            parsed = search_script.load_search_config(path)

        self.assertEqual(parsed["search_space"]["feature_perturbation"]["x_eps"], ["inf"])

    def test_load_search_config_defaults_missing_scale_to_one_for_raw(self):
        config = _base_config()
        del config["search_space"]["feature_transformation"]["scale"]  # type: ignore[index]

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_config(tmp_dir, config)
            parsed = search_script.load_search_config(path)

        self.assertEqual(parsed["search_space"]["feature_transformation"]["scale"], ["1"])

    def test_load_search_config_allows_nondefault_scale_when_raw_selected(self):
        config = _base_config()
        config["search_space"]["feature_transformation"]["scale"] = ["2"]  # type: ignore[index]

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_config(tmp_dir, config)
            parsed = search_script.load_search_config(path)
            batch = search_script.build_batch_spec(
                search_config=parsed,
                output_root=Path(tmp_dir) / "out",
                config_copy_source=path,
            )

        self.assertEqual(batch.jobs[0].job_spec["fixed_params"]["scale"], "2")
        self.assertIn("scale=2", str(batch.jobs[0].job_dir))

    def test_load_search_config_allows_feature_perturbation_shortcut_for_rewrite_features(self):
        config = _base_config()
        config["search_space"]["feature_transformation"]["features"] = ["shared"]  # type: ignore[index]
        config["search_space"]["feature_transformation"]["feature_dim"] = [8]  # type: ignore[index]
        config["search_space"]["feature_transformation"]["shared_value"] = ["1"]  # type: ignore[index]
        config["search_space"]["feature_perturbation"] = []  # type: ignore[assignment]

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_config(tmp_dir, config)
            parsed = search_script.load_search_config(path)
            batch = search_script.build_batch_spec(
                search_config=parsed,
                output_root=Path(tmp_dir) / "out",
                config_copy_source=path,
            )

        self.assertEqual(
            parsed["search_space"]["feature_perturbation"],
            {
                "mechanisms": ["mbm"],
                "x_eps": ["inf"],
                "m": ["best"],
            },
        )
        self.assertEqual(len(batch.jobs), 1)
        self.assertEqual(batch.jobs[0].job_spec["fixed_params"]["mechanism"], "mbm")
        self.assertEqual(batch.jobs[0].job_spec["fixed_params"]["x_eps"], "inf")
        self.assertEqual(batch.jobs[0].job_spec["fixed_params"]["m"], "best")

    def test_load_search_config_defaults_missing_scale_to_one_for_rewrite_only(self):
        config = _base_config()
        config["search_space"]["feature_transformation"]["features"] = ["shared"]  # type: ignore[index]
        config["search_space"]["feature_transformation"]["feature_dim"] = [8]  # type: ignore[index]
        config["search_space"]["feature_transformation"]["shared_value"] = ["1"]  # type: ignore[index]
        del config["search_space"]["feature_transformation"]["scale"]  # type: ignore[index]
        config["search_space"]["feature_perturbation"] = []  # type: ignore[assignment]

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_config(tmp_dir, config)
            parsed = search_script.load_search_config(path)
            batch = search_script.build_batch_spec(
                search_config=parsed,
                output_root=Path(tmp_dir) / "out",
                config_copy_source=path,
            )

        self.assertEqual(parsed["search_space"]["feature_transformation"]["scale"], ["1"])
        self.assertEqual(batch.jobs[0].job_spec["fixed_params"]["scale"], "1")
        self.assertNotIn("scale=", str(batch.jobs[0].job_dir))

    def test_load_search_config_allows_random_signed_onehot_rewrite_only_search(self):
        config = _base_config()
        config["search_space"]["feature_transformation"]["features"] = ["random_signed_onehot"]  # type: ignore[index]
        config["search_space"]["feature_transformation"]["feature_dim"] = [8]  # type: ignore[index]
        config["search_space"]["feature_perturbation"] = []  # type: ignore[assignment]

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_config(tmp_dir, config)
            parsed = search_script.load_search_config(path)
            batch = search_script.build_batch_spec(
                search_config=parsed,
                output_root=Path(tmp_dir) / "out",
                config_copy_source=path,
            )

        self.assertEqual(len(batch.jobs), 1)
        self.assertEqual(batch.jobs[0].job_spec["fixed_params"]["feature"], "random_signed_onehot")
        self.assertEqual(batch.jobs[0].job_spec["fixed_params"]["feature_dim"], 8)
        self.assertEqual(batch.jobs[0].job_spec["fixed_params"]["scale"], "1")
        self.assertIsNone(batch.jobs[0].job_spec["fixed_params"]["random_normal_mean"])
        self.assertIsNone(batch.jobs[0].job_spec["fixed_params"]["random_normal_std"])
        self.assertIsNone(batch.jobs[0].job_spec["fixed_params"]["shared_value"])

    def test_build_batch_spec_includes_nondefault_scale_in_job_dir(self):
        config = _base_config()
        config["search_space"]["feature_transformation"]["features"] = ["shared"]  # type: ignore[index]
        config["search_space"]["feature_transformation"]["feature_dim"] = [8]  # type: ignore[index]
        config["search_space"]["feature_transformation"]["shared_value"] = ["1"]  # type: ignore[index]
        config["search_space"]["feature_transformation"]["scale"] = ["2"]  # type: ignore[index]
        config["search_space"]["feature_perturbation"] = []  # type: ignore[assignment]

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_config(tmp_dir, config)
            parsed = search_script.load_search_config(path)
            batch = search_script.build_batch_spec(
                search_config=parsed,
                output_root=Path(tmp_dir) / "out",
                config_copy_source=path,
            )

        self.assertEqual(batch.jobs[0].job_spec["fixed_params"]["scale"], "2")
        self.assertIn("scale=2", str(batch.jobs[0].job_dir))

    def test_mixed_raw_and_rewrite_configs_use_default_scale_for_rewrite_jobs(self):
        config = _base_config()
        config["search_space"]["feature_transformation"]["features"] = ["raw", "shared"]  # type: ignore[index]
        config["search_space"]["feature_transformation"]["feature_dim"] = [8]  # type: ignore[index]
        config["search_space"]["feature_transformation"]["shared_value"] = ["1"]  # type: ignore[index]
        config["search_space"]["feature_transformation"]["scale"] = []  # type: ignore[index]

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_config(tmp_dir, config)
            parsed = search_script.load_search_config(path)
            batch = search_script.build_batch_spec(
                search_config=parsed,
                output_root=Path(tmp_dir) / "out",
                config_copy_source=path,
            )

        self.assertEqual(len(batch.jobs), 2)
        jobs_by_feature = {
            job.job_spec["fixed_params"]["feature"]: job.job_spec["fixed_params"]
            for job in batch.jobs
        }
        self.assertEqual(jobs_by_feature["raw"]["scale"], "1")
        self.assertEqual(jobs_by_feature["shared"]["scale"], "1")

    def test_operator_search_does_not_require_feature_dim(self):
        config = _base_config()
        config["search_space"]["feature_transformation"]["features"] = ["operator"]  # type: ignore[index]
        config["search_space"]["feature_perturbation"] = []  # type: ignore[assignment]

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_config(tmp_dir, config)
            parsed = search_script.load_search_config(path)
            batch = search_script.build_batch_spec(
                search_config=parsed,
                output_root=Path(tmp_dir) / "out",
                config_copy_source=path,
            )

        self.assertEqual(len(batch.jobs), 1)
        self.assertEqual(batch.jobs[0].job_spec["fixed_params"]["feature"], "operator")
        self.assertIsNone(batch.jobs[0].job_spec["fixed_params"]["feature_dim"])
        self.assertEqual(batch.jobs[0].job_spec["fixed_params"]["scale"], "1")

    def test_operator_preprojection_axes_expand_outer_variants_and_dedup_smoother(self):
        config = _base_config()
        config["search_space"]["feature_transformation"]["features"] = ["operator"]  # type: ignore[index]
        config["search_space"]["feature_transformation"]["feature_preprojection"] = [False, True]  # type: ignore[index]
        config["search_space"]["feature_transformation"]["preprojection_output_dim"] = [8, 16]  # type: ignore[index]
        config["search_space"]["feature_perturbation"] = []  # type: ignore[assignment]
        config["search_space"]["calibrator"]["x_steps"] = [0, 2]  # type: ignore[index]
        config["search_space"]["calibrator"]["smoother"] = ["kprop", "hoa"]  # type: ignore[index]

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_config(tmp_dir, config)
            parsed = search_script.load_search_config(path)
            batch = search_script.build_batch_spec(
                search_config=parsed,
                output_root=Path(tmp_dir) / "out",
                config_copy_source=path,
            )

        self.assertEqual(len(batch.jobs), 3)
        fixed_params = [job.job_spec["fixed_params"] for job in batch.jobs]
        preprojection_values = sorted(
            (params["feature_preprojection"], params["preprojection_output_dim"])
            for params in fixed_params
        )
        self.assertEqual(preprojection_values, [(False, None), (True, 8), (True, 16)])
        self.assertTrue(all(params["smoother"] is None for params in fixed_params))

    def test_load_search_config_allows_missing_norm_fields_for_rewrite_features(self):
        config = _base_config()
        config["search_space"]["feature_transformation"]["features"] = ["shared"]  # type: ignore[index]
        config["search_space"]["feature_transformation"]["feature_dim"] = [8]  # type: ignore[index]
        config["search_space"]["feature_transformation"]["shared_value"] = ["1"]  # type: ignore[index]
        del config["search_space"]["calibrator"]["norm"]  # type: ignore[index]
        del config["search_space"]["calibrator"]["norm_scale"]  # type: ignore[index]

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_config(tmp_dir, config)
            parsed = search_script.load_search_config(path)
            batch = search_script.build_batch_spec(
                search_config=parsed,
                output_root=Path(tmp_dir) / "out",
                config_copy_source=path,
            )

        self.assertEqual(parsed["search_space"]["calibrator"]["norm"], [False])
        self.assertEqual(parsed["search_space"]["calibrator"]["norm_scale"], [])
        self.assertEqual(len(batch.jobs), 1)
        self.assertEqual(batch.jobs[0].job_spec["fixed_params"]["norm"], False)
        self.assertEqual(batch.jobs[0].job_spec["fixed_params"]["norm_scale"], "none")

    def test_load_search_config_allows_missing_smoother_when_x_steps_are_zero(self):
        config = _base_config()
        del config["search_space"]["calibrator"]["smoother"]  # type: ignore[index]

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_config(tmp_dir, config)
            parsed = search_script.load_search_config(path)
            batch = search_script.build_batch_spec(
                search_config=parsed,
                output_root=Path(tmp_dir) / "out",
                config_copy_source=path,
            )

        self.assertEqual(parsed["search_space"]["calibrator"]["smoother"], ["kprop"])
        self.assertEqual(len(batch.jobs), 1)
        self.assertEqual(batch.jobs[0].job_spec["fixed_params"]["smoother"], "kprop")

    def test_operator_search_allows_missing_smoother_even_with_positive_x_steps(self):
        config = _base_config()
        config["search_space"]["feature_transformation"]["features"] = ["operator"]  # type: ignore[index]
        config["search_space"]["feature_perturbation"] = []  # type: ignore[assignment]
        config["search_space"]["calibrator"]["x_steps"] = [2]  # type: ignore[index]
        del config["search_space"]["calibrator"]["smoother"]  # type: ignore[index]

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_config(tmp_dir, config)
            parsed = search_script.load_search_config(path)
            batch = search_script.build_batch_spec(
                search_config=parsed,
                output_root=Path(tmp_dir) / "out",
                config_copy_source=path,
            )

        self.assertEqual(parsed["search_space"]["calibrator"]["smoother"], ["kprop"])
        self.assertEqual(batch.jobs[0].job_spec["fixed_params"]["smoother"], None)

    def test_load_search_config_still_requires_feature_perturbation_for_raw(self):
        config = _base_config()
        config["search_space"]["feature_perturbation"] = []  # type: ignore[assignment]

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_config(tmp_dir, config)
            with self.assertRaisesRegex(search_script.SearchError, "search_space.feature_perturbation"):
                search_script.load_search_config(path)

    def test_load_search_config_still_requires_norm_fields_for_raw(self):
        config = _base_config()
        del config["search_space"]["calibrator"]["norm"]  # type: ignore[index]
        del config["search_space"]["calibrator"]["norm_scale"]  # type: ignore[index]

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_config(tmp_dir, config)
            with self.assertRaisesRegex(search_script.SearchError, "search_space.calibrator"):
                search_script.load_search_config(path)

    def test_load_search_config_still_requires_smoother_when_positive_x_steps_exist(self):
        config = _base_config()
        config["search_space"]["calibrator"]["x_steps"] = [0, 2]  # type: ignore[index]
        del config["search_space"]["calibrator"]["smoother"]  # type: ignore[index]

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_config(tmp_dir, config)
            with self.assertRaisesRegex(search_script.SearchError, "search_space.calibrator"):
                search_script.load_search_config(path)

    def test_load_search_config_rejects_tao2_when_use_nfr_is_always_false(self):
        config = _base_config()
        config["search_space"]["nfr"]["tao2"] = ["0.2"]  # type: ignore[index]

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_config(tmp_dir, config)
            with self.assertRaisesRegex(search_script.SearchError, "search_space.nfr.tao2"):
                search_script.load_search_config(path)

    def test_build_batch_spec_uses_hardcoded_inner_candidate_axes(self):
        config = _base_config()
        config["search_space"]["calibrator"]["x_steps"] = [0, 2]  # type: ignore[index]
        config["search_space"]["trainer"]["learning_rate"] = ["0.001", "0.01"]  # type: ignore[index]
        config["search_space"]["trainer"]["weight_decay"] = ["0"]  # type: ignore[index]
        config["search_space"]["model"]["dropout"] = ["0.25", "0.5"]  # type: ignore[index]

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_config(tmp_dir, config)
            parsed = search_script.load_search_config(path)
            batch = search_script.build_batch_spec(
                search_config=parsed,
                output_root=Path(tmp_dir) / "out",
                config_copy_source=path,
            )

        self.assertEqual(len(batch.jobs), 1)
        job = batch.jobs[0]
        candidates = job.job_spec["candidates"]
        self.assertEqual(len(candidates), 2 * 2 * 1 * 2 * 1)
        self.assertIn("dataset=cora", str(job.job_dir))
        self.assertIn("feature=raw", str(job.job_dir))

    def test_build_batch_spec_uses_tao2_as_candidate_axis_when_nfr_enabled(self):
        config = _base_config()
        config["search_space"]["feature_perturbation"]["x_eps"] = ["1"]  # type: ignore[index]
        config["search_space"]["nfr"]["use_nfr"] = [True]  # type: ignore[index]
        config["search_space"]["nfr"]["tao2"] = ["0.1", "0.3"]  # type: ignore[index]

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_config(tmp_dir, config)
            parsed = search_script.load_search_config(path)
            batch = search_script.build_batch_spec(
                search_config=parsed,
                output_root=Path(tmp_dir) / "out",
                config_copy_source=path,
            )

        self.assertEqual(len(batch.jobs), 1)
        candidates = batch.jobs[0].job_spec["candidates"]
        self.assertEqual({candidate["tao2"] for candidate in candidates}, {"0.1", "0.3"})

    def test_build_batch_spec_uses_top_level_seed_as_job_base_seed(self):
        config = _base_config()
        config["seed"] = 2024

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_config(tmp_dir, config)
            parsed = search_script.load_search_config(path)
            batch = search_script.build_batch_spec(
                search_config=parsed,
                output_root=Path(tmp_dir) / "out",
                config_copy_source=path,
            )

        self.assertEqual(len(batch.jobs), 1)
        self.assertEqual(batch.jobs[0].job_spec["base_seed"], 2024)

    def test_load_search_config_canonicalizes_form_compatible_dataset_names(self):
        config = _base_config()
        config["search_space"]["dataset"]["datasets"] = ["books_children"]  # type: ignore[index]

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_config(tmp_dir, config)
            parsed = search_script.load_search_config(path)

        self.assertEqual(parsed["search_space"]["dataset"]["datasets"], ["Books-Children"])

    def test_load_search_config_accepts_citeseer_and_builds_canonical_job_dir(self):
        config = _base_config()
        config["search_space"]["dataset"]["datasets"] = ["CiteSeer"]  # type: ignore[index]

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_config(tmp_dir, config)
            parsed = search_script.load_search_config(path)
            batch = search_script.build_batch_spec(
                search_config=parsed,
                output_root=Path(tmp_dir) / "out",
                config_copy_source=path,
            )

        self.assertEqual(parsed["search_space"]["dataset"]["datasets"], ["citeseer"])
        self.assertEqual(len(batch.jobs), 1)
        self.assertEqual(batch.jobs[0].job_spec["fixed_params"]["dataset"], "citeseer")
        self.assertIn("dataset=citeseer", str(batch.jobs[0].job_dir))

    def test_load_search_config_rejects_semantic_dataset_aliases(self):
        config = _base_config()
        config["search_space"]["dataset"]["datasets"] = ["Arxiv"]  # type: ignore[index]

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_config(tmp_dir, config)
            with self.assertRaisesRegex(search_script.SearchError, "unsupported"):
                search_script.load_search_config(path)


if __name__ == "__main__":
    unittest.main()
