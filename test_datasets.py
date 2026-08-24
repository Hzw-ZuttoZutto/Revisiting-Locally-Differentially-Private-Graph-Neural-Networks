import tempfile
import unittest
from functools import partial
from pathlib import Path
from unittest import mock

import torch
import yaml
from torch_geometric.data import Data
from torch_geometric.datasets import (
    Actor,
    AttributedGraphDataset,
    Flickr,
    HeterophilousGraphDataset,
    Reddit,
    Reddit2,
    WikipediaNetwork,
)

import datasets
from hparams_search_scripts import run_mechanism_hparam_search


HETEROPHILOUS_DATASETS = (
    "actor",
    "chameleon",
    "squirrel",
    "roman-empire",
    "amazon-ratings",
    "attributedgraph-flickr",
    "flickr",
    "reddit",
)


class DatasetRegistryTests(unittest.TestCase):
    def test_heterophilous_datasets_are_supported(self):
        supported = datasets.list_supported_datasets()
        for name in HETEROPHILOUS_DATASETS:
            with self.subTest(dataset=name):
                self.assertIn(name, supported)
                self.assertTrue(datasets.is_dataset_supported(name))
                self.assertEqual(datasets.resolve_dataset_name(name.upper()), name)

    def test_pyg_dataset_builders(self):
        self.assertIs(datasets.BUILTIN_DATASET_BUILDERS["actor"], Actor)
        self.assertIs(datasets.BUILTIN_DATASET_BUILDERS["flickr"], Flickr)
        self.assertIs(datasets.BUILTIN_DATASET_BUILDERS["reddit"], Reddit)
        self.assertIsNot(datasets.BUILTIN_DATASET_BUILDERS["reddit"], Reddit2)

        for name in ("chameleon", "squirrel"):
            with self.subTest(dataset=name):
                builder = datasets.BUILTIN_DATASET_BUILDERS[name]
                self.assertIsInstance(builder, partial)
                self.assertIs(builder.func, WikipediaNetwork)
                self.assertEqual(builder.keywords["name"], name)
                self.assertIs(builder.keywords["geom_gcn_preprocess"], True)

        for name in ("roman-empire", "amazon-ratings"):
            with self.subTest(dataset=name):
                builder = datasets.BUILTIN_DATASET_BUILDERS[name]
                self.assertIsInstance(builder, partial)
                self.assertIs(builder.func, HeterophilousGraphDataset)
                self.assertEqual(builder.keywords["name"], name)

        builder = datasets.BUILTIN_DATASET_BUILDERS["attributedgraph-flickr"]
        self.assertIsInstance(builder, partial)
        self.assertIs(builder.func, AttributedGraphDataset)
        self.assertEqual(builder.keywords["name"], "Flickr")

    def test_builtin_loader_uses_dataset_specific_cache_root(self):
        raw = Data(
            x=torch.eye(3),
            y=torch.tensor([0, 1, 0]),
            edge_index=torch.tensor([[0, 1], [1, 2]]),
        )

        class FakeDataset:
            def __init__(self, root):
                self.root = root

            def __getitem__(self, index):
                self.test_case.assertEqual(index, 0)
                return raw

        with tempfile.TemporaryDirectory() as temp_dir:
            for name in HETEROPHILOUS_DATASETS:
                with self.subTest(dataset=name):
                    roots = []

                    def build(root):
                        roots.append(root)
                        instance = FakeDataset(root)
                        instance.test_case = self
                        return instance

                    with mock.patch.dict(datasets.BUILTIN_DATASET_BUILDERS, {name: build}):
                        bundle = datasets._load_builtin_raw_dataset(
                            datasets.DATASET_SPECS[name], temp_dir
                        )
                    self.assertIs(bundle.data, raw)
                    self.assertEqual(roots, [str(Path(temp_dir) / name)])

    def test_search_config_accepts_and_expands_pyg_datasets(self):
        test_config = {
            "seed": 12345,
            "device": {
                "device": "cpu",
                "cpu_worker_count": 1,
                "gpu_ids": [],
                "max_parallel_per_gpu": None,
                "gpu_launch_interval_sec": None,
            },
            "defaults": {
                "dataset": {
                    "data_range": [0.0, 1.0],
                    "val_ratio": 0.25,
                    "test_ratio": 0.25,
                },
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
                    "grid": {"patience": 1, "repeats": 1},
                    "verify": {"patience": 1, "repeats": 1},
                },
            },
            "search_space": {
                "dataset": {
                    "datasets": [
                        "FLICKR",
                        "Reddit",
                        "Roman_Empire",
                        "AMAZON-RATINGS",
                        "AttributedGraph_Flickr",
                    ]
                },
                "feature_transformation": {
                    "features": ["raw"],
                    "sim_reference_eps": [],
                    "feature_dim": [],
                    "scale": [],
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
                    "mechanisms": ["mbm"],
                    "x_eps": ["inf"],
                    "m": ["best"],
                },
                "calibrator": {
                    "norm": [False],
                    "norm_scale": [],
                    "x_steps": [0],
                    "smoother": ["kprop"],
                },
                "model": {"backbones": ["gcn"], "dropout": [0.5]},
                "trainer": {"learning_rate": [0.01], "weight_decay": [0.0]},
                "nfr": {"use_nfr": [False], "tao2": []},
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            config_path = temp_root / "search.yaml"
            with config_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(test_config, handle, sort_keys=False)

            parsed = run_mechanism_hparam_search.load_search_config(config_path)
            batch_spec = run_mechanism_hparam_search.build_batch_spec(
                search_config=parsed,
                output_root=temp_root / "output",
                config_copy_source=config_path,
            )

        parsed_datasets = parsed["search_space"]["dataset"]["datasets"]
        expanded_datasets = {
            job.job_spec["fixed_params"]["dataset"] for job in batch_spec.jobs
        }
        self.assertEqual(
            parsed_datasets,
            [
                "flickr",
                "reddit",
                "roman-empire",
                "amazon-ratings",
                "attributedgraph-flickr",
            ],
        )
        self.assertEqual(
            expanded_datasets,
            {
                "flickr",
                "reddit",
                "roman-empire",
                "amazon-ratings",
                "attributedgraph-flickr",
            },
        )


class DatasetStandardizationTests(unittest.TestCase):
    def test_standardization_replaces_upstream_multisplit_masks(self):
        num_nodes = 12
        raw = Data(
            x=torch.arange(num_nodes * 3, dtype=torch.float).view(num_nodes, 3),
            y=torch.tensor([0, 1, 2] * 4),
            edge_index=torch.tensor(
                [[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]], dtype=torch.long
            ),
            train_mask=torch.zeros((num_nodes, 10), dtype=torch.bool),
            val_mask=torch.zeros((num_nodes, 10), dtype=torch.bool),
            test_mask=torch.zeros((num_nodes, 10), dtype=torch.bool),
        )

        result = datasets._standardize_loaded_data(
            datasets.DATASET_SPECS["actor"],
            datasets.RawDatasetBundle(raw),
            data_range=(0.0, 1.0),
            val_ratio=0.25,
            test_ratio=0.25,
        )

        self.assertEqual(result.name, "actor")
        self.assertEqual(result.num_nodes, num_nodes)
        self.assertEqual(result.num_classes, 3)
        self.assertIsNotNone(result.adj_t)
        for mask in (result.train_mask, result.val_mask, result.test_mask):
            self.assertEqual(mask.dtype, torch.bool)
            self.assertEqual(tuple(mask.shape), (num_nodes,))
        self.assertFalse(bool((result.train_mask & result.val_mask).any()))
        self.assertFalse(bool((result.train_mask & result.test_mask).any()))
        self.assertFalse(bool((result.val_mask & result.test_mask).any()))
        self.assertTrue(bool((result.train_mask | result.val_mask | result.test_mask).all()))
        self.assertGreaterEqual(float(result.x.min()), 0.0)
        self.assertLessEqual(float(result.x.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
