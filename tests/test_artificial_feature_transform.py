import argparse
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional in minimal test envs
    torch = None

try:
    from main import validate_feature_rewrite_args, validate_sim_args
except Exception:  # pragma: no cover - optional in minimal test envs
    validate_feature_rewrite_args = None
    validate_sim_args = None

try:
    from transforms import FeaturePerturbation, FeatureTransform
except Exception:  # pragma: no cover - optional in minimal test envs
    FeaturePerturbation = None
    FeatureTransform = None

try:
    from utils import add_parameters_as_argument
except Exception:  # pragma: no cover - optional in minimal test envs
    add_parameters_as_argument = None


def make_graph_data():
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 0],
            [1, 0, 2, 1, 3, 2, 0, 3],
        ],
        dtype=torch.long,
    )
    x = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    return SimpleNamespace(x=x, edge_index=edge_index, num_nodes=4)


@unittest.skipIf(
    torch is None or FeatureTransform is None,
    'artificial feature rewrite dependencies are unavailable.',
)
class ArtificialFeatureTransformTests(unittest.TestCase):
    def test_supported_features_include_all_new_modes(self):
        self.assertEqual(
            FeatureTransform.supported_features,
            [
                'raw',
                'sim',
                'random_normal',
                'random_signed_onehot',
                'shared',
                'node_degree',
                'degree_bucket_range',
                'degree_bucket_distribution',
                'pagerank',
                'operator',
                'eigen',
                'eigen_norm',
                'deepwalk',
            ],
        )

    def test_rewrite_path_binds_to_current_submodule(self):
        data = make_graph_data()
        FeatureTransform(
            feature='shared',
            feature_dim=2,
            shared_value=1.0,
        )(data)

        module = sys.modules['artificial_node_feature_generator']
        self.assertIn(
            str((Path(__file__).resolve().parents[1] / 'submodule' / 'artificial-node-feature_generator').resolve()),
            str(Path(module.__file__).resolve()),
        )
        self.assertNotIn('/home/placitudo/artificial-node-feature_generator', str(Path(module.__file__).resolve()))

    def test_shared_feature_rewrites_dimension(self):
        data = make_graph_data()
        transform = FeatureTransform(
            feature='shared',
            feature_dim=2,
            shared_value=3.0,
        )
        rewritten = transform(data)
        self.assertIsNot(rewritten, data)
        self.assertEqual(tuple(rewritten.x.shape), (4, 2))
        self.assertTrue(torch.allclose(rewritten.x, torch.full((4, 2), 3.0)))

    def test_shared_feature_scale_is_not_applied_during_rewrite(self):
        data = make_graph_data()
        transform = FeatureTransform(
            feature='shared',
            feature_dim=2,
            shared_value=3.0,
            scale=2.5,
        )
        rewritten = transform(data)
        self.assertTrue(torch.allclose(rewritten.x, torch.full((4, 2), 3.0)))

    def test_random_normal_uses_repeat_seed(self):
        data = make_graph_data()
        first = FeatureTransform(feature='random_normal', feature_dim=5).set_rewrite_seed(7)(data)
        second = FeatureTransform(feature='random_normal', feature_dim=5).set_rewrite_seed(7)(data)
        third = FeatureTransform(feature='random_normal', feature_dim=5).set_rewrite_seed(8)(data)
        self.assertTrue(torch.allclose(first.x, second.x))
        self.assertFalse(torch.allclose(first.x, third.x))

    def test_random_signed_onehot_uses_repeat_seed_and_signed_onehot_structure(self):
        data = make_graph_data()
        first = FeatureTransform(feature='random_signed_onehot', feature_dim=5).set_rewrite_seed(7)(data)
        second = FeatureTransform(feature='random_signed_onehot', feature_dim=5).set_rewrite_seed(7)(data)
        third = FeatureTransform(feature='random_signed_onehot', feature_dim=5).set_rewrite_seed(8)(data)
        self.assertTrue(torch.allclose(first.x, second.x))
        self.assertFalse(torch.allclose(first.x, third.x))
        self.assertTrue(torch.equal(torch.count_nonzero(first.x, dim=1), torch.ones(4, dtype=torch.long)))
        self.assertTrue(torch.allclose(first.x.abs().sum(dim=1), torch.ones(4, dtype=first.x.dtype)))
        self.assertTrue(bool(torch.all((first.x == 0) | (first.x == 1) | (first.x == -1)).item()))


@unittest.skipIf(
    torch is None or FeaturePerturbation is None,
    'feature perturbation dependencies are unavailable.',
)
class ArtificialFeaturePerturbationBypassTests(unittest.TestCase):
    def test_non_raw_feature_skips_perturbation_even_with_finite_x_eps(self):
        base_x = torch.tensor([[0.1, 0.5], [0.2, 0.7]], dtype=torch.float32)
        data = SimpleNamespace(x=base_x.clone())
        perturb = FeaturePerturbation(
            mechanism='mbm',
            x_eps=0.25,
            m=1,
            norm=True,
            feature='shared',
        )
        perturb(data)

        self.assertTrue(torch.equal(data.x, base_x))
        self.assertIsNone(data.output_range)
        self.assertEqual(set(vars(data).keys()), {'x', 'feature_mechanism', 'output_range'})


@unittest.skipIf(
    validate_feature_rewrite_args is None or validate_sim_args is None or add_parameters_as_argument is None,
    'CLI validation dependencies are unavailable.',
)
class ArtificialFeatureValidationTests(unittest.TestCase):
    def _build_parser(self):
        parser = argparse.ArgumentParser(prog='feature-args-test')
        add_parameters_as_argument(FeatureTransform, parser)
        parser.add_argument('--mechanism', default='mbm')
        return parser

    def _validate(self, argv):
        parser = self._build_parser()
        args = parser.parse_args(argv)
        validate_sim_args(parser, args)
        validate_feature_rewrite_args(parser, args)
        return args

    def test_parser_accepts_new_feature_choice(self):
        args = self._validate(['--feature', 'shared', '--feature-dim', '4'])
        self.assertEqual(args.feature, 'shared')
        self.assertEqual(args.feature_dim, 4)
        self.assertEqual(args.scale, 1.0)

    def test_rewrite_feature_requires_feature_dim(self):
        with self.assertRaises(SystemExit):
            self._validate(['--feature', 'shared'])

    def test_raw_rejects_feature_dim(self):
        with self.assertRaises(SystemExit):
            self._validate(['--feature', 'raw', '--feature-dim', '4'])

    def test_raw_accepts_scale(self):
        args = self._validate(['--feature', 'raw', '--scale', '2'])
        self.assertEqual(args.scale, 2.0)

    def test_non_sim_rejects_sim_reference_eps(self):
        with self.assertRaises(SystemExit):
            self._validate(['--feature', 'shared', '--feature-dim', '4', '--sim-reference-eps', '1.0'])

    def test_sim_accepts_scale(self):
        args = self._validate(['--feature', 'sim', '--sim-reference-eps', '1.0', '--scale', '2'])
        self.assertEqual(args.scale, 2.0)

    def test_operator_does_not_require_feature_dim(self):
        args = self._validate(['--feature', 'operator'])
        self.assertEqual(args.feature, 'operator')
        self.assertIsNone(args.feature_dim)

    def test_operator_rejects_feature_dim(self):
        with self.assertRaises(SystemExit):
            self._validate(['--feature', 'operator', '--feature-dim', '4'])

    def test_feature_preprojection_requires_operator(self):
        with self.assertRaises(SystemExit):
            self._validate(['--feature', 'shared', '--feature-dim', '4', '--feature-preprojection'])

    def test_operator_preprojection_requires_output_dim(self):
        with self.assertRaises(SystemExit):
            self._validate(['--feature', 'operator', '--feature-preprojection'])

    def test_operator_accepts_preprojection_args(self):
        args = self._validate([
            '--feature', 'operator',
            '--feature-preprojection',
            '--preprojection-output-dim', '8',
        ])
        self.assertTrue(args.feature_preprojection)
        self.assertEqual(args.preprojection_output_dim, 8)

    def test_irrelevant_feature_specific_parameter_is_rejected(self):
        with self.assertRaises(SystemExit):
            self._validate(['--feature', 'shared', '--feature-dim', '4', '--random-normal-mean', '1.0'])

    def test_scale_must_be_positive(self):
        with self.assertRaises(SystemExit):
            self._validate(['--feature', 'shared', '--feature-dim', '4', '--scale', '0'])

    def test_random_normal_std_must_be_positive(self):
        with self.assertRaises(SystemExit):
            self._validate([
                '--feature', 'random_normal',
                '--feature-dim', '4',
                '--random-normal-std', '0',
            ])

    def test_deepwalk_positive_integer_constraints(self):
        with self.assertRaises(SystemExit):
            self._validate([
                '--feature', 'deepwalk',
                '--feature-dim', '4',
                '--deepwalk-workers', '0',
            ])


@unittest.skipIf(
    FeatureTransform is None,
    'artificial feature transform class is unavailable.',
)
class CoraArtificialFeatureSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.python_bin = Path('/home/placitudo/APP/miniconda3/envs/HZWDP/bin/python')
        if not cls.python_bin.exists():
            raise unittest.SkipTest('HZWDP python is unavailable')

    def test_cora_smoke_commands_cover_all_new_modes(self):
        feature_commands = {
            'random_normal': ['--feature-dim', '8'],
            'random_signed_onehot': ['--feature-dim', '8'],
            'shared': ['--feature-dim', '8'],
            'node_degree': ['--feature-dim', '8'],
            'degree_bucket_range': ['--feature-dim', '8', '--degree-bucket-num-buckets', '4', '--degree-bucket-range-max', '8'],
            'degree_bucket_distribution': ['--feature-dim', '8', '--degree-bucket-num-buckets', '4'],
            'pagerank': ['--feature-dim', '8'],
            'operator': [],
            'eigen': ['--feature-dim', '8'],
            'eigen_norm': ['--feature-dim', '8'],
            'deepwalk': [
                '--feature-dim', '8',
                '--deepwalk-walk-length', '4',
                '--deepwalk-number-walks', '2',
                '--deepwalk-window-size', '2',
                '--deepwalk-workers', '1',
                '--deepwalk-undirected', 'true',
            ],
        }
        self.assertEqual(set(feature_commands), set(FeatureTransform.rewrite_supported_features))

    def test_one_real_cora_invocation_template_is_well_formed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                str(self.python_bin),
                'main.py',
                '--dataset', 'cora',
                '--feature', 'shared',
                '--feature-dim', '8',
                '--device', 'cpu',
                '--max-epochs', '1',
                '--patience', '1',
                '--repeats', '1',
                '--output-dir', tmpdir,
            ]
            completed = subprocess.run(
                cmd,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0 and 'ModuleNotFoundError' in f'{completed.stdout}\n{completed.stderr}':
                self.skipTest('runtime dependencies for main.py are unavailable in HZWDP')
            self.assertEqual(completed.returncode, 0, msg=completed.stderr)

    def test_operator_cora_invocation_without_projection_is_well_formed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                str(self.python_bin),
                'main.py',
                '--dataset', 'cora',
                '--feature', 'operator',
                '--device', 'cpu',
                '--max-epochs', '1',
                '--patience', '1',
                '--repeats', '1',
                '--output-dir', tmpdir,
            ]
            completed = subprocess.run(
                cmd,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0 and 'ModuleNotFoundError' in f'{completed.stdout}\n{completed.stderr}':
                self.skipTest('runtime dependencies for main.py are unavailable in HZWDP')
            self.assertEqual(completed.returncode, 0, msg=completed.stderr)

    def test_operator_cora_invocation_with_projection_is_well_formed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                str(self.python_bin),
                'main.py',
                '--dataset', 'cora',
                '--feature', 'operator',
                '--feature-preprojection',
                '--preprojection-output-dim', '8',
                '--device', 'cpu',
                '--max-epochs', '1',
                '--patience', '1',
                '--repeats', '1',
                '--output-dir', tmpdir,
            ]
            completed = subprocess.run(
                cmd,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0 and 'ModuleNotFoundError' in f'{completed.stdout}\n{completed.stderr}':
                self.skipTest('runtime dependencies for main.py are unavailable in HZWDP')
            self.assertEqual(completed.returncode, 0, msg=completed.stderr)


if __name__ == '__main__':
    unittest.main()
