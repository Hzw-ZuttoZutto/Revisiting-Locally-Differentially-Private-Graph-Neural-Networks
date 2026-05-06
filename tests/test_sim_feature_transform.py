import argparse
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional in minimal test envs
    torch = None

try:
    from main import build_sim_epoch_refresh_callback, validate_sim_args
except Exception:  # pragma: no cover - optional in minimal test envs
    build_sim_epoch_refresh_callback = None
    validate_sim_args = None

try:
    from utils import add_parameters_as_argument
except Exception:  # pragma: no cover - optional in minimal test envs
    add_parameters_as_argument = None

try:
    from mechanisms import HighDimSquareWave, MultiBit, MultiDimPiecewise
    from transforms import FeaturePerturbation, FeatureTransform
except ModuleNotFoundError:
    FeatureTransform = None
    FeaturePerturbation = None
    MultiBit = None
    MultiDimPiecewise = None
    HighDimSquareWave = None


@unittest.skipIf(
    torch is None
    or FeatureTransform is None
    or MultiBit is None
    or MultiDimPiecewise is None
    or HighDimSquareWave is None,
    "sim transform dependencies are unavailable.",
)
class SimFeatureTransformTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_supported_features_keep_raw_and_sim_first(self):
        self.assertEqual(FeatureTransform.supported_features[:2], ["raw", "sim"])

    def test_raw_feature_is_noop(self):
        base_x = torch.tensor([[0.1, 0.5, 0.9]], dtype=torch.float32)
        data = SimpleNamespace(x=base_x.clone())
        transform = FeatureTransform(feature="raw")
        transform(data)
        self.assertTrue(torch.equal(data.x, base_x))

    @unittest.skipIf(add_parameters_as_argument is None, "argument builder is unavailable.")
    def test_argparse_rejects_removed_feature_modes(self):
        parser = argparse.ArgumentParser(prog="feature-args-test")
        add_parameters_as_argument(FeatureTransform, parser)

        for removed_feature in ("rnd", "one", "ohd"):
            with self.subTest(feature=removed_feature):
                with self.assertRaises(SystemExit):
                    parser.parse_args(["--feature", removed_feature])

    @staticmethod
    def _expected_kappa_and_support(mechanism, eps_per_dim):
        if mechanism == "mbm":
            return math.tanh(eps_per_dim / 2.0), 1.0
        if mechanism == "pm":
            return math.tanh(eps_per_dim / 4.0), 1.0
        if mechanism == "hds":
            one_minus_q = -math.expm1(-eps_per_dim)
            return 1.0 - one_minus_q / eps_per_dim, 2.0
        raise AssertionError(f"unsupported mechanism for test: {mechanism}")

    @staticmethod
    def _expected_output_range(norm, norm_scale, effective_gain, support):
        base_output_range = support / effective_gain
        if not norm:
            return base_output_range
        if str(norm_scale).strip().lower() == "none":
            return support
        return base_output_range / float(norm_scale)

    def test_sim_transform_rejects_legacy_sim_mode_kwarg(self):
        with self.assertRaises(TypeError):
            FeatureTransform(feature="sim", sim_mode=2, sim_reference_eps=1.0)

    def test_sim_calls_rectify_and_keeps_unsampled_center_at_zero(self):
        sim_reference_eps = 2.4
        m = 4
        shape = (7, 11)

        cases = [
            ("mbm", MultiBit),
            ("pm", MultiDimPiecewise),
            ("hds", HighDimSquareWave),
        ]

        for mechanism_name, mechanism_cls in cases:
            with self.subTest(mechanism=mechanism_name):
                data = SimpleNamespace(x=torch.zeros(shape, dtype=torch.float32))
                transform = FeatureTransform(
                    feature="sim",
                    mechanism=mechanism_name,
                    m=m,
                    sim_reference_eps=sim_reference_eps,
                )

                original_rectify = mechanism_cls.rectify
                rectify_calls = {"count": 0}

                def tracking_rectify(self, state):
                    rectify_calls["count"] += 1
                    return original_rectify(self, state)

                with patch.object(mechanism_cls, "rectify", new=tracking_rectify):
                    transform.refresh_sim_features(data)

                self.assertGreater(rectify_calls["count"], 0)
                centered_count = data.x.eq(0.0).sum(dim=1)
                self.assertTrue(torch.all(centered_count == data.x.size(1) - m).item())

                eps_per_dim = sim_reference_eps / m
                kappa, unit_support_scale = self._expected_kappa_and_support(
                    mechanism_name,
                    eps_per_dim,
                )
                effective_gain = (m / data.x.size(1)) * kappa
                expected_scale = unit_support_scale / effective_gain
                self.assertGreater(expected_scale, 0.0)
                self.assertAlmostEqual(data.output_range, expected_scale, places=12)
                self.assertEqual(set(vars(data).keys()), {"x", "output_range"})

    def test_sim_norm_true_none_matches_kappa_and_sampling_ratio_scaling(self):
        sim_reference_eps = 2.4
        m = 4
        shape = (7, 11)

        for mechanism_name in ["mbm", "pm", "hds"]:
            with self.subTest(mechanism=mechanism_name):
                torch.manual_seed(123)
                data_no_norm = SimpleNamespace(x=torch.zeros(shape, dtype=torch.float32))
                transform_no_norm = FeatureTransform(
                    feature="sim",
                    mechanism=mechanism_name,
                    m=m,
                    sim_reference_eps=sim_reference_eps,
                    norm=False,
                )
                transform_no_norm.refresh_sim_features(data_no_norm)

                torch.manual_seed(123)
                data_norm = SimpleNamespace(x=torch.zeros(shape, dtype=torch.float32))
                transform_norm = FeatureTransform(
                    feature="sim",
                    mechanism=mechanism_name,
                    m=m,
                    sim_reference_eps=sim_reference_eps,
                    norm=True,
                    norm_scale="none",
                )
                transform_norm.refresh_sim_features(data_norm)

                eps_per_dim = sim_reference_eps / m
                kappa, support = self._expected_kappa_and_support(mechanism_name, eps_per_dim)
                effective_gain = (m / shape[1]) * kappa
                expected = data_no_norm.x * kappa * (m / shape[1])
                self.assertTrue(torch.allclose(data_norm.x, expected, atol=1e-6, rtol=0.0))
                self.assertAlmostEqual(
                    data_no_norm.output_range,
                    self._expected_output_range(False, "none", effective_gain, support),
                    places=12,
                )
                self.assertAlmostEqual(
                    data_norm.output_range,
                    self._expected_output_range(True, "none", effective_gain, support),
                    places=12,
                )
                self.assertEqual(set(vars(data_norm).keys()), {"x", "output_range"})
                self.assertEqual(set(vars(data_no_norm).keys()), {"x", "output_range"})

    def test_sim_norm_true_with_norm_scale_uses_inverse_scale(self):
        sim_reference_eps = 2.4
        m = 4
        norm_scale = "2.5"
        shape = (7, 11)

        for mechanism_name in ["mbm", "pm", "hds"]:
            with self.subTest(mechanism=mechanism_name):
                torch.manual_seed(456)
                data_no_norm = SimpleNamespace(x=torch.zeros(shape, dtype=torch.float32))
                transform_no_norm = FeatureTransform(
                    feature="sim",
                    mechanism=mechanism_name,
                    m=m,
                    sim_reference_eps=sim_reference_eps,
                    norm=False,
                )
                transform_no_norm.refresh_sim_features(data_no_norm)

                torch.manual_seed(456)
                data_norm = SimpleNamespace(x=torch.zeros(shape, dtype=torch.float32))
                transform_norm = FeatureTransform(
                    feature="sim",
                    mechanism=mechanism_name,
                    m=m,
                    sim_reference_eps=sim_reference_eps,
                    norm=True,
                    norm_scale=norm_scale,
                )
                transform_norm.refresh_sim_features(data_norm)

                expected = data_no_norm.x * (1.0 / float(norm_scale))
                self.assertTrue(torch.allclose(data_norm.x, expected, atol=1e-6, rtol=0.0))
                eps_per_dim = sim_reference_eps / m
                kappa, support = self._expected_kappa_and_support(mechanism_name, eps_per_dim)
                effective_gain = (m / shape[1]) * kappa
                self.assertAlmostEqual(
                    data_no_norm.output_range,
                    self._expected_output_range(False, "none", effective_gain, support),
                    places=12,
                )
                self.assertAlmostEqual(
                    data_norm.output_range,
                    self._expected_output_range(True, norm_scale, effective_gain, support),
                    places=12,
                )
                self.assertEqual(set(vars(data_norm).keys()), {"x", "output_range"})
                self.assertEqual(set(vars(data_no_norm).keys()), {"x", "output_range"})

    def test_sim_rejects_unsupported_mechanism_in_transform(self):
        data = SimpleNamespace(x=torch.zeros((3, 5), dtype=torch.float32))
        transform = FeatureTransform(
            feature="sim",
            mechanism="lpm",
            m=2,
            sim_reference_eps=1.0,
        )
        with self.assertRaisesRegex(ValueError, "supports mechanism"):
            transform.refresh_sim_features(data)

    def test_sim_with_large_reference_eps_remains_finite(self):
        shape = (5, 4)
        huge_eps = 1_000_000.0

        for mechanism_name in ["mbm", "pm", "hds"]:
            with self.subTest(mechanism=mechanism_name):
                data = SimpleNamespace(x=torch.zeros(shape, dtype=torch.float32))
                transform = FeatureTransform(
                    feature="sim",
                    mechanism=mechanism_name,
                    m=2,
                    sim_reference_eps=huge_eps,
                )
                transform.refresh_sim_features(data)
                self.assertTrue(torch.isfinite(data.x).all().item())
                self.assertIsInstance(data.output_range, float)
                self.assertEqual(set(vars(data).keys()), {"x", "output_range"})


@unittest.skipIf(
    torch is None or FeaturePerturbation is None,
    "feature perturbation dependencies are unavailable.",
)
class SimFeaturePerturbationBypassTests(unittest.TestCase):
    def test_feature_sim_ignores_x_eps_and_keeps_input(self):
        base_x = torch.tensor([[0.1, 0.5, 0.9]], dtype=torch.float32)
        data = SimpleNamespace(x=base_x.clone())
        perturb = FeaturePerturbation(
            mechanism="mbm",
            x_eps=0.25,
            m=2,
            norm=True,
            feature="sim",
        )
        perturb(data)

        self.assertTrue(torch.equal(data.x, base_x))
        self.assertIsNone(data.output_range)
        self.assertEqual(set(vars(data).keys()), {"x", "feature_mechanism", "output_range"})

    def test_feature_sim_preserves_existing_output_range(self):
        base_x = torch.zeros((4, 5), dtype=torch.float32)
        data = SimpleNamespace(x=base_x.clone())
        transform = FeatureTransform(
            feature="sim",
            mechanism="mbm",
            m=2,
            sim_reference_eps=1.5,
        )
        torch.manual_seed(123)
        transform.refresh_sim_features(data)
        expected_x = data.x.clone()
        expected_output_range = data.output_range

        perturb = FeaturePerturbation(
            mechanism="mbm",
            x_eps=0.25,
            m=2,
            norm=True,
            feature="sim",
        )
        perturb(data)

        self.assertTrue(torch.equal(data.x, expected_x))
        self.assertEqual(data.output_range, expected_output_range)
        self.assertEqual(set(vars(data).keys()), {"x", "output_range", "feature_mechanism"})


@unittest.skipIf(
    torch is None or build_sim_epoch_refresh_callback is None,
    "sim refresh callback dependencies are unavailable.",
)
class SimEpochRefreshCallbackTests(unittest.TestCase):
    def test_refresh_callback_syncs_eval_output_range(self):
        args = SimpleNamespace(
            sim_epoch_refresh=True,
            feature="sim",
            sim_reference_eps=2.4,
            mechanism="pm",
            m=2,
            x_eps=float("inf"),
            norm=False,
            norm_scale="none",
        )
        callback = build_sim_epoch_refresh_callback(args)
        data = SimpleNamespace(x=torch.zeros((4, 5), dtype=torch.float32))
        eval_data = SimpleNamespace(
            x=torch.ones((4, 5), dtype=torch.float32),
            output_range=None,
        )

        torch.manual_seed(321)
        callback(epoch=1, data=data, eval_data=eval_data)

        self.assertTrue(torch.equal(eval_data.x, data.x))
        self.assertEqual(eval_data.output_range, data.output_range)
        self.assertIsInstance(data.output_range, float)


@unittest.skipIf(validate_sim_args is None, "main.validate_sim_args is unavailable.")
class SimValidationTests(unittest.TestCase):
    def _validate_raises(self, args):
        parser = argparse.ArgumentParser(prog="sim-args-test")
        with self.assertRaises(SystemExit):
            validate_sim_args(parser, args)

    def test_sim_requires_reference_eps_when_feature_sim(self):
        args = argparse.Namespace(
            sim_epoch_refresh=False,
            feature="sim",
            sim_reference_eps=None,
            mechanism="mbm",
            m="2",
        )
        self._validate_raises(args)

    def test_non_sim_feature_does_not_require_reference_eps(self):
        args = argparse.Namespace(
            sim_epoch_refresh=False,
            feature="raw",
            sim_reference_eps=None,
            mechanism="lpm",
            m="best",
        )
        validate_sim_args(argparse.ArgumentParser(prog="sim-args-test"), args)

    def test_non_sim_feature_rejects_reference_eps(self):
        args = argparse.Namespace(
            sim_epoch_refresh=False,
            feature="raw",
            sim_reference_eps=1.5,
            mechanism="lpm",
            m="best",
        )
        self._validate_raises(args)

    def test_sim_allows_pm_mechanism(self):
        args = argparse.Namespace(
            sim_epoch_refresh=False,
            feature="sim",
            sim_reference_eps=1.5,
            mechanism="pm",
            m="2",
            x_eps=0.1,
        )
        validate_sim_args(argparse.ArgumentParser(prog="sim-args-test"), args)

    def test_sim_rejects_unsupported_mechanism(self):
        args = argparse.Namespace(
            sim_epoch_refresh=False,
            feature="sim",
            sim_reference_eps=1.5,
            mechanism="lpm",
            m="2",
            x_eps=0.1,
        )
        self._validate_raises(args)


if __name__ == "__main__":
    unittest.main()
