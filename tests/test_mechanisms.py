import math
import unittest
from types import SimpleNamespace

import torch

from mechanisms import (
    AdditiveNoiseMechanism,
    AnalyticGaussian,
    Gaussian,
    HighDimSquareWave,
    Laplace,
    MultiBit,
    MultiDimPiecewise,
    OneBit,
)

try:
    from transforms import FeaturePerturbation
except ModuleNotFoundError:
    FeaturePerturbation = None


class MechanismRefactorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.tensor([[0.2, 0.7, 0.4, 0.9]], dtype=torch.float32)

    def _monte_carlo_call_mean(self, mechanism, x, num_samples=4000):
        repeated_x = x.repeat(num_samples, 1)
        outputs = mechanism(repeated_x)
        return outputs.view(num_samples, *x.shape).mean(dim=0)

    def _monte_carlo_rectified_unit_mean(self, mechanism, x_unit, num_samples=4000):
        repeated_x_unit = x_unit.repeat(num_samples, 1)
        state = mechanism.perturb(repeated_x_unit)
        rectified = mechanism.rectify(state)
        return rectified.view(num_samples, *x_unit.shape).mean(dim=0)

    def _monte_carlo_additive_mean(self, mechanism, x, num_samples=4000):
        repeated_x = x.repeat(num_samples, 1)
        outputs = mechanism(repeated_x)
        return outputs.view(num_samples, *x.shape).mean(dim=0)

    @staticmethod
    def _expected_four_stage_output_range(mechanism):
        base_output_range = mechanism.last_unit_support_scale / mechanism.last_effective_gain
        if not mechanism.norm:
            return float(base_output_range)
        if mechanism.norm_scale is None:
            return float(mechanism.last_unit_support_scale)
        return float(base_output_range / mechanism.norm_scale)

    def test_restore_scale_inverts_to_unit_interval(self):
        mechanism = MultiBit(eps=3.0, input_range=(2.0, 5.0), m=2)
        x = torch.tensor([[2.0, 2.75, 4.25, 5.0]], dtype=torch.float32)
        x_unit = mechanism.to_unit_interval(x)
        restored = mechanism.restore_scale(x_unit)
        self.assertTrue(torch.allclose(restored, x, atol=1e-6, rtol=0.0))

    def test_perturb_state_structure_and_rectifier(self):
        x = torch.tensor([[0.15, 0.35, 0.65, 0.85]], dtype=torch.float32)
        cases = [
            (MultiBit, 'mbm'),
            (MultiDimPiecewise, 'pm'),
            (HighDimSquareWave, 'hds'),
        ]

        for mechanism_cls, name in cases:
            with self.subTest(mechanism=name):
                mechanism = mechanism_cls(eps=6.0, input_range=(0.0, 1.0), m=2)
                x_unit = mechanism.to_unit_interval(x)
                state = mechanism.perturb(x_unit)
                self.assertEqual(state.z.shape, x.shape)
                self.assertEqual(state.d, x.size(1))
                self.assertEqual(state.m, 2)
                self.assertGreater(state.kappa, 0.0)
                self.assertGreater(state.effective_gain, 0.0)
                non_zero = state.z.ne(0).sum(dim=1)
                self.assertTrue(torch.all(non_zero <= state.m).item())
                self.assertTrue(torch.all(non_zero >= 1).item())
                rectified = mechanism.rectify(state)
                self.assertTrue(torch.allclose(rectified, state.z / state.effective_gain, atol=1e-6, rtol=0.0))

    def test_multibit_helper_formula(self):
        eta = 0.75
        x_unit = torch.tensor([[-0.8, -0.1, 0.3, 0.9]], dtype=torch.float32)
        mechanism = MultiBit(eps=3.0, input_range=(0.0, 1.0), m=2)
        y, kappa, unit_support_scale = mechanism._perturb_multibit_unit(x_unit, eta)
        u = math.expm1(eta)
        expected_kappa = u / (u + 2.0)
        self.assertEqual(y.shape, x_unit.shape)
        self.assertAlmostEqual(kappa, expected_kappa, places=12)
        self.assertEqual(unit_support_scale, 1.0)
        self.assertTrue(torch.all((y == -1.0) | (y == 1.0)).item())

    def test_piecewise_helper_formula(self):
        eta = 0.75
        x_unit = torch.tensor([[-0.8, -0.1, 0.3, 0.9]], dtype=torch.float32)
        mechanism = MultiDimPiecewise(eps=3.0, input_range=(0.0, 1.0), m=2)
        y, kappa, unit_support_scale = mechanism._perturb_piecewise_unit(x_unit, eta)
        u = math.expm1(eta / 2.0)
        expected_c = (u + 2.0) / u
        expected_kappa = u / (u + 2.0)
        self.assertEqual(y.shape, x_unit.shape)
        self.assertAlmostEqual(kappa, expected_kappa, places=12)
        self.assertEqual(unit_support_scale, 1.0)
        self.assertTrue(torch.all(y.abs() <= 1.0 + 1e-6).item())
        self.assertGreater(expected_c, 1.0)

    def test_squarewave_helper_formula(self):
        eta = 0.75
        x_unit = torch.tensor([[-0.8, -0.1, 0.3, 0.9]], dtype=torch.float32)
        mechanism = HighDimSquareWave(eps=3.0, input_range=(0.0, 1.0), m=2)
        y, kappa, unit_support_scale = mechanism._perturb_squarewave_unit(x_unit, eta)
        q = math.exp(-eta)
        one_minus_q = -math.expm1(-eta)
        b = q * (eta - one_minus_q) / (one_minus_q - eta * q)
        expected_kappa = 1.0 - one_minus_q / eta
        self.assertEqual(y.shape, x_unit.shape)
        self.assertAlmostEqual(kappa, expected_kappa, places=12)
        self.assertAlmostEqual(unit_support_scale, 1.0 + b, places=12)
        self.assertTrue(torch.all(y >= -(1.0 + b) - 1e-6).item())
        self.assertTrue(torch.all(y <= 1.0 + b + 1e-6).item())

    def test_diagnostics_follow_theory(self):
        x = self.x
        d = x.size(1)

        mbm = MultiBit(eps=2.0, input_range=(0.0, 1.0), m=2)
        mbm(x)
        eta = 1.0
        u = math.expm1(eta)
        kappa = u / (u + 2.0)
        effective_gain = (2 / d) * kappa
        self.assertAlmostEqual(mbm.last_kappa, kappa, places=12)
        self.assertAlmostEqual(mbm.last_effective_gain, effective_gain, places=12)
        self.assertAlmostEqual(mbm.last_c, 0.5 / effective_gain, places=12)
        self.assertAlmostEqual(mbm.last_unit_support_scale, 1.0, places=12)

        pm = MultiDimPiecewise(eps=2.0, input_range=(0.0, 1.0), m=2)
        pm(x)
        u = math.expm1(eta / 2.0)
        kappa = u / (u + 2.0)
        effective_gain = (2 / d) * kappa
        self.assertAlmostEqual(pm.last_kappa, kappa, places=12)
        self.assertAlmostEqual(pm.last_effective_gain, effective_gain, places=12)
        self.assertAlmostEqual(pm.last_c, 0.5 / effective_gain, places=12)
        self.assertAlmostEqual(pm.last_unit_support_scale, 1.0, places=12)

        hds = HighDimSquareWave(eps=2.0, input_range=(0.0, 1.0), m=2)
        hds(x)
        q = math.exp(-eta)
        one_minus_q = -math.expm1(-eta)
        b = q * (eta - one_minus_q) / (one_minus_q - eta * q)
        kappa = 1.0 - one_minus_q / eta
        effective_gain = (2 / d) * kappa
        self.assertAlmostEqual(hds.last_kappa, kappa, places=12)
        self.assertAlmostEqual(hds.last_effective_gain, effective_gain, places=12)
        self.assertAlmostEqual(hds.last_c, 0.5 / effective_gain, places=12)
        self.assertAlmostEqual(hds.last_unit_support_scale, 1.0 + b, places=12)

    def test_onebit_matches_multibit_max(self):
        x = self.x
        torch.manual_seed(123)
        onebit = OneBit(eps=3.0, input_range=(0.0, 1.0))
        out_onebit = onebit(x)

        torch.manual_seed(123)
        multibit = MultiBit(eps=3.0, input_range=(0.0, 1.0), m='max')
        out_multibit = multibit(x)

        self.assertTrue(torch.equal(out_onebit, out_multibit))
        self.assertAlmostEqual(onebit.last_kappa, multibit.last_kappa, places=12)
        self.assertAlmostEqual(onebit.last_c, multibit.last_c, places=12)
        self.assertAlmostEqual(onebit.output_range, multibit.output_range, places=12)

    def test_four_stage_output_range_matches_final_return_space(self):
        x = self.x
        cases = [
            (MultiBit, {'m': 2}),
            (OneBit, {}),
            (MultiDimPiecewise, {'m': 2}),
            (HighDimSquareWave, {'m': 2}),
        ]
        norm_cases = [
            {'norm': False},
            {'norm': True, 'norm_scale': 'none'},
            {'norm': True, 'norm_scale': '10'},
        ]

        for mechanism_cls, base_kwargs in cases:
            for norm_kwargs in norm_cases:
                with self.subTest(mechanism=mechanism_cls.__name__, norm_kwargs=norm_kwargs):
                    mechanism = mechanism_cls(
                        eps=3.0,
                        input_range=(0.0, 1.0),
                        **base_kwargs,
                        **norm_kwargs,
                    )
                    mechanism(x)
                    self.assertIsInstance(mechanism.output_range, float)
                    self.assertAlmostEqual(
                        mechanism.output_range,
                        self._expected_four_stage_output_range(mechanism),
                        places=12,
                    )

    def test_monte_carlo_unbiasedness(self):
        x = torch.tensor([[0.2, 0.7, 0.4]], dtype=torch.float32)
        cases = [
            MultiBit(eps=9.0, input_range=(0.0, 1.0), m=3),
            MultiDimPiecewise(eps=9.0, input_range=(0.0, 1.0), m=3),
            HighDimSquareWave(eps=9.0, input_range=(0.0, 1.0), m=3),
        ]

        for mechanism in cases:
            with self.subTest(mechanism=mechanism.__class__.__name__):
                x_unit = mechanism.to_unit_interval(x)
                unit_mean = self._monte_carlo_rectified_unit_mean(mechanism, x_unit)
                call_mean = self._monte_carlo_call_mean(mechanism, x)
                self.assertTrue(torch.allclose(unit_mean, x_unit, atol=0.08, rtol=0.0))
                self.assertTrue(torch.allclose(call_mean, x_unit, atol=0.04, rtol=0.0))

    def test_large_eps_four_stage_mechanisms_remain_finite(self):
        x = torch.tensor([[0.15, 0.35, 0.65, 0.85]], dtype=torch.float32)
        huge_eps = 1_000_000.0
        cases = [
            MultiBit(eps=huge_eps, input_range=(0.0, 1.0), m='max'),
            MultiDimPiecewise(eps=huge_eps, input_range=(0.0, 1.0), m='max'),
            HighDimSquareWave(eps=huge_eps, input_range=(0.0, 1.0), m='max'),
        ]

        for mechanism in cases:
            with self.subTest(mechanism=mechanism.__class__.__name__):
                out = mechanism(x)
                self.assertTrue(torch.isfinite(out).all().item())
                self.assertTrue(math.isfinite(mechanism.last_kappa))
                self.assertTrue(math.isfinite(mechanism.last_effective_gain))
                self.assertGreater(mechanism.last_kappa, 0.99)
                self.assertGreater(mechanism.last_effective_gain, 0.0)

    def test_large_eps_pm_and_hds_are_close_to_identity(self):
        x = torch.tensor([[0.15, 0.35, 0.65, 0.85]], dtype=torch.float32)
        x_unit = x * 2.0 - 1.0
        huge_eps = 1_000_000.0

        cases = [
            MultiDimPiecewise(eps=huge_eps, input_range=(0.0, 1.0), m='max'),
            HighDimSquareWave(eps=huge_eps, input_range=(0.0, 1.0), m='max'),
        ]

        for mechanism in cases:
            with self.subTest(mechanism=mechanism.__class__.__name__):
                torch.manual_seed(7)
                out = mechanism(x)
                self.assertTrue(torch.allclose(out, x_unit, atol=1e-4, rtol=0.0))

    def test_additive_noise_pipeline_structure(self):
        class DummyAdditive(AdditiveNoiseMechanism):
            def generate_noise(self, x):
                return torch.full_like(x, 0.25)

        mechanism = DummyAdditive(eps=1.0, input_range=(0.0, 1.0))
        x = torch.tensor([[0.1, 0.3, 0.5]], dtype=torch.float32)
        noise = mechanism.generate_noise(x)
        self.assertTrue(torch.equal(mechanism.add_noise(x, noise), x + noise))
        self.assertTrue(torch.equal(mechanism(x), x + noise))

    def test_four_stage_internal_norm_scales_with_kappa_and_sampling_ratio(self):
        x = torch.tensor([[0.2, 0.7, 0.4]], dtype=torch.float32)
        cases = [
            MultiBit(eps=3.0, input_range=(0.0, 1.0), m=2),
            MultiDimPiecewise(eps=3.0, input_range=(0.0, 1.0), m=2),
            HighDimSquareWave(eps=3.0, input_range=(0.0, 1.0), m=2),
        ]

        for raw_mechanism in cases:
            with self.subTest(mechanism=raw_mechanism.__class__.__name__):
                mechanism_cls = raw_mechanism.__class__
                kwargs = {
                    'eps': raw_mechanism.eps,
                    'input_range': (raw_mechanism.alpha, raw_mechanism.beta),
                }
                if hasattr(raw_mechanism, 'm'):
                    kwargs['m'] = raw_mechanism.m

                raw_kappa = mechanism_cls(norm=False, **kwargs)
                torch.manual_seed(123)
                raw_output = raw_kappa(x)

                norm_mechanism = mechanism_cls(norm=True, norm_scale='none', **kwargs)
                torch.manual_seed(123)
                norm_output = norm_mechanism(x)

                self.assertAlmostEqual(norm_mechanism.last_kappa, raw_kappa.last_kappa, places=12)
                expected = raw_output * raw_kappa.last_kappa * (raw_kappa.last_m / x.size(1))
                self.assertTrue(
                    torch.allclose(norm_output, expected, atol=1e-6, rtol=0.0)
                )

    def test_four_stage_internal_norm_uses_inverse_norm_scale(self):
        x = torch.tensor([[0.2, 0.7, 0.4]], dtype=torch.float32)
        norm_scale = 10.0
        cases = [
            MultiBit(eps=3.0, input_range=(0.0, 1.0), m=2),
            MultiDimPiecewise(eps=3.0, input_range=(0.0, 1.0), m=2),
            HighDimSquareWave(eps=3.0, input_range=(0.0, 1.0), m=2),
        ]

        for raw_mechanism in cases:
            with self.subTest(mechanism=raw_mechanism.__class__.__name__):
                mechanism_cls = raw_mechanism.__class__
                kwargs = {
                    'eps': raw_mechanism.eps,
                    'input_range': (raw_mechanism.alpha, raw_mechanism.beta),
                }
                if hasattr(raw_mechanism, 'm'):
                    kwargs['m'] = raw_mechanism.m

                raw_instance = mechanism_cls(norm=False, **kwargs)
                torch.manual_seed(123)
                raw_output = raw_instance(x)

                scaled_norm = mechanism_cls(norm=True, norm_scale=norm_scale, **kwargs)
                torch.manual_seed(123)
                scaled_output = scaled_norm(x)

                self.assertTrue(
                    torch.allclose(scaled_output, raw_output / norm_scale, atol=1e-6, rtol=0.0)
                )

    def test_four_stage_norm_scale_rejects_invalid_values(self):
        invalid_values = ['abc', '0', '-1', 'inf', 'nan', 0.0, -1.0, float('inf'), float('nan')]
        for invalid in invalid_values:
            with self.subTest(norm_scale=invalid):
                with self.assertRaisesRegex(ValueError, 'norm_scale'):
                    MultiBit(
                        eps=3.0,
                        input_range=(0.0, 1.0),
                        m=2,
                        norm=True,
                        norm_scale=invalid,
                    )

    def test_four_stage_norm_scale_is_ignored_when_norm_is_disabled(self):
        x = torch.tensor([[0.2, 0.7, 0.4]], dtype=torch.float32)
        cases = [
            MultiBit(eps=3.0, input_range=(0.0, 1.0), m=2),
            MultiDimPiecewise(eps=3.0, input_range=(0.0, 1.0), m=2),
            HighDimSquareWave(eps=3.0, input_range=(0.0, 1.0), m=2),
        ]

        for template in cases:
            with self.subTest(mechanism=template.__class__.__name__):
                mechanism_cls = template.__class__
                kwargs = {
                    'eps': template.eps,
                    'input_range': (template.alpha, template.beta),
                }
                if hasattr(template, 'm'):
                    kwargs['m'] = template.m

                base = mechanism_cls(norm=False, **kwargs)
                scaled = mechanism_cls(norm=False, norm_scale=10.0, **kwargs)

                torch.manual_seed(123)
                base_out = base(x)
                torch.manual_seed(123)
                scaled_out = scaled(x)
                self.assertTrue(torch.allclose(base_out, scaled_out, atol=1e-6, rtol=0.0))

    def test_additive_noise_generate_and_call_consistency(self):
        x = torch.tensor([[0.2, 0.7, 0.4]], dtype=torch.float32)
        cases = [
            Laplace(eps=4.0, input_range=(0.0, 1.0)),
            Gaussian(eps=4.0, input_range=(0.0, 1.0)),
            AnalyticGaussian(eps=4.0, input_range=(0.0, 1.0)),
        ]

        for mechanism in cases:
            with self.subTest(mechanism=mechanism.__class__.__name__):
                torch.manual_seed(123)
                noise = mechanism.generate_noise(x)
                torch.manual_seed(123)
                out = mechanism(x)
                self.assertEqual(noise.shape, x.shape)
                self.assertTrue(torch.allclose(out, mechanism.add_noise(x, noise), atol=1e-6, rtol=0.0))

    def test_additive_internal_norm_matches_max_abs(self):
        x = torch.tensor([[0.2, 0.7, 0.4]], dtype=torch.float32)
        cases = [
            Laplace(eps=4.0, input_range=(0.0, 1.0)),
            Gaussian(eps=4.0, input_range=(0.0, 1.0)),
            AnalyticGaussian(eps=4.0, input_range=(0.0, 1.0)),
        ]

        for raw_mechanism in cases:
            with self.subTest(mechanism=raw_mechanism.__class__.__name__):
                mechanism_cls = raw_mechanism.__class__
                kwargs = {
                    'eps': raw_mechanism.eps,
                    'input_range': (raw_mechanism.alpha, raw_mechanism.beta),
                }
                if hasattr(raw_mechanism, 'delta'):
                    kwargs['delta'] = raw_mechanism.delta

                raw_instance = mechanism_cls(norm=False, **kwargs)
                torch.manual_seed(123)
                raw_output = raw_instance(x)

                norm_instance = mechanism_cls(norm=True, **kwargs)
                torch.manual_seed(123)
                norm_output = norm_instance(x)

                expected = raw_output / raw_output.abs().max()
                self.assertTrue(torch.allclose(norm_output, expected, atol=1e-6, rtol=0.0))

    def test_additive_internal_norm_zero_tensor_returns_zero(self):
        class ZeroAdditive(AdditiveNoiseMechanism):
            def generate_noise(self, x):
                return -x

        mechanism = ZeroAdditive(eps=1.0, input_range=(0.0, 1.0), norm=True)
        x = torch.tensor([[0.2, -0.3, 0.5]], dtype=torch.float32)
        self.assertTrue(torch.equal(mechanism(x), torch.zeros_like(x)))

    def test_laplace_scale_formula(self):
        class InspectLaplace(Laplace):
            def _generate_laplace_noise(self, x, scale):
                self.captured_scale = float(scale)
                return torch.zeros_like(x)

        x = torch.tensor([[0.2, 0.7, 0.4, 0.9]], dtype=torch.float32)
        mechanism = InspectLaplace(eps=2.0, input_range=(0.0, 1.0))
        noise = mechanism.generate_noise(x)
        self.assertTrue(torch.equal(noise, torch.zeros_like(x)))
        self.assertAlmostEqual(mechanism.captured_scale, x.size(1) / 2.0, places=12)

    def test_gaussian_sigma_formula(self):
        x = torch.tensor([[0.2, 0.7, 0.4, 0.9]], dtype=torch.float32)
        mechanism = Gaussian(eps=2.0, input_range=(0.0, 1.0), delta=1e-10)
        noise = mechanism.generate_noise(x)
        expected_sensitivity = math.sqrt(x.size(1))
        expected_sigma = expected_sensitivity * math.sqrt(2.0 * math.log(1.25 / mechanism.delta)) / mechanism.eps
        self.assertEqual(noise.shape, x.shape)
        self.assertAlmostEqual(mechanism.sensitivity, expected_sensitivity, places=12)
        self.assertAlmostEqual(mechanism.sigma, expected_sigma, places=12)

    def test_analytic_gaussian_sigma_matches_calibration(self):
        x = torch.tensor([[0.2, 0.7, 0.4, 0.9]], dtype=torch.float32)
        mechanism = AnalyticGaussian(eps=2.0, input_range=(0.0, 1.0), delta=1e-10)
        noise = mechanism.generate_noise(x)
        expected_sensitivity = math.sqrt(x.size(1))
        expected_sigma = mechanism.calibrate_gaussian_mechanism()
        self.assertEqual(noise.shape, x.shape)
        self.assertAlmostEqual(mechanism.sensitivity, expected_sensitivity, places=12)
        self.assertAlmostEqual(mechanism.sigma, expected_sigma, places=12)

    def test_additive_noise_monte_carlo_unbiasedness(self):
        x = torch.tensor([[0.2, 0.7, 0.4]], dtype=torch.float32)
        cases = [
            Laplace(eps=8.0, input_range=(0.0, 1.0)),
            Gaussian(eps=8.0, input_range=(0.0, 1.0)),
            AnalyticGaussian(eps=8.0, input_range=(0.0, 1.0)),
        ]

        for mechanism in cases:
            with self.subTest(mechanism=mechanism.__class__.__name__):
                mean = self._monte_carlo_additive_mean(mechanism, x)
                self.assertTrue(torch.allclose(mean, x, atol=0.12, rtol=0.0))

    def test_feature_perturbation_norm_paths(self):
        if FeaturePerturbation is None:
            self.skipTest('FeaturePerturbation dependencies are unavailable in this environment.')

        base_x = torch.tensor([[0.0, 0.7, 0.4, 1.0]], dtype=torch.float32)
        input_range = (float(base_x.min().item()), float(base_x.max().item()))

        mbm_data = SimpleNamespace(x=base_x.clone())
        torch.manual_seed(7)
        FeaturePerturbation(mechanism='mbm', x_eps=3.0, m=2, norm=True)(mbm_data)
        direct_mbm = MultiBit(eps=3.0, input_range=input_range, m=2, norm=True)
        torch.manual_seed(7)
        expected_mbm = direct_mbm(base_x.clone())
        self.assertTrue(torch.allclose(mbm_data.x, expected_mbm, atol=1e-6, rtol=0.0))
        self.assertEqual(mbm_data.output_range, direct_mbm.output_range)

        pm_data = SimpleNamespace(x=base_x.clone())
        torch.manual_seed(7)
        FeaturePerturbation(mechanism='pm', x_eps=3.0, m=2, norm=True)(pm_data)
        direct_pm = MultiDimPiecewise(eps=3.0, input_range=input_range, m=2, norm=True)
        torch.manual_seed(7)
        expected_pm = direct_pm(base_x.clone())
        self.assertTrue(torch.allclose(pm_data.x, expected_pm, atol=1e-6, rtol=0.0))
        self.assertEqual(pm_data.output_range, direct_pm.output_range)

        hds_data = SimpleNamespace(x=base_x.clone())
        torch.manual_seed(7)
        FeaturePerturbation(mechanism='hds', x_eps=3.0, m=2, norm=True)(hds_data)
        direct_hds = HighDimSquareWave(eps=3.0, input_range=input_range, m=2, norm=True)
        torch.manual_seed(7)
        expected_hds = direct_hds(base_x.clone())
        self.assertTrue(torch.allclose(hds_data.x, expected_hds, atol=1e-6, rtol=0.0))
        self.assertEqual(hds_data.output_range, direct_hds.output_range)

        lpm_data = SimpleNamespace(x=base_x.clone())
        torch.manual_seed(7)
        FeaturePerturbation(mechanism='lpm', x_eps=3.0, norm=True)(lpm_data)
        direct_lpm = Laplace(eps=3.0, input_range=input_range, norm=True)
        torch.manual_seed(7)
        expected_lpm = direct_lpm(base_x.clone())
        self.assertTrue(torch.allclose(lpm_data.x, expected_lpm, atol=1e-6, rtol=0.0))
        self.assertIsNone(lpm_data.output_range)
        self.assertIsNone(direct_lpm.output_range)

        agm_data = SimpleNamespace(x=base_x.clone())
        torch.manual_seed(7)
        FeaturePerturbation(mechanism='agm', x_eps=3.0, norm=True)(agm_data)
        direct_agm = AnalyticGaussian(eps=3.0, input_range=input_range, norm=True)
        torch.manual_seed(7)
        expected_agm = direct_agm(base_x.clone())
        self.assertTrue(torch.allclose(agm_data.x, expected_agm, atol=1e-6, rtol=0.0))
        self.assertIsNone(agm_data.output_range)
        self.assertIsNone(direct_agm.output_range)

    def test_feature_perturbation_forwards_norm_scale_to_four_stage(self):
        if FeaturePerturbation is None:
            self.skipTest('FeaturePerturbation dependencies are unavailable in this environment.')

        base_x = torch.tensor([[0.0, 0.7, 0.4, 1.0]], dtype=torch.float32)
        input_range = (float(base_x.min().item()), float(base_x.max().item()))

        mbm_data = SimpleNamespace(x=base_x.clone())
        torch.manual_seed(7)
        FeaturePerturbation(
            mechanism='mbm',
            x_eps=3.0,
            m=2,
            norm=True,
            norm_scale='10',
        )(mbm_data)
        direct_mbm = MultiBit(eps=3.0, input_range=input_range, m=2, norm=True, norm_scale='10')
        torch.manual_seed(7)
        expected_mbm = direct_mbm(base_x.clone())
        self.assertTrue(torch.allclose(mbm_data.x, expected_mbm, atol=1e-6, rtol=0.0))
        self.assertEqual(mbm_data.output_range, direct_mbm.output_range)

    def test_feature_perturbation_norm_is_noop_without_mechanism(self):
        if FeaturePerturbation is None:
            self.skipTest('FeaturePerturbation dependencies are unavailable in this environment.')

        base_x = torch.tensor([[0.2, 0.7, 0.4, 0.9]], dtype=torch.float32)
        inf_data = SimpleNamespace(x=base_x.clone())
        FeaturePerturbation(mechanism='mbm', x_eps=float('inf'), m=2, norm=True)(inf_data)
        self.assertTrue(torch.equal(inf_data.x, base_x))
        self.assertIsNone(inf_data.output_range)

        zero_data = SimpleNamespace(x=base_x.clone())
        with self.assertRaisesRegex(ValueError, 'undefined'):
            FeaturePerturbation(
                mechanism='mbm',
                x_eps=0.0,
                m=2,
                norm=True,
            )(zero_data)

    def test_feature_perturbation_inf_eps_unit_map_enables_strict_comparison(self):
        if FeaturePerturbation is None:
            self.skipTest('FeaturePerturbation dependencies are unavailable in this environment.')

        base_x = torch.tensor([[0.0, 0.7, 0.4, 1.0]], dtype=torch.float32)
        mapped_data = SimpleNamespace(x=base_x.clone())
        FeaturePerturbation(
            mechanism='hds',
            x_eps=float('inf'),
            m='best',
            norm=True,
            inf_eps_unit_map=True,
        )(mapped_data)

        expected = base_x * 2.0 - 1.0
        self.assertTrue(torch.allclose(mapped_data.x, expected, atol=1e-6, rtol=0.0))
        self.assertIsNone(mapped_data.output_range)
        self.assertEqual(set(vars(mapped_data).keys()), {'x', 'feature_mechanism', 'output_range'})


if __name__ == '__main__':
    unittest.main()
