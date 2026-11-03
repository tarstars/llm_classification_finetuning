"""Validity contracts for v2 calibration, grouped selection and blend endpoints."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from arena.span_calibration import (SymmetricCalibration, grouped_folds, fit_blend,
                                    fit_selected, train_blend)


class CalibrationContracts(unittest.TestCase):
    def test_analytic_gradient_and_swap_equivariance(self):
        rng = np.random.default_rng(28)
        p = rng.dirichlet([2, 2, 2], 60)
        y = rng.integers(0, 3, 60)
        for family in ('temperature', 'temperature_tie', 'symmetric_vector'):
            model = SymmetricCalibration(family)
            design = model.design(p)
            theta = np.arange(design.shape[-1])*.2 + .7
            _, gradient = model.objective(theta, design, y)
            numerical = []
            for i in range(len(theta)):
                offset = np.eye(len(theta))[i]*1e-5
                numerical.append((model.objective(theta+offset, design, y)[0] -
                                  model.objective(theta-offset, design, y)[0])/2e-5)
            np.testing.assert_allclose(gradient, numerical, atol=1e-8)
            model.fit(p, y)
            np.testing.assert_allclose(model.predict_proba(p),
                model.predict_proba(p[:, [1,0,2]])[:, [1,0,2]], atol=1e-12)

    def test_tie_intercept_recovers_known_prior_from_uniform_predictions(self):
        p = np.full((400,3), 1/3)
        y = np.array([0]*100+[1]*100+[2]*200)
        for family in ('temperature_tie', 'symmetric_vector'):
            model = SymmetricCalibration(family).fit(p, y)
            np.testing.assert_allclose(model.predict_proba(p)[0], [.25,.25,.5], atol=1e-6)
            self.assertAlmostEqual(model.theta[-1], np.log(2), places=5)

    def test_blend_includes_exact_single_model_endpoints(self):
        y = np.arange(30) % 3
        perfect = np.eye(3)[y]*.8 + .2/3
        uniform = np.full_like(perfect, 1/3)
        self.assertEqual(fit_blend(perfect, uniform, y), 0.)
        self.assertEqual(fit_blend(uniform, perfect, y), 1.)

    def test_grouped_folds_and_training_only_fit_do_not_use_external_labels(self):
        rng = np.random.default_rng(29)
        first, second = rng.dirichlet([2,2,2], 210), rng.dirichlet([2,2,2], 210)
        y, groups = rng.integers(0,3,210), np.arange(210)//2
        train, evaluation = np.arange(180), np.arange(180,210)
        config = {'seed': 18, 'inner_group_folds': 3,
                  'calibration_families': ['temperature','temperature_tie','symmetric_vector'],
                  'classifier_temperature_bounds': [.5,3.], 'judge_temperature_bounds':[.25,8.],
                  'tie_intercept_bounds':[-1.,1.]}
        changed = y.copy()
        changed[evaluation] = (changed[evaluation]+1)%3
        for labels in (y, changed):
            for fit, valid in grouped_folds(labels, groups, 3, 18):
                self.assertFalse(set(groups[fit]) & set(groups[valid]))
        a, details_a = fit_selected(first[train], y[train], groups[train], config)
        b, details_b = fit_selected(first[train], changed[train], groups[train], config)
        self.assertEqual(details_a, details_b)
        np.testing.assert_array_equal(a.predict_proba(first[evaluation]), b.predict_proba(first[evaluation]))
        fits_a, weight_a = train_blend(first[train], second[train], y[train], groups[train], config)
        fits_b, weight_b = train_blend(first[train], second[train], changed[train], groups[train], config)
        self.assertEqual(weight_a, weight_b)
        for a, b in zip(fits_a, fits_b):
            np.testing.assert_array_equal(a.theta, b.theta)


if __name__ == '__main__':
    unittest.main()
