import unittest

import numpy as np

from arena.span016_reward import TieCalibration, conversation, crossfit, design, objective


class RewardMappingTests(unittest.TestCase):
    def test_multiturn_null_and_missing_messages(self):
        row = dict(prompt='["q1", "q2"]', response_a='["a1", null]', response_b='["b1"]')
        self.assertEqual([m['content'] for m in conversation(row, 'a')], ['q1', 'a1', 'q2', ''])
        self.assertEqual([m['content'] for m in conversation(row, 'b')], ['q1', 'b1', 'q2', ''])
        self.assertEqual([m['role'] for m in conversation(row, 'a')], ['user', 'assistant', 'user', 'assistant'])

    def test_swap_and_equal_rewards(self):
        model = TieCalibration()
        model.theta = np.array([.4, -.2, .1])
        d = np.array([-100., -3., 0., .5, 100.])
        p = model.predict_proba(d)
        np.testing.assert_allclose(model.predict_proba(-d), p[:, [1, 0, 2]], atol=1e-14)
        np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-14)
        self.assertEqual(p[2, 0], p[2, 1])

    def test_objective_gradient(self):
        x, y = design([-2., -.1, 0., 1., 4.]), np.array([1, 2, 0, 2, 0])
        theta = np.array([.2, -.3, .1])
        _, actual = objective(theta, x, y)
        numerical = []
        for k in range(3):
            step = np.zeros(3)
            step[k] = 1e-6
            numerical.append((objective(theta + step, x, y)[0] - objective(theta - step, x, y)[0]) / 2e-6)
        np.testing.assert_allclose(actual, numerical, rtol=1e-6, atol=1e-8)

    def test_crossfit_never_fits_held_out_group(self):
        rng = np.random.default_rng(8)
        d = rng.normal(size=180)
        y = np.tile([0, 1, 2], 60)
        groups = np.repeat(np.arange(90), 2)
        p, records = crossfit(d, y, groups)
        covered = []
        for record in records:
            fit, val = record['fit_indices'], record['validation_indices']
            self.assertFalse(set(groups[fit]) & set(groups[val]))
            # Changing held-out labels cannot change this fitted calibrator.
            model = TieCalibration().fit(d[fit], y[fit])
            np.testing.assert_allclose(model.theta, record['theta'], atol=1e-12)
            np.testing.assert_allclose(model.predict_proba(d[val]), p[val], atol=1e-12)
            covered.extend(val)
        self.assertEqual(sorted(covered), list(range(len(d))))

    def test_reject_nonfinite_and_invalid_labels(self):
        with self.assertRaises(ValueError):
            TieCalibration().fit([np.nan], [0])
        with self.assertRaises(ValueError):
            TieCalibration().fit([0.], [3])


if __name__ == '__main__':
    unittest.main()
