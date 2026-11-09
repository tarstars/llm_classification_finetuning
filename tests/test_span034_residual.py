import unittest
import numpy as np

from arena.span034_residual import predict, fit, nested
from arena.span_heads import loss_rows


class ResidualTests(unittest.TestCase):
    def test_zero_tie_and_bitwise_swap(self):
        p = np.array([[.1, .7, .2], [.6, .1, .3], [.2, .4, .4]])
        d = np.array([3., -2., 1.])
        np.testing.assert_array_equal(predict(p, d, 0, 1), p)
        out = predict(p, d, .7, 2)
        np.testing.assert_array_equal(out[:, 2], p[:, 2])
        np.testing.assert_array_equal(predict(p[:, [1, 0, 2]], -d, .7, 2), out[:, [1, 0, 2]])
        np.testing.assert_allclose(out[:, :2].sum(1), p[:, :2].sum(1), atol=1e-15)

    def test_useful_and_anticorrelated_reward(self):
        y = np.tile([0, 1], 30)
        p = np.tile([.4, .4, .2], (len(y), 1))
        d = np.where(y == 0, 1., -1.)
        model = fit(p, d, y, .01)
        self.assertGreater(model['alpha'], 0)
        self.assertLess(loss_rows(y, predict(p, d, model['alpha'], model['scale'])).mean(), loss_rows(y, p).mean())
        self.assertEqual(fit(p, -d, y, .01)['alpha'], 0)

    def test_nested_training_scale_excludes_evaluation(self):
        n = 120
        y = np.arange(n) % 3
        p = np.tile([.35, .35, .3], (n, 1))
        d = np.where(y == 0, 1., -1.) * (1 + np.arange(n))
        groups = np.array([str(k // 2) for k in range(n)])
        cfg = dict(seed=20261212, outer_folds=5, inner_folds=3, lambda_grid=[.01, .1, 1])
        _, _, _, _, records = nested(p, d, y, groups, cfg)
        for r in records:
            train, val = r['fit_indices'], r['validation_indices']
            self.assertFalse(set(groups[train]) & set(groups[val]))
            self.assertAlmostEqual(r['model']['scale'], np.sqrt(np.mean(d[train] ** 2)))


if __name__ == '__main__':
    unittest.main()
