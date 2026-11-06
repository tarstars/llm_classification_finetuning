import unittest
import numpy as np
import pandas as pd
from arena.span009_public import profile_indices, ensemble, validate_probabilities


class PublicEnsembleContracts(unittest.TestCase):
    def test_llama_swap_is_aligned_before_equal_mixture(self):
        a = np.array([[.8, .1, .1], [.1, .2, .7]])
        b = np.array([[.05, .85, .1], [.4, .1, .5]])
        np.testing.assert_allclose(ensemble(a, b), [[.825, .075, .1], [.1, .3, .6]])

    def test_profile_is_repeatable_disjoint_and_ignores_targets(self):
        frame = pd.DataFrame({'prompt': ['p']*400, 'response_a': ['x'*i for i in range(400)],
                              'response_b': ['b']*400, 'winner_model_a': np.arange(400) % 2})
        uniform, long = profile_indices(frame)
        changed = frame.assign(winner_model_a=1-frame.winner_model_a)
        for x, y in zip((uniform, long), profile_indices(changed)):
            np.testing.assert_array_equal(x, y)
        self.assertEqual(len(uniform), 256)
        self.assertEqual(len(long), 64)
        self.assertFalse(set(uniform) & set(long))
        remaining = sorted(set(range(400)) - set(uniform), reverse=True)[:64]
        self.assertEqual(long.tolist(), remaining)

    def test_corrupt_or_missing_probability_rows_fail(self):
        for values, rows in [(np.ones((2, 3)), 2), (np.array([[np.nan, 0, 1]]), 1),
                             (np.array([[.2, .3, .5]]), 2)]:
            with self.assertRaises(ValueError):
                validate_probabilities(values, rows)


if __name__ == '__main__':
    unittest.main()
