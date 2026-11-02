import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from arena.judge import calibrate, field_budgets, head_tail, label_probabilities


class PromptJudgeContracts(unittest.TestCase):
    def test_complete_label_distribution_and_missing_scores(self):
        response = {'probs': [{'top_probs': [{'id': 5, 'prob': .2}, {'id': 7, 'prob': .7}, {'id': 9, 'prob': .1}]}]}
        np.testing.assert_allclose(label_probabilities(response, [5, 7, 9]), [.2, .7, .1])
        with self.assertRaises(ValueError): label_probabilities(response, [5, 7, 10])
        response['probs'][0]['top_probs'][0]['prob'] = .9
        with self.assertRaises(ValueError): label_probabilities(response, [5, 7, 9])

    def test_supports_server_logprob_response(self):
        response = {'completion_probabilities': [{'top_logprobs': [
            {'id': i, 'logprob': float(np.log(p))} for i, p in zip([5, 7, 9], [.2, .7, .1])]}]}
        np.testing.assert_allclose(label_probabilities(response, [5, 7, 9]), [.2, .7, .1])

    def test_fields_survive_and_unused_budget_is_reallocated(self):
        for lengths in [[1000, 2000, 2000], [1, 3, 4000], [10, 4000, 3], [0, 0, 0]]:
            budgets = field_budgets(lengths, 1800, 384)
            self.assertLessEqual(sum(budgets), 1800)
            self.assertTrue(all(0 <= n <= m for n, m in zip(budgets, lengths)))
            for n, m in zip(budgets, lengths):
                if m: self.assertGreater(n, 0)
            self.assertEqual(sum(budgets), min(sum(lengths), 1800))
        tokens = head_tail(list(range(100)), 20, [-1, -2, -3])
        self.assertEqual(len(tokens), 20)
        self.assertEqual(tokens[:9], list(range(9)))
        self.assertEqual(tokens[-8:], list(range(92, 100)))

    def test_calibration_preserves_order_exchange(self):
        p = np.array([[.1, .7, .2], [.3, .2, .5]])
        np.testing.assert_allclose(calibrate(p, .8, .4).sum(1), 1)
        np.testing.assert_allclose(calibrate(p[:, [1, 0, 2]], .8, .4)[:, [1, 0, 2]], calibrate(p, .8, .4))
        np.testing.assert_allclose(calibrate(p), p)


if __name__ == '__main__': unittest.main()
