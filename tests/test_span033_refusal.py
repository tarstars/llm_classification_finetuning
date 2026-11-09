import unittest
import numpy as np
import pandas as pd

from arena.span033_refusal import describe, REFUSAL


class ActorTests(unittest.TestCase):
    def test_actor_and_action_boundaries(self):
        for text in ['I cannot provide that.', "I'm sorry, but I cannot help.",
                     'As an AI language model, I am unable to execute code.']:
            self.assertIsNotNone(REFUSAL.search(text), text)
        for text in ['We cannot see inside a black hole.', 'Understood. I will not talk about it.',
                     '1. "I cannot provide a sword."', '"I cannot help," said the character.',
                     'Here are several languages. However, I am not able to speak as humans do.']:
            self.assertIsNone(REFUSAL.search(text), text)

    def test_swap_and_off_slice(self):
        frame = pd.DataFrame(dict(prompt=['Who is real?', 'plain'],
            response_a=['I cannot determine that.', 'A'], response_b=['An answer', 'B']))
        x, affected, _ = describe(frame, encoded=False)
        reverse = frame.copy()
        reverse[['response_a', 'response_b']] = frame[['response_b', 'response_a']].values
        y, rev_affected, _ = describe(reverse, encoded=False)
        np.testing.assert_array_equal(y, -x)
        np.testing.assert_array_equal(affected, rev_affected)
        np.testing.assert_array_equal(x[~affected], 0)


if __name__ == '__main__':
    unittest.main()
