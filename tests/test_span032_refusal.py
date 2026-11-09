import unittest

import numpy as np
import pandas as pd

from arena.span032_refusal import describe


class DescriptorTests(unittest.TestCase):
    def test_swapping_is_antisymmetric_and_off_slice_is_zero(self):
        frame = pd.DataFrame(dict(prompt=['Who is real?', 'Browse this file', 'plain'],
            response_a=['I cannot do this.', 'Here is code ```', 'A'],
            response_b=['An answer', 'We are unable to access it.', 'B']))
        x, affected, _ = describe(frame, encoded=False)
        reverse = frame.copy()
        reverse[['response_a', 'response_b']] = frame[['response_b', 'response_a']].values
        rx, ra, _ = describe(reverse, encoded=False)
        np.testing.assert_array_equal(rx, -x)
        np.testing.assert_array_equal(ra, affected)
        np.testing.assert_array_equal(x[~affected], 0)
        self.assertEqual(affected.tolist(), [True, True, False])

    def test_encoded_arrays_and_opening_window(self):
        frame = pd.DataFrame(dict(prompt=['["Who?"]'], response_a=['["I cannot help."]'], response_b=['["ok"]']))
        self.assertTrue(describe(frame)[1][0])
        frame = pd.DataFrame(dict(prompt=['plain'], response_a=['x' * 601 + ' I cannot help.'], response_b=['ok']))
        self.assertFalse(describe(frame, encoded=False)[1][0])


if __name__ == '__main__':
    unittest.main()
