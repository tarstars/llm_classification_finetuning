import unittest
import numpy as np
from arena.gptq8 import dense_weight
from arena.span019_input import flatten_turns, native_pool_index


class MirrorSemanticsTests(unittest.TestCase):
    def test_native_pool_keeps_left_padding_and_embedded_pad_semantics(self):
        self.assertEqual(native_pool_index([0, 0, 2, 5, 1], 0), 4)
        self.assertEqual(native_pool_index([2, 5, 1], 0), 2)
        self.assertEqual(native_pool_index([2, 5, 0, 7, 1], 0), 1)
        self.assertEqual(flatten_turns('["first",null,"last"]'), 'first\n\nlast')

    def test_gptq8_signed_high_bytes_zero_point_and_transpose(self):
        # Independent scalar packing, including255 in the signed high byte and
        # stored zero255 (actual zero256), catches byte order and missing+1 bugs.
        levels = np.array([[0, 1, 127, 255], [128, 254, 3, 200],
                           [255, 4, 128, 0], [5, 6, 7, 8]], dtype=np.int64)
        packed = np.array([[sum(int(levels[i, j]) << (8*i) for i in range(4))
                            for j in range(4)]], dtype=np.uint32).view(np.int32)
        zeros = np.array([1, 128, 255, 256])
        qzeros = np.array([[sum(int(zeros[j]-1) << (8*j) for j in range(4))]], dtype=np.uint32).view(np.int32)
        scales = np.array([[.125, .25, .5, 1]], dtype=np.float16)
        actual = dense_weight(packed, qzeros, scales, np.zeros(4, dtype=np.int32))
        expected = ((levels-zeros).astype(np.float16)*scales).T
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue(actual.flags.c_contiguous)

    def test_invalid_group_is_not_silently_accepted(self):
        with self.assertRaises(ValueError):
            dense_weight(np.zeros((1,4),np.int32), np.zeros((1,1),np.int32),
                         np.ones((1,4),np.float16), np.array([0,0,0,1]))


if __name__ == '__main__': unittest.main()
