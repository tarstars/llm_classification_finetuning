import unittest
from arena.span020_infer import make_batches


class H66BatchTests(unittest.TestCase):
    def test_maximum_length_rows_fit_exactly_six_per_batch(self):
        rows=[dict(input_ids=[1]*2001) for _ in range(13)]
        batches=make_batches(rows)
        self.assertEqual([len(b) for b in batches],[6,6,1])
        self.assertEqual(sorted(i for b in batches for i in b),list(range(13)))
        for batch in batches:self.assertLessEqual(len(batch)*2001,12288)

    def test_short_rows_obey_new_row_cap_without_changing_order(self):
        rows=[dict(input_ids=[1]*64) for _ in range(65)]
        batches=make_batches(rows)
        self.assertEqual([len(b) for b in batches],[32,32,1])
        self.assertEqual([i for b in batches for i in b],list(range(65)))


if __name__=='__main__':unittest.main()
