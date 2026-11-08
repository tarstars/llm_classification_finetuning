import unittest
import numpy as np

from arena.span024_structural import EMPTY, categories, override


class StructuralRuleTests(unittest.TestCase):
    def test_published_constants_and_precedence(self):
        a=['[null]','["answer"]','["same"]','[]','[null]']
        b=['["answer"]','[]','["same"]','[]','[]']
        baseline=np.tile([.31,.32,.37],(5,1))
        p,c=override(baseline,a,b)
        np.testing.assert_allclose(p,[[.04,.88,.08],[.88,.04,.08],[.06,.06,.88],
            [.06,.06,.88],[.88,.04,.08]])
        self.assertEqual(c.tolist(),['empty_a','empty_b','identical','identical','empty_b'])
        # Different raw encodings of two empty responses retain the author's
        # B-empty precedence. This known asymmetry is not silently repaired.
        q,_=override(baseline,b,a)
        self.assertFalse(np.array_equal(p[-1],q[-1,[1,0,2]]))

    def test_no_normalization_and_exact_unchanged_probabilities(self):
        a=[' [null]','["same"]','["a"]'];b=['["x"]','[ "same" ]','["b"]']
        baseline=np.array([[.1,.2,.7],[.5,.1,.4],[.12,.49,.39]])
        p,c=override(baseline,a,b)
        np.testing.assert_array_equal(p,baseline)
        self.assertTrue(np.all(c=='unchanged'))
        self.assertEqual(categories(EMPTY,['["answer"]']*len(EMPTY)).tolist(),['empty_a']*len(EMPTY))

    def test_rejects_nonraw_or_mismatched_input(self):
        for a,b in [([None],['[]']),(['[]'],[]),([['a']],['[]'])]:
            with self.assertRaises(ValueError):categories(a,b)


if __name__=='__main__':unittest.main()
