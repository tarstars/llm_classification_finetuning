import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from arena.span_repeatability import difference, request_payload


class ReplayContracts(unittest.TestCase):
    def test_permutation_alignment_cannot_hide_bad_coverage(self):
        ids = np.array([10, 20, 30])
        p = np.array([[.8,.1,.1],[.2,.7,.1],[.2,.1,.7]])
        limits = {'mean': .001, 'maximum': .01}
        result = difference(ids,p,ids[::-1],p[::-1],limits)
        self.assertTrue(result['passed'])
        self.assertEqual(result['mean_probability_difference'],0)
        with self.assertRaises(AssertionError):
            difference(ids,p,[10,10,30],p,limits)
        with self.assertRaises(AssertionError):
            difference(ids,p,[10,20,40],p,limits)
        self.assertFalse(difference(ids,p,ids,p[:,[1,0,2]],limits)['passed'])

    def test_counterfactual_changes_only_the_cache_flag(self):
        original={'prompt':[2,9,4],'cache_prompt':True,'seed':20261113,'n_predict':1,
                  'logit_bias':[[7,100.0],[8,100.0],[9,100.0]]}
        saved=copy.deepcopy(original)
        observed=request_payload(original,False)
        self.assertEqual(original,saved)
        self.assertEqual(observed,{**original,'cache_prompt':False})
        self.assertEqual(request_payload(original,True),original)


if __name__ == '__main__':
    unittest.main()
