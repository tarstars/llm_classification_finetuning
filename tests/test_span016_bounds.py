import unittest

import numpy as np

from arena.span013_order import complete_gemma_order
from arena.span016_pipeline import mixture
from arena.span016_bounds import completion_gain_bounds


class UnfinishedCompletionBoundsTests(unittest.TestCase):
    def test_every_sampled_shared_completion_lies_inside_endpoint_bounds(self):
        rng = np.random.default_rng(2468)
        n = 31
        gemma,llama,reward = [rng.dirichlet([.3,.5,.2],n).astype(np.float32) for _ in range(3)]
        labels = rng.integers(0,3,n)
        old,new = np.array([0,3,7]),np.array([1,3,10])
        lower,upper = completion_gain_bounds(gemma,llama,reward,labels,old,new)
        for _ in range(100):
            reverse = rng.dirichlet([.2,.2,.2],n).astype(np.float32)
            b = complete_gemma_order(gemma,llama,old,reverse[old])
            c = mixture(gemma,reward,new,reverse[new])
            actual = np.log(np.maximum(c[np.arange(n),labels],1e-15))-np.log(np.maximum(b[np.arange(n),labels],1e-15))
            self.assertTrue(np.all(lower <= actual))
            self.assertTrue(np.all(actual <= upper))
        neither = np.ones(n,dtype=bool)
        neither[np.union1d(old,new)] = False
        self.assertTrue(np.all(upper[neither]-lower[neither] < 3e-12))

    def test_no_difference_has_zero_width_even_if_every_row_is_routed(self):
        p = np.array([[.5,.25,.25],[.125,.625,.25]])
        q = p[:,[1,0,2]]
        lower,upper = completion_gain_bounds(p,q,p,np.array([0,2]),np.array([0,1]),np.array([0,1]))
        np.testing.assert_allclose(lower,0,atol=2e-12)
        np.testing.assert_allclose(upper,0,atol=2e-12)

    def test_malformed_routes_and_targets_are_rejected(self):
        p = np.array([[.5,.25,.25]])
        for route,labels in [(np.array([0,0]),np.array([0])),(np.array([1]),np.array([0])),
                             (np.array([0]),np.array([3]))]:
            with self.subTest(route=route,labels=labels),self.assertRaises(ValueError):
                completion_gain_bounds(p,p,p,labels,route,np.array([],dtype=int))


if __name__ == '__main__':
    unittest.main()
