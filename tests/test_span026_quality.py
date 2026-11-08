import unittest
import numpy as np

from arena.span026_quality import gain_bounds


class ContextGainTests(unittest.TestCase):
    def test_bounds_cover_every_parent_route_with_replaced_original_component(self):
        rng=np.random.default_rng(42);n=100
        g,r,e,l=[rng.dirichlet([2,2,2],n) for _ in range(4)];y=rng.integers(0,3,n)
        low,high,p=gain_bounds(y,g,r,e,l)
        for _ in range(20):
            route=rng.random(n)<.2
            parent=p['parent_off'].copy();candidate=p['candidate_off'].copy()
            parent[route]=p['parent_on'][route];candidate[route]=p['candidate_on'][route]
            gain=np.log(candidate[np.arange(n),y])-np.log(parent[np.arange(n),y])
            self.assertTrue(np.all(low<=gain+1e-14));self.assertTrue(np.all(gain<=high+1e-14))

    def test_no_changed_gemma_means_exact_zero_gain(self):
        g=np.array([[.1,.2,.7],[.3,.5,.2]])
        low,high,_=gain_bounds(np.array([2,1]),g,g,g,g)
        np.testing.assert_array_equal(low,np.zeros(2));np.testing.assert_array_equal(high,np.zeros(2))


if __name__=='__main__':unittest.main()
