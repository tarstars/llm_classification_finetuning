import unittest
import numpy as np

from arena.span025_identical import FIXED_IDENTICAL, gain_bounds, outcomes


class IdenticalBoundsTests(unittest.TestCase):
    def test_bounds_contain_every_route_outcome(self):
        rng=np.random.default_rng(42)
        g=rng.dirichlet([1,2,3],300);l=rng.dirichlet([2,1,2],300)
        y=rng.integers(0,3,300);low,high=gain_bounds(y,g,l)
        off,on=outcomes(g,l)
        for _ in range(20):
            route=rng.random(300)<.3
            actual=off.copy();actual[route]=on[route]
            gain=np.log(FIXED_IDENTICAL[y])-np.log(actual[np.arange(300),y])
            self.assertTrue(np.all(low<=gain+1e-14))
            self.assertTrue(np.all(gain<=high+1e-14))

    def test_tie_channel_identical_and_column_alignment(self):
        g=np.array([[.6,.1,.3],[.2,.3,.5]])
        l=np.array([[.1,.4,.5],[.5,.2,.3]])
        off,on=outcomes(g,l)
        np.testing.assert_array_equal(off[:,2],on[:,2])
        np.testing.assert_allclose(on[0],[.225,.375,.4])
        low,high=gain_bounds(np.array([2,2]),g,l)
        np.testing.assert_array_equal(low,high)

    def test_invalid_targets_and_probabilities(self):
        g=np.array([[.2,.3,.5]])
        for y in [np.array([3]),np.array([1.0]),np.array([0,1])]:
            with self.assertRaises(ValueError):gain_bounds(y,g,g)
        with self.assertRaises(ValueError):outcomes(g,np.array([[.2,.3,.4]]))


if __name__=='__main__':unittest.main()
