import unittest
import numpy as np
from arena.span022_conditional import route,correct


class ConditionalTests(unittest.TestCase):
    def test_exact_tie_offroute_and_ab_mass(self):
        baseline=np.tile([.3,.3,.4],(100,1));mirror=np.tile([.8,.1,.1],(100,1))
        ids=np.arange(100)[::-1];lengths=np.full(100,80)
        selected,_,budget=route(ids,baseline,lengths)
        self.assertEqual(ids[selected].tolist(),[4,3,2,1,0])
        self.assertEqual(budget['selected_rows'],5)
        result=correct(baseline,mirror,selected)
        np.testing.assert_array_equal(result[:,2],baseline[:,2])
        off=np.ones(100,dtype=bool);off[selected]=False
        np.testing.assert_array_equal(result[off],baseline[off])
        np.testing.assert_allclose(result[:,:2].sum(1),baseline[:,:2].sum(1),rtol=0,atol=1e-15)
        expected=.6*(.75*np.array([.5,.5])+.25*np.array([8/9,1/9]))
        np.testing.assert_allclose(result[selected[0],:2],expected)

    def test_token_budget_can_prevent_all_uncertain_rows(self):
        baseline=np.tile([1.,0.,0.],(100,1));baseline[:5]=[.5,.5,0.]
        lengths=np.ones(100);lengths[:5]=512
        selected,_,budget=route(np.arange(100),baseline,lengths)
        self.assertEqual(len(selected),0);self.assertLess(budget['token_budget'],512)

    def test_long_inputs_excluded_and_selection_independent_of_row_order(self):
        rng=np.random.default_rng(42);ids=np.arange(100);base=np.tile([.3,.3,.4],(100,1))
        lengths=np.full(100,100);lengths[:10]=513
        selected,_,_=route(ids,base,lengths);self.assertTrue(np.all(lengths[selected]<=512))
        order=rng.permutation(100);other,_,_=route(ids[order],base[order],lengths[order])
        self.assertEqual(set(ids[selected]),set(ids[order][other]))

    def test_zero_ab_mirror_mass_is_uniform_and_invalid_route_rejected(self):
        baseline=np.tile([.3,.3,.4],(2,1));mirror=np.tile([0.,0.,1.],(2,1))
        np.testing.assert_array_equal(correct(baseline,mirror,np.array([0])),baseline)
        with self.assertRaises(ValueError):correct(baseline,mirror,np.array([0,0]))


if __name__=='__main__':unittest.main()
