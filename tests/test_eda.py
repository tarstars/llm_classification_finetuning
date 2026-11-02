import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from arena.eda_features import canonical_comparison
from arena.eda_analysis import GroupBootstrap,temper,tie_calibrate


class EdaStatistics(unittest.TestCase):
    def test_duplicate_comparison_respects_answer_exchange(self):
        self.assertEqual(canonical_comparison('question','a','b',0),
                         canonical_comparison('question','b','a',1))
        self.assertEqual(canonical_comparison('question','a','b',2),
                         canonical_comparison('question','b','a',2))
        self.assertNotEqual(canonical_comparison('question','a','b',0)[0],
                            canonical_comparison('another question','a','b',0)[0])

    def test_bootstrap_resamples_whole_prompt_groups(self):
        b=GroupBootstrap(['same','same','other'],repeats=200,seed=7)
        means=b.means(np.array([1.,1.,10.]))
        self.assertTrue(np.isin(means,[1.,4.,10.]).all())
        self.assertEqual(b.interval(np.zeros(3)),[0.,0.])
        np.testing.assert_array_equal(means,GroupBootstrap(['same','same','other'],200,7).means([1,1,10]))

    def test_postprocessing_preserves_probabilities_and_swap_symmetry(self):
        p=np.array([[.1,.7,.2],[.4,.3,.3]])
        q=tie_calibrate(p,.7,-.2)
        np.testing.assert_allclose(q.sum(1),1)
        self.assertTrue((q>0).all())
        np.testing.assert_allclose(tie_calibrate(p[:,[1,0,2]],.7,-.2)[:,[1,0,2]],q)
        np.testing.assert_allclose(temper(p,1),p)


if __name__=='__main__':unittest.main()
