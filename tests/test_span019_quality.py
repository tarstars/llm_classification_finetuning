import unittest
import numpy as np
from arena.span019_quality import paired_interval,parent_from_routed_outputs


class QualityTests(unittest.TestCase):
    def test_group_bootstrap_preserves_multiple_rows_in_one_group(self):
        delta=np.array([1.,1.,-2.]);groups=np.array(['a','a','b'])
        self.assertEqual(paired_interval(delta,groups),[-2.,1.])
        self.assertEqual(paired_interval(delta,groups),paired_interval(delta,groups))

    def test_actual_parent_requires_reverse_rows_and_aligns_answer_order(self):
        ids=np.array([11,22]);g=np.array([[.8,.1,.1],[.1,.8,.1]])
        l=np.array([[.2,.7,.1],[.6,.3,.1]])
        reverse=np.array([[.2,.6,.2]])
        result=parent_from_routed_outputs(ids,g,l,[1],np.array([22]),reverse)
        np.testing.assert_allclose(result[0],[.75,.15,.1])
        np.testing.assert_allclose(result[1],.25*g[1]+.25*reverse[0,[1,0,2]]+.5*l[1,[1,0,2]])
        with self.assertRaises(ValueError):
            parent_from_routed_outputs(ids,g,l,[1],np.array([11]),reverse)


if __name__=='__main__':unittest.main()
