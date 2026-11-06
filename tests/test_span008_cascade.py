from pathlib import Path
import sys,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from arena.span008_cascade import route_mask


class RouteTest(unittest.TestCase):
    def test_budget_stable_ties_and_no_below_threshold_fill(self):
        p=np.full((23,3),1/3)
        self.assertEqual(np.flatnonzero(route_mask(p,0,.1)).tolist(),[0,1])
        self.assertFalse(route_mask(p,2,.1).any())
        self.assertFalse(route_mask(p[:9],0,.1).any())


if __name__=='__main__':unittest.main()
