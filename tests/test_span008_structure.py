from pathlib import Path
import sys,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from arena.span008_structure import TreeResidual


class StructureTests(unittest.TestCase):
    def test_offset_zero_and_symmetrization(self):
        rng=np.random.default_rng(7);p=rng.dirichlet([2,2,2],size=120);y=rng.integers(0,3,size=120)
        x=np.log(p);z=x[:,[1,0,2]]
        cfg={'temperature_bounds':[.5,3.],'tie_intercept_bounds':[-1,1],'seed':7,'structural_trees':{'n_estimators':4,'num_leaves':3,'max_depth':2,'min_child_samples':5,'learning_rate':.1,'reg_lambda':1.}}
        fit=TreeResidual(cfg,.5).fit(p,x,z,y)
        np.testing.assert_allclose(fit.predict_proba(p,x,z,0),fit.cal.predict_proba(p),atol=1e-12)
        out=fit.predict_proba(p,x,z);back=fit.predict_proba(p[:,[1,0,2]],z,x)[:,[1,0,2]]
        np.testing.assert_allclose(out,back,atol=1e-12)
        self.assertGreater(np.max(np.abs(out-fit.cal.predict_proba(p))),1e-5)


if __name__=='__main__':unittest.main()
