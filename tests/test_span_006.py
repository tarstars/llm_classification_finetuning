from pathlib import Path
import json,sys,tempfile,unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from arena.span006_inputs import read_inputs,text_digest,token_digest
from arena.span006_source import freeze_head,save_head,load_head


class TransferContracts(unittest.TestCase):
    def test_input_reader_excludes_votes_and_source_names_and_respects_order(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'train.csv'
            pd.DataFrame({'id':[1,2],'prompt':['["first"]','["second"]'],'response_a':['["A"]']*2,
                          'response_b':['["B"]']*2,'winner_model_a':[1,0],'model_a':['SECRET1','SECRET2']}).to_csv(p,index=False)
            f=read_inputs(p,[2,1]);self.assertEqual(f.id.tolist(),[2,1]);self.assertEqual(set(f.columns),{'id','prompt','response_a','response_b'})
            first=text_digest(f);g=pd.read_csv(p);g['winner_model_a']=1-g['winner_model_a'];g['model_a']='ALTERED';g.to_csv(p,index=False)
            self.assertEqual(text_digest(read_inputs(p,[2,1])),first)

    def test_token_fingerprint_preserves_sequence_boundaries(self):
        self.assertNotEqual(token_digest([[1,2],[3]]),token_digest([[1],[2,3]]))

    def test_frozen_numeric_head_round_trip_and_swap(self):
        rng=np.random.default_rng(6);p=rng.dirichlet([2]*3,90);y=rng.integers(0,3,90);d=rng.normal(size=(90,2));s=rng.normal(size=(90,3));groups=np.arange(90)
        cfg={'inner_group_folds':3,'seed':1,'residual_regularization_grid':[.01,.1],
             'calibration_slope_bounds':[1/3,2.],'tie_intercept_bounds':[-1,1],'residual_max_iterations':500}
        model,detail=freeze_head(p,d,s,y,groups,cfg)
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/'head.npz';save_head(file,model);loaded=load_head(file)
            np.testing.assert_array_equal(model.predict_proba(p,d,s),loaded.predict_proba(p,d,s))
            np.testing.assert_allclose(loaded.predict_proba(p,d,s),loaded.predict_proba(p[:,[1,0,2]],-d,s)[:,[1,0,2]],atol=1e-10)
        self.assertTrue(detail['family_fixed_before_confirmation'])


if __name__=='__main__':unittest.main()
