import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import pandas as pd
from arena.data import parse_turns,structural_features,swapped,dense_pair_features,swapped_pair_features,split_data,prompt_groups
from arena.models import symmetric_predict


class Invariants(unittest.TestCase):
    def test_json_turns(self):
        self.assertEqual(parse_turns('["hello",null,"world"]'),"hello\n\n[TURN]\n\n\n\n[TURN]\n\nworld")
        self.assertEqual(parse_turns('[]'),"")
        with self.assertRaises(ValueError):parse_turns('not JSON')

    def test_dense_swapping_matches_rebuilding(self):
        f=pd.DataFrame({"prompt":["why?","test"],"response_a":["Long useful answer\n- item",""],"response_b":["sorry","code ```"]})
        rng=np.random.default_rng(4)
        embeddings=tuple(rng.normal(size=(2,7)) for _ in range(3))
        p,a,b=embeddings
        np.testing.assert_allclose(swapped_pair_features(embeddings,structural_features(f)),
                                   dense_pair_features((p,b,a),structural_features(swapped(f))))

    def test_prompt_groups_never_cross_splits(self):
        n=300
        f=pd.DataFrame({"prompt":[f'Question {i//2}' for i in range(n)],
                        "winner_model_a":[i%3==0 for i in range(n)],
                        "winner_model_b":[i%3==1 for i in range(n)],
                        "winner_tie":[i%3==2 for i in range(n)]})
        _,parts,fold=split_data(f)
        g=prompt_groups(f)
        for a,b in [('train','dev'),('train','holdout'),('dev','holdout')]:
            self.assertFalse(set(g.iloc[parts[a]]) & set(g.iloc[parts[b]]))
        np.testing.assert_array_equal(fold,split_data(f)[2])

    def test_symmetry_of_probability_average(self):
        class Mock:
            def predict_proba(self,x):return x
        a=np.array([[.5,.2,.3],[.1,.1,.8]])
        b=np.array([[.1,.7,.2],[.2,.5,.3]])
        np.testing.assert_allclose(symmetric_predict(Mock(),a,b),symmetric_predict(Mock(),b,a)[:,[1,0,2]])


if __name__=='__main__':unittest.main()
