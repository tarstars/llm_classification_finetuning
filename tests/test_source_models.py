import sys,unittest
from pathlib import Path
import numpy as np
from scipy.optimize import approx_fprime
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from arena.source_models import Strength,ResidualHead,source_features,answer_hash,purge_answers
from arena.source_evaluation import fit_selected
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_span_005_e07 import fit_head


class SourceContracts(unittest.TestCase):
    def test_strength_recovers_direction_and_preserves_swap(self):
        y=np.array([0]*70+[1]*10+[2]*20)
        m=Strength().fit(['strong']*100,['weak']*100,y,['strong','weak'])
        self.assertGreater(m.ratings(['strong'])[0],m.ratings(['weak'])[0])
        np.testing.assert_allclose(m.predict(['strong'],['weak']),m.predict(['weak'],['strong'])[:,[1,0,2]])

    def test_duplicate_purging_normalizes_case_whitespace(self):
        hashes=[answer_hash(x) for x in ['Hello  WORLD','retained','\nhello world ']]
        np.testing.assert_array_equal(purge_answers(hashes,[answer_hash('HELLO WORLD')]),[False,True,False])

    def test_residual_gradient_and_swap_with_uncertain_sources(self):
        rng=np.random.default_rng(11);p=rng.dirichlet([2,2,2],40);qa=rng.dirichlet([2]*5,40);qb=rng.dirichlet([2]*5,40)
        d,s=source_features(qa,qb,np.arange(5),'full');y=rng.integers(0,3,40)
        head=ResidualHead(.1).fit(p,d,s,y);theta=head.theta+.01
        features=head.transform(p,d,s)
        numeric=approx_fprime(theta,lambda t:head.objective(t,features,y)[0],1e-7)
        np.testing.assert_allclose(numeric,head.objective(theta,features,y)[1],atol=2e-6)
        ds,ss=source_features(qb,qa,np.arange(5),'full')
        np.testing.assert_allclose(head.predict_proba(p,d,s),head.predict_proba(p[:,[1,0,2]],ds,ss)[:,[1,0,2]],atol=1e-10)

    def test_residual_selection_does_not_read_outer_labels(self):
        rng=np.random.default_rng(12);p=rng.dirichlet([2,2,2],120);y=rng.integers(0,3,120);groups=np.arange(120)
        features={'empty':(np.empty((120,0)),np.empty((120,0)))};fit=np.arange(90);val=np.arange(90,120)
        cfg={'seed':1,'inner_group_folds':3,'residual_regularization_grid':[.1],
             'calibration_slope_bounds':[1/3,2.],'tie_intercept_bounds':[-1,1],'residual_max_iterations':500}
        first,detail,_=fit_selected(p,y,groups,features,fit,val,cfg,['empty'])
        changed=y.copy();changed[val]=(changed[val]+1)%3
        second,other,_=fit_selected(p,changed,groups,features,fit,val,cfg,['empty'])
        np.testing.assert_array_equal(first,second);self.assertEqual(detail,other)

    def test_attribution_fit_ignores_heldout_labels_and_vocabulary(self):
        text=np.array(['alpha beta '+('apple' if i%2 else 'pear') for i in range(48)]+
                      ['evaluationonlytoken alpha']*12,dtype=object)
        y=np.arange(60)%2;fit=np.arange(48);val=np.arange(48,60)
        cfg={'source_word_features':100,'source_char_features':100,'source_word_ngram_range':[1,2],
             'source_char_ngram_range':[3,5],'source_logistic_C':1.,'source_max_iterations':150,
             'source_tolerance':.001,'seed':1,'source_max_chars':4096}
        first,bundle,_=fit_head('tfidf',fit,val,text,np.empty((60,0)),y,2,cfg)
        changed=y.copy();changed[val]=1-changed[val]
        second,_,_=fit_head('tfidf',fit,val,text,np.empty((60,0)),changed,2,cfg)
        np.testing.assert_array_equal(first,second)
        self.assertNotIn('evaluationonlytoken',bundle['parts'][0].vocabulary_)


if __name__=='__main__':unittest.main()
