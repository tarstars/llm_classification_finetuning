from pathlib import Path
import sys,tempfile,unittest
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from arena.source_portable import write_source,PortableSource,compact_features,residual_predict
from arena.source_models import source_features,ResidualHead


class PortableContracts(unittest.TestCase):
    def test_sparse_weight_export_matches_estimator_including_empty_input(self):
        texts=['red apple','green pear','blue berry']*12;labels=np.arange(len(texts))%3
        parts=[TfidfVectorizer(dtype=np.float32),TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),dtype=np.float32)]
        x=hstack([v.fit_transform(texts) for v in parts],format='csr');model=LogisticRegression(solver='saga',random_state=1,max_iter=200).fit(x,labels)
        bundle={'parts':parts,'model':model,'prior':np.array([.2,.3,.5]),'classes':3,'max_chars':4096}
        with tempfile.TemporaryDirectory() as folder:
            write_source(folder,bundle);portable=PortableSource(folder);observed=portable.predict(texts+[''])
            np.testing.assert_allclose(observed[:-1],model.predict_proba(x),atol=1e-6);np.testing.assert_allclose(observed[-1],bundle['prior'],atol=1e-7)

    def test_compact_and_residual_numeric_portability(self):
        rng=np.random.default_rng(10);p=rng.dirichlet([2]*3,40);qa=rng.dirichlet([2]*6,40);qb=rng.dirichlet([2]*6,40);ratings=np.arange(6)
        d,s=source_features(qa,qb,ratings,'strength');other=compact_features(qa,qb,ratings)
        np.testing.assert_array_equal(d,other[0]);np.testing.assert_array_equal(s,other[1]);y=rng.integers(0,3,40)
        model=ResidualHead(.1).fit(p,d,s,y);head={'theta':model.theta,'dscale':model.dscale,'smean':model.smean,'sscale':model.sscale}
        np.testing.assert_allclose(model.predict_proba(p,d,s),residual_predict(p,*other,head),atol=1e-12)


if __name__=='__main__':unittest.main()
