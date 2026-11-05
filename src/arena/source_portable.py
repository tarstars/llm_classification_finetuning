"""Prediction-only portable sparse source and compact residual parameters."""
from pathlib import Path
import json
import numpy as np
from scipy.sparse import hstack
from scipy.special import softmax,logsumexp
from sklearn.feature_extraction.text import TfidfVectorizer


def write_source(folder,bundle):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);parts=[]
    allowed=['analyzer','ngram_range','lowercase','strip_accents','token_pattern','norm','use_idf','smooth_idf','sublinear_tf']
    arrays={'coef':bundle['model'].coef_,'intercept':bundle['model'].intercept_,'model_classes':bundle['model'].classes_,'prior':bundle['prior']}
    for i,v in enumerate(bundle['parts']):
        params=v.get_params();parts.append({'parameters':{key:params[key] for key in allowed},'vocabulary':{k:int(value) for k,value in v.vocabulary_.items()}})
        arrays['idf_'+str(i)]=v.idf_
    description={'max_chars':bundle['max_chars'],'source_classes':bundle['classes'],'parts':parts}
    (folder/'source.json').write_text(json.dumps(description,ensure_ascii=False,separators=(',',':')))
    np.savez_compressed(folder/'source_arrays.npz',**arrays)


class PortableSource:
    def __init__(self,folder):
        folder=Path(folder);self.config=json.loads((folder/'source.json').read_text());self.arrays=np.load(folder/'source_arrays.npz');self.parts=[]
        for i,part in enumerate(self.config['parts']):
            params=dict(part['parameters']);params['ngram_range']=tuple(params['ngram_range'])
            v=TfidfVectorizer(vocabulary=part['vocabulary'],dtype=np.float32,**params);v.idf_=self.arrays['idf_'+str(i)]
            self.parts.append(v)

    def predict(self,texts):
        cap=self.config['max_chars'];texts=[t if len(t)<=cap else t[:cap//2]+'\n[TRUNCATED]\n'+t[-cap//2:] for t in texts]
        x=hstack([v.transform(texts) for v in self.parts],format='csr',dtype=np.float32)
        logits=x@self.arrays['coef'].T+self.arrays['intercept'];assert logits.shape[1]>=3
        q=np.full((len(texts),self.config['source_classes']),1e-12);q[:,self.arrays['model_classes']]=softmax(logits,axis=1)
        q[x.getnnz(axis=1)==0]=self.arrays['prior'];q/=q.sum(1,keepdims=True)
        return q.astype(np.float32)


def compact_features(qa,qb,ratings):
    qa=np.asarray(qa,float);qb=np.asarray(qb,float);ea,eb=qa@ratings,qb@ratings
    va=np.maximum(0,qa@(ratings**2)-ea**2);vb=np.maximum(0,qb@(ratings**2)-eb**2)
    ha=-(qa*np.log(np.maximum(qa,1e-12))).sum(1);hb=-(qb*np.log(np.maximum(qb,1e-12))).sum(1)
    return np.column_stack([ea-eb,ha-hb]),np.column_stack([np.abs(ea-eb),(ea+eb)/2,va+vb,ha+hb,np.abs(ha-hb)])


def residual_predict(p,d,s,head):
    p=np.clip(np.asarray(p,float),1e-12,1);p/=p.sum(1,keepdims=True);lp=np.log(p);theta=head['theta'];k=d.shape[1]
    direction=theta[0]*(lp[:,0]-lp[:,1])/2+(d/head['dscale'])@theta[3:3+k]
    tie=theta[1]*(lp[:,2]-(lp[:,0]+lp[:,1])/2)+theta[2]+((s-head['smean'])/head['sscale'])@theta[3+k:]
    logits=np.column_stack([direction,-direction,tie]);return np.exp(logits-logsumexp(logits,axis=1,keepdims=True))
