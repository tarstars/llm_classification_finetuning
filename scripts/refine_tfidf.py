"""Check whether conditioning, rather than representation, limits sparse logistic quality."""
import os,sys
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import joblib,numpy as np,pandas as pd
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from arena.data import ROOT,TEXT_COLUMNS,load_data,structural_features,swapped,json_write
from arena.models import fit_classifiers

out=ROOT/'runs/attempt_001'
if (out/'final_evaluation.json').exists():
    raise SystemExit('Attempt 001 is frozen; use a fresh artifact root.')
train,test=load_data();frame=pd.concat([train,test],ignore_index=True)
y=np.load(out/'labels.npy');parts={k:np.load(out/f'{k}_idx.npy') for k in ['train','dev','holdout']}
vectorizer,_=joblib.load(out/'tfidf_svd.joblib')
n=len(frame)
matrix=vectorizer.transform(frame[TEXT_COLUMNS].values.T.ravel().tolist())
p,a,b=[matrix[i*n:(i+1)*n].tocsr() for i in range(3)]
delta=a-b;absolute=delta.copy();absolute.data=np.abs(absolute.data)
stats=structural_features(frame);swapstats=structural_features(swapped(frame))
scaler=StandardScaler().fit(np.concatenate([stats[parts['train']],swapstats[parts['train']]]))
scaled=scaler.transform(stats)/np.sqrt(stats.shape[1]);ss=scaler.transform(swapstats)/np.sqrt(stats.shape[1])
x=sparse.hstack([p,a,b,delta,absolute,sparse.csr_matrix(scaled)],format='csr')
xs=sparse.hstack([p,b,a,-delta,absolute,sparse.csr_matrix(ss)],format='csr')
joblib.dump(scaler,out/'tfidf_structural_scaler.joblib')
fit_classifiers('tfidf_sparse_scaled',x,xs,y,parts,out,kinds=('logistic',))
model=joblib.load(out/'tfidf_sparse_scaled__logistic.joblib')[-1]
path=out/'tfidf_sparse_scaled__logistic.json'
import json
metadata=json.loads(path.read_text())
metadata['settings']['selected_n_iter']=model.n_iter_.tolist()
metadata['settings']['selected_converged']=bool(np.all(model.n_iter_<model.max_iter))
metadata['settings']['structural_preprocessing']='training-only standardization / sqrt(45)'
json_write(path,metadata)
