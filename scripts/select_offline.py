"""Freeze deployable ensemble choices using development labels only."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from scipy.optimize import minimize,minimize_scalar
from arena.data import ROOT,json_write

out=ROOT/'runs/attempt_001'
names=sorted(['structural__gbdt','tfidf_sparse__logistic','tfidf_svd__gbdt','rnn__end_to_end','rnn__gbdt'])
rows={n:json.loads((out/f'{n}.json').read_text()) for n in names}
stack=np.stack([np.load(out/f'{n}_predictions.npy') for n in names])
y=np.load(out/'labels.npy');dev=np.load(out/'dev_idx.npy')
def loss(w):
    p=np.tensordot(w,stack[:,dev,:],axes=1)
    return float(-np.log(np.clip(p[np.arange(len(dev)),y[dev]],1e-12,1)).mean())
fit=minimize(loss,np.full(len(names),1/len(names)),method='SLSQP',bounds=[(0,1)]*len(names),
    constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],options={'maxiter':300,'ftol':1e-10})
if not fit.success:raise RuntimeError(fit.message)
w=np.maximum(fit.x,0);w[w<1e-4]=0;w/=w.sum()
blend=np.tensordot(w,stack[:,dev,:],axes=1)
def temper(p,t):
    logits=np.log(np.clip(p,1e-12,1))/t;logits-=logits.max(1,keepdims=True)
    exp=np.exp(logits);return exp/exp.sum(1,keepdims=True)
t=minimize_scalar(lambda t:float(-np.log(temper(blend,t)[np.arange(len(dev)),y[dev]]).mean()),bounds=(.7,2),method='bounded').x
config={'weights':dict(zip(names,w.tolist())),'temperature':t,
    'settings':{n:rows[n].get('settings',{}) for n in names},
    'rnn_best_epoch':int(min(json.loads((out/'rnn_history.json').read_text()),key=lambda r:r['dev_log_loss'])['epoch'])}
json_write(out/'offline_config.json',config)
print(json.dumps(config,indent=2))
print('Development log loss:',float(-np.log(temper(blend,t)[np.arange(len(dev)),y[dev]]).mean()))
