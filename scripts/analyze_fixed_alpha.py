"""Evaluate one global confidence assigned to the frozen judge's hard class."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
from scipy.special import xlogy
from sklearn.model_selection import StratifiedGroupKFold

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT

p=argparse.ArgumentParser()
p.add_argument('--job',default='development')
p.add_argument('--model',default='gemma26')
p.add_argument('--config',type=Path,default=REPO/'configs/attempt_003_judge.json')
a=p.parse_args()
config=json.loads(a.config.read_text())
out=ROOT/'runs/attempt_003/jobs'/a.job/'output'/a.model
summary=json.loads((out/'judge_summary.json').read_text())
assert summary['config_sha256']==hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
data=np.load(out/'judge_predictions.npz')
features=pd.read_parquet(ROOT/'eda/attempt_002/features.parquet',columns=['id','label','split','prompt_group']).set_index('id').loc[data['ids']]
y=data['labels'];assert features.split.eq('dev').all() and np.array_equal(features.label,y)
groups=features.prompt_group.to_numpy()
folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=config['seed']).split(np.zeros(len(y)),y,groups))
codes,unique=pd.factorize(groups,sort=True)
counts=np.bincount(codes)
weights=np.random.default_rng(config['seed']).multinomial(len(unique),np.full(len(unique),1/len(unique)),size=2000)


def bound(accuracy):
    return float(-xlogy(accuracy,accuracy)-xlogy(1-accuracy,(1-accuracy)/2))


def probabilities(prediction,alpha):
    result=np.full((len(prediction),3),(1-alpha)/2,dtype=np.float64)
    result[np.arange(len(prediction)),prediction]=alpha
    return result


def losses(probability):
    return -np.log(np.clip(probability[np.arange(len(y)),y],np.finfo(float).eps,1))


rows=[]
for view in ['original','average']:
    prediction=data[view].argmax(1)
    accuracy=float((prediction==y).mean());alpha=accuracy
    fitted=probabilities(prediction,alpha)
    np.testing.assert_allclose(fitted.sum(1),1)
    np.testing.assert_allclose(losses(fitted).mean(),bound(accuracy))
    numeric=minimize_scalar(lambda x:-accuracy*np.log(x)-(1-accuracy)*np.log((1-x)/2),
                            bounds=(1e-8,1-1e-8),method='bounded')
    assert abs(numeric.x-alpha)<1e-5
    oof=np.zeros_like(fitted);fold_alpha=[]
    for train,valid in folds:
        assert not set(groups[train]) & set(groups[valid])
        value=float((prediction[train]==y[train]).mean())
        fold_alpha.append(value)
        oof[valid]=probabilities(prediction[valid],value)
    heldout_losses=losses(oof)
    numer=np.bincount(codes,weights=heldout_losses)
    boot=(weights@numer)/(weights@counts)
    rows.append({'view':view,'rows':len(y),'correct':int((prediction==y).sum()),
        'hard_class_definition':'argmax of saved judge probabilities; no new prompting',
        'accuracy':accuracy,'optimal_alpha_fit_same_rows':alpha,'other_class_probability':(1-alpha)/2,
        'best_same_rows_log_loss':bound(accuracy),
        'cross_fitted_log_loss':float(heldout_losses.mean()),
        'cross_fitted_log_loss_95_interval':np.quantile(boot,[.025,.975]).tolist(),
        'fold_alpha':fold_alpha,'alpha_keeps_selected_class_most_probable':bool(alpha>=1/3)})
result={'job':a.job,'model':a.model,'max_length':config['max_length'],
        'config_sha256':summary['config_sha256'],'rows':rows,
        'formula':'L(alpha)=-accuracy*ln(alpha)-(1-accuracy)*ln((1-alpha)/2); optimum alpha=accuracy',
        'accuracy_for_log_loss_0_8_above_chance':brentq(lambda x:bound(x)-.8,1/3+1e-8,1-1e-8),
        'note':'Same-row optimum is optimistic; held-out alpha uses five prompt-group folds. Development was used for previous selection. No new GPU inference, model training or submission.'}
name=f"fixed_alpha_{config['max_length']}.json"
(REPO/'results/attempt_003'/name).write_text(json.dumps(result,indent=2)+'\n')
(ROOT/'runs/attempt_003'/name).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
