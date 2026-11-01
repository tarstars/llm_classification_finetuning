"""Freeze a dev-selected ensemble, then evaluate all frozen candidates once."""
import hashlib
import json
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import pandas as pd
from scipy.optimize import minimize,minimize_scalar
from arena.data import ROOT,CLASSES,SEED,load_data,prompt_groups,json_write
from arena.models import scores

out=ROOT/"runs/attempt_001"
repo=Path(__file__).resolve().parents[1]
report_dir=repo/"results/attempt_001"
report_dir.mkdir(parents=True,exist_ok=True)
if (out/"final_evaluation.json").exists():
    raise SystemExit("Final holdout was already evaluated. Use the saved results; do not retune on it.")
required={f'{representation}__{head}' for representation in ['tfidf_svd','rnn','qwen3_256'] for head in ['logistic','gbdt','neural']}
rows=[];predictions={}
for path in sorted(out.glob('*.json')):
    row=json.loads(path.read_text())
    if isinstance(row,dict) and 'model' in row and 'dev' in row:
        rows.append(row)
        predictions[row['model']]=np.load(out/f"{row['model']}_predictions.npy")
missing=required-set(predictions)
if missing:raise SystemExit(f"Required experiments incomplete: {sorted(missing)}")
y=np.load(out/'labels.npy')
dev=np.load(out/'dev_idx.npy');hold=np.load(out/'holdout_idx.npy');train_idx=np.load(out/'train_idx.npy')
train,test=load_data()
prior=np.bincount(y[train_idx],minlength=3).astype(float)
prior/=prior.sum()
predictions['training_prior']=np.tile(prior,(len(train)+len(test),1))
rows.append({'model':'training_prior','representation':'none','classifier':'prior','dev':scores(y[dev],predictions['training_prior'][dev])})
predictions['uniform']=np.full((len(train)+len(test),3),1/3,dtype=float)
rows.append({'model':'uniform','representation':'none','classifier':'uniform','dev':scores(y[dev],predictions['uniform'][dev])})
majority_class=int(np.bincount(y[train_idx],minlength=3).argmax())
predictions['most_frequent_class']=np.tile(np.eye(3)[majority_class],(len(train)+len(test),1))
rows.append({'model':'most_frequent_class','representation':'none','classifier':'majority',
             'dev':scores(y[dev],predictions['most_frequent_class'][dev]),
             'settings':{'majority_class':CLASSES[majority_class],'fit_partition':'train',
                         'log_loss_note':'Hard one-hot prediction; sklearn clips zero probabilities at float64 machine epsilon. This baseline intentionally always predicts the original training-majority class, without swap averaging.'}})

# Include all fixed candidates; nonnegative weights and one temperature use only dev labels.
names=sorted(n for n in predictions if n not in ['most_frequent_class','training_prior'])
stack=np.stack([predictions[n] for n in names])
def loss(weights):
    p=np.tensordot(weights,stack[:,dev,:],axes=1)
    return float(-np.log(np.clip(p[np.arange(len(dev)),y[dev]],1e-12,1)).mean())
fit=minimize(loss,np.full(len(names),1/len(names)),method='SLSQP',bounds=[(0,1)]*len(names),
             constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],options={'maxiter':300,'ftol':1e-10})
if not fit.success:raise RuntimeError(fit.message)
weights=np.maximum(fit.x,0);weights/=weights.sum()
blend=np.tensordot(weights,stack,axes=1)
def temper(p,t):
    logits=np.log(np.clip(p,1e-12,1))/t
    logits-=logits.max(1,keepdims=True)
    exp=np.exp(logits);return exp/exp.sum(1,keepdims=True)
temp=minimize_scalar(lambda t:float(-np.log(temper(blend[dev],t)[np.arange(len(dev)),y[dev]]).mean()),bounds=(.7,2),method='bounded').x
ensemble=temper(blend,temp)
predictions['calibrated_ensemble']=ensemble
rows.append({'model':'calibrated_ensemble','representation':'ensemble','classifier':'calibrated_blend','dev':scores(y[dev],ensemble[dev])})

# A second, independently dev-selected ensemble can run without API calls on Kaggle.
offline_names=[n for n in names if n in ['structural__gbdt','tfidf_sparse__logistic','tfidf_svd__gbdt','rnn__end_to_end','rnn__gbdt']]
offline_stack=np.stack([predictions[n] for n in offline_names])
def offline_loss(w):
    p=np.tensordot(w,offline_stack[:,dev,:],axes=1)
    return float(-np.log(np.clip(p[np.arange(len(dev)),y[dev]],1e-12,1)).mean())
offline_fit=minimize(offline_loss,np.full(len(offline_names),1/len(offline_names)),method='SLSQP',
    bounds=[(0,1)]*len(offline_names),constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],options={'maxiter':300,'ftol':1e-10})
if not offline_fit.success:raise RuntimeError(offline_fit.message)
ow=np.maximum(offline_fit.x,0);ow[ow<1e-4]=0;ow/=ow.sum()
offline_blend=np.tensordot(ow,offline_stack,axes=1)
ot=minimize_scalar(lambda t:float(-np.log(temper(offline_blend[dev],t)[np.arange(len(dev)),y[dev]]).mean()),bounds=(.7,2),method='bounded').x
predictions['offline_ensemble']=temper(offline_blend,ot)
rows.append({'model':'offline_ensemble','representation':'offline_ensemble','classifier':'calibrated_blend',
             'dev':scores(y[dev],predictions['offline_ensemble'][dev])})
offline_config={'weights':dict(zip(offline_names,ow.tolist())),'temperature':ot,
    'settings':{r['model']:r.get('settings',{}) for r in rows if r['model'] in offline_names},
    'rnn_best_epoch':int(min(json.loads((out/'rnn_history.json').read_text()),key=lambda r:r['dev_log_loss'])['epoch'])}
json_write(out/'offline_config.json',offline_config)
json_write(report_dir/'offline_config.json',offline_config)
best_single=min((r for r in rows if r['classifier']!='calibrated_blend'),key=lambda r:r['dev']['log_loss'])['model']
selected=min(rows,key=lambda r:r['dev']['log_loss'])['model']
selection={'weights':dict(zip(names,weights.tolist())),'temperature':temp,'best_single_on_dev':best_single,
           'selected_on_dev':selected,'fit_partition':'dev','holdout_used_for_selection':False,
           'candidate_prediction_sha256':{n:hashlib.sha256(predictions[n].tobytes()).hexdigest() for n in names}}
json_write(out/'frozen_selection.json',selection)

# This is the only point at which holdout labels are used to evaluate these models.
for row in rows:
    row['dev']=scores(y[dev],predictions[row['model']][dev])
    row['holdout']=scores(y[hold],predictions[row['model']][hold])
group=prompt_groups(train).iloc[hold].to_numpy()
groups,inverse=np.unique(group,return_inverse=True)
chosen=predictions[selected][hold]
reference=predictions['training_prior'][hold]
delta=-np.log(chosen[np.arange(len(hold)),y[hold]])+np.log(reference[np.arange(len(hold)),y[hold]])
sums=np.bincount(inverse,weights=delta);counts=np.bincount(inverse)
rng=np.random.default_rng(SEED)
boot=[]
for _ in range(2000):
    sampled=rng.integers(0,len(groups),len(groups))
    boot.append(sums[sampled].sum()/counts[sampled].sum())
confirmation={'selected_model':selected,'holdout_log_loss':scores(y[hold],chosen)['log_loss'],
    'difference_vs_prior':float(delta.mean()),'group_bootstrap_95pct_ci':np.quantile(boot,[.025,.975]).tolist(),
    'n_holdout':len(hold),'n_prompt_groups':len(groups),'bootstrap_replicates':2000}
evaluation={'selection':selection,'confirmation':confirmation,'models':rows,
    'limitations':['One fixed prompt-grouped split; no repeated-seed or full cross-validation evidence.',
        'Visible test has three examples; submission CSV is a format check, not a leaderboard score.',
        'The sparse TF-IDF logistic solver reached its 250-iteration cap; treat it as an optimization-limited baseline.',
        'Qwen features require an external service and are unsuitable for internet-disabled Kaggle inference without a local model.',
        'The from-scratch GRU has limited context and cannot be assumed to match pretrained language-model quality.']}
json_write(out/'final_evaluation.json',evaluation)
json_write(report_dir/'metrics.json',evaluation)
json_write(report_dir/'data_summary.json',json.loads((out/'data_summary.json').read_text()))
for name in ['rnn_history.json','rnn_metadata.json','tfidf_metadata.json']:
    json_write(report_dir/name,json.loads((out/name).read_text()))
json_write(report_dir/'qwen_metadata.json',json.loads((ROOT/'embeddings/qwen3_256.json').read_text()))
np.save(out/'calibrated_ensemble_predictions.npy',ensemble)
np.save(out/'offline_ensemble_predictions.npy',predictions['offline_ensemble'])
submission=pd.DataFrame(predictions[selected][len(train):],columns=CLASSES)
submission.insert(0,'id',test.id.to_numpy())
sample=pd.read_csv(ROOT/'raw/sample_submission.csv')
assert submission.columns.tolist()==sample.columns.tolist()
assert submission.id.tolist()==sample.id.tolist()
assert np.allclose(submission[CLASSES].sum(1),1)
submission.to_csv(out/'submission_preview.csv',index=False)
flat=[]
for row in rows:
    flat.append({'model':row['model'],'dev_log_loss':row['dev']['log_loss'],'holdout_log_loss':row['holdout']['log_loss'],
                 'holdout_accuracy':row['holdout']['accuracy'],'holdout_balanced_accuracy':row['holdout']['balanced_accuracy'],
                 'holdout_macro_f1':row['holdout']['macro_f1'],'holdout_brier':row['holdout']['brier']})
table=pd.DataFrame(flat).sort_values('dev_log_loss')
table.to_json(report_dir/'comparison.json',orient='records',indent=2)
print(table.to_string(index=False))
print(json.dumps(confirmation,indent=2))
