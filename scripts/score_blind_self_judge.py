"""Unseal targets only after checking committed hashes of all 300 judgments."""
from __future__ import annotations
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import xlogy
from scipy.stats import binomtest
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedGroupKFold

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src'))
from arena.data import ROOT, CLASSES, json_write
from arena.judge import calibrate, metrics

directory = ROOT/'runs/attempt_004_self_judge'
destination = REPO/'results/attempt_004_self_judge'
frozen = json.loads((directory/'frozen.json').read_text())
seal_commit = subprocess.check_output(['git', 'log', '-1', '--format=%H', '--',
    'results/attempt_004_self_judge/frozen.json'], cwd=REPO, text=True).strip()
assert seal_commit
committed = json.loads(subprocess.check_output(['git', 'show',
    seal_commit+':results/attempt_004_self_judge/frozen.json'], cwd=REPO, text=True))
assert committed == frozen
for name in ['inputs', 'mapping', 'judgments']:
    path = directory/(name+('.json' if name == 'mapping' else '.jsonl'))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == frozen[name+'_sha256']
records = [json.loads(line) for line in (directory/'judgments.jsonl').read_text().splitlines()]
mapping = json.loads((directory/'mapping.json').read_text())
assert len(records) == len(mapping) == 300
records = sorted(records, key=lambda r:r['case'])
assert [r['case'] for r in records] == list(range(1,301))
ids = np.array([r['id'] for r in mapping])
assert len(set(ids)) == 300

# First target-column read in this experiment occurs below, after the committed seal.
target = pd.read_csv(ROOT/'raw/train.csv', usecols=['id', *CLASSES]).set_index('id')
y = target.loc[ids, CLASSES].to_numpy().argmax(1)
features = pd.read_parquet(ROOT/'eda/attempt_002/features.parquet',
    columns=['id', 'prompt_group', 'split', 'label']).set_index('id')
f = features.loc[ids]
assert f.split.eq('dev').all() and np.array_equal(f.label, y)
groups = f.prompt_group.to_numpy()
assert len(set(groups)) == 300
folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=20261116).split(
    np.zeros(len(y)), y, groups))
rng = np.random.default_rng(20261116)
weights = rng.multinomial(len(y), np.full(len(y),1/len(y)), size=5000)


def losses(p):
    return -np.log(np.clip(p[np.arange(len(y)), y], np.finfo(float).eps,1))


def interval(values):
    return np.quantile(weights@np.asarray(values)/len(y), [.025,.975]).tolist()


def wilson(correct):
    n = len(y); a = correct/n; z=1.959963984540054
    mid = (a+z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(a*(1-a)/n+z*z/(4*n*n))/(1+z*z/n)
    return [float(mid-half),float(mid+half)]


def fixed_probability(prediction, alpha):
    p = np.full((len(prediction),3), (1-alpha)/2, dtype=float)
    p[np.arange(len(prediction)),prediction]=alpha
    return p


def cross_calibration(p):
    result = np.zeros_like(p)
    parameters=[]
    for train, valid in folds:
        def objective(x):
            q=calibrate(p[train],x[0],x[1])
            return -np.log(np.clip(q[np.arange(len(train)),y[train]],1e-15,1)).mean()
        fit=minimize(objective,[1.,0.],method='L-BFGS-B',bounds=[(.25,8.),(-4.,4.)])
        assert fit.success, fit.message
        result[valid]=calibrate(p[valid],*fit.x)
        parameters.append(fit.x.tolist())
    return result,parameters


probs={}
self_pred=[]
for record, row in zip(records,mapping):
    prediction={'A':0,'B':1,'tie':2}[record['class']]
    if row['swapped'] and prediction!=2: prediction=1-prediction
    self_pred.append(prediction)
confidence=np.array([frozen['confidence_mapping'][r['confidence']] for r in records])
self_prob=np.repeat(((1-confidence)/2)[:,None],3,axis=1)
self_prob[np.arange(300),self_pred]=confidence
probs['codex_blind_subjective']=self_prob

for job,key in [('development','gemma26_2048_original'),
                ('development_ctx1024','gemma26_1024_original')]:
    with np.load(ROOT/'runs/attempt_003/jobs'/job/'output/gemma26/judge_predictions.npz') as data:
        positions={int(rid):i for i,rid in enumerate(data['ids'])}
        index=[positions[int(rid)] for rid in ids]
        assert np.array_equal(data['labels'][index],y)
        probs[key]=data['original'][index].astype(float)
reference=ROOT/'runs/attempt_002/jobs/qwen4b1024/output/joint_run/dev_predictions.npz'
with np.load(reference) as data:
    positions={int(rid):i for i,rid in enumerate(data['ids'])}
    index=[positions[int(rid)] for rid in ids]
    assert np.array_equal(data['labels'][index],y)
    selection=json.loads((ROOT/'runs/attempt_002/model_selection.json').read_text())
    probs['qwen3_4b_lora_frozen']=calibrate(data['probabilities'][index],selection['temperature'])
counts=np.bincount(features.loc[features.split.eq('train'),'label'].to_numpy(dtype=int),minlength=3)
probs['training_prior']=np.tile(counts/counts.sum(),(300,1))

results=[]
per_row=pd.DataFrame({'case':np.arange(1,301),'id':ids,'label':y,
    'blind_class':[r['class'] for r in records], 'confidence':[r['confidence'] for r in records],
    'reason':[r['reason'] for r in records]})
for key,p in probs.items():
    assert np.isfinite(p).all() and (p>=0).all()
    np.testing.assert_allclose(p.sum(1),1,atol=1e-5)
    prediction=p.argmax(1); correct=prediction==y; accuracy=float(correct.mean())
    oof=np.zeros_like(p); fold_alpha=[]
    for train,valid in folds:
        a=float(correct[train].mean());fold_alpha.append(a)
        oof[valid]=fixed_probability(prediction[valid],a)
    fitted=float(-xlogy(accuracy,accuracy)-xlogy(1-accuracy,(1-accuracy)/2))
    row={'model':key,**metrics(y,p),'correct':int(correct.sum()),
         'accuracy_wilson_95':wilson(int(correct.sum())),
         'log_loss_bootstrap_95':interval(losses(p)),
         'classification_report':classification_report(y,prediction,labels=[0,1,2],
             target_names=CLASSES,output_dict=True,zero_division=0),
         'predicted_class_counts':np.bincount(prediction,minlength=3).tolist(),
         'fixed_alpha_fit_same_rows':accuracy,'fixed_alpha_fitted_log_loss':fitted,
         'fixed_alpha_cross_fitted_log_loss':float(losses(oof).mean()),
         'fixed_alpha_cross_fitted_95':interval(losses(oof)),'fold_alpha':fold_alpha}
    if key!='training_prior':
        cp,parameters=cross_calibration(p)
        row['cross_fitted_temperature_tie']={**metrics(y,cp),
            'log_loss_bootstrap_95':interval(losses(cp)),'fold_parameters':parameters}
        per_row[key+'_cross_fitted_loss']=losses(cp)
    per_row[key+'_prediction']=prediction;per_row[key+'_loss']=losses(p)
    for c in range(3):per_row[key+'_p'+str(c)]=p[:,c]
    results.append(row)

paired=[]
mine=self_prob.argmax(1)==y
for key,p in probs.items():
    if key=='codex_blind_subjective':continue
    other=p.argmax(1)==y
    won=int((mine&~other).sum());lost=int((~mine&other).sum())
    paired.append({'against':key,'accuracy_gain':float(mine.mean()-other.mean()),
        'accuracy_gain_95':interval(mine.astype(float)-other.astype(float)),
        'self_only_correct':won,'other_only_correct':lost,
        'mcnemar_exact_p':float(binomtest(won,won+lost,.5).pvalue) if won+lost else 1.,
        'raw_probability_log_loss_gain':float((losses(p)-losses(self_prob)).mean()),
        'raw_probability_log_loss_gain_95':interval(losses(p)-losses(self_prob))})

confidence_rows=[]
for level, nominal in frozen['confidence_mapping'].items():
    mask=np.array([r['confidence']==level for r in records])
    if mask.any():confidence_rows.append({'level':level,'nominal_probability':nominal,
        'rows':int(mask.sum()),'accuracy':float(mine[mask].mean())})
report={'attempt':'004_self_judge','scored_utc':datetime.now(timezone.utc).isoformat(),
    'rows':300,'seal_commit':seal_commit,'frozen':frozen,'models':results,'paired':paired,
    'confidence_reliability':confidence_rows,
    'human_label_counts':np.bincount(y,minlength=3).tolist(),
    'protocol':json.loads((directory/'manifest.json').read_text()),
    'notes':['Accuracy measures agreement with recorded human preference, not objective truth.',
      'All self judgments were completed and committed by hash before the target-column read.',
      'Self probabilities are subjective fixed confidence bins, not model token likelihoods.',
      'Fixed-alpha fitted loss uses all labels and is optimistic; cross-fitted loss uses five prompt-group folds.',
      'Temperature/tie calibration is exploratory five-fold fitting on these same 300 rows, with no language-model training.',
      'Comparison combines model, quantization, full versus truncated context, reasoning and judging protocol; it does not isolate parameter count.',
      'Self labels are not automatically valid training truth. No model was retrained and no new submission was made by this analysis.']}
per_row.to_parquet(directory/'scored_judgments.parquet',index=False)
json_write(directory/'results.json',report)
json_write(destination/'results.json',report)
print(json.dumps({'seal_commit':seal_commit,'rows':300,'human_label_counts':report['human_label_counts'],
    'models':[{k:r[k] for k in ['model','accuracy','correct','log_loss','accuracy_wilson_95',
        'fixed_alpha_fitted_log_loss','fixed_alpha_cross_fitted_log_loss','predicted_class_counts']}|
        {'cross_fitted_temperature_tie_log_loss':r.get('cross_fitted_temperature_tie',{}).get('log_loss')}
        for r in results], 'paired':paired,'confidence_reliability':confidence_rows},indent=2))
