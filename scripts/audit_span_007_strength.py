"""Reconstruct H47 transforms, predictions, folds and reported paired effects."""
from pathlib import Path
import gc
import json
import signal
import sys
import time
from datetime import datetime,timezone
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from scipy.special import logsumexp
from sklearn.feature_extraction.text import TfidfVectorizer

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,CLASSES,prompt_groups
from arena.experiment import read_json,atomic_json,sha256,utc_now
from arena.source_models import answer_hash,Strength,source_features,style_features
from arena.span_heads import temperature,loss_rows


def interval(values,groups,seed):
    _,inverse=np.unique(groups,return_inverse=True);sums=np.bincount(inverse,weights=values);counts=np.bincount(inverse)
    rng=np.random.default_rng(seed);samples=rng.integers(0,len(counts),size=(2000,len(counts)))
    return float(np.mean(values)),np.quantile(sums[samples].sum(1)/counts[samples].sum(1),[.025,.975])


def main():
    start=time.monotonic();pub=REPO/'results/span_007';out=ROOT/'runs/span_007/strength';cfg=read_json(REPO/'configs/span_007.json')
    result=read_json(pub/'H47_results.json');contract=read_json(pub/'H47_contract.json')
    assert sha256(REPO/'scripts/run_span_007_strength.py')==contract['source_sha256']==result['source_sha256']
    for path,digest in contract['input_sha256'].items():assert sha256(path)==digest,path
    def expired(*unused):raise TimeoutError('H47 audit cap')
    clock=read_json(ROOT/'runs/span_007/strength_audit_clock.json')
    remaining=(datetime.fromisoformat(clock['deadline'])-datetime.now(timezone.utc)).total_seconds()
    assert remaining>0,'Original audit/recovery cap exhausted'
    signal.signal(signal.SIGALRM,expired);signal.alarm(int(remaining))
    old=ROOT/'runs/span_005';a=np.load(old/'inputs.npz');train=pd.read_parquet(old/'train.parquet');dev=pd.read_parquet(old/'dev.parquet')
    vocab=read_json(old/'source_vocabulary.json');names=vocab['all_training_sources'];classes=vocab['source_classes']
    source=np.load(ROOT/'runs/span_006/freeze/source_predictions.npz');q=np.load(old/'E07/tfidf_predictions.npz')['oof']
    saved=np.load(out/'predictions.npz');n=len(train);texts=np.r_[train.response_a,train.response_b];source_names=np.r_[train.model_a,train.model_b]
    hashes=np.array([answer_hash(t) for t in texts]);groups=prompt_groups(train).to_numpy(dtype=str)
    packed=np.array([t if len(t)<=4096 else t[:2048]+'\n[TRUNCATED]\n'+t[-2048:] for t in texts],dtype=object)
    ytrain=train[CLASSES].to_numpy().argmax(1);folds=a['source_fold'];coverage=np.zeros(2*n,int)
    def class_ratings(r,pairs):
        counts=pd.Series(np.r_[train.model_a.iloc[pairs],train.model_b.iloc[pairs]]).value_counts()
        rare=[name for name in names if name not in classes and counts.get(name,0)>0]
        other=float(np.average([r[name] for name in rare],weights=[counts[name] for name in rare])) if rare else 0.
        return np.array([r.get(name,other) for name in classes])
    for fold in sorted(set(folds)):
        record=np.load(out/f'fold{fold}.npz');fit_pairs=np.flatnonzero(folds!=fold);val_pairs=np.flatnonzero(folds==fold)
        val=np.r_[val_pairs,val_pairs+n];fit=np.r_[fit_pairs,fit_pairs+n];fit=fit[~np.isin(hashes[fit],hashes[val])]
        np.testing.assert_array_equal(fit,record['fit_indices']);np.testing.assert_array_equal(val,record['evaluation_indices'])
        assert not set(groups[fit_pairs])&set(groups[val_pairs]);coverage[val]+=1
        rating=Strength().fit(train.model_a.iloc[fit_pairs],train.model_b.iloc[fit_pairs],ytrain[fit_pairs],names)
        np.testing.assert_allclose(record['ratings'],rating.theta[:-1],atol=1e-8)
        r=dict(zip(names,record['ratings']));np.testing.assert_allclose(record['target'],[r[name] for name in source_names[val]],atol=1e-10)
        np.testing.assert_allclose(record['identity'],q[val]@class_ratings(r,fit_pairs),atol=1e-10)
        # Rebuild only unsupervised feature transforms from the frozen fitting IDs;
        # replay the saved coefficients, never fit another ridge model.
        common={'min_df':2,'sublinear_tf':True,'dtype':np.float32}
        parts=[TfidfVectorizer(max_features=12000,ngram_range=(1,2),**common),TfidfVectorizer(analyzer='char_wb',max_features=24000,ngram_range=(3,5),**common)]
        for v in parts:v.fit(packed[fit].tolist())
        z=hstack([v.transform(packed[val].tolist()) for v in parts],format='csr',dtype=np.float32)
        np.testing.assert_allclose(z@record['coefficients']+record['intercept'],record['prediction'],atol=1e-6)
        for field,key in [('prediction','direct_oof'),('identity','identity_oof'),('target','rating_target')]:np.testing.assert_array_equal(record[field],saved[key][val])
        del z,parts;gc.collect()
    assert np.all(coverage==1) and len(result['source_fold_details'])==3
    assert result['final_alpha']==float(np.median([r['alpha'] for r in result['source_fold_details']]))
    delta=(saved['identity_oof']-saved['rating_target'])**2-(saved['direct_oof']-saved['rating_target'])**2
    gain,ci=interval(delta,np.tile(groups,2),cfg['bootstrap_seed'])
    np.testing.assert_allclose(gain,result['source_mse']['gain']['gain'],atol=1e-12)
    np.testing.assert_allclose(ci,result['source_mse']['gain']['interval_95'],atol=1e-12)
    bundle=joblib.load(out/'regressor.joblib');dt=np.r_[dev.response_a,dev.response_b]
    dt=[t if len(t)<=4096 else t[:2048]+'\n[TRUNCATED]\n'+t[-2048:] for t in dt]
    z=hstack([v.transform(dt) for v in bundle['parts']],format='csr',dtype=np.float32)
    np.testing.assert_array_equal(bundle['model'].predict(z),saved['direct_dev'])
    final_rating=Strength().fit(train.model_a,train.model_b,ytrain,names);r=dict(zip(names,final_rating.theta[:-1]));rclass=class_ratings(r,np.arange(n))
    qa,qb=source['dev_a'],source['dev_b'];ra,rb=qa@rclass,qb@rclass;direct=saved['direct_dev'];nd=len(dev)
    def features(x,y):return (x-y)[:,None],np.column_stack([np.abs(x-y),(x+y)/2])
    designs={'calibration':(np.empty((nd,0)),np.empty((nd,0))),'style':style_features(dev.response_a,dev.response_b),
             'direct_strength':features(direct[:nd],direct[nd:]),'posterior_three':features(ra,rb),'old_compact':source_features(qa,qb,rclass,'strength')}
    p=temperature(a['probabilities'],.9441963863130951);lp=np.log(p);dg=prompt_groups(dev).to_numpy(dtype=str);y=dev[CLASSES].to_numpy().argmax(1)
    np.testing.assert_array_equal(saved['ids'],dev.id);np.testing.assert_array_equal(saved['labels'],y)
    for fold in read_json(out/'meta_folds.json'):
        train_index=np.flatnonzero(a['outer_fold']!=fold['fold']);val=np.flatnonzero(a['outer_fold']==fold['fold'])
        for name,(d,s) in designs.items():
            d=np.asarray(d,dtype=np.float64);s=np.asarray(s,dtype=np.float64)
            theta=np.array(fold['families'][name]['theta']);k=d.shape[1]
            ds=np.maximum(np.sqrt((d[train_index]**2).mean(0)),.05);mean=s[train_index].mean(0);scale=np.maximum(s[train_index].std(0),.05)
            direction=theta[0]*(lp[val,0]-lp[val,1])/2+(d[val]/ds)@theta[3:3+k]
            tie=theta[1]*(lp[val,2]-(lp[val,0]+lp[val,1])/2)+theta[2]+((s[val]-mean)/scale)@theta[3+k:]
            logits=np.column_stack([direction,-direction,tie]);prediction=np.exp(logits-logsumexp(logits,axis=1,keepdims=True))
            np.testing.assert_allclose(prediction,saved[name][val],atol=1e-10)
    np.testing.assert_array_equal(saved['nested_selected'],saved['direct_strength'])
    for name,value in result['preference_metrics'].items():np.testing.assert_allclose(loss_rows(y,saved[name]).mean(),value['log_loss'],atol=1e-12)
    for name,value in result['contrasts'].items():
        gain,ci=interval(loss_rows(y,saved[name])-loss_rows(y,saved['direct_strength']),dg,cfg['bootstrap_seed'])
        np.testing.assert_allclose(gain,value['gain'],atol=1e-12);np.testing.assert_allclose(ci,value['interval_95'],atol=1e-12)
    proof={'passed':True,'checked_at':utc_now(),'source_outer_folds':3,'preference_outer_folds':5,
           'source_predictions_replayed':2*n,'final_serialized_predictions_replayed':2*nd,
           'meta_predictions_reconstructed':5*nd,'source_fold_targets_verified':True,'confirmation_targets_used':False,
           'audit_seconds':time.monotonic()-start,'H47_results_sha256':sha256(pub/'H47_results.json'),'source_sha256':sha256(Path(__file__)),
           'output_sha256':{p.name:sha256(p) for p in out.glob('*') if p.suffix in ['.npz','.joblib']}}
    atomic_json(pub/'H47_audit.json',proof);signal.alarm(0);print(proof)


if __name__=='__main__':main()
