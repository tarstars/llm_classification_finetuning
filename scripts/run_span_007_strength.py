"""Nested direct-strength regression; no held-out model names or confirmation votes."""
from pathlib import Path
import gc
import os
import resource
import signal
import sys
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,CLASSES,prompt_groups
from arena.experiment import read_json,atomic_json,write_intent,sha256,EventJournal,utc_now
from arena.source_models import Strength,answer_hash,source_features,style_features
from arena.source_evaluation import evaluate,summarize
from arena.span_calibration import grouped_folds
from arena.span_heads import temperature,paired_group_difference,paired_group_gain


def main():
    out=ROOT/'runs/span_007/strength';out.mkdir(parents=True,exist_ok=True);pub=REPO/'results/span_007'
    cfg=read_json(REPO/'configs/span_007.json');oldcfg=read_json(REPO/'configs/span_005.json')
    contract=read_json(pub/'H47_contract.json');assert sha256(Path(__file__))==contract['source_sha256']
    for path,digest in contract['input_sha256'].items():assert sha256(path)==digest,path
    write_intent(out/'started.json',{'at':utc_now(),'pid':os.getpid(),'source_sha256':sha256(Path(__file__))})
    started=time.monotonic();journal=EventJournal(out)
    def expired(*unused):raise TimeoutError('Original H47 CPU cap reached')
    signal.signal(signal.SIGALRM,expired);signal.alarm(cfg['cpu_stage_seconds_caps']['strength'])
    def guard():assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2<6,'H47 RSS guard'
    inv=read_json(pub/'artifact_inventory.json');train=pd.read_parquet(inv['input_artifacts']['train.parquet']['path'])
    dev=pd.read_parquet(inv['input_artifacts']['dev.parquet']['path'],columns=['id','prompt','response_a','response_b',*CLASSES])
    confirm=pd.read_parquet(inv['input_artifacts']['confirmation_inputs']['path'])
    oldroot=ROOT/'runs/span_005';arrays=np.load(oldroot/'inputs.npz');folds=arrays['source_fold'];groups=prompt_groups(train).to_numpy(dtype=str)
    vocab=read_json(oldroot/'source_vocabulary.json');names=vocab['all_training_sources'];classes=vocab['source_classes']
    text=np.r_[train.response_a,train.response_b,dev.response_a,dev.response_b];n=len(train);nd=len(dev)
    hashes=np.array([answer_hash(t) for t in text]);source_names=np.r_[train.model_a,train.model_b]
    packed=np.array([t if len(t)<=4096 else t[:2048]+'\n[TRUNCATED]\n'+t[-2048:] for t in text],dtype=object)
    cached=np.load(oldroot/'E07/tfidf_predictions.npz')['oof']
    direct=np.full(2*n,np.nan);identity=np.full_like(direct,np.nan);target=np.full_like(direct,np.nan)
    pair_y=train[CLASSES].to_numpy().argmax(1);all_pairs=np.arange(n);selected=[];details=[]

    def regress(x,y,alpha):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model=Ridge(alpha=alpha,solver='lsqr',max_iter=500,tol=1e-4).fit(x,y)
        assert max(model.n_iter_)<500,'Preserve nonconverged ridge; do not expand its cap'
        if caught:journal.emit('strength','solver_warning',metrics={'warning_count':len(caught)})
        return model

    def ratings(fit_pairs):
        model=Strength().fit(train.model_a.iloc[fit_pairs],train.model_b.iloc[fit_pairs],pair_y[fit_pairs],names)
        return dict(zip(names,model.theta[:-1]))

    def class_ratings(r,fit_pairs):
        counts=pd.Series(np.r_[train.model_a.iloc[fit_pairs],train.model_b.iloc[fit_pairs]]).value_counts()
        rare=[name for name in names if name not in classes and counts.get(name,0)>0]
        other=float(np.average([r[name] for name in rare],weights=[counts[name] for name in rare])) if rare else 0.
        return np.array([r.get(name,other) for name in classes])

    def fit_matrices(fit_pairs,val_pairs,extra_held=()):
        fit=np.r_[fit_pairs,fit_pairs+n];val=np.r_[val_pairs,val_pairs+n]
        held=np.r_[val,np.r_[extra_held,np.asarray(extra_held)+n]] if len(extra_held) else val
        fit=fit[~np.isin(hashes[fit],hashes[held])]
        common={'min_df':2,'sublinear_tf':True,'dtype':np.float32}
        parts=[TfidfVectorizer(max_features=12000,ngram_range=(1,2),**common),
               TfidfVectorizer(analyzer='char_wb',max_features=24000,ngram_range=(3,5),**common)]
        x=hstack([v.fit_transform(packed[fit].tolist()) for v in parts],format='csr',dtype=np.float32)
        z=hstack([v.transform(packed[val].tolist()) for v in parts],format='csr',dtype=np.float32)
        return fit,val,parts,x,z

    for fold in sorted(set(folds)):
        fit_pairs=all_pairs[folds!=fold];val_pairs=all_pairs[folds==fold]
        assert not set(groups[fit_pairs])&set(groups[val_pairs])
        losses={a:[] for a in [1.,10.,100.]}
        for inner,(a,b) in enumerate(grouped_folds(pair_y[fit_pairs],groups[fit_pairs],3,cfg['seed'])):
            fp,vp=fit_pairs[a],fit_pairs[b];r=ratings(fp)
            fit,val,parts,x,z=fit_matrices(fp,vp,val_pairs)
            yt=np.array([r[name] for name in source_names[fit]]);yv=np.array([r[name] for name in source_names[val]])
            for alpha in losses:
                reg=regress(x,yt,alpha)
                losses[alpha].extend((reg.predict(z)-yv)**2)
            guard();journal.emit('strength','inner_complete',rank=int(fold),step=inner,rows=len(val))
            del x,z,parts;gc.collect()
        chosen=min(losses,key=lambda a:float(np.mean(losses[a])));selected.append(chosen)
        r=ratings(fit_pairs);fit,val,parts,x,z=fit_matrices(fit_pairs,val_pairs)
        oldfold=np.load(oldroot/f'E07/tfidf_fold{fold}.npz');np.testing.assert_array_equal(fit,oldfold['fit_indices'])
        yt=np.array([r[name] for name in source_names[fit]]);yv=np.array([r[name] for name in source_names[val]])
        reg=regress(x,yt,chosen)
        direct[val]=reg.predict(z);identity[val]=cached[val]@class_ratings(r,fit_pairs);target[val]=yv
        np.savez(out/f'fold{fold}.npz',fit_indices=fit,evaluation_indices=val,prediction=direct[val],identity=identity[val],target=yv,
                 coefficients=reg.coef_,intercept=reg.intercept_,ratings=np.array([r[name] for name in names]))
        details.append({'fold':int(fold),'alpha':chosen,'inner_mse':{str(a):float(np.mean(v)) for a,v in losses.items()},
                        'fit_answers':len(fit),'validation_answers':len(val),'iterations':reg.n_iter_.tolist()})
        guard();journal.emit('strength','outer_complete',rank=int(fold),rows=len(val))
        del x,z,parts;gc.collect()
    assert np.isfinite(direct).all() and np.isfinite(identity).all() and np.isfinite(target).all()
    # One final source fit, with no development/confirmation source target access.
    full_r=ratings(all_pairs);alpha=float(np.median(selected));all_answers=np.arange(2*n)
    held_hashes={answer_hash(t) for t in np.r_[dev.response_a,dev.response_b,confirm.response_a,confirm.response_b]}
    fit=all_answers[[h not in held_hashes for h in hashes[:2*n]]]
    common={'min_df':2,'sublinear_tf':True,'dtype':np.float32}
    parts=[TfidfVectorizer(max_features=12000,ngram_range=(1,2),**common),TfidfVectorizer(analyzer='char_wb',max_features=24000,ngram_range=(3,5),**common)]
    x=hstack([v.fit_transform(packed[fit].tolist()) for v in parts],format='csr',dtype=np.float32)
    z=hstack([v.transform(packed[2*n:].tolist()) for v in parts],format='csr',dtype=np.float32)
    reg=regress(x,np.array([full_r[name] for name in source_names[fit]]),alpha)
    direct_dev=reg.predict(z);joblib.dump({'parts':parts,'model':reg},out/'regressor.joblib',compress=3)
    guard();del x,z;gc.collect()
    source=np.load(inv['input_artifacts']['source_predictions']['path']);rclass=class_ratings(full_r,all_pairs)
    np.testing.assert_array_equal(source['dev_ids'],dev.id)
    np.testing.assert_array_equal(arrays['ids'],dev.id)
    qa,qb=source['dev_a'],source['dev_b'];ra,rb=qa@rclass,qb@rclass
    def features(a,b):return (a-b)[:,None],np.column_stack([np.abs(a-b),(a+b)/2])
    p=temperature(arrays['probabilities'],.9441963863130951);y=dev[CLASSES].to_numpy().argmax(1);dg=prompt_groups(dev).to_numpy(dtype=str)
    meta=dict(oldcfg,seed=cfg['seed']);design={'calibration':(np.empty((nd,0)),np.empty((nd,0))),'style':style_features(dev.response_a,dev.response_b),
         'direct_strength':features(direct_dev[:nd],direct_dev[nd:]),'posterior_three':features(ra,rb),'old_compact':source_features(qa,qb,rclass,'strength')}
    predictions,meta_details=evaluate(p,y,dg,arrays['outer_fold'],design,meta,['direct_strength'])
    bootstrap=dict(cfg,seed=cfg['bootstrap_seed']);contrasts={name:paired_group_gain(y,predictions[name],predictions['direct_strength'],dg,bootstrap)
        for name in ['calibration','style','posterior_three','old_compact']}
    mse_gain=paired_group_difference((identity-target)**2-(direct-target)**2,np.tile(groups,2),bootstrap)
    gate=bool(mse_gain['gain']>0 and mse_gain['interval_95'][0]>0 and contrasts['calibration']['gain']>=.001
              and all(contrasts[k]['interval_95'][0]>0 for k in ['calibration','style','posterior_three']))
    from arena.span_heads import metrics
    result={'task':'H47','gate':{'passed':gate},'source_mse':{'direct':float(np.mean((direct-target)**2)),
        'identity':float(np.mean((identity-target)**2)),'gain':mse_gain},'preference_metrics':{k:metrics(y,v) for k,v in predictions.items()},
        'contrasts':contrasts,'source_fold_details':details,'final_alpha':alpha,'final_fit_answers':len(fit),
        'cpu_seconds':time.monotonic()-started,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        'confirmation_targets_read':False,'new_gpu_seconds':0,'source_sha256':sha256(Path(__file__))}
    np.savez(out/'predictions.npz',ids=dev.id,labels=y,groups=dg,direct_oof=direct,identity_oof=identity,rating_target=target,
             direct_dev=direct_dev,**predictions)
    atomic_json(out/'meta_folds.json',meta_details)
    for folder in [out,pub]:atomic_json(folder/'H47_results.json',result)
    journal.emit('strength','complete',complete=True,passed=gate,seconds=result['cpu_seconds']);signal.alarm(0)


if __name__=='__main__':main()
