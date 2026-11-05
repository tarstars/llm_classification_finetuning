"""Execute the preregistered CPU-only model-strength and oracle residual pilot."""
from pathlib import Path
import hashlib,json,os,resource,signal,sys,time
from datetime import datetime,timezone
import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,CLASSES,json_write,prompt_groups
from arena.span_heads import metrics,temperature
from arena.source_models import Strength,source_features,style_features
from arena.source_evaluation import evaluate,summarize


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    start=time.monotonic();out=ROOT/'runs/span_005';pub=REPO/'results/span_005'
    assert not (out/'E06/summary.json').exists(),'Completed E06 is immutable'
    cfg=json.loads((REPO/'configs/span_005.json').read_text());manifest=json.loads((pub/'artifact_inventory.json').read_text())
    recovery=json.loads((pub/'E06_encoding_recovery.json').read_text()) if (pub/'E06_encoding_recovery.json').exists() else None
    for name,sha in manifest['source_sha256'].items():
        expected=recovery['revised_runner_sha256'] if recovery and name=='scripts/run_span_005_e06.py' else sha
        assert digest(REPO/name)==expected,name
    for name,item in manifest['input_artifacts'].items():assert digest(item['path'])==item['sha256'],name
    directory=out/'E06';directory.mkdir(exist_ok=True)
    if (directory/'started.json').exists():raise RuntimeError('E06 already started; inspect its actual outputs before any recovery')
    json_write(directory/'started.json',{'at':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'source_sha256':digest(Path(__file__))})
    def log(*args,**kwargs):
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
        assert rss<cfg['rss_guard_gib'],f'Resident memory guard {rss:.2f}GiB'
        print(*args,**kwargs)
    def expired(signum,frame):raise TimeoutError('Original E06 CPU elapsed cap reached')
    signal.signal(signal.SIGALRM,expired);signal.alarm(cfg['cpu_stage_seconds_caps']['E06']-(60 if recovery else 0))
    train=pd.read_parquet(out/'train.parquet');dev=pd.read_parquet(out/'dev.parquet');a=np.load(out/'inputs.npz')
    vocabulary=json.loads((out/'source_vocabulary.json').read_text());names=vocabulary['all_training_sources']
    # Recompute canonical hashes from frozen prompt text; never unpickle arrays.
    y=a['labels'];yt=a['train_labels'];groups=prompt_groups(dev).to_numpy(dtype=str);p=temperature(a['probabilities'],.9441963863130951)
    strength=Strength(cfg['strength_l2_sum'],cfg['strength_max_iterations']).fit(train.model_a,train.model_b,yt,names)
    np.savez(directory/'strength.npz',theta=strength.theta,row_scores=strength.row_scores,hessian=strength.hessian)
    _,ginv=np.unique(prompt_groups(train).to_numpy(dtype=str),return_inverse=True);sums=np.zeros((ginv.max()+1,len(names)+1));np.add.at(sums,ginv,strength.row_scores)
    bread=np.linalg.inv(strength.hessian);cov=bread@(sums.T@sums)@bread;se=np.sqrt(np.maximum(0,np.diag(cov)))
    exposure=pd.concat([train.model_a,train.model_b]).value_counts()
    wins=pd.concat([train.loc[yt==0,'model_a'],train.loc[yt==1,'model_b']]).value_counts()
    ties=pd.concat([train.loc[yt==2,'model_a'],train.loc[yt==2,'model_b']]).value_counts()
    rankings=[{'model':name,'answers':int(exposure[name]),'wins':int(wins.get(name,0)),'ties':int(ties.get(name,0)),
               'raw_win_rate':float(wins.get(name,0)/exposure[name]),'rating':float(strength.theta[i]),
               'conditional_cluster_normal_interval_95':[float(strength.theta[i]-1.96*se[i]),float(strength.theta[i]+1.96*se[i])]} for i,name in enumerate(names)]
    rankings.sort(key=lambda r:r['rating'],reverse=True)
    onehot=np.eye(len(names));lookup={n:i for i,n in enumerate(names)}
    qa=np.array([onehot[lookup[n]] if n in lookup else np.full(len(names),1/len(names)) for n in dev.model_a])
    qb=np.array([onehot[lookup[n]] if n in lookup else np.full(len(names),1/len(names)) for n in dev.model_b])
    features={'calibration':(np.empty((len(y),0)),np.empty((len(y),0))),
              'style':style_features(dev.response_a,dev.response_b),
              'oracle_strength':source_features(qa,qb,strength.theta[:-1],'strength'),
              'oracle_full':source_features(qa,qb,strength.theta[:-1],'full')}
    predictions,details=evaluate(p,y,groups,a['outer_fold'],features,cfg,['oracle_strength','oracle_full'],log)
    predictions['frozen_classifier']=p
    prior=np.bincount(yt,minlength=3)/len(yt);predictions['training_prior']=np.tile(prior,(len(y),1))
    predictions['uniform']=np.full((len(y),3),1/3);predictions['majority_clipped_1e-12']=np.tile(np.eye(3)[prior.argmax()],(len(y),1))
    predictions['strength_only']=strength.predict(dev.model_a,dev.model_b)
    paircounts={}
    for left,right,label in zip(train.model_a,train.model_b,yt):
        pair=tuple(sorted([left,right]));key=int(label) if left<=right or label==2 else 1-int(label)
        paircounts.setdefault(pair,np.zeros(3))[key]+=1
    pair=[];symmetric_prior=np.array([(prior[0]+prior[1])/2]*2+[prior[2]])
    for left,right in zip(dev.model_a,dev.model_b):
        counts=paircounts.get(tuple(sorted([left,right])),np.zeros(3));row=(counts+30*symmetric_prior)/(counts.sum()+30)
        if left==right:row[:2]=row[:2].mean()
        pair.append(row if left<=right else row[[1,0,2]])
    predictions['pair_frequency']=np.asarray(pair)
    result=summarize(y,predictions,groups,cfg)
    gain=result['contrasts']['nested_selected_vs_calibration'];coverage=float(np.mean(dev.model_a.isin(names)&dev.model_b.isin(names)))
    gate=bool(gain['gain']>0 and gain['interval_95'][1]>=cfg['practical_gain'] and coverage>=.95)
    np.savez(directory/'predictions.npz',ids=a['ids'],labels=y,groups=groups,outer_fold=a['outer_fold'],**predictions)
    json_write(directory/'folds.json',details)
    result.update(task='E06',rows=len(y),groups=len(set(groups)),source_models=len(names),train_rows=len(train),
        source_known_pair_coverage=coverage,rankings=rankings,
        ranking_uncertainty='Conditional prompt-cluster sandwich normal intervals at the penalized fit; not refitted bootstrap intervals, intrinsic skill, or correction for shrinkage/selection bias.',
        gate={'passed':gate,'rule':'Nested-oracle gain >0,95% upper bound >=0.001,and >=95% known-pair coverage; an information gate, not adoption.',
              'evidence':gain},strength_fit=strength.details,cpu_elapsed_seconds=time.monotonic()-start,
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,official_score_unchanged=.95481,
        confirmation_used=False,gpu_seconds=0,oracle_is_diagnostic_only=True,
        prior=prior.tolist(),majority_loss_note='Exact one-hot loss infinite on errors; finite table clips at1e-12.',
        prediction_sha256=digest(directory/'predictions.npz'),folds_sha256=digest(directory/'folds.json'),
        source_sha256=digest(Path(__file__)),config_sha256=digest(REPO/'configs/span_005.json'))
    result['failed_attempt_cpu_seconds_upper_bound']=60 if recovery else 0
    result['cpu_accounted_seconds_upper_bound']=result['cpu_elapsed_seconds']+result['failed_attempt_cpu_seconds_upper_bound']
    for folder in [directory,pub]:json_write(folder/'E06_results.json',result)
    json_write(directory/'summary.json',{'complete':True,'gate_passed':gate,'cpu_elapsed_seconds':result['cpu_elapsed_seconds']})
    signal.alarm(0)
    log(json.dumps({'task':'E06','gate':result['gate'],'losses':{k:v['log_loss'] for k,v in result['metrics'].items()},'seconds':result['cpu_elapsed_seconds']},indent=2),flush=True)


if __name__=='__main__':main()
