"""Refit fixed sparse source head and freeze all meta weights before target reveal."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,resource,signal,sys,time
import joblib
import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'));sys.path.insert(0,str(REPO/'scripts'))
from arena.data import ROOT,CLASSES,TEXT_COLUMNS,prompt_groups,json_write
from arena.source_models import answer_hash,source_features,style_features
from arena.span_heads import temperature
from arena.span006_source import freeze_head,save_head
from run_span_005_e07 import fit_head,packed


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    start=time.monotonic();out=ROOT/'runs/span_006';pub=REPO/'results/span_006';cfg=json.loads((REPO/'configs/span_006.json').read_text());manifest=json.loads((pub/'artifact_inventory.json').read_text())
    directory=out/'freeze';directory.mkdir(exist_ok=True);assert not (directory/'started.json').exists(),'Freeze already started; inspect actual state before recovery'
    for name,digest in manifest['source_sha256'].items():assert sha(REPO/name)==digest,name
    for key,item in manifest['input_artifacts'].items():assert sha(item['path'])==item['sha256'],key
    json_write(directory/'started.json',{'at':datetime.now(timezone.utc).isoformat(),'pid':os.getpid()})
    def expired(*unused):raise TimeoutError('Original CPU freeze cap reached')
    signal.signal(signal.SIGALRM,expired);signal.alarm(cfg['cpu_stage_seconds_caps']['freeze'])
    entries=manifest['input_artifacts'];train=pd.read_parquet(entries['train.parquet']['path']);dev=pd.read_parquet(entries['dev.parquet']['path']);confirm=pd.read_parquet(entries['confirmation_inputs']['path'])
    assert set(confirm.columns)=={'id',*TEXT_COLUMNS},'No confirmation labels/names permitted before freezing'
    vocab=json.loads(Path(entries['source_vocabulary.json']['path']).read_text());classes=vocab['source_classes'];lookup={n:i for i,n in enumerate(classes)}
    texts=np.r_[train.response_a,train.response_b,dev.response_a,dev.response_b,confirm.response_a,confirm.response_b]
    n=2*len(train);nd=2*len(dev);nc=2*len(confirm);hashes=np.array([answer_hash(t) for t in texts]);held=set(hashes[n:])
    fit=np.flatnonzero([h not in held for h in hashes[:n]]);eval_idx=np.arange(n,n+nd+nc)
    assert not set(hashes[fit])&held
    labels=np.full(len(texts),-1,int);labels[:n]=[lookup.get(s,lookup['__OTHER__']) for s in np.r_[train.model_a,train.model_b]]
    source_text=np.array([packed(t,cfg['source_max_chars']) for t in texts],dtype=object)
    probabilities,bundle,source_details=fit_head('tfidf',fit,eval_idx,source_text,np.empty((len(texts),0)),labels,len(classes),cfg)
    assert not source_details['warnings'] and source_details['rss_gib']<cfg['rss_guard_gib']
    joblib.dump(bundle,directory/'source_bundle.joblib',compress=3)
    qd_a,qd_b=probabilities[:len(dev)],probabilities[len(dev):nd]
    qc_a,qc_b=probabilities[nd:nd+len(confirm)],probabilities[nd+len(confirm):]
    np.savez(directory/'source_predictions.npz',dev_ids=dev.id.to_numpy(),confirmation_ids=confirm.id.to_numpy(),dev_a=qd_a,dev_b=qd_b,confirmation_a=qc_a,confirmation_b=qc_b)
    np.savez(directory/'source_fit_indices.npz',fit_indices=fit,evaluation_indices=eval_idx)
    strengths=np.load(entries['strength']['path'])['theta'];name_r=dict(zip(vocab['all_training_sources'],strengths[:-1]));rare=[name for name in name_r if name not in classes]
    other=np.average([name_r[x] for x in rare],weights=[vocab['source_counts'][x] for x in rare]) if rare else 0.
    ratings=np.array([name_r.get(name,other) for name in classes])
    np.savez(directory/'ratings.npz',ratings=ratings,tie_intercept=strengths[-1])
    cache=np.load(entries['classifier']['path']);idx={int(row):i for i,row in enumerate(cache['ids'])};order=[idx[int(row)] for row in dev.id]
    p=temperature((cache['original'][order]+cache['aligned_swap'][order])/2,cfg['temperature']);y=dev[CLASSES].values.argmax(1);groups=prompt_groups(dev).to_numpy(dtype=str)
    permutation=np.random.default_rng(cfg['bootstrap_seed']).permutation(len(dev))
    designs={'calibration':(np.empty((len(dev),0)),np.empty((len(dev),0))),'style':style_features(dev.response_a,dev.response_b),
             'strength':source_features(qd_a,qd_b,ratings,'strength'),'shuffled':source_features(qd_a[permutation],qd_b[permutation],ratings,'strength')}
    details={}
    for name,(d,s) in designs.items():
        model,details[name]=freeze_head(p,d,s,y,groups,cfg);save_head(directory/(name+'_head.npz'),model)
        print(json.dumps({'frozen_family':name,**details[name]}),flush=True)
    json_write(directory/'head_selection.json',details)
    files=['source_bundle.joblib','source_predictions.npz','source_fit_indices.npz','ratings.npz','head_selection.json',*[name+'_head.npz' for name in designs]]
    result={'span':6,'frozen_at':datetime.now(timezone.utc).isoformat(),'family':'compact_inferred_strength','source_family':'tfidf',
            'confirmation_votes_read':False,'confirmation_source_names_read':False,'development_rows':len(dev),'confirmation_rows':len(confirm),
            'source_fit_answers':len(fit),'purged_against_dev_and_confirmation':n-len(fit),'source_details':source_details,
            'head_selection':details,'recipe_artifacts':{name:{'path':str(directory/name),'sha256':sha(directory/name)} for name in files},
            'input_manifest_sha256':sha(pub/'artifact_inventory.json'),'config_sha256':sha(REPO/'configs/span_006.json'),
            'source_sha256':sha(Path(__file__)),'cpu_elapsed_seconds':time.monotonic()-start,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2}
    for folder in [directory,pub]:json_write(folder/'frozen_recipe.json',result)
    json_write(directory/'summary.json',{'complete':True,'cpu_elapsed_seconds':result['cpu_elapsed_seconds']});signal.alarm(0)
    print(json.dumps({k:result[k] for k in ['frozen_at','source_fit_answers','purged_against_dev_and_confirmation','cpu_elapsed_seconds','peak_rss_gib','confirmation_votes_read']},indent=2),flush=True)


if __name__=='__main__':main()
