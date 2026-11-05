"""Verify actual v5 inputs, fold probabilities, source purges, metrics and gates."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys,time
import numpy as np
import pandas as pd
from scipy.special import logsumexp
import joblib
from scipy.sparse import hstack

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,CLASSES,prompt_groups,json_write
from arena.source_models import answer_hash,source_features,style_features


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def loss(y,p):
    p=np.clip(np.asarray(p,float),1e-12,1);p/=p.sum(1,keepdims=True)
    return -np.log(p[np.arange(len(y)),y])


def interval(values,groups,cfg):
    _,inverse=np.unique(groups,return_inverse=True);tot=np.bincount(inverse,weights=values);counts=np.bincount(inverse)
    draws=np.random.default_rng(cfg['seed']).integers(0,len(counts),size=(cfg['bootstrap_group_replicates'],len(counts)))
    return np.quantile(tot[draws].sum(1)/counts[draws].sum(1),[.025,.975])


def reconstruct_head(p,d,s,train,val,theta):
    # Independent reconstruction of every outer prediction, including fold-local
    # scaling. No fit and no validation label is needed.
    dscale=np.maximum(np.sqrt((d[train]**2).mean(0)),.05)
    mean=s[train].mean(0);scale=np.maximum(s[train].std(0),.05)
    lp=np.log(p[val]);direction=(lp[:,0]-lp[:,1])/2;tie=lp[:,2]-(lp[:,0]+lp[:,1])/2
    k=d.shape[1];theta=np.asarray(theta)
    z=theta[0]*direction+(d[val]/dscale)@theta[3:3+k]
    t=theta[1]*tie+theta[2]+((s[val]-mean)/scale)@theta[3+k:]
    logits=np.column_stack([z,-z,t]);return np.exp(logits-logsumexp(logits,axis=1,keepdims=True))


def main():
    start=time.monotonic();out=ROOT/'runs/span_005';pub=REPO/'results/span_005';cfg=read(REPO/'configs/span_005.json');manifest=read(pub/'artifact_inventory.json')
    assert datetime.now(timezone.utc)<datetime.fromisoformat(cfg['deadline'])
    for name,entry in manifest['input_artifacts'].items():assert sha(entry['path'])==entry['sha256'],name
    for name,digest in manifest['old_snapshot_sha256'].items():assert sha(REPO/name)==digest,name
    for name,digest in manifest['user_notebook_sha256'].items():assert sha(REPO/name)==digest,name
    recovery=read(pub/'E06_encoding_recovery.json')
    for name,digest in manifest['source_sha256'].items():
        expected=recovery['revised_runner_sha256'] if name=='scripts/run_span_005_e06.py' else digest
        assert sha(REPO/name)==expected,name
    assert recovery['fitted_models_before_failure']==0 and (out/'E06_failed_encoding/started.json').exists()
    train=pd.read_parquet(out/'train.parquet');dev=pd.read_parquet(out/'dev.parquet');inputs=np.load(out/'inputs.npz')
    y=dev[CLASSES].values.argmax(1);yt=train[CLASSES].values.argmax(1);groups=prompt_groups(dev).to_numpy(dtype=str)
    np.testing.assert_array_equal(y,inputs['labels']);np.testing.assert_array_equal(yt,inputs['train_labels'])
    assert not set(prompt_groups(train))&set(groups)
    lp=np.log(inputs['probabilities'])/.9441963863130951;p=np.exp(lp-logsumexp(lp,axis=1,keepdims=True))
    vocab=read(out/'source_vocabulary.json');allnames=vocab['all_training_sources'];lookup={n:i for i,n in enumerate(allnames)}
    theta=np.load(out/'E06/strength.npz')['theta'];oa=np.eye(len(allnames))[[lookup[n] for n in dev.model_a]];ob=np.eye(len(allnames))[[lookup[n] for n in dev.model_b]]
    features={'calibration':(np.empty((len(y),0)),np.empty((len(y),0))),'style':style_features(dev.response_a,dev.response_b),
              'oracle_strength':source_features(oa,ob,theta[:-1],'strength'),'oracle_full':source_features(oa,ob,theta[:-1],'full')}
    proof={'span':5,'passed':True,'original_start':cfg['started_at'],'original_deadline':cfg['deadline'],'confirmation_used':False,'gpu_seconds':0,
           'snapshot_files_preserved':len(manifest['old_snapshot_sha256']),'input_artifact_hashes_verified':True,'tasks':{}}
    for task in ['E06','H45']:
        result=read(pub/(task+'_results.json'));a=np.load(out/task/'predictions.npz');detail=read(out/task/'folds.json')
        assert result['prediction_sha256']==sha(out/task/'predictions.npz') and result['folds_sha256']==sha(out/task/'folds.json')
        np.testing.assert_array_equal(a['ids'],dev.id);np.testing.assert_array_equal(a['labels'],y);np.testing.assert_array_equal(a['groups'],groups)
        if task=='H45':
            source=np.load(out/'E07/tfidf_predictions.npz');classes=vocab['source_classes'];ratings=dict(zip(allnames,theta[:-1]));rare=[n for n in allnames if n not in classes]
            other=np.average([ratings[n] for n in rare],weights=[vocab['source_counts'][n] for n in rare]) if rare else 0.
            r=np.array([ratings.get(n,other) for n in classes]);qa,qb=source['dev_a'],source['dev_b'];perm=a['shuffled_source_permutation']
            features={'calibration':features['calibration'],'style':features['style'],
                      'source_strength':source_features(qa,qb,r,'strength'),'source_full':source_features(qa,qb,r,'full'),
                      'shuffled_source':source_features(qa[perm],qb[perm],r,'strength')}
        reconstructed=0
        for record in detail:
            fit=np.flatnonzero(inputs['outer_fold']!=record['fold']);val=np.flatnonzero(inputs['outer_fold']==record['fold'])
            assert not set(groups[fit])&set(groups[val])
            for family,values in record['families'].items():
                d,s=features[family];prediction=reconstruct_head(p,d,s,fit,val,values['theta'])
                np.testing.assert_allclose(prediction,a[family][val],atol=1e-10,rtol=1e-10);reconstructed+=len(val)
                for option in values['inner_scores']:assert option['family']==family
            chosen=record['selection']['family'];assert chosen==min(record['selection']['inner_choices'],key=lambda k:record['selection']['inner_choices'][k]['inner_loss'])
            np.testing.assert_array_equal(a['nested_selected'][val],a[chosen][val])
        for name,m in result['metrics'].items():
            probability=a[name];assert probability.shape==(len(y),3) and np.isfinite(probability).all()
            # Hard majority is intentionally clipped in its metric, not its saved array.
            np.testing.assert_allclose(loss(y,probability).mean(),m['log_loss'],atol=1e-12)
            np.testing.assert_allclose((probability.argmax(1)==y).mean(),m['accuracy'],atol=1e-12)
        for name,value in result['contrasts'].items():
            if name=='selected_vs_style':first,last='style','nested_selected'
            else:last,first=name.split('_vs_')
            delta=loss(y,a[first])-loss(y,a[last]);np.testing.assert_allclose(delta.mean(),value['gain'],atol=1e-12)
            np.testing.assert_allclose(interval(delta,groups,cfg),value['interval_95'],atol=1e-12)
        proof['tasks'][task]={'outer_predictions_reconstructed':reconstructed,'metric_rows':len(y),'contrasts_recomputed':len(result['contrasts']),
                             'cpu_seconds':result['cpu_elapsed_seconds'],'source_sha256':result['source_sha256'],'gate':result['gate']}
    # Audit all source OOF coverage and duplicate purges independently.
    source_result=read(pub/'E07_results.json');classes=vocab['source_classes'];clook={n:i for i,n in enumerate(classes)}
    texts=np.r_[train.response_a,train.response_b,dev.response_a,dev.response_b];hashes=np.array([answer_hash(t) for t in texts]);n=2*len(train)
    target=np.array([clook.get(n,clook['__OTHER__']) for n in np.r_[train.model_a,train.model_b]])
    folds=np.tile(inputs['source_fold'],2);sg=np.tile(prompt_groups(train).to_numpy(dtype=str),2)
    for family,stats in source_result['families'].items():
        saved=np.load(out/'E07'/(family+'_predictions.npz'));np.testing.assert_array_equal(saved['source_labels'],target)
        assert sha(out/'E07'/(family+'_predictions.npz'))==stats['prediction_sha256']
        assert sha(out/'E07'/(family+'_bundle.joblib'))==stats['bundle_sha256']
        for fold in range(3):
            trace=np.load(out/'E07'/(family+f'_fold{fold}.npz'));fit=trace['fit_indices'];val=trace['evaluation_indices']
            np.testing.assert_array_equal(val,np.flatnonzero(folds==fold))
            held=set(hashes[val]);expected=np.array([i for i in np.flatnonzero(folds!=fold) if hashes[i] not in held])
            np.testing.assert_array_equal(fit,expected);assert not set(sg[fit])&set(sg[val])
            np.testing.assert_array_equal(trace['probabilities'],saved['oof'][val])
            counts=np.bincount(target[fit],minlength=len(classes));prior=(counts+1)/(len(fit)+len(classes))
            np.testing.assert_allclose(saved['oof_prior'][val],np.tile(prior,(len(val),1)),atol=1e-7)
        value=-np.log(np.maximum(saved['oof'][np.arange(n),target],1e-12))
        np.testing.assert_allclose(value.mean(),stats['oof']['log_loss'],rtol=1e-6)
        proof.setdefault('source_families',{})[family]={'oof_answers':n,'duplicate_purges_verified':True,'training_source_loss':float(value.mean())}
    # Local offline replay of the actual serialized portable feature/head bundle.
    bundle=joblib.load(out/'E07/tfidf_bundle.joblib');cap=bundle['max_chars']
    packed=[t if len(t)<=cap else t[:cap//2]+'\n[TRUNCATED]\n'+t[-cap//2:] for t in texts[n:]]
    x=hstack([v.transform(packed) for v in bundle['parts']],format='csr',dtype=np.float32)
    q=np.full((len(packed),len(classes)),1e-12);q[:,bundle['model'].classes_]=bundle['model'].predict_proba(x);q[x.getnnz(axis=1)==0]=bundle['prior'];q/=q.sum(1,keepdims=True)
    saved=np.load(out/'E07/tfidf_predictions.npz');np.testing.assert_allclose(q,np.vstack([saved['dev_a'],saved['dev_b']]),atol=1e-6)
    proof['portable_bundle_replay_rows']=len(packed);proof['source_prediction_gate']=source_result['gate']
    proof['cpu_accounted_upper_seconds']=read(pub/'E06_results.json')['cpu_accounted_seconds_upper_bound']+source_result['cpu_elapsed_seconds']+read(pub/'H45_results.json')['cpu_elapsed_seconds']
    assert proof['cpu_accounted_upper_seconds']<27000
    proof['checked_at']=datetime.now(timezone.utc).isoformat();proof['audit_seconds']=time.monotonic()-start;proof['source_sha256']=sha(Path(__file__))
    proof['artifact_sha256']={str(p.relative_to(REPO)):sha(p) for p in pub.glob('*results.json')}
    for folder in [pub,out]:json_write(folder/'scientific_audit.json',proof)
    print(json.dumps(proof,indent=2))


if __name__=='__main__':main()
