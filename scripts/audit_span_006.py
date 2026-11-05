"""Independent post-reveal verification of frozen transfer, input packing and metrics."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys,time
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import hstack
from scipy.special import logsumexp
from transformers import AutoTokenizer
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,CLASSES,prompt_groups,json_write
from arena.joint_data import JointTokenizer
from arena.span006_inputs import text_digest,token_digest
from arena.source_models import answer_hash,source_features,style_features


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def losses(y,p):
    p=np.clip(p,1e-12,1);p=p/p.sum(1,keepdims=True);return -np.log(p[np.arange(len(y)),y])


def main():
    started=time.monotonic();out=ROOT/'runs/span_006';pub=REPO/'results/span_006';cfg=read(REPO/'configs/span_006.json');inputs=read(pub/'artifact_inventory.json');recipe=read(pub/'frozen_recipe.json')
    result=read(pub/'H46_results.json');reveal=read(pub/'reveal_intent.json');job=out/'jobs/confirmation'
    assert datetime.fromisoformat(recipe['frozen_at'])<datetime.fromisoformat(read(job/'push_intent.json')['at'])<datetime.fromisoformat(reveal['at'])
    assert not recipe['confirmation_votes_read'] and reveal['model_fit_operations']==0 and reveal['all_predictions_computed_before_votes']
    assert sha(pub/'frozen_recipe.json')==reveal['frozen_recipe_sha256']==result['frozen_recipe_sha256']
    for name,artifact in recipe['recipe_artifacts'].items():assert sha(artifact['path'])==artifact['sha256'],name
    for name,artifact in inputs['input_artifacts'].items():assert sha(artifact['path'])==artifact['sha256'],name
    for p,digest in inputs['source_sha256'].items():assert sha(REPO/p)==digest,p
    for p,digest in inputs['old_snapshot_sha256'].items():assert sha(REPO/p)==digest,p
    assert sha(REPO/'scripts/evaluate_span_006.py')==result['source_sha256']
    confirm=pd.read_parquet(inputs['input_artifacts']['confirmation_inputs']['path']);dev=pd.read_parquet(inputs['input_artifacts']['dev.parquet']['path']);train=pd.read_parquet(inputs['input_artifacts']['train.parquet']['path'])
    assert set(confirm.columns)=={'id','prompt','response_a','response_b'} and text_digest(confirm)==inputs['confirmation_text_digest']
    groups=prompt_groups(confirm).to_numpy(dtype=str);assert not set(groups)&(set(prompt_groups(train))|set(prompt_groups(dev)))
    held={answer_hash(t) for t in np.r_[dev.response_a,dev.response_b,confirm.response_a,confirm.response_b]}
    trainhash=np.array([answer_hash(t) for t in np.r_[train.response_a,train.response_b]])
    expected=np.flatnonzero([h not in held for h in trainhash]);fit=np.load(out/'freeze/source_fit_indices.npz')
    np.testing.assert_array_equal(fit['fit_indices'],expected);assert len(expected)==recipe['source_fit_answers']
    bundle=joblib.load(recipe['recipe_artifacts']['source_bundle.joblib']['path']);source=np.load(out/'freeze/source_predictions.npz')
    cap=bundle['max_chars'];texts=np.r_[dev.response_a,dev.response_b,confirm.response_a,confirm.response_b]
    packed=[t if len(t)<=cap else t[:cap//2]+'\n[TRUNCATED]\n'+t[-cap//2:] for t in texts]
    x=hstack([v.transform(packed) for v in bundle['parts']],format='csr',dtype=np.float32)
    q=np.full((len(packed),bundle['classes']),1e-12);q[:,bundle['model'].classes_]=bundle['model'].predict_proba(x);q[x.getnnz(axis=1)==0]=bundle['prior'];q/=q.sum(1,keepdims=True)
    np.testing.assert_allclose(q,np.vstack([source['dev_a'],source['dev_b'],source['confirmation_a'],source['confirmation_b']]),atol=1e-6)
    raw=np.load(job/'output/confirmation/confirmation_predictions.npz');summary=read(job/'output/confirmation/confirmation_summary.json');smoke=read(job/'output/confirmation/smoke_summary.json')
    assert sha(job/'output/confirmation/confirmation_predictions.npz')==summary['prediction_sha256']==reveal['raw_prediction_sha256']
    np.testing.assert_array_equal(raw['ids'],confirm.id)
    ids=[];first=[];second=[]
    for c in summary['chunks']:
        file=job/'output/confirmation'/f"chunk_{c['start']:05d}.npz";assert sha(file)==c['sha256'];a=np.load(file)
        ids.extend(a['ids']);first.append(a['original']);second.append(a['aligned_swap'])
    np.testing.assert_array_equal(ids,raw['ids']);np.testing.assert_array_equal(np.vstack(first),raw['original']);np.testing.assert_array_equal(np.vstack(second),raw['aligned_swap'])
    tokenizer=AutoTokenizer.from_pretrained(str(Path(inputs['input_artifacts']['adapter']['path']).parent),local_files_only=True,padding_side='right')
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    original,swapped,_=JointTokenizer(tokenizer,1024).encode_frame(confirm)
    assert token_digest(original)==summary['original_tokens_sha256'] and token_digest(swapped)==summary['swapped_tokens_sha256']
    ref=np.load(out/'smoke_reference.npz');observed=np.load(job/'output/confirmation/smoke_predictions.npz')
    np.testing.assert_array_equal(ref['ids'],observed['ids']);delta=np.abs((ref['original']+ref['aligned_swap']-observed['original']-observed['aligned_swap'])/2)
    np.testing.assert_allclose([delta.mean(),delta.max()],[smoke['mean_raw_average_difference'],smoke['max_raw_average_difference']],atol=1e-12)
    assert delta.mean()<=.002 and delta.max()<=.03
    hardware=[read(job/'output/confirmation'/f'gpu_rank{k}.json') for k in [0,1]]
    assert {r['device'] for r in hardware}=={0,1} and all('T4' in r['name'] and r['peak_allocated_bytes']>0 for r in hardware)
    saved=np.load(out/'H46/predictions.npz');assert sha(out/'H46/predictions.npz')==result['prediction_sha256']
    labels=pd.read_csv(inputs['input_artifacts']['raw_train']['path'],usecols=['id',*CLASSES]).set_index('id').loc[confirm.id,CLASSES].to_numpy().argmax(1)
    np.testing.assert_array_equal(saved['labels'],labels);np.testing.assert_array_equal(saved['ids'],confirm.id);np.testing.assert_array_equal(saved['groups'],groups)
    rawp=(raw['original']+raw['aligned_swap'])/2;lp=np.log(np.clip(rawp,1e-12,1))/.9441963863130951;p=np.exp(lp-logsumexp(lp,axis=1,keepdims=True))
    np.testing.assert_allclose(p,saved['frozen_classifier'],atol=1e-12)
    source=np.load(out/'freeze/source_predictions.npz');qa,qb=source['confirmation_a'],source['confirmation_b'];r=np.load(out/'freeze/ratings.npz')['ratings'];perm=saved['shuffled_permutation']
    np.testing.assert_array_equal(perm,np.random.default_rng(cfg['bootstrap_seed']).permutation(len(confirm)))
    features={'calibration':(np.empty((len(confirm),0)),np.empty((len(confirm),0))),'style':style_features(confirm.response_a,confirm.response_b),
              'strength':source_features(qa,qb,r,'strength'),'shuffled':source_features(qa[perm],qb[perm],r,'strength')}
    for name,(d,s) in features.items():
        head=np.load(out/'freeze'/(name+'_head.npz'));theta=head['theta'];k=d.shape[1];logp=np.log(p)
        z=theta[0]*(logp[:,0]-logp[:,1])/2+(d/head['dscale'])@theta[3:3+k]
        t=theta[1]*(logp[:,2]-(logp[:,0]+logp[:,1])/2)+theta[2]+((s-head['smean'])/head['sscale'])@theta[3+k:]
        logits=np.column_stack([z,-z,t]);prediction=np.exp(logits-logsumexp(logits,axis=1,keepdims=True))
        np.testing.assert_allclose(prediction,saved[name],atol=1e-10)
    for name,metric in result['metrics'].items():
        probability=saved[name];assert probability.shape==(5746,3) and np.isfinite(probability).all() and np.allclose(probability.sum(1),1,atol=1e-6)
        rowloss=losses(labels,probability);np.testing.assert_allclose(rowloss.mean(),metric['log_loss'],atol=1e-12)
        guess=probability.argmax(1);matrix=np.zeros((3,3),int);np.add.at(matrix,(labels,guess),1)
        np.testing.assert_array_equal(matrix,metric['confusion_matrix']);np.testing.assert_allclose((guess==labels).mean(),metric['accuracy'],atol=1e-12)
        np.testing.assert_allclose([rowloss[labels==k].mean() for k in range(3)],metric['class_loss'],atol=1e-12)
    _,inverse=np.unique(groups,return_inverse=True);counts=np.bincount(inverse);draws=np.random.default_rng(cfg['bootstrap_seed']).integers(0,len(counts),size=(2000,len(counts)))
    for key,value in result['contrasts'].items():
        comparator=key.removeprefix('strength_vs_');diff=losses(labels,saved[comparator])-losses(labels,saved['strength']);sums=np.bincount(inverse,weights=diff)
        np.testing.assert_allclose(diff.mean(),value['gain'],atol=1e-12);np.testing.assert_allclose(np.quantile(sums[draws].sum(1)/counts[draws].sum(1),[.025,.975]),value['interval_95'],atol=1e-12)
    gain=result['contrasts']['strength_vs_calibration'];style=result['contrasts']['strength_vs_style']
    assert result['gate']['passed']==bool(gain['gain']>=.001 and gain['interval_95'][0]>0 and style['gain']>0 and style['interval_95'][0]>0 and result['projected_joint_seconds']<=25200)
    progress={'passed':True,'checked_at':datetime.now(timezone.utc).isoformat(),'rows':5746,'groups':len(counts),'chunks_verified':len(summary['chunks']),
              'token_packing_recomputed':True,'head_predictions_reconstructed':4*5746,'contrasts_recomputed':len(result['contrasts']),
              'frozen_before_vote_access':True,'source_union_purge_verified':True,'old_snapshot_files_preserved':len(inputs['old_snapshot_sha256']),
              'serialized_source_replay_answers':len(packed),
              'hardware':hardware,'gate':result['gate'],'charged_gpu_seconds':result['charged_gpu_seconds'],'audit_seconds':time.monotonic()-started,
              'H46_results_sha256':sha(pub/'H46_results.json'),'frozen_recipe_sha256':sha(pub/'frozen_recipe.json'),'source_sha256':sha(Path(__file__))}
    for folder in [pub,out]:json_write(folder/'scientific_audit.json',progress)
    print(json.dumps(progress,indent=2))


if __name__=='__main__':main()
