"""One frozen confirmation evaluation; no fit operation or label-based selection."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,resource,signal,subprocess,sys,time
import numpy as np
import pandas as pd
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,CLASSES,prompt_groups,json_write
from arena.source_models import source_features,style_features,softmax
from arena.span006_source import load_head
from arena.span_heads import temperature,metrics,paired_group_gain,order_diagnostics


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    started=time.monotonic();out=ROOT/'runs/span_006';pub=REPO/'results/span_006';directory=out/'H46';directory.mkdir(exist_ok=True)
    assert not (pub/'H46_results.json').exists() and not (out/'reveal_intent.json').exists(),'Confirmation evaluation already began; preserve it rather than retuning/repeating'
    cfg=read(REPO/'configs/span_006.json');manifest=read(pub/'artifact_inventory.json');recipe=read(pub/'frozen_recipe.json');job=out/'jobs/confirmation'
    assert read(job/'controller_state.json')['stage']=='complete' and read(job/'status.json')['status']=='COMPLETE'
    for name,artifact in recipe['recipe_artifacts'].items():assert sha(artifact['path'])==artifact['sha256'],name
    assert sha(REPO/'configs/span_006.json')==recipe['config_sha256']
    committed=subprocess.check_output(['git','show','HEAD:results/span_006/frozen_recipe.json'],cwd=REPO)
    assert hashlib.sha256(committed).hexdigest()==sha(pub/'frozen_recipe.json')
    bundle=read(pub/'confirmation_bundle.json')
    assert bundle['source_sha256']['scripts/evaluate_span_006.py']==sha(Path(__file__))
    summary=read(job/'output/confirmation/confirmation_summary.json');smoke=read(job/'output/confirmation/smoke_summary.json')
    assert summary['complete'] and smoke['passed'] and summary['rows']==5746
    assert summary['frozen_recipe_sha256']==sha(pub/'frozen_recipe.json') and summary['confirmation_text_digest']==manifest['confirmation_text_digest']
    rawfile=job/'output/confirmation/confirmation_predictions.npz';assert sha(rawfile)==summary['prediction_sha256']
    confirm=pd.read_parquet(manifest['input_artifacts']['confirmation_inputs']['path']);raw=np.load(rawfile);source=np.load(out/'freeze/source_predictions.npz')
    np.testing.assert_array_equal(raw['ids'],confirm.id);np.testing.assert_array_equal(source['confirmation_ids'],confirm.id)
    p=temperature((raw['original']+raw['aligned_swap'])/2,cfg['temperature']);qa,qb=source['confirmation_a'],source['confirmation_b'];ratings=np.load(out/'freeze/ratings.npz')
    r=ratings['ratings'];permutation=np.random.default_rng(cfg['bootstrap_seed']).permutation(len(confirm))
    designs={'calibration':(np.empty((len(confirm),0)),np.empty((len(confirm),0))),'style':style_features(confirm.response_a,confirm.response_b),
             'strength':source_features(qa,qb,r,'strength'),'shuffled':source_features(qa[permutation],qb[permutation],r,'strength')}
    predictions={'frozen_classifier':p};swaps={}
    for name,(d,s) in designs.items():
        model=load_head(out/'freeze'/(name+'_head.npz'));predictions[name]=model.predict_proba(p,d,s)
        swap=model.predict_proba(p[:,[1,0,2]],-d,s)[:,[1,0,2]];swaps[name]=float(np.max(np.abs(predictions[name]-swap)))
        assert swaps[name]<1e-10
    delta=(r[:,None]-r[None,:])/2;kernels=softmax(np.column_stack([delta.ravel(),-delta.ravel(),np.full(delta.size,float(ratings['tie_intercept']))])).reshape(len(r),len(r),3)
    predictions['source_only']=np.column_stack([np.sum((qa@kernels[:,:,k])*qb,axis=1) for k in range(3)])
    predictions['source_only']/=predictions['source_only'].sum(1,keepdims=True)
    intent={'at':datetime.now(timezone.utc).isoformat(),'frozen_recipe_sha256':sha(pub/'frozen_recipe.json'),'raw_prediction_sha256':sha(rawfile),
            'evaluation_source_sha256':sha(Path(__file__)),'all_predictions_computed_before_votes':True,'model_fit_operations':0}
    for folder in [out,pub]:json_write(folder/'reveal_intent.json',intent)
    def expired(*unused):raise TimeoutError('Original evaluation CPU cap reached')
    signal.signal(signal.SIGALRM,expired);signal.alarm(cfg['cpu_stage_seconds_caps']['evaluate'])
    # First access to this span's confirmation winner/source-name columns.
    labels=pd.read_csv(manifest['input_artifacts']['raw_train']['path'],usecols=['id',*CLASSES,'model_a','model_b']).set_index('id').loc[confirm.id]
    assert labels[CLASSES].isin([0,1]).all().all() and labels[CLASSES].sum(1).eq(1).all()
    y=labels[CLASSES].to_numpy().argmax(1);groups=prompt_groups(confirm).to_numpy(dtype=str)
    train=pd.read_parquet(manifest['input_artifacts']['train.parquet']['path'],columns=CLASSES);prior=train[CLASSES].sum().to_numpy(float);prior/=prior.sum()
    predictions.update(training_prior=np.tile(prior,(len(confirm),1)),uniform=np.full((len(confirm),3),1/3),majority_clipped_1e_12=np.tile(np.eye(3)[prior.argmax()],(len(confirm),1)))
    bootstrap=dict(cfg,seed=cfg['bootstrap_seed']);contrasts={}
    for first,last in [('calibration','strength'),('style','strength'),('shuffled','strength'),('frozen_classifier','strength')]:
        contrasts['strength_vs_'+first]=paired_group_gain(y,predictions[first],predictions[last],groups,bootstrap)
    charged=read(job/'quota.json')['used_seconds']-read(job/'push_intent.json')['quota_before']['used_seconds']
    prior_profile=read(REPO/'results/span_004/repeatability_results.json')['classifier_profile_seconds_per_row']
    classifier_rate=max(prior_profile,summary['seconds_per_row']);source_rate=recipe['source_details']['score_seconds']/(2*recipe['development_rows']+2*recipe['confirmation_rows'])
    setup=max(0,charged-summary['prediction_seconds']);projection=25000*(classifier_rate+2*source_rate)*1.2+setup+600
    g=contrasts['strength_vs_calibration'];s=contrasts['strength_vs_style'];passed=bool(g['gain']>=.001 and g['interval_95'][0]>0 and s['gain']>0 and s['interval_95'][0]>0 and projection<=25200)
    vocab=read(manifest['input_artifacts']['source_vocabulary.json']['path']);classes=vocab['source_classes'];idx={name:i for i,name in enumerate(classes)}
    source_y=np.array([idx.get(name,idx['__OTHER__']) for name in np.r_[labels.model_a,labels.model_b]]);q=np.vstack([qa,qb])
    source_diag={'accuracy':float((q.argmax(1)==source_y).mean()),'source_log_loss':float(-np.log(np.maximum(q[np.arange(len(q)),source_y],1e-12)).mean()),
                 'mean_entropy':float(-(q*np.log(np.maximum(q,1e-12))).sum(1).mean()),'source_names_used_only_after_prediction':True}
    np.savez(directory/'predictions.npz',ids=confirm.id.to_numpy(),labels=y,groups=groups,shuffled_permutation=permutation,**predictions)
    result={'task':'H46','rows':len(confirm),'groups':len(set(groups)),'metrics':{n:metrics(y,v) for n,v in predictions.items()},'contrasts':contrasts,
            'gate':{'passed':passed,'rule':'Fixed strength gain>=0.001 and95%lower>0 versus calibration, positive95%evidence versus style, full valid coverage and conditional runtime<=25200s.'},
            'source_diagnostics':source_diag,'raw_classifier_order':order_diagnostics(raw['original'],raw['aligned_swap']),'head_swap_max_error':swaps,
            'classifier_seconds_per_row':classifier_rate,'source_seconds_per_answer':source_rate,'charged_gpu_seconds':charged,'cold_setup_allowance_seconds':setup,
            'projected_joint_seconds':projection,'projection_note':'Assumed25000hidden rows, measured prior/current classifier rate and local CPU source rate,20%margin,600sreserve. End-to-end offline profile still required.',
            'cpu_freeze_seconds':recipe['cpu_elapsed_seconds'],'cpu_evaluate_seconds':time.monotonic()-started,'peak_eval_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
            'confirmation_historically_inspected':True,'parameters_frozen_before_vote_access':True,'new_official_submissions':0,'official_score_unchanged':.95481,
            'prediction_sha256':sha(directory/'predictions.npz'),'source_sha256':sha(Path(__file__)),'frozen_recipe_sha256':sha(pub/'frozen_recipe.json')}
    for folder in [directory,pub]:json_write(folder/'H46_results.json',result)
    json_write(directory/'summary.json',{'complete':True,'gate_passed':passed,'cpu_elapsed_seconds':result['cpu_evaluate_seconds']});signal.alarm(0)
    print(json.dumps({'gate':result['gate'],'contrasts':contrasts,'losses':{n:v['log_loss'] for n,v in result['metrics'].items()},'source_diagnostics':source_diag,'projection':projection,'charged_gpu_seconds':charged},indent=2))


if __name__=='__main__':main()
