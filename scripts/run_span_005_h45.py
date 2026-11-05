"""Nested preference correction from inferred source posteriors; no dev names."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,resource,signal,sys,time
import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,json_write,prompt_groups
from arena.span_heads import temperature,normalize
from arena.source_models import source_features,style_features,softmax
from arena.source_evaluation import evaluate,summarize


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    started=time.monotonic();out=ROOT/'runs/span_005';pub=REPO/'results/span_005';directory=out/'H45';directory.mkdir(exist_ok=True)
    assert not (directory/'started.json').exists(),'H45 already started; inspect outputs rather than repeating'
    cfg=json.loads((REPO/'configs/span_005.json').read_text());contract=json.loads((pub/'H45_contract.json').read_text())
    assert digest(Path(__file__))==contract['runner_sha256'] and digest(REPO/'configs/span_005.json')==contract['config_sha256']
    for key in ['E06','E07']:
        assert digest(pub/(key+'_results.json'))==contract[key+'_result_sha256']
        assert json.loads((pub/(key+'_results.json')).read_text())['gate']['passed']
    assert digest(out/'E06/strength.npz')==contract['strength_sha256']
    assert digest(out/'E07/tfidf_predictions.npz')==contract['source_prediction_sha256']
    assert digest(REPO/'src/arena/source_evaluation.py')==contract['evaluation_source_sha256']
    json_write(directory/'started.json',{'at':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'source_sha256':digest(Path(__file__))})
    def expired(signum,frame):raise TimeoutError('Original H45 CPU elapsed cap reached')
    signal.signal(signal.SIGALRM,expired);signal.alarm(cfg['cpu_stage_seconds_caps']['H45'])
    # Actual held-out source names are not even loaded by the practical worker.
    dev=pd.read_parquet(out/'dev.parquet',columns=['id','prompt','response_a','response_b'])
    a=np.load(out/'inputs.npz');source=np.load(out/'E07/tfidf_predictions.npz')
    y=a['labels'];groups=prompt_groups(dev).to_numpy(dtype=str);p=temperature(a['probabilities'],.9441963863130951)
    qa,qb=source['dev_a'],source['dev_b'];assert qa.shape==qb.shape and qa.shape[0]==len(y)
    vocab=json.loads((out/'source_vocabulary.json').read_text());classes=vocab['source_classes'];names=vocab['all_training_sources']
    theta=np.load(out/'E06/strength.npz')['theta'];ratings_by_name=dict(zip(names,theta[:-1]));counts=vocab['source_counts']
    rare=[name for name in names if name not in classes]
    other=float(np.average([ratings_by_name[name] for name in rare],weights=[counts[name] for name in rare])) if rare else 0.
    ratings=np.array([ratings_by_name.get(name,other) for name in classes])
    permutation=np.random.default_rng(cfg['seed']).permutation(len(y))
    features={'calibration':(np.empty((len(y),0)),np.empty((len(y),0))),
              'style':style_features(dev.response_a,dev.response_b),
              'source_strength':source_features(qa,qb,ratings,'strength'),
              'source_full':source_features(qa,qb,ratings,'full'),
              'shuffled_source':source_features(qa[permutation],qb[permutation],ratings,'strength')}
    def log(*args,**kwargs):
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2<cfg['rss_guard_gib'],'Resident memory guard'
        print(*args,**kwargs)
    predictions,details=evaluate(p,y,groups,a['outer_fold'],features,cfg,['source_strength','source_full'],log)
    diff=(ratings[:,None]-ratings[None,:])/2
    kernels=softmax(np.column_stack([diff.ravel(),-diff.ravel(),np.full(diff.size,theta[-1])])).reshape(len(ratings),len(ratings),3)
    prior=np.column_stack([np.sum((qa@kernels[:,:,k])*qb,axis=1) for k in range(3)])
    predictions['source_only']=normalize(prior);predictions['frozen_classifier']=p
    old=np.load(out/'E06/predictions.npz')
    for key in ['calibration','style']:np.testing.assert_allclose(predictions[key],old[key],atol=1e-12,rtol=1e-12)
    for key in ['training_prior','uniform','majority_clipped_1e-12']:predictions[key]=old[key]
    result=summarize(y,predictions,groups,cfg);g=result['contrasts']['nested_selected_vs_calibration'];s=result['contrasts']['selected_vs_style']
    admitted=bool(g['gain']>=cfg['practical_gain'] and g['interval_95'][0]>0 and s['gain']>0 and s['interval_95'][0]>0)
    np.savez(directory/'predictions.npz',ids=a['ids'],labels=y,groups=groups,outer_fold=a['outer_fold'],shuffled_source_permutation=permutation,**predictions)
    json_write(directory/'folds.json',details)
    result.update(task='H45',rows=len(y),groups=len(set(groups)),source_family='tfidf',source_development_names_loaded=False,
                  auxiliary_models_frozen_from_training_only=True,development_and_confirmation_previously_inspected=True,
                  confirmation_used=False,gpu_seconds=0,cpu_elapsed_seconds=time.monotonic()-started,
                  peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
                  gate={'passed':admitted,'rule':'Gain>=0.001 and paired95%lower bound>0 versus both calibration and style; no official submission in this span.'},
                  prediction_sha256=digest(directory/'predictions.npz'),folds_sha256=digest(directory/'folds.json'),
                  source_sha256=digest(Path(__file__)),official_score_unchanged=.95481)
    for folder in [directory,pub]:json_write(folder/'H45_results.json',result)
    json_write(directory/'summary.json',{'complete':True,'adoption_gate_passed':admitted,'cpu_elapsed_seconds':result['cpu_elapsed_seconds']});signal.alarm(0)
    log(json.dumps({'task':'H45','gate':result['gate'],'contrasts':result['contrasts'],'losses':{k:v['log_loss'] for k,v in result['metrics'].items()},'cpu_seconds':result['cpu_elapsed_seconds']},indent=2),flush=True)


if __name__=='__main__':main()
