"""Run the fixed CPU structural residual and its nested, matched controls."""
from pathlib import Path
import os,resource,signal,sys,time
import joblib
import numpy as np
import pandas as pd
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,CLASSES,prompt_groups
from arena.experiment import read_json,atomic_json,write_intent,sha256,utc_now,EventJournal
from arena.span008_structure import feature_views,fit_selected
from arena.span_calibration import SymmetricCalibration,grouped_folds
from arena.span_heads import metrics,paired_group_gain
from arena.source_models import style_features
from arena.source_evaluation import fit_selected as fit_linear


def main():
    cfg=read_json(REPO/'configs/span_008.json');inv=read_json(REPO/'results/span_008/artifact_inventory.json')
    root=ROOT/'runs/span_008/structure';root.mkdir(parents=True,exist_ok=True)
    write_intent(root/'started.json',{'at':utc_now(),'pid':os.getpid(),'source_sha256':sha256(Path(__file__))});start=time.monotonic()
    def expired(*unused):raise TimeoutError('Original H36 CPU budget exhausted')
    signal.signal(signal.SIGALRM,expired);signal.alarm(cfg['cpu_stage_seconds_caps']['structure']);journal=EventJournal(root)
    dev=pd.read_parquet(inv['input_artifacts']['dev.parquet']['path'],columns=['id','prompt','response_a','response_b',*CLASSES])
    raw=np.load(inv['input_artifacts']['classifier']['path']);np.testing.assert_array_equal(raw['ids'],dev.id)
    p=(raw['original']+raw['aligned_swap'])/2;y=dev[CLASSES].to_numpy().argmax(1);groups=prompt_groups(dev).to_numpy(dtype=str)
    views={name:feature_views(dev,p,name=='structure') for name in ['probability_only','structure']}
    features={'style':style_features(dev.response_a,dev.response_b)};linear_cfg=dict(cfg,residual_regularization_grid=[.001,.01,.1],calibration_slope_bounds=[1/3,2.],residual_max_iterations=500)
    outputs={k:np.full_like(p,np.nan,dtype=float) for k in ['calibration','probability_only','structure','style']};details=[]
    for fold,(train,val) in enumerate(grouped_folds(y,groups,cfg['outer_group_folds'],cfg['seed'])):
        record={'fold':fold,'fit_rows':len(train),'validation_rows':len(val)}
        cal=SymmetricCalibration('symmetric_vector',cfg['temperature_bounds'],cfg['tie_intercept_bounds']).fit(p[train],y[train]);outputs['calibration'][val]=cal.predict_proba(p[val])
        for name,(x,z) in views.items():
            fit,choice=fit_selected(p[train],y[train],groups[train],x[train],z[train],cfg)
            pred=fit.predict_proba(p[val],x[val],z[val]);reverse=fit.predict_proba(p[val][:,[1,0,2]],z[val],x[val])[:,[1,0,2]]
            assert np.allclose(pred,reverse,atol=1e-12);outputs[name][val]=pred;record[name]=choice
            joblib.dump(fit,root/f'{name}_fold{fold}.joblib',compress=3)
        outputs['style'][val],record['style'],_=fit_linear(p,y,groups,features,train,val,linear_cfg,['style'])
        details.append(record);assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2<cfg['rss_guard_gib']
        journal.emit('structure','outer_complete',step=fold,rows=len(val),seconds=time.monotonic()-start)
    contrasts={name:paired_group_gain(y,outputs[name],outputs['structure'],groups,dict(cfg,seed=cfg['bootstrap_seed'])) for name in ['calibration','probability_only','style']}
    eligible=contrasts['calibration']['gain']>=cfg['development_practical_gain'] and all(v['interval_95'][0]>0 for v in contrasts.values())
    fit=None
    if eligible:
        x,z=views['structure'];fit,choice=fit_selected(p,y,groups,x,z,cfg);joblib.dump(fit,root/'frozen_model.joblib',compress=3)
        atomic_json(root/'frozen_selection.json',choice)
    np.savez_compressed(root/'predictions.npz',ids=dev.id,labels=y,groups=groups,**outputs)
    result={'task':'H36','complete':True,'eligible':bool(eligible),'metrics':{k:metrics(y,v) for k,v in outputs.items()},'contrasts':contrasts,'outer_details':details,
        'seconds':time.monotonic()-start,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,'numerical_threads':2,
        'confirmation_targets_read':False,'exact_swap_equivariance':True,'backbone_predictions':'Held-out development predictions; meta heads fitted within nested prompt groups',
        'prediction_sha256':sha256(root/'predictions.npz'),'source_sha256':{n:sha256(REPO/n) for n in ['scripts/run_span_008_structure.py','src/arena/span008_structure.py','src/arena/data.py']}}
    atomic_json(REPO/'results/span_008/H36_results.json',result);print({k:result[k] for k in ['eligible','seconds','peak_rss_gib','contrasts']},flush=True)


if __name__=='__main__':main()
