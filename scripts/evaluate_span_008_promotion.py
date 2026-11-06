"""One frozen context confirmation contrast, after complete raw/CSV/source verification."""
from pathlib import Path
import json,resource,signal,sys,time
import numpy as np
import pandas as pd
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,CLASSES,prompt_groups
from arena.experiment import read_json,atomic_json,write_intent,sha256,verify_sources,utc_now
from arena.span_heads import temperature,metrics,paired_group_gain,order_diagnostics
from arena.span006_inputs import text_digest


def main():
    start=time.monotonic();root=ROOT/'runs/span_008';pub=REPO/'results/span_008';job=root/'jobs/promotion';out=job/'output/promotion';cfg=read_json(REPO/'configs/span_008.json')
    assert not (pub/'H50_results.json').exists() and not (root/'confirmation_reveal_intent.json').exists()
    def expired(*unused):raise TimeoutError('Original promotion analysis CPU cap')
    signal.signal(signal.SIGALRM,expired);signal.alarm(cfg['cpu_stage_seconds_caps']['analysis'])
    state=read_json(job/'controller_state.json');assert state['stage']=='complete' and state['status']=='COMPLETE'
    bundle=read_json(pub/'promotion_bundle.json');verify_sources(REPO,bundle['source_sha256']);assert read_json(pub/'jobs/promotion/source_verification.json')['passed']
    for file,digest in read_json(job/'retrieval.json')['artifact_sha256'].items():assert sha256(job/'output'/file)==digest
    frozen=read_json(pub/'candidate_selection.json');assert frozen['eligible'] and frozen['candidate_kind']=='context_completion'
    summary=read_json(out/'summary.json');assert summary['complete'] and summary['qualify'] and summary['identity']['passed'] and not summary['competition_target_columns_loaded']
    assert summary['candidate_selection_sha256']==sha256(pub/'candidate_selection.json') and summary['temperature']==.9441963863130951
    hardware=[read_json(out/f'gpu_rank{rank}.json') for rank in [0,1]];assert all(h['rank']==h['device']==i and 'T4' in h['name'] for i,h in enumerate(hardware))
    inv=read_json(pub/'artifact_inventory.json');frame=pd.read_parquet(inv['input_artifacts']['confirmation_inputs']['path']);raw=np.load(out/'confirmation/predictions.npz');base=np.load(inv['input_artifacts']['confirmation_reference']['path'])
    for name in ['confirmation_inputs','confirmation_reference','raw_train']:
        assert sha256(inv['input_artifacts'][name]['path'])==inv['input_artifacts'][name]['sha256']
    np.testing.assert_array_equal(raw['ids'],frame.id);np.testing.assert_array_equal(base['ids'],frame.id)
    cs=read_json(out/'confirmation/summary.json');assert cs['complete'] and cs['rows']==5746 and sha256(out/'confirmation/predictions.npz')==cs['prediction_sha256'] and text_digest(frame)==cs['text_sha256']
    t=summary['temperature'];p=temperature((base['original']+base['aligned_swap'])/2,t);q=temperature((raw['original']+raw['aligned_swap'])/2,t)
    np.testing.assert_allclose(raw['probabilities'],q,atol=1e-6)
    mask=raw['changed'];delta=np.abs(np.r_[raw['original'][~mask]-base['original'][~mask],raw['aligned_swap'][~mask]-base['aligned_swap'][~mask]])
    limits=cfg['raw_identity_limits'];unchanged={'rows':int((~mask).sum()),'mean':float(delta.mean()),'maximum':float(delta.max()),'passed':bool(delta.mean()<=limits['mean'] and delta.max()<=limits['maximum'])}
    test=np.load(out/'test/predictions.npz');csv=pd.read_csv(job/'output/submission.csv');visible=pd.read_csv(ROOT/'raw/test.csv',usecols=['id'])
    assert csv.columns.tolist()==['id']+CLASSES;np.testing.assert_array_equal(csv.id,visible.id);np.testing.assert_array_equal(test['ids'],visible.id)
    np.testing.assert_allclose(csv[CLASSES],test['probabilities'],atol=1e-8);assert np.isfinite(csv[CLASSES]).all().all() and csv[CLASSES].ge(0).all().all()
    np.testing.assert_allclose(csv[CLASSES].sum(1),1,atol=1e-6);assert sha256(job/'output/submission.csv')==summary['csv_sha256']
    session=read_json(job/'output/session_summary.json');cold=max(0,session['seconds']-cs['prediction_seconds']-summary['test']['prediction_seconds'])
    dev=read_json(pub/'H51_results.json')['runtime'];dev_rate=(dev['prospective_projected_seconds']-dev['cold_seconds']-600)/(25000*1.2)
    rate=max(dev_rate,cs['prediction_seconds']/5746);projection=25000*rate*1.2+cold+600
    write_intent(root/'confirmation_reveal_intent.json',{'at':utc_now(),'frozen_candidate_sha256':sha256(pub/'candidate_selection.json'),'bundle_sha256':sha256(pub/'promotion_bundle.json'),
        'raw_prediction_sha256':sha256(out/'confirmation/predictions.npz'),'evaluation_source_sha256':sha256(Path(__file__)),'fit_operations':0,'predictions_verified_before_votes':True})
    labels=pd.read_csv(inv['input_artifacts']['raw_train']['path'],usecols=['id']+CLASSES).set_index('id').loc[frame.id];assert labels.sum(1).eq(1).all();y=labels.to_numpy().argmax(1);groups=prompt_groups(frame).to_numpy(dtype=str)
    gain=paired_group_gain(y,p,q,groups,dict(cfg,seed=cfg['bootstrap_seed']));quality=gain['gain']>=cfg['confirmation_practical_gain'] and gain['interval_95'][0]>0
    result={'task':'H50','executed':True,'confirmation_complete':True,'confirmation_historically_inspected':True,'fit_operations':0,'parameters_frozen_before_vote_access':True,
        'metrics':{'frozen_classifier':metrics(y,p),'context':metrics(y,q)},'contrast':gain,'quality_gate_passed':bool(quality),'unchanged_raw_identity':unchanged,
        'runtime':{'prediction_seconds_per_row':rate,'confirmation_seconds_per_row':cs['prediction_seconds']/5746,'cold_setup_seconds':cold,'projected_seconds':projection,'passed':projection<=25200},
        'eligible_for_official':bool(quality and unchanged['passed'] and projection<=25200),'csv_verified':True,'csv_sha256':sha256(job/'output/submission.csv'),
        'changed_rows':int(mask.sum()),'hardware':hardware,'charged_gpu_seconds':state['charged_seconds'],'new_official_submissions':0,'official_submission_ref':None,
        'order_diagnostics':order_diagnostics(raw['original'],raw['aligned_swap']),'source_sha256':sha256(Path(__file__)),'cpu_seconds':time.monotonic()-start,
        'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,'candidate_selection_sha256':sha256(pub/'candidate_selection.json')}
    np.savez_compressed(root/'confirmation_comparison.npz',ids=frame.id,labels=y,groups=groups,baseline=p,candidate=q)
    atomic_json(pub/'H50_results.json',result);print({k:result[k] for k in ['contrast','runtime','unchanged_raw_identity','eligible_for_official']},flush=True)


if __name__=='__main__':main()
