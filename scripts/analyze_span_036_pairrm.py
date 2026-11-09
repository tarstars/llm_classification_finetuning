"""One published and group-nested information decision after all native guards."""
from pathlib import Path
import hashlib
import resource
import signal
import subprocess
import sys
import time
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,sha256,utc_now,verify_sources
from arena.span034_residual import nested,predict
from arena.span_heads import metrics,paired_group_gain


def main():
    started=time.monotonic();pub,run=REPO/'results/span_036',ROOT/'runs/span_036'
    assert not (pub/'H79_results.json').exists() and not (pub/'target_access_intent.json').exists()
    cfg,prep=read_json(REPO/'configs/span_036.json'),read_json(pub/'preparation.json')
    def expired(*unused):raise TimeoutError('Analysis cap')
    signal.signal(signal.SIGALRM,expired);signal.alarm(cfg['cpu_analysis_seconds_cap'])
    bundle=read_json(pub/'pairrm_pilot_bundle.json');verify_sources(REPO,bundle['source_sha256'])
    assert read_json(pub/'jobs/pairrm_pilot/source_verification.json')['passed']
    state=read_json(run/'jobs/pairrm_pilot/controller_state.json');assert state['status']=='COMPLETE'
    job=run/'jobs/pairrm_pilot';receipt=read_json(job/'retrieval.json')
    for n,h in receipt['artifact_sha256'].items():assert sha256(job/'output'/n)==h,n
    out=job/'output/pairrm_audit';assert read_json(out/'pilot_complete.json')['complete']
    assert sha256(run/'baseline.npz')==prep['baseline_sha256'];baseline=np.load(run/'baseline.npz')
    spec=read_json(run/'input_asset/upload/token_spec.json'); arrays={}
    workers={}
    for ori in ['ab','ba']:
        workers[ori]=w=read_json(out/('worker_'+ori+'.json'))
        assert w['passed'] and w['strict_state_loaded'] and w['canonical_max_difference']<=1e-5 and w['memory_fraction']<.95
        assert w['cohorts']['main']['token_ids_sha256']==spec[ori]['token_ids_sha256']
        arrays[ori]=a=np.load(out/(ori+'.npz'));np.testing.assert_array_equal(a['ids'],baseline['ids'])
        assert a['lengths'].tolist()==spec[ori]['lengths'] and np.isfinite(a['logits']).all()
    d=.5*(arrays['ab']['logits']-arrays['ba']['logits']);p=baseline['baseline']
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    assert revision==subprocess.check_output(['git','rev-parse','origin/main'],cwd=REPO,text=True).strip()
    for n,h in bundle['source_sha256'].items():assert hashlib.sha256(subprocess.check_output(['git','show',revision+':'+n],cwd=REPO)).hexdigest()==h,n
    atomic_json(pub/'target_access_intent.json',dict(at=utc_now(),publication_commit=revision,variants=1,feature='symmetric_joint_pairrm',targets_after_all_native_guards=True))
    target=np.load(ROOT/'runs/span_016/cohort_matched/labels.npz')
    np.testing.assert_array_equal(target['ids'],baseline['ids']);y,g=target['labels'],target['groups']
    output,prior,majority,assignment,records=nested(p,d,y,g,cfg)
    np.testing.assert_array_equal(output[:,2],p[:,2]);np.testing.assert_array_equal(predict(p,d,0,1),p)
    for r in records:
        v,m=r['validation_indices'],r['model'];np.testing.assert_array_equal(predict(p[v][:,[1,0,2]],-d[v],m['alpha'],m['scale']),output[v][:,[1,0,2]])
    gain=paired_group_gain(y,p,output,g,dict(cfg,seed=cfg['bootstrap_seed']))
    useful=gain['gain']>=cfg['advancement_min_gain'] and gain['interval_95'][0]>0
    np.savez_compressed(run/'evaluation.npz',ids=baseline['ids'],labels=y,groups=g,feature=d,
        baseline=p,candidate=output,prior=prior,majority=majority,outer_fold=assignment)
    atomic_json(run/'fold_records.json',records)
    result=dict(at=utc_now(),task='H79',complete=True,quality_measured=True,publication_commit=revision,
        rows=len(y),groups=len(np.unique(g)),baseline_metrics=metrics(y,p),candidate_metrics=metrics(y,output),
        prior_metrics=metrics(y,prior),hard_majority_metrics=metrics(y,majority),raw_paired_gain=gain,
        useful_information=bool(useful),folds=[dict(fold=r['fold'],model=r['model'],inner_scores=r['inner_scores']) for r in records],
        tie_channel_exact=True,swap_equivariance_exact=True,zero_correction_exact=True,
        complete_current_H60=False,independent_competition_cv=False,old_service_status='ERROR',
        model_last_modified='2024-01-22',source_partitions_historically_inspected=True,
        new_gpu_executions=1,charged_gpu_seconds=state['charged_seconds'],official_submissions=0,official_submission_eligible=False,
        decision='prepare_separate_current_H60_transfer' if useful else 'stop_joint_comparator_information_gate',
        no_final_deployment_fit=True,worker_diagnostics=workers,evaluation_sha256=sha256(run/'evaluation.npz'),
        folds_sha256=sha256(run/'fold_records.json'),elapsed_seconds=time.monotonic()-started,
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    assert result['peak_rss_gib']<cfg['rss_guard_gib']
    atomic_json(pub/'H79_results.json',result)
    print({k:v for k,v in result.items() if not k.endswith('_metrics') and k not in ['worker_diagnostics','folds']})


if __name__=='__main__':main()
