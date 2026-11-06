"""Check completed exact GPU artifacts before revealing the frozen v8 comparisons."""
from pathlib import Path
import argparse,resource,signal,sys,time,json,hashlib
import joblib
import numpy as np
import pandas as pd
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,CLASSES,prompt_groups
from arena.experiment import read_json,atomic_json,sha256
from arena.span_heads import normalize,temperature,metrics,paired_group_gain,order_diagnostics
from arena.span_calibration import grouped_folds
from arena.span008_cascade import fit_selected


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('role',choices=['serial_cascade','context_completion']);args=parser.parse_args();role=args.role;task='H52' if role=='serial_cascade' else 'H51'
    cfg=read_json(REPO/'configs/span_008.json');pub=REPO/'results/span_008';root=ROOT/'runs/span_008';job=root/'jobs'/role;out=job/'output'/role
    assert not (pub/(task+'_results.json')).exists(),'No repeated quality selection'
    def expired(*unused):raise TimeoutError('Original analysis cap')
    signal.signal(signal.SIGALRM,expired);signal.alarm(cfg['cpu_stage_seconds_caps']['analysis']);started=time.monotonic()
    state=read_json(job/'controller_state.json');receipt=read_json(job/'retrieval.json')
    for file,digest in receipt['artifact_sha256'].items():assert sha256(job/'output'/file)==digest
    source=read_json(pub/'jobs'/role/'source_verification.json');assert source['passed']
    charge=state['charged_seconds'];summary=read_json(out/'summary.json') if (out/'summary.json').exists() else {}
    base_result={'task':task,'service_status':state['status'],'charged_gpu_seconds':charge,'source_verification_sha256':sha256(pub/'jobs'/role/'source_verification.json'),
        'retrieval_sha256':sha256(job/'retrieval.json'),'source_sha256':sha256(Path(__file__)),'eligible':False,'complete':False}
    if state['status']!='COMPLETE' or not summary.get('complete') or (role=='serial_cascade' and not summary.get('cache_complete')):
        atomic_json(pub/(task+'_results.json'),dict(base_result,worker_summary=summary,reason=summary.get('reason','Incomplete/failed worker; no quality claim')));return
    hardware=out/'classifier' if role=='serial_cascade' else out
    gpu=[read_json(hardware/f'gpu_rank{rank}.json') for rank in [0,1]]
    assert [x['rank'] for x in gpu]==[0,1] and all('T4' in x['name'] for x in gpu)
    inv=read_json(pub/'artifact_inventory.json');dev=pd.read_parquet(inv['input_artifacts']['dev.parquet']['path'],columns=['id','prompt',*CLASSES])
    y=dev[CLASSES].to_numpy().argmax(1);groups=prompt_groups(dev).to_numpy(dtype=str);raw=np.load(inv['input_artifacts']['classifier']['path']);np.testing.assert_array_equal(raw['ids'],dev.id)
    baseline=normalize((raw['original']+raw['aligned_swap'])/2);lookup={int(v):i for i,v in enumerate(dev.id)};selection=read_json(root/'sample_ids.json')
    destination=root/'analysis'/role;destination.mkdir(parents=True,exist_ok=False);bootstrap=dict(cfg,seed=cfg['bootstrap_seed'])
    if role=='serial_cascade':
        assert summary['native_both_devices_verified'] and summary['stability_passed'] and summary['runtime']['passed']
        from arena.judge import label_probabilities
        from arena.span_repeatability import difference
        # Replay every fixed-cohort response independently before fitting any labels.
        hw=read_json(job/'output/hardware.json');assert 'not found' not in hw['ldd'] and all(v in hw['devices'] for v in ['CUDA0: Tesla T4','CUDA1: Tesla T4'])
        token_ids=read_json(ROOT/'runs/span_007/jobs/judge_stability/output/judge_stability/diagnostic_summary.json')['configurations']['serial']['label_tokens']
        expected={name:selection['judge_sentinel'] for name in ['sentinel_initial','sentinel_repeat','sentinel_restart']}
        expected.update(route_profile=selection['judge_profile'],swap_diagnostic=selection['judge_swap'])
        expected.update({f'cache_{begin}':selection['judge_cache'][begin:begin+128] for begin in range(0,cfg['judge_cache_rows'],128)})
        replayed=0;token_maps={}
        for name,expected_ids in expected.items():
            folder=out/name;pred=np.load(folder/'predictions.npz');np.testing.assert_array_equal(pred['ids'],expected_ids)
            traces=[json.loads(line) for line in (folder/'requests.jsonl').read_text().splitlines()];assert len(traces)==len(expected_ids)
            byid={int(i):v for i,v in zip(pred['ids'],pred['original'])};assert {t['id'] for t in traces}==set(expected_ids)
            token_maps[name]={}
            for t in traces:
                assert t['cache_prompt'] is False and not t['response'].get('truncated',False)
                tokens=t['prepared_tokens'];assert 0<len(tokens)<1024 and all(isinstance(v,int) for v in tokens)
                assert hashlib.sha256(json.dumps(tokens).encode()).hexdigest()==t['prepared_tokens_sha256']
                np.testing.assert_allclose(label_probabilities(t['response'],token_ids),byid[t['id']],atol=1e-12,rtol=0)
                token_maps[name][str(t['id'])]=t['prepared_tokens_sha256'];replayed+=1
        assert token_maps['sentinel_initial']==token_maps['sentinel_repeat']==token_maps['sentinel_restart']==read_json(root/'judge_reference_tokens.json')
        first=np.load(out/'sentinel_initial/predictions.npz')
        for name in ['sentinel_repeat','sentinel_restart']:
            observed=np.load(out/name/'predictions.npz');assert difference(first['ids'],first['original'],observed['ids'],observed['original'],cfg['repeatability_limits'])['passed']
        atomic_json(pub/'H52_request_audit.json',{'passed':True,'responses_independently_replayed':replayed,'targets_used':False,'native_both_devices_verified':True,'label_tokens_from_same_pinned_model':token_ids,
            'historical_sentinel_token_maps_identical':True,'source_sha256':sha256(Path(__file__))})
        cache=np.load(out/'judge_cache.npz');np.testing.assert_array_equal(cache['ids'],selection['judge_cache']);assert sha256(out/'judge_cache.npz')==summary['cache_sha256']
        idx=np.array([lookup[int(v)] for v in cache['ids']]);p=baseline[idx];q=normalize(cache['original']);y=y[idx];groups=groups[idx]
        predictions=np.full_like(p,np.nan);control=np.full_like(p,np.nan);masks=np.zeros(len(p),bool);details=[]
        for fold,(train,val) in enumerate(grouped_folds(y,groups,cfg['outer_group_folds'],cfg['seed'])):
            fit,chosen=fit_selected(p[train],q[train],y[train],groups[train],cfg)
            predictions[val],control[val],masks[val]=fit.predict(p[val],q[val]);details.append({'fold':fold,'threshold':fit.threshold,'routed_rows':int(masks[val].sum()),**chosen})
            joblib.dump(fit,destination/f'fold{fold}.joblib',compress=3)
        contrast=paired_group_gain(y,control,predictions,groups,bootstrap)
        eligible=contrast['gain']>=cfg['development_practical_gain'] and contrast['interval_95'][0]>0
        outputs={'calibration':control,'cascade':predictions};ids=cache['ids']
        if eligible:
            fit,chosen=fit_selected(p,q,y,groups,cfg);joblib.dump(fit,destination/'frozen_model.joblib',compress=3);atomic_json(destination/'frozen_selection.json',chosen)
        result=dict(base_result,complete=True,eligible=bool(eligible),metrics={k:metrics(y,v) for k,v in outputs.items()},contrast=contrast,
            outer_details=details,route_fraction=float(masks.mean()),routed_rows=int(masks.sum()),cache_rows=len(p),runtime=summary['runtime'],
            swap_diagnostic=summary['swap_diagnostic'],competition_targets_used_by_gpu=False,confirmation_targets_read=False,
            routed_metrics={k:metrics(y[masks],v[masks]) for k,v in outputs.items()} if masks.any() else {})
    else:
        oldroot=ROOT/'runs/span_007/jobs/context/output/context';old_audit=read_json(REPO/'results/span_007/context_partial_audit.json');assert old_audit['passed']
        for size in ['1024','2048']:
            for file,digest in old_audit['coverage'][size]['prediction_files_sha256'].items():assert sha256(oldroot/file)==digest
        for name in ['smoke','overlap','unchanged_profile']:assert read_json(out/(name+'_identity.json'))['passed']
        assert summary['context_text_sha256']==inv['context_text_sha256'] and summary['context_token_sha256']==inv['context_token_sha256']
        short=np.load(oldroot/'predictions_1024.npz');longparts=[np.load(oldroot/f'2048_{start:04d}.npz') for start in range(0,896,128)]
        new=np.load(out/'completed_long.npz');longparts.append(new);allids=np.concatenate([x['ids'] for x in longparts]);np.testing.assert_array_equal(allids,selection['context_all']);np.testing.assert_array_equal(short['ids'],allids)
        joined={v:np.vstack([x[v] for x in longparts]) for v in ['original','aligned_swap']};idx=np.array([lookup[int(v)] for v in allids])
        reference_orders=[raw['original'].copy(),raw['aligned_swap'].copy()]
        for i,v in enumerate(['original','aligned_swap']):reference_orders[i][idx]=short[v]
        candidate_orders=[p.copy() for p in reference_orders]
        for i,v in enumerate(['original','aligned_swap']):candidate_orders[i][idx]=joined[v]
        deployed_baseline=baseline.copy();baseline=(reference_orders[0]+reference_orders[1])/2;candidate=(candidate_orders[0]+candidate_orders[1])/2
        t=.9441963863130951;outputs={'short':temperature(baseline,t),'long':temperature(candidate,t)};ids=dev.id.to_numpy();contrast=paired_group_gain(y,outputs['short'],outputs['long'],groups,bootstrap)
        events=[__import__('json').loads(line) for line in (out/'stages/events.jsonl').read_text().splitlines()]
        measured=summary['long_seconds']+sum(e.get('seconds',0) for e in events if e['stage'] in ['smoke','overlap','unchanged_profile'])
        cold=max(0,charge-measured);fraction=len(idx)/len(dev);oldrate=old_audit['coverage']['2048']['seconds_per_completed_row'];newrate=summary['long_seconds']/256;longrate=max(oldrate,newrate)
        def projection(short_rate):return cfg['hidden_rows_assumption']*((1-fraction)*short_rate+fraction*longrate)*cfg['runtime_margin']+cold+cfg['runtime_reserve_seconds']
        runtime={'conservative_projected_seconds':projection(.6527916988221372),'prospective_projected_seconds':projection(summary['unchanged_profile_seconds']/128),
            'unchanged_seconds_per_row':summary['unchanged_profile_seconds']/128,'long_old_seconds_per_row':oldrate,'long_new_seconds_per_row':newrate,
            'adopted_long_seconds_per_row':longrate,'cold_seconds':cold,'changed_fraction':fraction,'passed':projection(summary['unchanged_profile_seconds']/128)<=25200}
        eligible=contrast['gain']>=cfg['development_practical_gain'] and contrast['interval_95'][0]>0 and runtime['passed']
        np.savez_compressed(destination/'long_raw.npz',ids=allids,**joined)
        result=dict(base_result,complete=True,eligible=bool(eligible),metrics={k:metrics(y,v) for k,v in outputs.items()},contrast=contrast,
            changed_slice={'rows':len(idx),'contrast':paired_group_gain(y[idx],outputs['short'][idx],outputs['long'][idx],groups[idx],bootstrap),
                'metrics':{k:metrics(y[idx],v[idx]) for k,v in outputs.items()}},unchanged_rows=len(dev)-len(idx),runtime=runtime,
            order_diagnostics={'short':order_diagnostics(*reference_orders),'long':order_diagnostics(*candidate_orders)},
            deployed_baseline_sensitivity={'contrast':paired_group_gain(y,temperature(deployed_baseline,t),outputs['long'],groups,bootstrap),
                'changed_short_cache_drift_mean':float(np.abs(baseline[idx]-deployed_baseline[idx]).mean()),
                'changed_short_cache_drift_max':float(np.abs(baseline[idx]-deployed_baseline[idx]).max()),'primary_gate_unchanged':True},
            frozen_temperature=t,confirmation_targets_read=False,competition_targets_used_by_gpu=False)
    np.savez_compressed(destination/'predictions.npz',ids=ids,labels=y,groups=groups,**outputs)
    result.update(prediction_sha256=sha256(destination/'predictions.npz'),cpu_seconds=time.monotonic()-started,peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    atomic_json(pub/(task+'_results.json'),result);print({k:result[k] for k in ['task','complete','eligible','contrast','runtime']},flush=True)


if __name__=='__main__':main()
