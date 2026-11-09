"""Audit the two fresh size arms and compare only after reconciling actual exposure."""
from collections import Counter
from pathlib import Path
import json,resource,signal,sys,time
import numpy as np
import pandas as pd
from safetensors.torch import load_file
from transformers import AutoTokenizer
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,CLASSES
from arena.experiment import read_json,atomic_json,sha256,verify_sources,utc_now
from arena.exposure_validation import validate_exposure
from arena.joint_data import JointTokenizer
from arena.span006_inputs import token_digest
from arena.span_heads import metrics,paired_group_gain,crossfit_temperature
from arena.data import prompt_groups


def main():
    start=time.monotonic();cfg=read_json(REPO/'configs/span_031.json');pub=REPO/'results/span_031';run=ROOT/'runs/span_031';job=run/'jobs/size_pilot';out=job/'output/size_pilot'
    assert not (pub/'H02_results.json').exists()
    def expired(*unused):raise TimeoutError('Size audit CPU cap')
    signal.signal(signal.SIGALRM,expired);signal.alarm(cfg['cpu_stage_seconds_caps']['analysis'])
    state=read_json(job/'controller_state.json');receipt=read_json(job/'retrieval.json')
    for name,digest in receipt['artifact_sha256'].items():assert sha256(job/'output'/name)==digest
    assert read_json(pub/'jobs/size_pilot/source_verification.json')['passed'];verify_sources(REPO,read_json(pub/'size_pilot_bundle.json')['source_sha256'])
    result={'task':'H02','complete':False,'service_status':state['status'],'charged_gpu_seconds':state['charged_seconds'],'arms':{},'quality_comparison_valid':False,
        'official_submission':False,'confirmation_targets_used':False,'source_sha256':sha256(Path(__file__))}
    inv=read_json(pub/'artifact_inventory.json');selection=read_json(run/'pilot_ids.json');columns=['id','prompt','response_a','response_b',*CLASSES]
    frames={name:pd.read_parquet(inv['input_artifacts'][name+'.parquet']['path'],columns=columns).set_index('id',drop=False).loc[selection[name]] for name in ['train','dev']}
    train,dev=frames['train'],frames['dev'];labels=dict(zip(map(int,train.id),train[CLASSES].to_numpy().argmax(1)));y=dev[CLASSES].to_numpy().argmax(1);groups=prompt_groups(dev).to_numpy(dtype=str)
    predictions={};exposures={};hashes={}
    for size in ['4b','8b']:
        arm=out/size
        if not (arm/'summary.json').exists():
            result['arms'][size]={'complete':False,'reason':'No complete arm summary; partial execution remains preserved'};continue
        summary=read_json(arm/'summary.json');assert summary['complete'];assert summary['global_batch']==32;assert summary['fresh_classifier_head'] and not summary['confirmation_targets_used']
        assert summary['source_sha256']==sha256(REPO/'src/arena/span031_size.py')
        assert sha256(arm/'adapter/adapter_model.safetensors')==summary['adapter_sha256'];tensors=load_file(str(arm/'adapter/adapter_model.safetensors'))
        assert all(t.isfinite().all() for t in tensors.values());del tensors
        tok=AutoTokenizer.from_pretrained(str(arm/'adapter'),local_files_only=True,padding_side='right')
        original,swapped,_=JointTokenizer(tok,1024).encode_frame(train);do,ds,_=JointTokenizer(tok,1024).encode_frame(dev)
        hashes[size]={name:token_digest(data) for name,data in [('train_original',original),('train_swapped',swapped),('dev_original',do),('dev_swapped',ds)]}
        assert hashes[size]==summary['token_hashes'];lengths={int(r):{len(a),len(b)} for r,a,b in zip(train.id,original,swapped)}
        accepted=Counter();token_count=rows=0;ranks=[];vectors=[]
        for rank in [0,1]:
            assert read_json(arm/f'packing_rank{rank}.json')==hashes[size]
            events=[json.loads(line) for line in (arm/f'exposure_rank{rank}.jsonl').read_text().splitlines()];record=read_json(arm/f'audit_state_rank{rank}.json')
            audit=validate_exposure(events,record,0,summary['nominal_cycles'],labels,1024);vectors.append(audit['skip_vector'])
            assert events[0]['scale_before']==cfg['size_pilot_initial_amp_scale']
            assert all(len(e['ids'])==16 for e in events),'Every global cycle must process32rows'
            for event in events:
                for rid,n in zip(event['ids'],event['token_lengths']):assert n in lengths[rid]
                if not event['skipped']:accepted.update(event['ids']);token_count+=sum(event['token_lengths']);rows+=len(event['ids'])
            gpu=read_json(arm/f'gpu_rank{rank}.json');assert gpu['device']==rank and 'T4' in gpu['name'] and gpu['peak_allocated_bytes']>0
            assert gpu['total_memory_bytes']>0
            gpu['peak_memory_fraction']=gpu['peak_allocated_bytes']/gpu['total_memory_bytes']
            audit['hardware']=gpu;audit.pop('skip_vector');ranks.append(audit)
        assert vectors[0]==vectors[1];assert ranks[0]['accepted_updates']==summary['accepted_updates']==ranks[1]['accepted_updates']
        raw=np.load(arm/'dev_predictions.npz');np.testing.assert_array_equal(raw['ids'],dev.id);p=(raw['original']+raw['aligned_swap'])/2;predictions[size]=p;exposures[size]=(accepted,token_count)
        telemetry=[json.loads(line) for line in (arm/'telemetry/events.jsonl').read_text().splitlines()]
        curves=[{'step':e['step'],'accepted_updates':e['counts']['accepted_updates'],'elapsed_seconds':e['elapsed_seconds'],**e['metrics']} for e in telemetry if e['event']=='cycle']
        result['arms'][size]={'complete':True,'raw_metrics':metrics(y,p),'accepted_updates':summary['accepted_updates'],'nominal_cycles':summary['nominal_cycles'],
            'global_accepted_rows':rows,'global_accepted_unique_rows':len(accepted),'global_accepted_tokens':token_count,'per_rank':ranks,
            'training_seconds':summary['training_seconds'],'prediction_seconds':summary['prediction_seconds'],'prediction_seconds_per_row':summary['prediction_seconds']/len(dev),
            'model_config_sha256':summary['model_config_sha256'],'adapter_sha256':summary['adapter_sha256'],'packed_token_hashes':hashes[size],
            'training_curve':curves,'loss_curve':[{'step':e['step'],'elapsed_seconds':e['elapsed_seconds'],**e['metrics']} for e in telemetry if e['event']=='metrics'],
            'stopped_for_budget':summary['stopped_for_budget'],
            'setup_save_overhead_seconds':max(0,summary['worker_seconds']-summary['training_seconds']-summary['prediction_seconds']),
            'order_diagnostics':__import__('arena.span_heads',fromlist=['order_diagnostics']).order_diagnostics(raw['original'],raw['aligned_swap'])}
        result['arms'][size]['raw_metrics']['balanced_accuracy']=float(np.mean(result['arms'][size]['raw_metrics']['class_recall']))
    if len(predictions)==2:
        matched=hashes['4b']==hashes['8b'] and exposures['4b']==exposures['8b'] and all(result['arms'][s]['accepted_updates']==cfg['size_pilot_accepted_updates'] for s in ['4b','8b'])
        result.update(complete=state['status']=='COMPLETE',quality_comparison_valid=matched,raw_paired_gain=paired_group_gain(y,predictions['4b'],predictions['8b'],groups,dict(cfg,seed=cfg['bootstrap_seed'])),
            exposure_note='Same packed inputs, accepted row multiset and token count' if matched else 'Unequal accepted exposure or packing; do not attribute observed loss difference causally to size')
        calibrated={}
        for size in ['4b','8b']:
            calibrated[size],temperatures=crossfit_temperature(y,predictions[size],groups,cfg)
            result['arms'][size].update(temperature_cv_metrics=metrics(y,calibrated[size]),fold_temperatures=temperatures)
        result['temperature_cv_diagnostic']=paired_group_gain(y,calibrated['4b'],calibrated['8b'],groups,dict(cfg,seed=cfg['bootstrap_seed']))
        result['primary_comparison']='Raw paired loss, subject to matched exposure; temperature CV is a fixed secondary calibration diagnostic, not an alternative promotion gate'
        np.savez_compressed(run/'size_comparison.npz',ids=dev.id,labels=y,groups=groups,small=predictions['4b'],large=predictions['8b'],calibrated_small=calibrated['4b'],calibrated_large=calibrated['8b'])
    result.update(cpu_seconds=time.monotonic()-start,peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,checked_at=utc_now(),
        comparison_limitations=['Two pretrained family checkpoints also differ in architecture and potentially pretraining; matched finetuning exposure does not isolate parameter count alone.',
            'Per-device microbatch and accumulation differ to fit memory; report their numerical recipe and actual accepted exposure.',
            'Only32accepted updates from fresh classifier heads on1024rows, with512 historically inspected development examples; initial128-update draft stopped before GPU on runtime planning evidence; no full-training or official-score estimate follows.'],
        decision='Short fresh-head capacity/throughput pilot only; no full-training quality or official-score claim. Full8Btraining needs a new bounded compute plan.')
    for size,arm in result['arms'].items():
        if arm.get('complete'):
            arm['projection_seconds']=1.2*(arm['prediction_seconds_per_row']*25000+arm['setup_save_overhead_seconds'])+600
    gain=result.get('raw_paired_gain',{})
    memory_ok=all(a.get('complete') and all(r['hardware']['peak_memory_fraction']<=.95 for r in a.get('per_rank',[])) for a in result['arms'].values()) and len(result['arms'])==2
    raw_gate=bool(gain and gain['gain']>=cfg['advancement_min_gain'] and gain['interval_95'][0]>0)
    runtime_ok=bool(result['arms'].get('8b',{}).get('complete') and result['arms']['8b']['projection_seconds']<32400)
    result.update(raw_quality_gate_passed=raw_gate,memory_guard_passed=memory_ok,runtime_screen_passed=runtime_ok,
        longer_training_justified=bool(result['complete'] and result['quality_comparison_valid'] and raw_gate and memory_ok and runtime_ok),
        official_submission_eligible=False,protocol_revision=2,goal_achieved=False)
    result['decision']='separately_budget_full_training' if result['longer_training_justified'] else 'stop_no_qualified_size_advantage'
    atomic_json(pub/'H02_results.json',result);print({k:v for k,v in result.items() if k!='arms'},flush=True)


if __name__=='__main__':main()
