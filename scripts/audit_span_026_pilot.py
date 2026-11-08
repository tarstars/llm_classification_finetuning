"""Prespecified engineering checks, then descriptive loss and regression guard."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,CLASSES,prompt_groups,parse_turns
from arena.experiment import atomic_json,read_json,sha256,utc_now,verify_sources
from arena.span019_quality import metrics,paired_interval
from arena.span026_quality import gain_bounds
from arena.span026_runtime import projection
from arena.span010_order import validate_probabilities


def main():
    pub=REPO/'results/span_026';root=ROOT/'runs/span_026';role='context_pilot'
    assert not (pub/'pilot_audit.json').exists(),'Preserve first fixed comparison'
    bundle=read_json(pub/(role+'_bundle.json'));verify_sources(REPO,bundle['source_sha256'])
    job=root/'jobs'/role;out=job/'output/context_audit'
    assert read_json(job/'status.json')['status']=='COMPLETE'
    assert read_json(pub/'jobs'/role/'source_verification.json')['passed']
    local=read_json(root/'notebooks'/role/'kernel-metadata.json')
    remote=read_json(root/'verified_sources'/role/'kernel-metadata.json')
    for k in ['docker_image','machine_shape','is_private','enable_gpu','enable_internet']:assert local[k]==remote[k]
    assert sorted('/'.join(x.split('/')[:2]) for x in local['dataset_sources'])==sorted('/'.join(x.split('/')[:2]) for x in remote['dataset_sources'])
    for ref in remote['dataset_sources']:assert len(ref.split('/'))==2 or ref.split('/')[2:]==['1']
    retrieval=read_json(job/'retrieval.json')
    for n,h in retrieval['artifact_sha256'].items():assert sha256(job/'output'/n)==h
    assert read_json(job/'output/session_summary.json')['exit_code']==0
    hardware=read_json(job/'output/hardware.json');assert len(hardware['gpus'])==2 and all('T4' in x for x in hardware['gpus'])
    spec=read_json(root/'pilot_spec.json');actual_spec=read_json(out/'runtime_spec.json')
    for key,value in spec.items():assert actual_spec[key]==value
    assert actual_spec['frozen_spec_sha256']==sha256(root/'pilot_spec.json')
    probe=read_json(out/'engineering_probe.json');done=read_json(out/'information_complete.json')
    # Replay the engineering decision from measured cohorts before reading labels.
    summary=read_json(out/'m0_summary.json');timings=summary['cohorts'];parent=spec['parent_runtime']
    startup=max(0,summary['cold_seconds']-parent['g_cold_seconds'])+max(0,timings['warmup']['seconds']-parent['g_warmup_seconds'])
    expected_probe=projection(parent['projection_seconds'],parent['g_uniform_seconds'],parent['worst_g_uniform_seconds'],
        timings['uniform']['seconds'],timings['context_probe']['seconds'],timings['context_probe']['rows'],
        spec['changed_fraction'],timings['context_warmup']['seconds'],actual_spec['runtime_selector_seconds'],
        extra_parent_startup_seconds=startup)
    for key,value in expected_probe.items():
        if key!='engineering_screen_passed':assert probe[key]==value
    totals=summary['device_total_memory_bytes'];assert len(totals)==2 and all(x>0 for x in totals)
    fractions=[timings['context_probe']['peak_allocated_bytes'][g]/totals[g] for g in range(2)]
    assert probe['memory_allocated_fractions']==fractions
    memory_pass=all(x<=spec['memory_allocated_fraction_cap'] for x in fractions)
    assert probe['memory_guard_passed']==memory_pass
    assert probe['engineering_screen_passed']==(expected_probe['engineering_screen_passed'] and memory_pass)
    models=['m0','m3','m0_reverse'] if done['complete'] else ['m0']
    controls={};arrays={}
    for model in models:
        s=read_json(out/(model+'_summary.json'))
        assert s['loading_exact'] and s['versions']['python'].startswith('3.10.')
        assert s['versions']['torch'].split('+')[0]=='2.1.2' and s['versions']['transformers']=='4.42.3'
        assert read_json(out/(model+'_checkpoint.json'))==spec['checkpoint_references'][model]
        for name,details in s['cohorts'].items():
            a=np.load(out/(model+'_'+name+'.npz'));expected=spec['token_references'][model][name]
            assert a['ids'].tolist()==expected['ids'] and a['lengths'].tolist()==expected['lengths']
            assert details['token_ids_sha256']==expected['token_ids_sha256']
            validate_probabilities(a['probabilities'],len(a['ids']))
            assert int(a['lengths'].max())<=s['max_position_embeddings']
        c=np.load(out/(model+'_control.npz'));old=np.load(root/'controls'/(model+'.npz'))
        assert np.array_equal(c['ids'],old['ids'])
        diff=float(np.max(np.abs(c['probabilities']-old['probabilities'])));assert diff<=1e-5
        controls[model]=dict(rows=len(c['ids']),max_abs_probability_difference=diff)
        if done['complete']:arrays[model]=np.load(out/(model+'_changed_old.npz'))['probabilities']
    if not probe['engineering_screen_passed']:
        assert not done['complete'] and done['stopped_for_engineering']
        r=dict(at=utc_now(),complete_information=False,engineering_screen_passed=False,probe=probe,
            controls=controls,labels_read=False,quality_measured=False,decision='stop_6k_engineering',
            official_submission_eligible=False,source_sha256=bundle['source_sha256'])
        atomic_json(pub/'pilot_audit.json',r);print(json.dumps(r,indent=2));return
    assert done['complete'] and done['rows']==len(spec['changed_ids'])==357
    # Human targets are first read only after accepted source, tokens and controls.
    csv=ROOT/'raw/train.csv';assert sha256(csv)==spec['train_csv_sha256']
    df=pd.read_csv(csv);pos=df.set_index('id').index.get_indexer(spec['changed_ids']);assert np.all(pos>=0)
    yall=df[CLASSES].to_numpy().argmax(1);y=yall[pos]
    extended=np.load(out/'m0_changed_extended.npz')['probabilities']
    lower,upper,p= gain_bounds(y,arrays['m0'],arrays['m0_reverse'][:,[1,0,2]],extended,arrays['m3'][:,[1,0,2]])
    full=np.zeros(len(df));full[pos]=lower
    groups=prompt_groups(pd.DataFrame({'prompt':df.prompt.map(parse_turns)})).to_numpy()
    interval=paired_interval(full,groups);passes=bool(full.mean()>=-.005)
    target=root/'context_information_analysis.npz';assert not target.exists()
    np.savez_compressed(target,ids=df.id.iloc[pos].to_numpy(),labels=y,groups=groups[pos],
        lower_gain=lower,upper_gain=upper,**p)
    r=dict(at=utc_now(),complete_information=True,engineering_screen_passed=True,probe=probe,controls=controls,
        labels_read=True,quality_measured=True,rows=len(y),full_original_rows=len(df),
        weighted_lower_gain=float(lower.sum()/len(df)),weighted_upper_gain=float(upper.sum()/len(df)),
        weighted_lower_gain_paired_group95=interval,bootstrap_seed=42,bootstrap_resamples=2000,
        class_counts=np.bincount(y,minlength=3).tolist(),slice_metrics={k:metrics(y,v) for k,v in p.items()},
        fixed_gross_regression_threshold=-.005,gross_regression_guard_passed=passes,
        local_positive_gain_required=False,original_training_overlap=True,independent_cv=False,hidden_score_forecast=False,
        decision='prepare_actual_full_qualification' if passes else 'stop_gross_regression',
        official_submission_eligible=False,analysis_sha256=sha256(target),source_sha256=bundle['source_sha256'],
        retrieval_sha256=sha256(job/'retrieval.json'))
    atomic_json(pub/'pilot_audit.json',r);print(json.dumps(r,indent=2))


if __name__=='__main__':main()
