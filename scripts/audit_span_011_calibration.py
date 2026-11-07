"""Audit exact accepted source, raw controls, alignment, CSV and original-cohort timing."""
from pathlib import Path
import argparse
import json
import sys

import nbformat
import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,sha256,verify_sources,utc_now
from arena.span004_support import api_client,read_call
from arena.span011_calibration import ensemble,temperature_scale,validate_probabilities,CLASSES


def runtime_estimate(summaries,session_seconds):
    seconds=sum(s['cohorts']['uniform']['seconds'] for s in summaries.values())
    cold_warm=sum(s['cold_seconds']+s['cohorts']['warmup']['seconds'] for s in summaries.values())
    measured=sum(s['cold_seconds']+sum(c['seconds'] for c in s['cohorts'].values()) for s in summaries.values())
    assert session_seconds>=measured-1,'Session timing contradicts sequential model timing'
    other=max(0.,session_seconds-measured)
    worker_projection=25000*seconds/512*1.2+cold_warm+600
    return dict(projection_seconds=worker_projection+other,startup_seconds=cold_warm+other,
        cold_and_warmup_seconds=cold_warm,other_session_overhead_seconds=other,
        worker_reported_projection_seconds=worker_projection,uniform_rows=512,measured_here=True,
        session_seconds=session_seconds,includes_all_observed_session_overhead=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['learning','confirmation'],default='learning')
    parser.add_argument('--source-only',action='store_true');args=parser.parse_args()
    stage=args.stage;role='calibration_'+stage;root=ROOT/'runs/span_011';pub=REPO/'results/span_011';job=root/'jobs'/role
    bundle=read_json(pub/(role+'_bundle.json'));verify_sources(REPO,bundle['source_sha256'])
    push=read_json(job/'push.json');exact=push['ref'].removeprefix('/code/')+'/'+str(push['version_number'])
    original=root/'notebooks'/role/'job.ipynb';assert sha256(original)==bundle['notebook_sha256']
    api=api_client();folder=root/'audited_sources'/role;folder.mkdir(parents=True,exist_ok=True)
    read_call(lambda:api.kernels_pull(exact,path=str(folder),metadata=True,quiet=True))
    notebooks=list(folder.glob('*.ipynb'));assert len(notebooks)==1
    def cells(path):return [(c.cell_type,c.source) for c in nbformat.read(path,4).cells]
    assert cells(original)==cells(notebooks[0]),'Accepted source differs from frozen notebook'
    metadata=read_json(folder/'kernel-metadata.json');local=read_json(root/'notebooks'/role/'kernel-metadata.json')
    assert metadata['enable_internet'] is False and metadata['docker_image']==local['docker_image']
    assert sorted(x.removesuffix('/1') for x in local['dataset_sources'])==sorted(metadata['dataset_sources'])
    proof=dict(passed=True,at=utc_now(),kernel_version=exact,notebook_sha256=bundle['notebook_sha256'],
        cell_sources_identical=True,internet_enabled=False,remote_docker_image=metadata['docker_image'],
        remote_dataset_sources=metadata['dataset_sources'],temperature=bundle['temperature'])
    atomic_json(pub/(role+'_source_verification.json'),proof)
    if args.source_only:print(json.dumps(proof));return
    status=str(read_call(lambda:api.kernels_status(exact)).status).split('.')[-1]
    assert status=='COMPLETE','Preserve pending/failed execution; no quality admission'
    receipt=read_json(job/'retrieval.json')
    for name,digest in receipt['artifact_sha256'].items():assert sha256(job/'output'/name)==digest
    out=job/'output/calibration';qualification=read_json(out/'qualification.json')
    assert qualification['complete'] and qualification['experiment_stage']==stage
    assert qualification['base_weights']==[.5,.5] and qualification['packing_tokens']==8192 and qualification['conversation_cap']==4096
    assert not qualification['training_or_target_fields_read']
    cohort=read_json(pub/'cohort_manifest.json');ids=cohort['split_ids'][stage]
    mapping=pd.read_parquet(root/'cohort'/(stage+'_mapping.parquet'))
    assert sha256(root/'cohort'/(stage+'_mapping.parquet'))==cohort['artifact_sha256'][stage+'_mapping.parquet']
    previous=ROOT/'runs/span_009/jobs/public_ensemble_resume/output/public_ensemble'
    identities={};summaries={}
    for model in ['m0','m3']:
        assert read_json(out/(model+'_checkpoint.json'))==read_json(previous/(model+'_checkpoint.json'))
        actual=np.load(out/(model+'_control.npz'));old=np.load(previous/(model+'_uniform.npz'))
        assert np.array_equal(actual['ids'],old['ids']) and np.array_equal(actual['lengths'],old['lengths'])
        delta=float(np.max(np.abs(actual['probabilities']-old['probabilities'])))
        identities[model]=dict(rows=256,max_abs_probability_difference=delta,passed=delta<1e-5)
        summary=read_json(out/(model+'_summary.json'));summaries[model]=summary;assert summary['loading_exact']
        assert summary['cohorts']['control']['token_ids_sha256']==read_json(previous/(model+'_summary.json'))['cohorts']['uniform']['token_ids_sha256']
        for value in summary['cohorts'].values():
            if value['rows']:assert len(value['peak_allocated_bytes'])==2 and all(n>0 for n in value['peak_allocated_bytes'])
    expected={'control':np.load(previous/'m0_uniform.npz')['ids'],stage:np.asarray(ids),
              'test':pd.read_csv(ROOT/'raw/test.csv',usecols=['id']).id.to_numpy()}
    if stage=='learning':
        selected=read_json(out/'profile_selection.json');old_selection=read_json(REPO/'results/span_010/profile_selection.json')
        for name in ['control','uniform','long']:assert selected[name+'_ids']==old_selection[name+'_ids']
        for name in ['uniform','long']:expected[name]=np.asarray(selected[name+'_ids'])
    for name,expected_ids in expected.items():
        a=np.load(out/('m0_'+name+'.npz'));b=np.load(out/('m3_'+name+'.npz'));actual=np.load(out/('ensemble_'+name+'.npz'))
        assert all(np.array_equal(data['ids'],expected_ids) for data in [a,b,actual])
        base=ensemble(a['probabilities'],b['probabilities'])
        np.testing.assert_array_equal(base,actual['base_probabilities'])
        np.testing.assert_allclose(temperature_scale(base,bundle['temperature']),actual['probabilities'],rtol=1e-13,atol=1e-14)
        if name==stage:np.testing.assert_array_equal(a['lengths'],mapping.prepared_tokens)
    csv=job/'output/submission.csv';submission=pd.read_csv(csv)
    assert submission.columns.tolist()==['id',*CLASSES] and np.array_equal(submission.id,expected['test'])
    validate_probabilities(submission[CLASSES].to_numpy(),len(submission))
    np.testing.assert_allclose(submission[CLASSES],np.load(out/'ensemble_test.npz')['probabilities'],atol=1e-7)
    hardware=read_json(job/'output/hardware.json')
    assert len(hardware['gpus'])==2 and all('T4' in name for name in hardware['gpus'])
    assert hardware['python'].startswith('3.10.')
    if stage=='learning':
        timing=runtime_estimate(summaries,read_json(job/'output/session_summary.json')['seconds'])
        assert np.isclose(timing['worker_reported_projection_seconds'],qualification['projected_hidden_seconds'])
    else:
        learning=read_json(pub/'calibration_learning_qualification.json')
        assert learning['all_controls_passed'] and learning['runtime_gate_passed']
        assert sha256(pub/'frozen_temperature.json')==bundle['frozen_temperature_sha256']
        timing=dict(learning['runtime'],measured_here=False)
        assert read_json(pub/'calibration_learning_bundle.json')['source_sha256']['src/arena/span011_calibration.py']==bundle['source_sha256']['src/arena/span011_calibration.py']
    cap=read_json(REPO/'configs/span_011.json')['job_seconds_caps'][role]
    assert read_json(job/'output/session_summary.json')['seconds']<=cap
    result=dict(at=utc_now(),complete=True,role=role,source=proof,control_identity=identities,
        all_controls_passed=all(v['passed'] for v in identities.values()),csv_verified=True,csv_sha256=sha256(csv),
        both_t4_verified=True,rows_scored=len(ids),artifact_files_verified=len(receipt['artifact_sha256']),
        runtime=timing,runtime_gate_passed=timing['projection_seconds']<32400,measured_roles=summaries,
        confirmation_labels_inspected=False,quality_not_evaluated=True)
    atomic_json(pub/(role+'_qualification.json'),result)
    print(json.dumps({k:result[k] for k in ['complete','all_controls_passed','runtime_gate_passed','runtime','rows_scored']}))


if __name__=='__main__':main()
