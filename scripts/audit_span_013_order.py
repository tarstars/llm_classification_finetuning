"""Audit exact accepted source and the actual fixed Gemma-only deployment."""
from pathlib import Path
import argparse
import json
import sys

import nbformat
import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,parse_turns,prompt_groups
from arena.experiment import read_json,atomic_json,sha256,verify_sources,utc_now
from arena.span004_support import api_client,read_call
from arena.span010_order import route_indices,ensemble,validate_probabilities,CLASSES
from arena.span013_order import complete_gemma_order
from audit_span_010_order import loss,paired_interval


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-only',action='store_true')
    parser.add_argument('--role',choices=['gemma_order','gemma_order_followup'],default='gemma_order');args=parser.parse_args()
    role=args.role;root=ROOT/'runs/span_013';pub=REPO/'results/span_013';job=root/'jobs'/role
    bundle=read_json(pub/(role+'_bundle.json'));verify_sources(REPO,bundle['source_sha256'])
    push=read_json(job/'push.json');exact=push['ref'].removeprefix('/code/')+'/'+str(push['version_number'])
    original=root/'notebooks'/role/'job.ipynb';assert sha256(original)==bundle['notebook_sha256']
    api=api_client();folder=root/'audited_sources'/role;folder.mkdir(parents=True,exist_ok=True)
    read_call(lambda:api.kernels_pull(exact,path=str(folder),metadata=True,quiet=True))
    copies=list(folder.glob('*.ipynb'));assert len(copies)==1
    def cells(path):return [(c.cell_type,c.source) for c in nbformat.read(path,4).cells]
    assert cells(original)==cells(copies[0]),'Accepted source differs from frozen notebook'
    metadata=read_json(folder/'kernel-metadata.json');local=read_json(root/'notebooks'/role/'kernel-metadata.json')
    assert metadata['enable_internet'] is False and metadata['docker_image']==local['docker_image']
    assert sorted(x.removesuffix('/1') for x in local['dataset_sources'])==sorted(metadata['dataset_sources'])
    source=dict(at=utc_now(),passed=True,kernel_version=exact,notebook_sha256=bundle['notebook_sha256'],
                cell_sources_identical=True,internet_enabled=False,remote_docker_image=metadata['docker_image'])
    atomic_json(pub/(role+'_source_verification.json'),source)
    if args.source_only:print(json.dumps(source));return
    assert str(read_call(lambda:api.kernels_status(exact)).status).split('.')[-1]=='COMPLETE'
    receipt=read_json(job/'retrieval.json')
    for name,digest in receipt['artifact_sha256'].items():assert sha256(job/'output'/name)==digest
    out=job/'output/order_completion';done=read_json(out/'qualification.json')
    assert done['complete'] and done['qualify'] and not done['uses_targets'] and not done['independent_local_quality']
    assert done['inference_roles']==['m0','m3','m0_reverse'] and done['packing_tokens']==8192 and done['temperature']==1
    assert done['warmup_rows']==80 and done['routed_weights']==[.25,.25,.5]
    selection=read_json(pub/'profile_selection.json')
    assert selection==read_json(REPO/'results/span_010/profile_selection.json')==read_json(out/'profile_selection.json')
    test=pd.read_csv(ROOT/'raw/test.csv',usecols=['id'])
    expected={n:np.asarray(selection[n+'_ids']) for n in ['control','uniform','long']};expected['test']=test.id.to_numpy()
    old=ROOT/'runs/span_010/jobs/order_pilot/output/order_completion'
    prior=ROOT/'runs/span_009/jobs/public_ensemble_resume/output/public_ensemble'
    roles=['m0','m3','m0_reverse'];summaries={r:read_json(out/(r+'_summary.json')) for r in roles}
    identities={};coverage={}
    for role_name,summary in summaries.items():
        assert summary['loading_exact'] and summary['cohorts']['warmup']['rows']==80
        reference_role='m0' if role_name=='m0_reverse' else role_name
        assert read_json(out/(role_name+'_checkpoint.json'))==read_json(prior/(reference_role+'_checkpoint.json'))
        for value in summary['cohorts'].values():
            if value['rows']:assert len(value['peak_allocated_bytes'])==2 and all(v>0 for v in value['peak_allocated_bytes'])
    assert set(summaries['m0_reverse']['cohorts'])=={'warmup',*['routed_'+n for n in expected]}
    train=pd.read_csv(ROOT/'raw/train.csv',usecols=['id','prompt',*CLASSES]).set_index('id')
    for name,ids in expected.items():
        a=np.load(out/('m0_'+name+'.npz'));b=np.load(out/('m3_'+name+'.npz'))
        for model,actual in [('m0',a),('m3',b)]:
            archived=np.load(old/(model+'_'+name+'.npz'))
            assert np.array_equal(actual['ids'],ids) and np.array_equal(actual['lengths'],archived['lengths'])
            assert summaries[model]['cohorts'][name]['token_ids_sha256']==read_json(old/(model+'_summary.json'))['cohorts'][name]['token_ids_sha256']
            if name=='control':
                original_control=np.load(prior/(model+'_uniform.npz'))
                assert np.array_equal(actual['ids'],original_control['ids'])
                diff=float(np.max(np.abs(actual['probabilities']-original_control['probabilities'])))
                identities[model]=dict(rows=len(ids),max_abs_probability_difference=diff,passed=diff<1e-5)
        selected,js,work=route_indices(ids,a['probabilities'],b['probabilities'][:,[1,0,2]],np.c_[a['lengths'],b['lengths']])
        route=np.load(out/('routing_'+name+'.npz'))
        np.testing.assert_array_equal(route['ids'],ids);np.testing.assert_array_equal(route['selected_indices'],selected)
        np.testing.assert_allclose(route['disagreement'],js,rtol=1e-12,atol=1e-14);np.testing.assert_array_equal(route['work'],work)
        assert len(selected)<=int(.2*len(ids)) and work[selected].sum()<=.1*work.sum()
        reverse=np.load(out/('m0_reverse_routed_'+name+'.npz'));archived=np.load(old/('m0_reverse_routed_'+name+'.npz'))
        assert np.array_equal(reverse['ids'],ids[selected]) and np.array_equal(reverse['ids'],archived['ids'])
        assert np.array_equal(reverse['lengths'],archived['lengths'])
        if len(selected):
            assert summaries['m0_reverse']['cohorts']['routed_'+name]['token_ids_sha256']==read_json(old/'m0_reverse_summary.json')['cohorts']['routed_'+name]['token_ids_sha256']
        diff=float(np.max(np.abs(reverse['probabilities']-archived['probabilities']))) if len(selected) else 0.
        identities['reverse_'+name]=dict(rows=len(selected),max_abs_probability_difference=diff,passed=diff<1e-5)
        candidate=complete_gemma_order(a['probabilities'],b['probabilities'],selected,reverse['probabilities'])
        saved=np.load(out/('ensemble_'+name+'.npz'));np.testing.assert_array_equal(saved['ids'],ids)
        np.testing.assert_array_equal(saved['probabilities'],candidate)
        coverage[name]=dict(rows=len(ids),selected_rows=len(selected),selected_work_fraction=float(work[selected].sum()/work.sum()),
                            route_replayed=True,mixture_replayed=True,independent=False)
        if name!='test':
            frame=train.loc[ids];y=frame[CLASSES].to_numpy().argmax(1);base=ensemble(a['probabilities'],b['probabilities'])
            delta=loss(y,base)-loss(y,candidate);groups=prompt_groups(frame.assign(prompt=frame.prompt.map(parse_turns))).to_numpy()
            coverage[name].update(baseline_log_loss=float(loss(y,base).mean()),candidate_log_loss=float(loss(y,candidate).mean()),
                loss_reduction=float(delta.mean()),descriptive_paired_group95=paired_interval(delta,groups),
                class_loss_reduction={CLASSES[k]:float(delta[y==k].mean()) for k in range(3) if np.any(y==k)})
    session=read_json(job/'output/session_summary.json');cap=read_json(REPO/'configs/span_013.json')['job_seconds_caps'][role]
    assert session['exit_code']==0 and session['seconds']<=cap
    cold=sum(s['cold_seconds'] for s in summaries.values());warm=sum(s['cohorts']['warmup']['seconds'] for s in summaries.values())
    measured=cold+sum(c['seconds'] for s in summaries.values() for c in s['cohorts'].values())
    assert session['seconds']>=measured-1
    other=max(0.,session['seconds']-measured)
    base=sum(summaries[r]['cohorts']['uniform']['seconds'] for r in ['m0','m3'])
    reverse=summaries['m0_reverse']['cohorts']['routed_uniform']['seconds']
    projection=25000*(base+reverse)/512*1.2+cold+warm+other+600
    hardware=read_json(job/'output/hardware.json')
    assert len(hardware['gpus'])==2 and all('T4' in n for n in hardware['gpus']) and hardware['python'].startswith('3.10.')
    csv=job/'output/submission.csv';submission=pd.read_csv(csv)
    assert submission.columns.tolist()==['id',*CLASSES] and np.array_equal(submission.id,test.id)
    validate_probabilities(submission[CLASSES].to_numpy(),len(test))
    np.testing.assert_allclose(submission[CLASSES],np.load(out/'ensemble_test.npz')['probabilities'],atol=1e-7)
    result=dict(at=utc_now(),role=role,complete=True,source=source,all_controls_passed=all(v['passed'] for v in identities.values()),
        control_identity=identities,coverage=coverage,independent_local_quality=False,artifact_files_verified=len(receipt['artifact_sha256']),
        measured_roles=summaries,session_seconds=session['seconds'],runtime_gate_passed=projection<32400,
        runtime=dict(projection_seconds=projection,base_uniform_seconds=base,reverse_routed_uniform_seconds=reverse,
                     cold_seconds=cold,full_warmup_seconds=warm,other_seconds=other,margin=1.2,reserve_seconds=600),
        both_t4_verified=True,csv_verified=True,csv_sha256=sha256(csv))
    atomic_json(pub/(role+'_qualification.json'),result)
    print(json.dumps({k:result[k] for k in ['complete','all_controls_passed','runtime_gate_passed','runtime']}))


if __name__=='__main__':main()
