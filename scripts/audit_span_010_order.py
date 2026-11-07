"""Audit accepted source, exact controls, order routing and measured runtime."""
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
from arena.kaggle_control import output_catalog
from arena.span004_support import api_client,read_call
from arena.span010_order import route_indices,complete_orders,ensemble,validate_probabilities,CLASSES


def loss(y,p):return -np.log(np.clip(p[np.arange(len(y)),y],1e-15,1))


def paired_interval(delta,groups):
    _,inverse=np.unique(groups,return_inverse=True)
    totals=np.bincount(inverse,weights=delta);counts=np.bincount(inverse)
    rng=np.random.default_rng(42);draws=[]
    for _ in range(2000):
        sampled=rng.integers(0,len(counts),len(counts))
        draws.append(float(totals[sampled].sum()/counts[sampled].sum()))
    return np.quantile(draws,[.025,.975]).tolist()


def write_qualification(directory,role,result):
    if role not in ['order_pilot','order_packed'] or result.get('role')!=role:
        raise ValueError('Qualification must identify its actual experiment attempt')
    atomic_json(Path(directory)/('H57_'+role+'_qualification.json'),result)
    atomic_json(Path(directory)/'H57_qualification.json',result)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-only',action='store_true')
    parser.add_argument('--role',choices=['order_pilot','order_packed'],default='order_pilot');args=parser.parse_args()
    role=args.role
    root=ROOT/'runs/span_010';pub=REPO/'results/span_010';job=root/'jobs'/role
    bundle=read_json(pub/(role+'_bundle.json'));verify_sources(REPO,bundle['source_sha256'])
    push=read_json(job/'push.json');exact=push['ref'].removeprefix('/code/')+'/'+str(push['version_number'])
    original=root/'notebooks'/role/'job.ipynb';assert sha256(original)==bundle['notebook_sha256']
    api=api_client();folder=root/'audited_sources'/role;folder.mkdir(parents=True,exist_ok=True)
    read_call(lambda:api.kernels_pull(exact,path=str(folder),metadata=True,quiet=True))
    notebooks=list(folder.glob('*.ipynb'));assert len(notebooks)==1
    cells=lambda p:[(c.cell_type,c.source) for c in nbformat.read(p,4).cells]
    assert cells(original)==cells(notebooks[0]),'Accepted code differs from frozen notebook'
    metadata=read_json(folder/'kernel-metadata.json')
    local_metadata=read_json(root/'notebooks'/role/'kernel-metadata.json')
    assert metadata['enable_internet'] is False and metadata['docker_image']==local_metadata['docker_image']
    assert sorted(x.rsplit('/1',1)[0] for x in local_metadata['dataset_sources'])==sorted(metadata['dataset_sources'])
    proof=dict(passed=True,at=utc_now(),kernel_version=exact,notebook_sha256=bundle['notebook_sha256'],
        cell_sources_identical=True,internet_enabled=False,remote_docker_image=metadata['docker_image'],
        remote_dataset_sources=metadata['dataset_sources'])
    atomic_json(pub/(role+'_source_verification.json'),proof)
    if args.source_only:
        print(json.dumps(proof));return
    status=str(read_call(lambda:api.kernels_status(exact)).status).split('.')[-1]
    assert status=='COMPLETE','Preserve failed/pending service state; no quality admission'
    retrieval=read_json(job/'retrieval.json')
    for name,digest in retrieval['artifact_sha256'].items():assert sha256(job/'output'/name)==digest
    out=job/'output/order_completion';q=read_json(out/'qualification.json')
    assert q['complete'] and q['qualify'] and not q['independent_local_quality']
    selected=read_json(pub/'profile_selection.json');assert read_json(out/'profile_selection.json')==selected
    test=pd.read_csv(ROOT/'raw/test.csv',usecols=['id'])
    expected={name:np.asarray(selected[name+'_ids']) for name in ['control','uniform','long']}
    expected['test']=test.id.to_numpy();records={};metrics=[];identity={}
    train=pd.read_csv(ROOT/'raw/train.csv',usecols=['id','prompt',*CLASSES]).set_index('id')
    for name,ids in expected.items():
        frame=train.loc[ids] if name!='test' else None
        a=np.load(out/('m0_'+name+'.npz'));b=np.load(out/('m3_'+name+'.npz'))
        actual=np.load(out/('ensemble_'+name+'.npz'))
        assert all(np.array_equal(x['ids'],ids) for x in [a,b,actual])
        indices,js,work=route_indices(ids,a['probabilities'],b['probabilities'][:,[1,0,2]],np.c_[a['lengths'],b['lengths']])
        saved=np.load(out/('routing_'+name+'.npz'))
        np.testing.assert_array_equal(indices,saved['selected_indices'])
        # NumPy/libm versions can differ at float64 roundoff in log(). Route
        # indices remain exact; this tolerance is only for the JS diagnostic.
        np.testing.assert_allclose(js,saved['disagreement'],rtol=1e-12,atol=1e-14)
        np.testing.assert_array_equal(work,saved['work'])
        ar=np.load(out/('m0_reverse_routed_'+name+'.npz'));br=np.load(out/('m3_reverse_routed_'+name+'.npz'))
        assert all(np.array_equal(x['ids'],ids[indices]) for x in [ar,br])
        calculated=complete_orders(a['probabilities'],b['probabilities'],indices,ar['probabilities'],br['probabilities'])
        np.testing.assert_array_equal(calculated,actual['probabilities'])
        assert len(indices)<=int(.2*len(ids)) and work[indices].sum()<=.1*work.sum()
        records[name]=dict(rows=len(ids),selected_rows=len(indices),route_replayed=True,mixture_replayed=True,
            max_abs_js_replay_difference=float(np.max(np.abs(js-saved['disagreement']))),
            selected_work_fraction=float(work[indices].sum()/work.sum()),mean_disagreement=float(js.mean()),
            selected_mean_disagreement=float(js[indices].mean()) if len(indices) else None)
        if frame is not None:
            y=frame[CLASSES].to_numpy().argmax(1)
            base=ensemble(a['probabilities'],b['probabilities'])
            all_ar=np.load(out/('m0_reverse_'+name+'.npz'));all_br=np.load(out/('m3_reverse_'+name+'.npz'))
            assert np.array_equal(all_ar['ids'],ids) and np.array_equal(all_br['ids'],ids)
            variants=dict(base=base,candidate=calculated,
                gemma_tta=(a['probabilities']+all_ar['probabilities'][:,[1,0,2]])/2,
                llama_tta=(b['probabilities'][:,[1,0,2]]+all_br['probabilities'])/2,
                full_four=(base+(all_ar['probabilities'][:,[1,0,2]]+all_br['probabilities'])/2)/2)
            for model,p in variants.items():
                validate_probabilities(p,len(ids))
                metrics.append(dict(cohort=name,model=model,rows=len(ids),descriptive_log_loss=float(loss(y,p).mean()),
                    accuracy=float((p.argmax(1)==y).mean()),independent=False,
                    class_log_loss={CLASSES[k]:float(loss(y,p)[y==k].mean()) for k in range(3) if np.any(y==k)}))
            groups=prompt_groups(frame.assign(prompt=frame.prompt.map(parse_turns))).to_numpy()
            delta=loss(y,base)-loss(y,calculated)
            records[name].update(descriptive_loss_reduction=float(delta.mean()),
                descriptive_paired_group95=paired_interval(delta,groups),
                independent=False,original_votes_seen_in_checkpoint_training=True)
    prior=ROOT/'runs/span_009/jobs/public_ensemble_resume/output/public_ensemble'
    for model_role in ['m0','m3']:
        loaded=read_json(out/(model_role+'_checkpoint.json'))
        assert loaded==read_json(prior/(model_role+'_checkpoint.json'))
        assert loaded==read_json(out/(model_role+'_reverse_checkpoint.json'))
        a=np.load(out/(model_role+'_control.npz'));b=np.load(prior/(model_role+'_uniform.npz'))
        assert np.array_equal(a['ids'],b['ids']) and np.array_equal(a['lengths'],b['lengths'])
        difference=float(np.max(np.abs(a['probabilities']-b['probabilities'])))
        identity[model_role]=dict(rows=256,max_abs_probability_difference=difference,passed=difference<1e-5)
    summaries={r:read_json(out/(r+'_summary.json')) for r in ['m0','m3','m3_reverse','m0_reverse']}
    for s in summaries.values():
        assert s['loading_exact']
        for c in s['cohorts'].values():
            if c['rows']:assert len(c['peak_allocated_bytes'])==2 and all(x>0 for x in c['peak_allocated_bytes'])
    base=sum(summaries[r]['cohorts']['uniform']['seconds'] for r in ['m0','m3'])/512
    reverse=sum(summaries[r]['cohorts']['routed_uniform']['seconds'] for r in ['m3_reverse','m0_reverse'])/512
    startup=sum(s['cold_seconds']+s['cohorts']['warmup']['seconds'] for s in summaries.values())
    projection=25000*(base+reverse)*1.2+startup+600
    assert np.isclose(projection,q['projected_hidden_seconds'])
    csv=job/'output/submission.csv';submission=pd.read_csv(csv)
    assert submission.columns.tolist()==['id',*CLASSES] and np.array_equal(submission.id,test.id)
    validate_probabilities(submission[CLASSES].to_numpy(),len(test))
    np.testing.assert_allclose(submission[CLASSES],np.load(out/'ensemble_test.npz')['probabilities'],atol=1e-7)
    result=dict(at=utc_now(),role=role,complete=True,source=proof,coverage=records,control_identity=identity,
        all_controls_passed=all(x['passed'] for x in identity.values()),descriptive_metrics=metrics,
        independent_local_quality=False,runtime_projection_seconds=projection,runtime_gate_passed=projection<32400,
        csv_sha256=sha256(csv),csv_verified=True,both_t4_verified=True,
        artifact_files_verified=len(retrieval['artifact_sha256']),measured_roles=summaries,
        base_seconds_per_row=base,reverse_seconds_per_all_row=reverse,startup_seconds=startup)
    assert role==args.role
    write_qualification(pub,role,result)
    print(json.dumps({k:result[k] for k in ['complete','all_controls_passed','runtime_projection_seconds','runtime_gate_passed','both_t4_verified']}))


if __name__=='__main__':main()
