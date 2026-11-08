"""Replay actual H72 routing/components and qualify all-role deployment runtime."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,parse_turns,prompt_groups
from arena.experiment import atomic_json,read_json,sha256,verify_sources,utc_now
from arena.span010_order import CLASSES,route_indices,validate_probabilities
from arena.span013_order import complete_gemma_order
from arena.span026_candidate import replace_original
from arena.span026_qualification import deployment_runtime
from arena.span019_quality import compare


def main():
    role='context_qualification';pub=REPO/'results/span_026';root=ROOT/'runs/span_026'
    assert not (pub/'candidate_qualification.json').exists(),'Preserve initial full qualification'
    bundle=read_json(pub/(role+'_bundle.json'));verify_sources(REPO,bundle['source_sha256'])
    job=root/'jobs'/role;out=job/'output/context_candidate'
    assert read_json(job/'status.json')['status']=='COMPLETE'
    proof=read_json(pub/'jobs'/role/'source_verification.json');assert proof['passed']
    assert proof['notebook_sha256']==bundle['notebook_sha256']
    local=read_json(root/'notebooks'/role/'kernel-metadata.json');remote=read_json(root/'verified_sources'/role/'kernel-metadata.json')
    for k in ['docker_image','machine_shape','is_private','enable_gpu','enable_internet']:assert local[k]==remote[k]
    assert sorted('/'.join(x.split('/')[:2]) for x in local['dataset_sources'])==sorted('/'.join(x.split('/')[:2]) for x in remote['dataset_sources'])
    for x in remote['dataset_sources']:assert len(x.split('/'))==2 or x.split('/')[2:]==['1']
    receipt=read_json(job/'retrieval.json')
    for n,h in receipt['artifact_sha256'].items():assert sha256(job/'output'/n)==h
    spec=read_json(root/'candidate_spec.json');actual_spec=read_json(out/'runtime_spec.json')
    for k,v in spec.items():assert actual_spec[k]==v
    done=read_json(out/'qualification.json');assert done['complete'] and done['qualify'] and not done['uses_targets']
    assert done['context_tokens']==6144 and done['parent_tokens']==4096 and done['preserves_old_route']
    assert not done['public_prediction_cache_used'] and done['actual_hidden_inference']
    assert done['temperature']==1 and done['packing_tokens']==8192
    session=read_json(job/'output/session_summary.json');assert session['exit_code']==0 and session['seconds']<=3600
    hardware=read_json(job/'output/hardware.json');assert len(hardware['gpus'])==2 and all('T4' in x for x in hardware['gpus'])
    roles=['m0','m3','m0_reverse'];summaries={r:read_json(out/(r+'_summary.json')) for r in roles}
    memory={}
    for role_name,s in summaries.items():
        assert s['loading_exact'] and s['versions']['python'].startswith('3.10.')
        assert s['versions']['torch'].split('+')[0]=='2.1.2' and s['versions']['transformers']=='4.42.3'
        assert read_json(out/(role_name+'_checkpoint.json'))==spec['checkpoint_references'][role_name]
        assert set(s['cohorts'])==set(spec['token_references'][role_name])
        totals=s['device_total_memory_bytes'];assert len(totals)==2 and min(totals)>0
        memory[role_name]=[]
        for name,c in s['cohorts'].items():
            a=np.load(out/(role_name+'_'+name+'.npz'));expected=spec['token_references'][role_name][name]
            assert a['ids'].tolist()==expected['ids'] and a['lengths'].tolist()==expected['lengths']
            validate_probabilities(a['probabilities'],len(a['ids']))
            if len(a['ids']):
                assert c['token_ids_sha256']==expected['token_ids_sha256']
                assert int(a['lengths'].max())<=s['max_position_embeddings']
                memory[role_name].extend(c['peak_allocated_bytes'][g]/totals[g] for g in range(2))
    parent=ROOT/'runs/span_013/jobs/gemma_order/output/order_completion'
    parent_receipt=read_json(REPO/'results/span_013/jobs/gemma_order/retrieval.json')
    for n,h in parent_receipt['artifact_sha256'].items():assert sha256(parent.parent/n)==h
    pilot_audit=read_json(pub/'pilot_audit.json');pilot_job=root/'jobs/context_pilot'
    assert sha256(pilot_job/'retrieval.json')==pilot_audit['retrieval_sha256']
    for n,h in read_json(pilot_job/'retrieval.json')['artifact_sha256'].items():assert sha256(pilot_job/'output'/n)==h
    assert sha256(root/'pilot_spec.json')==read_json(pub/'pilot_preparation.json')['spec_sha256']
    identities={};replay={}
    for name in ['control','uniform','long','test']:
        a=np.load(out/('m0_'+name+'.npz'));b=np.load(out/('m3_'+name+'.npz'));ids=a['ids']
        for model,current in [('m0',a),('m3',b)]:
            old=np.load(parent/(model+'_'+name+'.npz'))
            assert np.array_equal(current['ids'],old['ids']) and np.array_equal(current['lengths'],old['lengths'])
            diff=float(np.abs(current['probabilities']-old['probabilities']).max());assert diff<=1e-5
            identities[model+'_'+name]=diff
        selected,js,work=route_indices(ids,a['probabilities'],b['probabilities'][:,[1,0,2]],np.c_[a['lengths'],b['lengths']])
        saved_route=np.load(out/('routing_'+name+'.npz'));old_route=np.load(parent/('routing_'+name+'.npz'))
        np.testing.assert_array_equal(saved_route['selected_indices'],selected)
        np.testing.assert_array_equal(selected,old_route['selected_indices'])
        np.testing.assert_array_equal(saved_route['work'],work);np.testing.assert_array_equal(saved_route['disagreement'],js)
        reverse=np.load(out/('m0_reverse_routed_'+name+'.npz'));old_reverse=np.load(parent/('m0_reverse_routed_'+name+'.npz'))
        assert np.array_equal(reverse['ids'],ids[selected]) and np.array_equal(reverse['ids'],old_reverse['ids'])
        diff=float(np.abs(reverse['probabilities']-old_reverse['probabilities']).max()) if len(selected) else 0
        assert diff<=1e-5;identities['m0_reverse_'+name]=diff
        extended=np.load(out/('m0_extended_'+name+'.npz'));expected=spec['token_references']['m0']['extended_'+name]['ids']
        assert extended['ids'].tolist()==expected
        replaced=replace_original(ids,a['probabilities'],extended['ids'],extended['probabilities'])
        candidate=complete_gemma_order(replaced,b['probabilities'],selected,reverse['probabilities'])
        baseline=complete_gemma_order(a['probabilities'],b['probabilities'],selected,reverse['probabilities'])
        saved=np.load(out/('ensemble_'+name+'.npz'));assert np.array_equal(saved['ids'],ids)
        np.testing.assert_array_equal(saved['probabilities'],candidate);np.testing.assert_array_equal(saved['parent_probabilities'],baseline)
        changed=np.isin(ids,extended['ids']);np.testing.assert_array_equal(saved['changed'],changed)
        np.testing.assert_array_equal(candidate[~changed],baseline[~changed])
        replay[name]=dict(rows=len(ids),changed_rows=int(changed.sum()),routed_rows=len(selected),route_replayed=True,mixture_replayed=True,unchanged_rows_exact=True)
    csv=job/'output/submission.csv';submission=pd.read_csv(csv);visible=pd.read_csv(ROOT/'raw/test.csv',usecols=['id'])
    assert submission.columns.tolist()==['id',*CLASSES] and np.array_equal(submission.id,visible.id)
    validate_probabilities(submission[CLASSES].to_numpy(),len(visible))
    np.testing.assert_allclose(submission[CLASSES].to_numpy(),np.load(out/'ensemble_test.npz')['probabilities'],rtol=0,atol=1e-12)
    parent_roles={r:read_json(parent/(r+'_summary.json')) for r in roles}
    pilot_dir=root/'jobs/context_pilot/output/context_audit';pilot_roles={r:read_json(pilot_dir/(r+'_summary.json')) for r in roles}
    pilot_spec=read_json(root/'pilot_spec.json');pilot_runtime=read_json(pilot_dir/'runtime_spec.json')
    parent_runtime=read_json(REPO/'results/span_013/gemma_order_qualification.json')['runtime']
    runtime=deployment_runtime(parent_roles,summaries,pilot_roles,parent_runtime,session['seconds'],done['selector_seconds'],
        pilot_runtime['runtime_selector_seconds'],pilot_spec['changed_fraction'],pilot_spec['parent_runtime']['worst_g_uniform_seconds'])
    memory_pass=all(max(v)<=.95 for v in memory.values())
    result=dict(at=utc_now(),complete=True,kernel_version=proof['kernel_version'],source_verified=True,
        all_controls_passed=True,canonical_max_abs=identities,tokens_exact=True,runtime=runtime,
        runtime_gate_passed=runtime['runtime_gate_passed'],memory_guard_passed=memory_pass,
        peak_memory_fractions={r:max(v) for r,v in memory.items()},coverage=replay,csv_verified=True,csv_sha256=sha256(csv),
        both_t4_verified=True,original_training_overlap=True,independent_cv=False,hidden_score_forecast=False,
        labels_read=False,official_submission_eligible=bool(runtime['runtime_gate_passed'] and memory_pass),
        pilot_audit_sha256=sha256(pub/'pilot_audit.json'),source_sha256=bundle['source_sha256'])
    if result['official_submission_eligible']:
        # Descriptive only; this stage does not fit or select a recipe.
        assert sha256(ROOT/'raw/train.csv')==spec['train_csv_sha256']
        train=pd.read_csv(ROOT/'raw/train.csv',usecols=['id','prompt',*CLASSES]).set_index('id')
        quality={}
        for name in ['control','uniform','long']:
            a=np.load(out/('ensemble_'+name+'.npz'));frame=train.loc[a['ids']]
            groups=prompt_groups(frame.assign(prompt=frame.prompt.map(parse_turns))).to_numpy()
            quality[name]=compare(frame[CLASSES].to_numpy().argmax(1),a['parent_probabilities'],a['probabilities'],groups)
        result.update(labels_read=True,descriptive_quality=quality)
    atomic_json(pub/'candidate_qualification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
