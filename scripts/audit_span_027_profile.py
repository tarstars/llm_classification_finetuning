"""Terminal H74source/input/control/equivalence/runtime checks; never read votes."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,read_json,sha256,verify_sources,utc_now
from arena.span010_order import CLASSES,route_indices,validate_probabilities
from arena.span013_order import complete_gemma_order
from arena.span027_runtime import projection


def main():
    pub=REPO/'results/span_027';root=ROOT/'runs/span_027';role='packing_profile';job=root/'jobs'/role;out=job/'output/packing_profile'
    assert not (pub/'profile_audit.json').exists(),'Preserve first engineering outcome'
    bundle=read_json(pub/(role+'_bundle.json'));verify_sources(REPO,bundle['source_sha256'])
    assert read_json(job/'status.json')['status']=='COMPLETE'
    proof=read_json(pub/'jobs'/role/'source_verification.json');assert proof['passed'] and proof['notebook_sha256']==bundle['notebook_sha256']
    local=read_json(root/'notebooks'/role/'kernel-metadata.json');remote=read_json(root/'verified_sources'/role/'kernel-metadata.json')
    for k in ['docker_image','machine_shape','is_private','enable_gpu','enable_internet']:assert local[k]==remote[k]
    assert sorted('/'.join(x.split('/')[:2]) for x in local['dataset_sources'])==sorted('/'.join(x.split('/')[:2]) for x in remote['dataset_sources'])
    for x in remote['dataset_sources']:assert len(x.split('/'))==2 or x.split('/')[2:]==['1']
    retrieval=read_json(job/'retrieval.json')
    for n,h in retrieval['artifact_sha256'].items():assert sha256(job/'output'/n)==h
    session=read_json(job/'output/session_summary.json');assert session['exit_code']==0 and session['seconds']<=3600
    hardware=read_json(job/'output/hardware.json');assert len(hardware['gpus'])==2 and all('T4' in x for x in hardware['gpus'])
    done=read_json(out/'profile_complete.json');assert not done['uses_targets'] and done['packing_tokens']==10240
    spec=read_json(root/'profile_spec.json');actual_spec=read_json(out/'runtime_spec.json')
    for k,v in spec.items():assert actual_spec[k]==v
    result=dict(at=utc_now(),kernel_version=proof['kernel_version'],profile_complete=done['complete'],labels_read=False,
        quality_measured=False,packing_tokens=10240,conversation_tokens=4096,official_submission_eligible=False,
        source_sha256=bundle['source_sha256'],artifact_files_verified=retrieval['files'],
        engineering_gate_passed=False,original_probability_tolerance=1e-5)
    if not done['complete']:
        assert done['stopped_for_guard'];failure=read_json(out/'engineering_stop.json');r=failure['role'];name=failure['cohort']
        values=np.load(out/(r+'_'+name+'.npz'));summary=read_json(out/(r+'_summary.json'));details=summary['cohorts'][name]
        fractions=[details['peak_allocated_bytes'][g]/summary['device_total_memory_bytes'][g] for g in range(2)]
        assert failure['checks']['memory_fractions']==fractions
        assert failure['checks']['memory_passed']==all(x<=.95 for x in fractions)
        if name in ['control','routed_control']:
            reference=np.load(root/'controls'/(r+'.npz'));assert np.array_equal(values['ids'],reference['ids'])
            difference=float(np.abs(values['probabilities'].astype(float)-reference['probabilities']).max())
            assert difference==failure['checks']['canonical_max_abs']
            assert failure['checks']['canonical_passed']==(difference<=1e-5)
        result.update(decision='stop_packing_guard',guard_failure=failure,failed_guard_independently_replayed=True)
        atomic_json(pub/'profile_audit.json',result);print(json.dumps(result,indent=2));return
    roles=['m0','m3','m0_reverse'];summaries={r:read_json(out/(r+'_summary.json')) for r in roles}
    cpu=read_json(pub/'packing_cpu_audit.json');identity={};memory={};parent=ROOT/'runs/span_013/jobs/gemma_order/output'
    for n,h in read_json(REPO/'results/span_013/jobs/gemma_order/retrieval.json')['artifact_sha256'].items():assert sha256(parent/n)==h
    old=parent/'order_completion'
    for r,s in summaries.items():
        assert s['loading_exact'] and s['packing_tokens']==10240 and s['versions']['python'].startswith('3.10.')
        assert s['versions']['torch'].split('+')[0]=='2.1.2' and s['versions']['transformers']=='4.42.3'
        assert read_json(out/(r+'_checkpoint.json'))==spec['checkpoint_references'][r]
        assert set(s['cohorts'])==set(spec['token_references'][r])
        memory[r]=[];checks=read_json(out/(r+'_checks.json'))
        for name,c in s['cohorts'].items():
            a=np.load(out/(r+'_'+name+'.npz'));expected=spec['token_references'][r][name]
            assert a['ids'].tolist()==expected['ids'] and a['lengths'].tolist()==expected['lengths']
            validate_probabilities(a['probabilities'],len(a['ids']))
            if len(a['ids']):
                assert checks[name]['passed'] and c['token_ids_sha256']==expected['token_ids_sha256']
                cp=cpu['actual_native_reference_cohorts'][r][name]['10240']
                assert c['micro_batches']==cp['micro_batches'] and c['peak_packed_tokens']==cp['peak_packed_tokens']
                assert c['max_tokens']<=s['max_position_embeddings']
                memory[r].extend(c['peak_allocated_bytes'][g]/s['device_total_memory_bytes'][g] for g in range(2))
            previous=np.load(old/(r+'_'+name+'.npz'));assert np.array_equal(a['ids'],previous['ids'])
            difference=float(np.abs(a['probabilities'].astype(float)-previous['probabilities']).max()) if len(a['ids']) else 0
            assert difference<=1e-5;identity[r+'_'+name]=difference
    for name in ['control','uniform','long','test']:
        a=np.load(out/('m0_'+name+'.npz'));b=np.load(out/('m3_'+name+'.npz'));ids=a['ids']
        selected,js,work=route_indices(ids,a['probabilities'],b['probabilities'][:,[1,0,2]],np.c_[a['lengths'],b['lengths']])
        saved=np.load(out/('routing_'+name+'.npz'));old_route=np.load(old/('routing_'+name+'.npz'))
        np.testing.assert_array_equal(saved['selected_indices'],selected);np.testing.assert_array_equal(selected,old_route['selected_indices'])
        np.testing.assert_allclose(saved['disagreement'],js,rtol=1e-12,atol=1e-14);np.testing.assert_array_equal(saved['work'],work)
        reverse=np.load(out/('m0_reverse_routed_'+name+'.npz'));assert np.array_equal(reverse['ids'],ids[selected])
        p=complete_gemma_order(a['probabilities'],b['probabilities'],selected,reverse['probabilities'])
        ensemble=np.load(out/('ensemble_'+name+'.npz'));np.testing.assert_array_equal(ensemble['ids'],ids);np.testing.assert_array_equal(ensemble['probabilities'],p)
    csv=job/'output/submission.csv';visible=pd.read_csv(ROOT/'raw/test.csv',usecols=['id']);submission=pd.read_csv(csv)
    assert submission.columns.tolist()==['id',*CLASSES] and np.array_equal(submission.id,visible.id)
    validate_probabilities(submission[CLASSES].to_numpy(),len(submission));np.testing.assert_allclose(submission[CLASSES],np.load(out/'ensemble_test.npz')['probabilities'],rtol=0,atol=1e-12)
    timing_reference=read_json(pub/'runtime_reference.json')
    for n,h in timing_reference['database_reference_sha256'].items():assert sha256(ROOT/n)==h
    for n,h in timing_reference['project_reference_sha256'].items():assert sha256(REPO/n)==h
    runtime=projection(summaries,session['seconds'],timing_reference['overhead_floor'])
    memory_pass=all(max(v)<=.95 for v in memory.values());passes=bool(runtime['runtime_gate_passed'] and memory_pass)
    result.update(all_controls_passed=True,canonical_max_abs=identity,tokens_and_packing_shapes_exact=True,
        original_routes_exact=True,mixtures_replayed=True,csv_verified=True,csv_sha256=sha256(csv),memory_guard_passed=memory_pass,
        peak_memory_fractions={r:max(v) for r,v in memory.items()},runtime=runtime,runtime_gate_passed=runtime['runtime_gate_passed'],
        engineering_gate_passed=passes,decision='allow_H73_fixed_CPU_replay' if passes else 'stop_packing_runtime_or_memory',
        source_prediction_equivalence_is_not_new_quality_gain=True)
    atomic_json(pub/'profile_audit.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
