"""Independent terminal two-role qualification; never reads target labels."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,read_json,sha256,verify_sources,utc_now
from arena.span010_order import CLASSES,validate_probabilities
from arena.span030_mixture import two_expert_prior
from arena.span030_runtime import projection


def main():
    pub=REPO/'results/span_030';root=ROOT/'runs/span_030';role='weight_qualification'
    job=root/'jobs'/role;out=job/'output/two_expert'
    assert not (pub/'candidate_qualification.json').exists(),'Preserve first qualification outcome'
    bundle=read_json(pub/(role+'_bundle.json'));verify_sources(REPO,bundle['source_sha256'])
    assert read_json(job/'status.json')['status']=='COMPLETE'
    proof=read_json(pub/'jobs'/role/'source_verification.json');assert proof['passed'] and proof['notebook_sha256']==bundle['notebook_sha256']
    local=read_json(root/'notebooks'/role/'kernel-metadata.json');remote=read_json(root/'verified_sources'/role/'kernel-metadata.json')
    for k in ['docker_image','machine_shape','is_private','enable_gpu','enable_internet']:assert local[k]==remote[k]
    assert remote['is_private'] and remote['enable_gpu'] and not remote['enable_internet']
    assert sorted('/'.join(s.split('/')[:2]) for s in local['dataset_sources'])==sorted('/'.join(s.split('/')[:2]) for s in remote['dataset_sources'])
    for s in remote['dataset_sources']:assert len(s.split('/'))==2 or s.split('/')[2:]==['1']
    retrieval=read_json(job/'retrieval.json')
    for n,h in retrieval['artifact_sha256'].items():assert sha256(job/'output'/n)==h,n
    session=read_json(job/'output/session_summary.json');assert session['exit_code']==0 and session['seconds']<=3600
    hardware=read_json(job/'output/hardware.json');assert len(hardware['gpus'])==2 and all('T4' in g for g in hardware['gpus'])
    spec=read_json(root/'qualification_spec.json');actual=read_json(out/'runtime_spec.json')
    for k,v in spec.items():assert actual[k]==v
    done=read_json(out/'inference_complete.json')
    assert done['mode']=='qualification' and not done['uses_targets'] and done['packing_tokens']==8192
    result=dict(at=utc_now(),complete=done['complete'],source=proof,labels_read=False,quality_measured=False,
        official_submission_eligible=False,expert_weights=[.4,.6],temperature=1,row_normalization=True,
        reverse_pass=False,route=False,artifact_files_verified=retrieval['files'],source_sha256=bundle['source_sha256'])
    if not done['complete']:
        assert done['stopped_for_guard'];failure=read_json(out/'engineering_stop.json');r=failure['role'];name=failure['cohort']
        summary=read_json(out/(r+'_summary.json'));c=summary['cohorts'][name];a=np.load(out/(r+'_'+name+'.npz'))
        expected=spec['token_references'][r][name];batch=spec['batch_references'][r][name]
        replay=dict(checkpoint_match=read_json(out/(r+'_checkpoint.json'))==spec['checkpoint_references'][r],
            producer_joined=c['producer_joined'],ids_match=a['ids'].tolist()==spec['frame_ids'][name],
            memory_passed=all(c['peak_allocated_bytes'][g]/summary['device_total_memory_bytes'][g]<=.95 for g in range(2)),
            lengths_match=a['lengths'].tolist()==expected['lengths'],token_hash_match=c['token_ids_sha256']==expected['token_ids_sha256'],
            microbatches_match=c['batch_micro_sequence_lengths']==batch['groups'],host_tensors_match=c['host_tensor_sha256']==batch['host_tensor_sha256'])
        if name=='control':
            gold=np.load(root/'controls'/(r+'.npz'));np.testing.assert_array_equal(gold['ids'],a['ids'])
            difference=float(np.abs(gold['probabilities'].astype(float)-a['probabilities']).max())
            assert failure['checks']['canonical_max_abs']==difference;replay['canonical_passed']=difference<=1e-5
        for k,v in replay.items():assert failure['checks'][k]==v
        assert not all(replay.values()) and not failure['checks']['passed']
        result.update(decision='stop_public_guard',guard_failure=failure,failed_guard_independently_replayed=True)
        atomic_json(pub/'candidate_qualification.json',result);print(result);return
    assert done['expert_weights']==[.4,.6] and not done['route'] and not done['reverse_pass'] and done['roles']==['m0','m3']
    parent_job=ROOT/'runs/span_028/jobs/prefetch_profile';parent=parent_job/'output/prefetch_profile'
    for n,h in read_json(parent_job/'retrieval.json')['artifact_sha256'].items():assert sha256(parent_job/'output'/n)==h,n
    summaries={r:read_json(out/(r+'_summary.json')) for r in ['m0','m3']};memory={};controls={};mixtures={}
    for r,s in summaries.items():
        assert s['loading_exact'] and s['versions']['python'].startswith('3.10.') and s['versions']['torch'].split('+')[0]=='2.1.2'
        assert s['versions']['transformers']=='4.42.3' and s['packing_tokens']==8192 and s['cpu_prefetch_enabled'] and s['queue_depth']==2
        assert read_json(out/(r+'_checkpoint.json'))==spec['checkpoint_references'][r]
        assert set(s['cohorts'])==set(spec['token_references'][r]);memory[r]=[]
        checks=read_json(out/(r+'_checks.json'))
        for name,c in s['cohorts'].items():
            a=np.load(out/(r+'_'+name+'.npz'));expected=spec['token_references'][r][name];batch=spec['batch_references'][r][name]
            assert a['ids'].tolist()==expected['ids'] and a['lengths'].tolist()==expected['lengths']
            validate_probabilities(a['probabilities'],len(a['ids']))
            assert checks[name]['passed'] and c['token_ids_sha256']==expected['token_ids_sha256']
            assert c['host_tensor_sha256']==batch['host_tensor_sha256'] and c['batch_micro_sequence_lengths']==batch['groups']
            assert c['producer_joined'] and c['cpu_prefetch_metrics']['prepared_batches']==len(batch['groups'])
            assert c['max_tokens']<=s['max_position_embeddings']
            memory[r].extend(c['peak_allocated_bytes'][g]/s['device_total_memory_bytes'][g] for g in range(2))
            previous=np.load(parent/(r+'_'+name+'.npz'));np.testing.assert_array_equal(a['ids'],previous['ids'])
            difference=float(np.abs(a['probabilities'].astype(float)-previous['probabilities']).max())
            assert difference<=1e-5;controls[r+'_'+name]=difference
    for name in ['control','uniform','long','test']:
        a=np.load(out/('m0_'+name+'.npz'));b=np.load(out/('m3_'+name+'.npz'));saved=np.load(out/('candidate_'+name+'.npz'))
        np.testing.assert_array_equal(a['ids'],b['ids']);np.testing.assert_array_equal(saved['ids'],a['ids'])
        p=two_expert_prior(a['probabilities'],b['probabilities'])
        np.testing.assert_allclose(p,saved['probabilities'],rtol=1e-12,atol=1e-14)
        mixtures[name]=float(np.abs(p-saved['probabilities']).max())
    csv=job/'output/submission.csv';submission=pd.read_csv(csv);test=pd.read_csv(ROOT/'raw/test.csv',usecols=['id'])
    assert submission.columns.tolist()==['id',*CLASSES];np.testing.assert_array_equal(submission.id,test.id)
    validate_probabilities(submission[CLASSES].to_numpy(),len(test))
    np.testing.assert_allclose(submission[CLASSES],np.load(out/'candidate_test.npz')['probabilities'],rtol=0,atol=1e-12)
    reference=read_json(pub/'runtime_reference.json')
    for n,h in {**reference['database_reference_sha256'],**reference['parent_summary_sha256']}.items():assert sha256(ROOT/n)==h,n
    for n,h in reference['project_reference_sha256'].items():assert sha256(REPO/n)==h,n
    old={r:read_json(ROOT/n) for r,n in reference['parent_summary_paths'].items()}
    runtime=projection(summaries,session['seconds'],reference['overhead_floor'],old)
    memory_pass=all(max(v)<=.95 for v in memory.values());passes=bool(memory_pass and runtime['runtime_gate_passed'])
    result.update(all_controls_passed=True,canonical_max_abs=controls,raw_probability_tolerance=1e-5,
        host_tensors_and_microbatches_exact=True,tokens_and_positions_exact=True,both_t4_verified=True,
        mixtures_replayed=True,mixture_max_abs=mixtures,derived_mixture_roundoff_rtol=1e-12,derived_mixture_roundoff_atol=1e-14,
        csv_verified=True,csv_sha256=sha256(csv),memory_guard_passed=memory_pass,peak_memory_fractions={r:max(v) for r,v in memory.items()},
        runtime=runtime,runtime_gate_passed=runtime['runtime_gate_passed'],official_submission_eligible=passes,
        independent_local_quality=False,quality_replay_sha256=sha256(pub/'quality_replay.json'),
        decision='allow_one_frozen_official' if passes else 'stop_runtime_or_memory')
    atomic_json(pub/'candidate_qualification.json',result);print(result)


if __name__=='__main__':main()
