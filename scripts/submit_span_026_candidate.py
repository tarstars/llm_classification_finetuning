"""Submit one fully qualified H72 version; resume only its durable receipt."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import json
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,write_intent,exclusive_lock,sha256,verify_sources,utc_now
from arena.span004_support import api_client,read_call,inventory,public_inventory
from arena.span026_qualification import deployment_runtime
from submit_span_013_order import submissions,TERMINAL,dataset_dependencies
from submit_span_010_order import fields


def check_gate(gate,pilot,bundle):
    assert all(gate[k] for k in ['complete','source_verified','all_controls_passed','tokens_exact','runtime_gate_passed',
        'memory_guard_passed','csv_verified','both_t4_verified','official_submission_eligible'])
    assert gate['original_training_overlap'] and not gate['independent_cv'] and not gate['hidden_score_forecast']
    expected={r+'_'+n for r in ['m0','m3','m0_reverse'] for n in ['control','uniform','long','test']}
    assert set(gate['canonical_max_abs'])==expected
    assert all(0<=x<=1e-5 for x in gate['canonical_max_abs'].values())
    assert set(gate['coverage'])=={'control','uniform','long','test'}
    assert all(all(c[k] for k in ['route_replayed','mixture_replayed','unchanged_rows_exact']) for c in gate['coverage'].values())
    assert pilot['complete_information'] and pilot['engineering_screen_passed'] and pilot['gross_regression_guard_passed']
    assert pilot['fixed_gross_regression_threshold']==-.005 and pilot['weighted_lower_gain']>=-.005
    assert not pilot['local_positive_gain_required']
    assert bundle['public_cap_seconds']==3600 and bundle['hidden_cap_seconds']==32400 and not bundle['public_prediction_cache_embedded']
    assert gate['runtime']['runtime_gate_passed'] and 0<gate['runtime']['projection_seconds']<32400
    assert set(gate['peak_memory_fractions'])=={'m0','m3','m0_reverse'}
    assert all(0<x<=.95 for x in gate['peak_memory_fractions'].values())


def candidate(root,pub):
    role='context_qualification';job=root/'jobs'/role;out=job/'output/context_candidate'
    gate=read_json(pub/'candidate_qualification.json');bundle=read_json(pub/(role+'_bundle.json'))
    verify_sources(REPO,bundle['source_sha256'])
    pilot=read_json(pub/'pilot_audit.json')
    check_gate(gate,pilot,bundle)
    assert gate['pilot_audit_sha256']==bundle['pilot_audit_sha256']==sha256(pub/'pilot_audit.json')
    roles=['m0','m3','m0_reverse'];parent=ROOT/'runs/span_013/jobs/gemma_order/output/order_completion'
    for n,h in read_json(REPO/'results/span_013/jobs/gemma_order/retrieval.json')['artifact_sha256'].items():assert sha256(parent.parent/n)==h
    pilot_job=root/'jobs/context_pilot'
    assert sha256(pilot_job/'retrieval.json')==pilot['retrieval_sha256']
    for n,h in read_json(pilot_job/'retrieval.json')['artifact_sha256'].items():assert sha256(pilot_job/'output'/n)==h
    assert sha256(root/'pilot_spec.json')==read_json(pub/'pilot_preparation.json')['spec_sha256']
    pilot_dir=root/'jobs/context_pilot/output/context_audit';spec=read_json(root/'pilot_spec.json')
    parent_roles={r:read_json(parent/(r+'_summary.json')) for r in roles}
    current_roles={r:read_json(out/(r+'_summary.json')) for r in roles}
    pilot_roles={r:read_json(pilot_dir/(r+'_summary.json')) for r in roles}
    runtime=deployment_runtime(parent_roles,current_roles,pilot_roles,
        read_json(REPO/'results/span_013/gemma_order_qualification.json')['runtime'],
        read_json(job/'output/session_summary.json')['seconds'],read_json(out/'qualification.json')['selector_seconds'],
        read_json(pilot_dir/'runtime_spec.json')['runtime_selector_seconds'],spec['changed_fraction'],spec['parent_runtime']['worst_g_uniform_seconds'])
    assert runtime==gate['runtime'] and runtime['runtime_gate_passed'] and runtime['projection_seconds']<32400
    assert bundle['public_cap_seconds']==3600 and bundle['hidden_cap_seconds']==32400 and not bundle['public_prediction_cache_embedded']
    assert sha256(root/'candidate_spec.json')==bundle['candidate_spec_sha256']
    assert sha256(root/'hidden_asset_spec.json')==bundle['hidden_spec_sha256']
    assert sha256(root/'notebooks'/role/'job.ipynb')==bundle['notebook_sha256']
    assert sha256(root/'notebooks'/role/'kernel-metadata.json')==bundle['metadata_sha256']
    assert read_json(pub/(role+'_cell_rehearsals.json'))['passed']
    proof=read_json(pub/'jobs'/role/'source_verification.json');push=read_json(job/'push.json')
    ref=push['ref'].removeprefix('/code/');version=push['version_number'];exact=ref+'/'+str(version)
    assert proof['passed'] and proof['notebook_sha256']==bundle['notebook_sha256']
    assert exact==proof['kernel_version']==gate['kernel_version'] and read_json(job/'status.json')['status']=='COMPLETE'
    for n,h in read_json(job/'retrieval.json')['artifact_sha256'].items():assert sha256(job/'output'/n)==h
    assert sha256(job/'output/submission.csv')==gate['csv_sha256']
    evidence=['pilot_audit.json','candidate_qualification.json',role+'_bundle.json',role+'_cell_rehearsals.json']
    decision=dict(submit=True,role=role,kernel_version=exact,temperature=1,context_tokens=6144,
        retains_original_H60_route=True,weights_unchanged=True,parameters_selected_without_leaderboard_feedback=True,
        original_training_overlap=True,independent_cv=False,hidden_score_forecast=False,
        projection_seconds=runtime['projection_seconds'],notebook_sha256=bundle['notebook_sha256'],
        evidence_sha256={n:sha256(pub/n) for n in evidence})
    source=dict(role=role,reference=ref,version=version,exact=exact,notebook_sha256=bundle['notebook_sha256'],
        qualification_sha256=sha256(pub/'candidate_qualification.json'),projection_seconds=runtime['projection_seconds'])
    return source,decision


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-decision',action='store_true');parser.add_argument('--observe-only',action='store_true')
    args=parser.parse_args();root=ROOT/'runs/span_026';pub=REPO/'results/span_026';folder=root/'official/candidate'
    folder.mkdir(parents=True,exist_ok=True)
    with exclusive_lock(folder/'controller.lock'):
        def save(n,v):atomic_json(folder/n,v);atomic_json(pub/('official_'+n),v)
        def state(stage,**v):
            record=dict(at=utc_now(),stage=stage,**v);save('state.json',record);print(json.dumps(record),flush=True)
        if not (folder/'receipt.json').exists():
            assert not args.observe_only and not (folder/'submission_intent.json').exists(),'Reconcile existing intent; never submit twice'
            registry=read_json(REPO/'backlog/hypotheses.json')
            assert registry['active_version']==26 and next(v for v in registry['versions'] if v['number']==26)['focus_ids']==['H72']
            assert next(i for i in registry['items'] if i['id']=='H72')['status']=='running'
            source,decision=candidate(root,pub)
            if args.prepare_decision:
                assert not (pub/'deployment_decision.json').exists(),'Preserve initial fixed decision'
                atomic_json(pub/'deployment_decision.json',dict(at=utc_now(),**decision));print(json.dumps(decision));return
            frozen=read_json(pub/'deployment_decision.json');assert {k:v for k,v in frozen.items() if k!='at'}==decision
            verify_sources(REPO,{str(Path(__file__).relative_to(REPO)):sha256(Path(__file__)),
                'scripts/submit_span_013_order.py':sha256(REPO/'scripts/submit_span_013_order.py'),
                'scripts/observe_leaderboard.py':sha256(REPO/'scripts/observe_leaderboard.py'),
                'results/span_026/deployment_decision.json':sha256(pub/'deployment_decision.json')})
            api=api_client();assert str(read_call(lambda:api.kernels_status(source['exact'])).status).split('.')[-1]=='COMPLETE'
            plan=read_json(pub/'span_plan.json');fresh=public_inventory(inventory(api,plan['started_at']));q=fresh['quota']
            assert not fresh['nonterminal_jobs'] and q['reserved_seconds']==0 and not q['paid_scaling']
            assert q['refresh_time']==plan['initial_quota_refresh_time'] and q['remaining_seconds']>=32400
            charge=q['used_seconds']-plan['quota_used_at_start'];assert -.1<=charge<=10800.1
            assert datetime.now(timezone.utc).timestamp()+32400+600<datetime.fromisoformat(plan['deadline']).timestamp()
            assert len(list((root/'jobs').glob('*/push.json')))==2
            assert len(list((root/'official').glob('*/receipt.json')))<plan['official_candidate_limit']==1
            message='Span026 fixed6144 Gemma evidence, original H60 route and weights, T1'
            history=submissions(api);assert all(str(s.status).split('.')[-1] in TERMINAL for s in history)
            assert not any(getattr(s,'description','')==message for s in history),'Reconcile possible accepted duplicate'
            deps=dataset_dependencies(api,read_json(root/'notebooks/context_qualification/kernel-metadata.json'))
            save('admission.json',dict(at=utc_now(),inventory=fresh,dependencies=deps,prior_submissions=len(history),
                all_official_terminal=True,public_gpu_charged_seconds=charge,**source))
            intent=dict(at=utc_now(),**source,quota_before=q,hidden_seconds_cap=32400,message=message,
                decision_sha256=sha256(pub/'deployment_decision.json'),controller_sha256=sha256(Path(__file__)))
            write_intent(folder/'submission_intent.json',intent);atomic_json(pub/'official_submission_intent.json',intent)
            state('submitting_one_frozen_version',kernel_version=source['exact'])
            response=api.competition_submit_code(file_name='submission.csv',message=message,competition='llm-classification-finetuning',
                kernel=source['reference'],kernel_version=source['version'])
            receipt=fields(response);save('receipt.json',receipt);assert receipt.get('ref'),'Reconcile accepted state; never duplicate'
            state('accepted',receipt=receipt['ref'])
        else:assert not args.prepare_decision,'Already submitted; preserve decision'
        api=api_client();receipt=read_json(folder/'receipt.json');plan=read_json(pub/'span_plan.json')
        while time.time()<datetime.fromisoformat(plan['deadline']).timestamp():
            try:
                matches=[fields(s) for s in submissions(api) if str(s.ref)==str(receipt['ref'])]
                if matches:
                    assert len(matches)==1;result=matches[0];save('result.json',result);status=result['status'].split('.')[-1]
                    state('monitoring',receipt=receipt['ref'],status=status)
                    if status in TERMINAL:
                        if status=='COMPLETE':
                            subprocess.run([sys.executable,str(REPO/'scripts/observe_leaderboard.py'),
                                '--output','results/span_026/official_leaderboard.json'],cwd=REPO,check=True,timeout=180)
                            save('leaderboard.json',read_json(pub/'official_leaderboard.json'))
                        state('terminal',receipt=receipt['ref'],status=status,score=result.get('public_score'));return
                else:state('receipt_not_yet_visible',receipt=receipt['ref'])
            except Exception as error:state('observation_retry',receipt=receipt['ref'],error_type=type(error).__name__)
            time.sleep(45)
        state('original_deadline_reached_resume_same_receipt',receipt=receipt['ref'])


if __name__=='__main__':main()
