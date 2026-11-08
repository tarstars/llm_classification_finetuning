"""One qualified frozen official; reconcile an uncertain reply, never resubmit."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,read_json,sha256,verify_sources,utc_now,exclusive_lock,write_intent
from arena.span004_support import api_client,read_call,inventory,public_inventory
from submit_span_013_order import submissions,TERMINAL,dataset_dependencies
from submit_span_010_order import fields

MESSAGE='Span030 H76 fixed 40/60 two-expert prior T1'


def candidate(root,pub):
    gate=read_json(pub/'candidate_qualification.json');quality=read_json(pub/'quality_replay.json')
    bundle=read_json(pub/'weight_qualification_bundle.json');proof=read_json(pub/'jobs/weight_qualification/source_verification.json')
    for key in ['complete','all_controls_passed','host_tensors_and_microbatches_exact','tokens_and_positions_exact',
                'both_t4_verified','mixtures_replayed','csv_verified','memory_guard_passed','runtime_gate_passed','official_submission_eligible']:
        assert gate[key],key
    assert gate['expert_weights']==quality['expert_weights']==[.4,.6] and gate['temperature']==1
    assert not gate['reverse_pass'] and not gate['route'] and not gate['independent_local_quality']
    assert quality['gross_regression_guard_passed'] and quality['quality_variants_tested']==1
    assert quality['cumulative_twoexpert_priors_tested']==[[.6,.4],[.4,.6]]
    assert quality['original_training_overlap'] and not quality['independent_cv'] and not quality['hidden_score_forecast']
    assert gate['quality_replay_sha256']==sha256(pub/'quality_replay.json')
    assert gate['raw_probability_tolerance']==1e-5 and all(v<=1e-5 for v in gate['canonical_max_abs'].values())
    assert len(gate['canonical_max_abs'])==10
    runtime=gate['runtime'];assert runtime['margin']==1.2 and runtime['reserve_seconds']==600
    assert runtime['retains_worse_matching_parent_rates'] and 0<runtime['projection_seconds']<32400
    assert proof['passed'] and proof['notebook_sha256']==bundle['notebook_sha256']
    assert proof['kernel_version']==gate['source']['kernel_version'] and proof['internet_disabled']
    verify_sources(REPO,bundle['source_sha256'])
    role='weight_qualification';folder=root/'notebooks'/role;job=root/'jobs'/role
    assert sha256(folder/'job.ipynb')==bundle['notebook_sha256'] and sha256(folder/'kernel-metadata.json')==bundle['metadata_sha256']
    assert read_json(job/'status.json')['status']=='COMPLETE' and sha256(job/'output/submission.csv')==gate['csv_sha256']
    for n,h in read_json(job/'retrieval.json')['artifact_sha256'].items():assert sha256(job/'output'/n)==h
    push=read_json(job/'push.json');ref=push['ref'].removeprefix('/code/');version=push['version_number'];exact=ref+'/'+str(version)
    assert exact==proof['kernel_version']
    decision=dict(at=utc_now(),submit=True,role=role,kernel_version=exact,expert_weights=[.4,.6],temperature=1,
        row_normalization=True,reverse_pass=False,route=False,packing_tokens=8192,cpu_prefetch=True,
        independent_local_quality=False,sequentially_selected=True,cumulative_priors=[[.6,.4],[.4,.6]],
        interpretation='Original-training-overlapping local screen; one official result measures transfer, no hidden forecast.',
        projection_seconds=runtime['projection_seconds'],evidence_sha256={n:sha256(pub/n) for n in
            ['candidate_qualification.json','quality_replay.json','weight_qualification_bundle.json','weight_qualification_cell_rehearsals.json']})
    return dict(role=role,reference=ref,version=version,exact=exact,notebook_sha256=bundle['notebook_sha256']),decision


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-decision',action='store_true');parser.add_argument('--observe-only',action='store_true')
    args=parser.parse_args();root=ROOT/'runs/span_030';pub=REPO/'results/span_030';folder=root/'official/candidate'
    folder.mkdir(parents=True,exist_ok=True)
    with exclusive_lock(folder/'controller.lock'):
        def save(name,value):
            atomic_json(folder/name,value);atomic_json(pub/('official_'+name),value)
        def state(stage,**values):
            value=dict(at=utc_now(),stage=stage,**values);save('state.json',value);print(value,flush=True)
        if args.prepare_decision:
            assert not (folder/'submission_intent.json').exists() and not (folder/'receipt.json').exists()
            source,decision=candidate(root,pub);atomic_json(pub/'deployment_decision.json',decision);print(decision);return
        api=api_client();plan=read_json(pub/'span_plan.json')
        if not (folder/'receipt.json').exists() and not (folder/'submission_intent.json').exists():
            assert not args.observe_only,'No accepted intent or receipt to observe'
            registry=read_json(REPO/'backlog/hypotheses.json')
            assert registry['active_version']==30 and next(i for i in registry['items'] if i['id']=='H76')['status']=='running'
            source,decision=candidate(root,pub);frozen=read_json(pub/'deployment_decision.json')
            assert {k:v for k,v in frozen.items() if k!='at'}=={k:v for k,v in decision.items() if k!='at'}
            verify_sources(REPO,{str(Path(__file__).relative_to(REPO)):sha256(Path(__file__)),
                'results/span_030/candidate_qualification.json':sha256(pub/'candidate_qualification.json'),
                'results/span_030/deployment_decision.json':sha256(pub/'deployment_decision.json')})
            assert str(read_call(lambda:api.kernels_status(source['exact'])).status).split('.')[-1]=='COMPLETE'
            fresh=public_inventory(inventory(api,plan['started_at']));q=fresh['quota']
            assert not fresh['nonterminal_jobs'] and q['reserved_seconds']==0 and not q['paid_scaling']
            assert q['refresh_time']==plan['initial_quota_refresh_time'] and q['remaining_seconds']>=32400
            charge=q['used_seconds']-plan['quota_used_at_start'];assert -.1<=charge<=3600.1
            assert datetime.now(timezone.utc).timestamp()+32400+600<datetime.fromisoformat(plan['deadline']).timestamp()
            assert not list((root/'official').glob('*/receipt.json'))
            prior=submissions(api);assert all(str(s.status).split('.')[-1] in TERMINAL for s in prior)
            assert not any(getattr(s,'description','')==MESSAGE for s in prior)
            deps=dataset_dependencies(api,read_json(root/'notebooks'/source['role']/'kernel-metadata.json'))
            save('admission.json',dict(at=utc_now(),inventory=fresh,dependencies=deps,prior_submissions=len(prior),
                public_gpu_charged_seconds=charge,**source))
            intent=dict(at=utc_now(),**source,quota_before=q,hidden_seconds_cap=32400,message=MESSAGE,
                prior_submission_refs=[str(s.ref) for s in prior],decision_sha256=sha256(pub/'deployment_decision.json'),
                controller_sha256=sha256(Path(__file__)))
            write_intent(folder/'submission_intent.json',intent);atomic_json(pub/'official_submission_intent.json',intent)
            state('submitting_one_frozen_version',kernel=source['exact'])
            try:
                reply=api.competition_submit_code(file_name='submission.csv',message=MESSAGE,competition='llm-classification-finetuning',
                    kernel=source['reference'],kernel_version=source['version'])
                receipt=fields(reply)
                if receipt.get('ref'):save('receipt.json',receipt);state('accepted',receipt=receipt['ref'])
                else:state('reply_without_receipt_reconcile_same_intent')
            except Exception as error:
                code=getattr(getattr(error,'response',None),'status_code',None)
                state('submission_reply_error',error_type=type(error).__name__,http_status=code)
                if code in {400,401,403,404}:raise
                # Never invoke the mutation again: reconcile only read-only history.
        while time.time()<datetime.fromisoformat(plan['deadline']).timestamp():
            try:
                history=submissions(api)
                if not (folder/'receipt.json').exists():
                    intent=read_json(folder/'submission_intent.json')
                    matches=[s for s in history if getattr(s,'description','')==intent['message'] and str(s.ref) not in intent['prior_submission_refs']]
                    assert len(matches)<=1,'Ambiguous receipt history; no resubmission'
                    if matches:
                        save('receipt.json',dict(ref=str(matches[0].ref),reconciled_from_same_intent=True,observed_at=utc_now()))
                        state('accepted_reconciled',receipt=str(matches[0].ref))
                    else:state('same_intent_not_yet_visible')
                if (folder/'receipt.json').exists():
                    receipt=read_json(folder/'receipt.json');matches=[fields(s) for s in history if str(s.ref)==str(receipt['ref'])]
                    if matches:
                        assert len(matches)==1;result=matches[0];save('result.json',result)
                        status=result['status'].split('.')[-1];state('monitoring',receipt=receipt['ref'],status=status)
                        if status in TERMINAL:
                            if status=='COMPLETE':
                                subprocess.run([sys.executable,str(REPO/'scripts/observe_leaderboard.py'),
                                    '--output','results/span_030/official_leaderboard.json'],cwd=REPO,check=True,timeout=180)
                            state('terminal',receipt=receipt['ref'],status=status);return
                    else:state('receipt_not_yet_visible',receipt=receipt['ref'])
            except Exception as error:state('observation_retry',error_type=type(error).__name__)
            time.sleep(45)
        state('original_observation_deadline_reached')


if __name__=='__main__':main()
