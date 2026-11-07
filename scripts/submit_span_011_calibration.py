"""Submit one independently qualified frozen scalar, or observe the same receipt."""
from datetime import datetime,timezone
from pathlib import Path
import argparse
import json
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,write_intent,exclusive_lock,sha256,verify_sources,utc_now
from arena.span004_support import api_client,read_call,inventory,public_inventory
from submit_span_010_order import leaderboard,fields

TERMINAL={'COMPLETE','ERROR','FAILED','INVALID','CANCELED','CANCELLED'}


def candidate(root,pub):
    decision=read_json(pub/'deployment_decision.json')
    assert decision['submit'] and decision['domain_review_completed']
    assert decision['parameters_selected_without_leaderboard_feedback']
    assert set(decision['evidence_sha256'])=={'confirmation_metrics.json','frozen_temperature.json',
        'learning_metrics.json','calibration_learning_qualification.json','calibration_confirmation_qualification.json',
        'calibration_confirmation_bundle.json','calibration_confirmation_source_verification.json'}
    for name,digest in decision['evidence_sha256'].items():assert sha256(pub/name)==digest
    metrics=read_json(pub/'confirmation_metrics.json')
    assert metrics['information_gate_passed'] and not metrics['temperature_fit_on_confirmation']
    assert read_json(pub/'learning_metrics.json')['information_gate_passed']
    learning=read_json(pub/'calibration_learning_qualification.json')
    assert learning['all_controls_passed'] and learning['runtime_gate_passed']
    assert metrics['comparison']['loss_reduction']>=.001 and metrics['comparison']['paired_group95'][0]>0
    frozen=read_json(pub/'frozen_temperature.json');assert frozen['learning_gate_passed']
    assert metrics['temperature']==frozen['temperature']==decision['temperature']
    role='calibration_confirmation';gate=read_json(pub/(role+'_qualification.json'))
    assert all(gate[k] for k in ['complete','all_controls_passed','runtime_gate_passed','csv_verified','both_t4_verified'])
    assert gate['runtime']['projection_seconds']<32400
    bundle=read_json(pub/(role+'_bundle.json'));verify_sources(REPO,bundle['source_sha256'])
    assert bundle['temperature']==frozen['temperature']
    assert bundle['frozen_temperature_sha256']==sha256(pub/'frozen_temperature.json')
    assert sha256(root/'notebooks'/role/'job.ipynb')==bundle['notebook_sha256']
    job=root/'jobs'/role;push=read_json(job/'push.json')
    ref=push['ref'].removeprefix('/code/');version=push['version_number'];exact=ref+'/'+str(version)
    assert gate['source']['kernel_version']==exact and gate['source']['notebook_sha256']==bundle['notebook_sha256']
    proof=read_json(pub/'calibration_confirmation_source_verification.json')
    assert proof['passed'] and proof['kernel_version']==exact and proof['notebook_sha256']==bundle['notebook_sha256']
    assert sha256(job/'output/submission.csv')==gate['csv_sha256']
    return dict(role=role,reference=ref,version=version,exact=exact,temperature=frozen['temperature'],
        notebook_sha256=bundle['notebook_sha256'],qualification_sha256=sha256(pub/(role+'_qualification.json')),
        decision_sha256=sha256(pub/'deployment_decision.json'),projection_seconds=gate['runtime']['projection_seconds'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--observe-only',action='store_true')
    parser.add_argument('--dry-run',action='store_true');args=parser.parse_args()
    root=ROOT/'runs/span_011';pub=REPO/'results/span_011';folder=root/'official/calibration'
    folder.mkdir(parents=True,exist_ok=True);api=api_client()
    with exclusive_lock(folder/'controller.lock'):
        def save(name,value):
            atomic_json(folder/name,value);atomic_json(pub/('official_'+name),value)
        def state(stage,**values):
            value=dict(at=utc_now(),stage=stage,**values);save('state.json',value);print(json.dumps(value),flush=True)
        if not (folder/'receipt.json').exists():
            assert not args.observe_only,'No accepted receipt to observe'
            assert not (folder/'submission_intent.json').exists(),'Reconcile an ambiguous intent; never resubmit'
            registry=read_json(REPO/'backlog/hypotheses.json')
            assert registry['active_version']==11 and next(i for i in registry['items'] if i['id']=='H59')['status']=='running'
            source=candidate(root,pub)
            assert str(read_call(lambda:api.kernels_status(source['exact'])).status).split('.')[-1]=='COMPLETE'
            plan=read_json(pub/'span_plan.json');fresh=public_inventory(inventory(api,plan['started_at']));q=fresh['quota']
            assert not fresh['nonterminal_jobs'] and q['reserved_seconds']==0 and not q['paid_scaling']
            assert q['refresh_time']==plan['initial_quota_refresh_time']
            charge=q['used_seconds']-plan['quota_used_at_start'];assert -.1<=charge<=21600.1
            assert q['remaining_seconds']>=32400
            assert datetime.now(timezone.utc).timestamp()+32400+600<datetime.fromisoformat(plan['deadline']).timestamp()
            assert len(list((root/'official').glob('*/receipt.json')))<plan['official_candidate_limit']
            message='Span011 frozen independent later-vote shared temperature; unchanged equal two-expert inference'
            prior=list(read_call(lambda:api.competition_submissions('llm-classification-finetuning')) or [])
            assert all(str(s.status).split('.')[-1] in TERMINAL for s in prior),'Official scoring already pending'
            assert not any(getattr(s,'description','')==message for s in prior),'Reconcile possible accepted duplicate'
            save('admission.json',dict(at=utc_now(),inventory=fresh,public_gpu_charged_seconds=charge,**source))
            if args.dry_run:state('dry_run_admissible',**source);return
            verify_sources(REPO,{str(Path(__file__).relative_to(REPO)):sha256(Path(__file__)),
                'scripts/submit_span_010_order.py':sha256(REPO/'scripts/submit_span_010_order.py')})
            intent=dict(at=utc_now(),**source,quota_before=q,hidden_seconds_cap=32400,message=message,
                controller_sha256=sha256(Path(__file__)))
            write_intent(folder/'submission_intent.json',intent);atomic_json(pub/'official_submission_intent.json',intent)
            state('submitting_one_frozen_version',kernel=source['exact'],temperature=source['temperature'])
            reply=api.competition_submit_code(file_name='submission.csv',message=message,competition='llm-classification-finetuning',
                                            kernel=source['reference'],kernel_version=source['version'])
            receipt=fields(reply);save('receipt.json',receipt)
            assert receipt.get('ref'),'Ambiguous accepted state; preserve intent and reconcile'
            state('accepted',receipt=receipt['ref'])
        receipt=read_json(folder/'receipt.json');plan=read_json(pub/'span_plan.json')
        while time.time()<datetime.fromisoformat(plan['deadline']).timestamp():
            try:
                submissions=list(read_call(lambda:api.competition_submissions('llm-classification-finetuning')) or [])
                matches=[fields(s) for s in submissions if str(s.ref)==str(receipt['ref'])]
                if matches:
                    assert len(matches)==1;result=matches[0];save('result.json',result)
                    status=result['status'].split('.')[-1];state('monitoring',receipt=receipt['ref'],status=status)
                    if status in TERMINAL:
                        if status=='COMPLETE':
                            score=float(result['public_score']);live=leaderboard(api);save('leaderboard.json',live)
                            state('terminal',receipt=receipt['ref'],score=score,leaderboard=live)
                        else:state('terminal',receipt=receipt['ref'],status=status)
                        return
                else:state('receipt_not_yet_visible',receipt=receipt['ref'])
            except Exception as error:state('observation_retry',receipt=receipt['ref'],error_type=type(error).__name__)
            time.sleep(45)
        state('original_observation_deadline_reached',receipt=receipt['ref'])


if __name__=='__main__':main()
