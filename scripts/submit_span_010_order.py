"""One audited official candidate; observe its durable receipt without re-submission."""
from pathlib import Path
import argparse
import json
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,write_intent,exclusive_lock,sha256,verify_sources,utc_now
from arena.span004_support import api_client,read_call,inventory,public_inventory,quota


def leaderboard(api):
    from kaggle.api.kaggle_api_extended import ApiGetLeaderboardRequest
    rows=[];cursor=None;seen=set()
    for _ in range(100):
        request=ApiGetLeaderboardRequest();request.competition_name='llm-classification-finetuning'
        api._set_paging(request,100,cursor)
        with api.build_kaggle_client() as client:
            response=read_call(lambda:client.competitions.competition_api_client.get_leaderboard(request))
        rows.extend(x.to_dict() for x in response.submissions);cursor=response.next_page_token
        if not cursor:break
        if cursor in seen:raise RuntimeError('Repeated leaderboard cursor')
        seen.add(cursor)
    else:raise RuntimeError('Incomplete leaderboard')
    ours=[(i+1,r) for i,r in enumerate(rows) if r.get('teamId')==17244411]
    assert len(ours)==1 and len(rows)>=20
    rank,row=ours[0]
    return dict(observed_at=utc_now(),complete=True,team_id=17244411,rank=rank,teams=len(rows),
        score=float(row['score']),rank19_score=float(rows[18]['score']),rank20_score=float(rows[19]['score']),
        leader_score=float(rows[0]['score']),target_achieved=rank<=19,
        latest_team_submission_date=row.get('submissionDate'))


def fields(obj):
    names=['ref','status','public_score','private_score','date','description']
    return {k:str(getattr(obj,k)) if k in ['status','date'] else getattr(obj,k) for k in names if hasattr(obj,k)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--observe-only',action='store_true')
    parser.add_argument('--role',choices=['order_pilot','order_packed'],default='order_pilot')
    args=parser.parse_args();root=ROOT/'runs/span_010';pub=REPO/'results/span_010'
    role=args.role
    folder=root/'official'/role;folder.mkdir(parents=True,exist_ok=True)
    api=api_client()
    with exclusive_lock(folder/'controller.lock'):
        def save(name,value):
            atomic_json(folder/name,value);atomic_json(pub/('official_'+name),value)
        def state(stage,**values):
            value=dict(at=utc_now(),stage=stage,**values);save('state.json',value)
            print(json.dumps(value),flush=True)
        if not (folder/'receipt.json').exists():
            assert not args.observe_only,'No accepted receipt to observe'
            assert not (folder/'submission_intent.json').exists(),'Reconcile existing intent; never resubmit'
            verify_sources(REPO,{str(Path(__file__).relative_to(REPO)):sha256(Path(__file__)),
                'scripts/audit_span_010_order.py':sha256(REPO/'scripts/audit_span_010_order.py')})
            qualification=pub/('H57_'+role+'_qualification.json')
            gate=read_json(qualification)
            assert gate['role']==role,'Qualification belongs to another experiment attempt'
            assert all(gate[k] for k in ['complete','all_controls_passed','runtime_gate_passed','csv_verified','both_t4_verified'])
            assert not gate['independent_local_quality'] and gate['runtime_projection_seconds']<32400
            bundle=read_json(pub/(role+'_bundle.json'));verify_sources(REPO,bundle['source_sha256'])
            job=root/'jobs'/role;push=read_json(job/'push.json')
            reference=push['ref'].removeprefix('/code/');version=push['version_number'];exact=reference+'/'+str(version)
            assert gate['source']['kernel_version']==exact
            assert gate['source']['notebook_sha256']==bundle['notebook_sha256']
            assert sha256(root/'notebooks'/role/'job.ipynb')==bundle['notebook_sha256']
            assert sha256(job/'output/submission.csv')==gate['csv_sha256']
            assert str(read_call(lambda:api.kernels_status(exact)).status).endswith('COMPLETE')
            plan=read_json(pub/'span_plan.json');fresh=public_inventory(inventory(api,plan['started_at']))
            q=fresh['quota'];assert not fresh['nonterminal_jobs'] and q['reserved_seconds']==0 and not q['paid_scaling']
            assert q['refresh_time']==plan['initial_quota_refresh_time'],'Quota refresh requires explicit re-admission'
            charge=q['used_seconds']-plan['quota_used_at_start'];assert -.1<=charge<=21600.1
            assert len(list((root/'official').glob('*/receipt.json')))<plan['official_candidate_limit']
            message=f'Span010 {role} frozen10pct-work equal-order completion; no local-CV claim'
            prior=list(read_call(lambda:api.competition_submissions('llm-classification-finetuning')) or [])
            assert not any(getattr(x,'description','')==message for x in prior),'Reconcile possible accepted duplicate'
            intent=dict(at=utc_now(),kernel=reference,version=version,notebook_sha256=bundle['notebook_sha256'],
                qualification_sha256=sha256(qualification),controller_sha256=sha256(Path(__file__)),
                quota_before=q,public_gpu_charged_seconds=charge,hidden_seconds_cap=32400,
                projection_seconds=gate['runtime_projection_seconds'],message=message)
            write_intent(folder/'submission_intent.json',intent);atomic_json(pub/'official_submission_intent.json',intent)
            state('submitting_one_frozen_version',kernel=exact)
            response=api.competition_submit_code(file_name='submission.csv',message=message,
                competition='llm-classification-finetuning',kernel=reference,kernel_version=version)
            receipt=fields(response);save('receipt.json',receipt)
            assert receipt.get('ref'),'Accepted state ambiguous; preserve intent and reconcile'
            state('accepted',receipt=receipt['ref'])
        receipt=read_json(folder/'receipt.json');intent=read_json(folder/'submission_intent.json')
        deadline=time.monotonic()+3600
        while time.monotonic()<deadline:
            try:
                submissions=list(read_call(lambda:api.competition_submissions('llm-classification-finetuning')) or [])
                matches=[fields(x) for x in submissions if str(x.ref)==str(receipt['ref'])]
                if matches:
                    assert len(matches)==1
                    result=matches[0];save('result.json',result);status=result['status'].split('.')[-1]
                    state('monitoring',receipt=receipt['ref'],status=status)
                    if status in ['COMPLETE','ERROR','FAILED','INVALID','CANCELED','CANCELLED']:
                        after=quota(api)
                        save('accounting.json',dict(at=utc_now(),quota_after=after,
                            same_quota_period=after['refresh_time']==intent['quota_before']['refresh_time'],
                            hidden_seconds_cap=32400,method='Receipt wall time includes queue/obfuscation, not measured GPU time.'))
                        if status=='COMPLETE':
                            float(result['public_score']);live=leaderboard(api);save('leaderboard.json',live)
                            state('terminal',receipt=receipt['ref'],score=result['public_score'],leaderboard=live)
                        else:state('terminal',receipt=receipt['ref'],status=status)
                        return
                else:state('receipt_not_yet_visible',receipt=receipt['ref'])
            except Exception as error:state('observation_retry',error_type=type(error).__name__)
            time.sleep(45)
        state('observation_window_ended_resume_same_receipt',receipt=receipt['ref'])


if __name__=='__main__':main()
