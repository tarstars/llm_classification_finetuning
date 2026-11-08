"""Submit one fully qualified H66version, or observe its original receipt."""
from datetime import datetime,timezone
from pathlib import Path
import argparse
import json
import math
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,exclusive_lock,read_json,sha256,utc_now,verify_sources,write_intent
from arena.leaderboard import collect_board
from arena.span004_support import api_client,inventory,public_inventory,quota,read_call
from submit_span_013_order import submissions,TERMINAL
from submit_span_010_order import fields


def check_gate(gate,bundle,proof):
    assert bundle['role']=='batch_candidate'
    assert gate['engineering_checks_passed'] and gate['quality_evaluated'] and gate['human_votes_read']
    assert gate['official_submission_eligible'] and gate['quality_gate_passed']
    assert gate['original_uniform_gate'] and gate['later_robustness_gate']
    assert not gate['independent_competition_cv'] and not gate['official_score_forecast']
    assert gate['original_training_overlap'] and gate['later_previously_inspected_domain_shift']
    assert gate['fixed_candidate_weights']==bundle['fixed_weights']==[.5,.5]
    assert gate['temperature']==bundle['temperature']==1
    assert bundle['padded_tokens_cap']==12288 and bundle['max_rows_per_batch']==32
    assert bundle['hidden_cap_seconds']==32400 and not bundle['internet_enabled']
    assert proof['passed'] and proof['remote_cells_match'] and proof['internet_disabled']
    assert proof['kernel_version']==gate['kernel_version']
    assert proof['notebook_sha256']==bundle['notebook_sha256']
    assert all(k in gate['native_controls'] for k in ['native_parallel_logits_max_abs',
        'native_parallel_probability_max_abs','repeated_probability_max_abs'])
    for control in [gate['native_controls'],gate['repeated_logits']]:
        assert control['passed'] and control['rows']==12 and control['tolerance']==1e-5
        values=[v for k,v in control.items() if k.endswith('_max_abs')]
        assert values and all(math.isfinite(v) and 0<=v<=1e-5 for v in values)
    assert set(gate['canonical_probability_max_abs'])=={'control','uniform','long','test','reverse_control'}
    assert all(0<=v<=1e-5 for v in gate['canonical_probability_max_abs'].values())
    assert gate['all_mirror_cohorts_recomputed'] and not gate['old_new_shape_identity_claimed']
    for name in ['uniform','source']:
        evidence=gate['evaluations'][name]
        assert evidence['bootstrap_seed']==42 and evidence['bootstrap_resamples']==2000
        assert math.isfinite(evidence['loss_reduction'])
        assert len(evidence['paired_group95'])==2 and all(math.isfinite(v) for v in evidence['paired_group95'])
    assert gate['evaluations']['uniform']['loss_reduction']>.001
    assert gate['evaluations']['source']['loss_reduction']>=0
    assert gate['evaluations']['source']['paired_group95'][0]>-.01
    r=gate['runtime']
    assert r['runtime_gate_passed'] and r['actual_rows']==512 and r['hidden_rows_assumed']==25000
    assert r['margin']==1.2 and r['reserve_seconds']==600 and r['replica_wall_time_not_divided_again']
    costs=[r[k] for k in ['retained_seconds_per_row','mirror_parallel_seconds_per_row',
        'retained_startup_warmup_other_seconds','mirror_startup_warmup_other_seconds','setup_seconds']]
    assert all(math.isfinite(v) and v>=0 for v in costs)
    expected=25000*1.2*(costs[0]+costs[1])+sum(costs[2:])+600
    assert 0<expected<32400 and math.isclose(expected,r['projected_hidden_seconds'],rel_tol=1e-12)


def candidate(root,pub):
    role='batch_candidate';gate=read_json(pub/'candidate_qualification.json')
    bundle=read_json(pub/(role+'_bundle.json'));proof=read_json(pub/'jobs'/role/'source_verification.json')
    check_gate(gate,bundle,proof);verify_sources(REPO,bundle['source_sha256'])
    job=root/'jobs'/role;push=read_json(job/'push.json')
    ref=push['ref'].removeprefix('/code/');version=push['version_number'];exact=ref+'/'+str(version)
    assert exact==proof['kernel_version'] and read_json(job/'status.json')['status']=='COMPLETE'
    assert sha256(root/'notebooks'/role/'job.ipynb')==bundle['notebook_sha256']
    assert sha256(root/'notebooks'/role/'kernel-metadata.json')==bundle['metadata_sha256']
    assert gate['source_sha256']==bundle['source_sha256']['scripts/audit_span_020_candidate.py']
    assert gate['quality_helper_sha256']==bundle['source_sha256']['src/arena/span019_quality.py']
    for name,digest in read_json(job/'retrieval.json')['artifact_sha256'].items():assert sha256(job/'output'/name)==digest
    decision=dict(role=role,kernel_version=exact,notebook_sha256=bundle['notebook_sha256'],
        qualification_sha256=sha256(pub/'candidate_qualification.json'),csv_sha256=sha256(job/'output/submission.csv'),
        weights=[.5,.5],temperature=1,padded_tokens_cap=12288,max_rows_per_batch=32,
        projected_hidden_seconds=gate['runtime']['projected_hidden_seconds'],
        parameters_selected_without_leaderboard_feedback=True,independent_competition_cv=False,
        interpretation='One fixed transfer trial; original training overlap and inspected later votes are not a hidden-score forecast')
    return dict(reference=ref,version=version,exact=exact,role=role),decision


def leaderboard(api):
    from kaggle.api.kaggle_api_extended import ApiGetLeaderboardRequest
    def fetch(cursor):
        request=ApiGetLeaderboardRequest();request.competition_name='llm-classification-finetuning'
        api._set_paging(request,100,cursor)
        with api.build_kaggle_client() as client:
            result=read_call(lambda:client.competitions.competition_api_client.get_leaderboard(request))
        return [r.to_dict() for r in result.submissions],result.next_page_token
    rows,pagination=collect_board(fetch,retries=8)
    own=[(i+1,r) for i,r in enumerate(rows) if int(r['teamId'])==17244411]
    assert len(own)==1 and len(rows)>=20
    rank,row=own[0]
    return dict(observed_at=utc_now(),complete=True,team_id=17244411,rank=rank,teams=len(rows),
        score=float(row['score']),rank19_score=float(rows[18]['score']),rank20_score=float(rows[19]['score']),
        leader_score=float(rows[0]['score']),target_achieved=rank<=19,pagination=pagination,
        latest_team_submission_date=row.get('submissionDate'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-decision',action='store_true')
    parser.add_argument('--observe-only',action='store_true');args=parser.parse_args()
    root=ROOT/'runs/span_020';pub=REPO/'results/span_020';folder=root/'official/candidate'
    folder.mkdir(parents=True,exist_ok=True)
    with exclusive_lock(folder/'controller.lock'):
        def save(name,value):
            atomic_json(folder/name,value);atomic_json(pub/('official_'+name),value)
        def state(stage,**values):
            record=dict(at=utc_now(),stage=stage,**values);save('state.json',record)
            print(json.dumps(record),flush=True)
        if args.prepare_decision:
            assert not (folder/'submission_intent.json').exists(),'Preserve existing submission'
            _,decision=candidate(root,pub);assert not (pub/'deployment_decision.json').exists()
            atomic_json(pub/'deployment_decision.json',decision);print(json.dumps(decision));return
        api=api_client();plan=read_json(pub/'span_plan.json')
        if not (folder/'receipt.json').exists() and (folder/'submission_intent.json').exists():
            # Reconcile only the original unique message; never repeat a mutation.
            intent=read_json(folder/'submission_intent.json')
            matches=[fields(r) for r in submissions(api) if getattr(r,'description','')==intent['message']]
            assert len(matches)<=1,'Ambiguous receipt identity'
            if not matches:
                state('original_intent_receipt_not_yet_visible',resubmitted=False);return
            save('receipt.json',matches[0]);state('original_receipt_reconciled',receipt=matches[0]['ref'])
        if not (folder/'receipt.json').exists():
            assert not args.observe_only,'No accepted receipt or intent to observe'
            registry=read_json(REPO/'backlog/hypotheses.json')
            assert registry['active_version']==20 and next(i for i in registry['items'] if i['id']=='H66')['status']=='running'
            source,decision=candidate(root,pub)
            assert decision==read_json(pub/'deployment_decision.json')
            verify_sources(REPO,{str(Path(__file__).relative_to(REPO)):sha256(Path(__file__)),
                'src/arena/leaderboard.py':sha256(REPO/'src/arena/leaderboard.py'),
                'results/span_020/deployment_decision.json':sha256(pub/'deployment_decision.json')})
            assert str(read_call(lambda:api.kernels_status(source['exact'])).status).split('.')[-1]=='COMPLETE'
            fresh=public_inventory(inventory(api,plan['started_at']));q=fresh['quota']
            assert not fresh['nonterminal_jobs'] and q['reserved_seconds']==0 and not q['paid_scaling']
            assert q['refresh_time']==plan['initial_quota_refresh_time'] and q['remaining_seconds']>=32400
            public_charge=q['used_seconds']-plan['quota_used_at_start'];assert -.1<=public_charge<=18000.1
            assert datetime.now(timezone.utc).timestamp()+32400+600<datetime.fromisoformat(plan['deadline']).timestamp()
            assert len(list((ROOT/'runs/span_019/jobs').glob('*/push.json')))==1
            assert len(list((root/'jobs').glob('*/push.json')))==1
            assert not list((root/'official').glob('*/receipt.json'))
            assert not list((root/'official').glob('*/submission_intent.json'))
            prior=submissions(api)
            assert all(str(r.status).split('.')[-1] in TERMINAL for r in prior),'Official scoring pending'
            message='Span020 sole12288-max32 native paired-classifier replacement; equal experts,T1'
            assert not any(getattr(r,'description','')==message for r in prior),'Reconcile existing unique submission'
            from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
            dependencies=[]
            for item in read_json(root/'notebooks/batch_candidate/kernel-metadata.json')['dataset_sources']:
                owner,slug,version=item.split('/');assert version=='1'
                request=ApiGetDatasetRequest();request.owner_slug,request.dataset_slug=owner,slug
                with api.build_kaggle_client() as client:
                    data=read_call(lambda:client.datasets.dataset_api_client.get_dataset(request))
                assert data.current_version_number==1 and data.is_private==(owner=='arseniytrushin')
                dependencies.append(dict(source=item,version=1,private=data.is_private))
            save('admission.json',dict(at=utc_now(),inventory=fresh,dependencies=dependencies,
                prior_submissions=len(prior),all_official_terminal=True,public_gpu_seconds=public_charge,**source))
            intent=dict(at=utc_now(),**source,message=message,quota_before=q,hidden_seconds_cap=32400,
                decision_sha256=sha256(pub/'deployment_decision.json'),controller_sha256=sha256(Path(__file__)))
            write_intent(folder/'submission_intent.json',intent);atomic_json(pub/'official_submission_intent.json',intent)
            state('submitting_one_frozen_version',kernel=source['exact'])
            reply=api.competition_submit_code(file_name='submission.csv',message=message,
                competition='llm-classification-finetuning',kernel=source['reference'],kernel_version=source['version'])
            receipt=fields(reply)
            if not receipt.get('ref'):
                save('response_without_receipt.json',receipt)
                state('accepted_state_ambiguous_observe_original_intent',resubmitted=False);return
            save('receipt.json',receipt)
            state('accepted',receipt=receipt['ref'])
        receipt=read_json(folder/'receipt.json')
        while time.time()<datetime.fromisoformat(plan['deadline']).timestamp():
            try:
                matches=[fields(r) for r in submissions(api) if str(r.ref)==str(receipt['ref'])]
                if matches:
                    assert len(matches)==1;result=matches[0];save('result.json',result)
                    status=result['status'].split('.')[-1];state('monitoring',receipt=receipt['ref'],status=status)
                    if status in TERMINAL:
                        save('accounting.json',dict(at=utc_now(),quota_after=quota(api),
                            hidden_seconds_cap=32400,wall_time_is_not_gpu_time=True))
                        if status=='COMPLETE':
                            score=float(result['public_score']);assert math.isfinite(score) and score>=0
                            board=leaderboard(api);save('leaderboard.json',board)
                            state('terminal',receipt=receipt['ref'],score=score,leaderboard=board)
                        else:state('terminal',receipt=receipt['ref'],status=status)
                        return
                else:state('receipt_not_yet_visible',receipt=receipt['ref'])
            except Exception as error:
                state('observation_retry',receipt=receipt['ref'],error_type=type(error).__name__)
            time.sleep(45)
        state('original_observation_deadline_reached',receipt=receipt['ref'])


if __name__=='__main__':main()
