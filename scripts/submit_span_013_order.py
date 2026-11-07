"""Submit one qualified frozen v13 recipe, or observe its durable receipt."""
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
from arena.experiment import read_json,atomic_json,write_intent,exclusive_lock,sha256,verify_sources,utc_now,collect_pages
from arena.span004_support import api_client,read_call,inventory,public_inventory
from submit_span_010_order import leaderboard,fields

TERMINAL={'COMPLETE','ERROR','FAILED','INVALID','CANCELED','CANCELLED'}


def submissions(api):
    """The SDK convenience method discards cursors and defaults to20records."""
    from kaggle.api.kaggle_api_extended import ApiListSubmissionsRequest
    def fetch(cursor):
        request=ApiListSubmissionsRequest();request.competition_name='llm-classification-finetuning'
        api._set_paging(request,100,cursor)
        with api.build_kaggle_client() as client:
            return read_call(lambda:client.competitions.competition_api_client.list_submissions(request))
    return collect_pages(fetch,records=lambda result:result.submissions)


def dataset_dependencies(api,metadata):
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
    records=[]
    for source in metadata['dataset_sources']:
        owner,slug,version=source.split('/')
        assert owner=='tascj0' and version=='1','Unexpected or unpinned dependency'
        request=ApiGetDatasetRequest();request.owner_slug=owner;request.dataset_slug=slug
        with api.build_kaggle_client() as client:
            dataset=read_call(lambda:client.datasets.dataset_api_client.get_dataset(request))
        assert dataset.current_version_number==1 and not dataset.is_private,'Pinned public asset changed'
        records.append(dict(source=source,public=True,current_version=dataset.current_version_number))
    assert len(records)==4
    return dict(at=utc_now(),datasets=records)


def check_gate(role,gate,proof,bundle,replay):
    assert role==gate['role']==bundle['role'],'Qualification belongs to another attempt'
    assert all(gate[k] for k in ['complete','all_controls_passed','runtime_gate_passed','csv_verified','both_t4_verified'])
    assert not gate['independent_local_quality'] and not replay['independent_local_quality']
    assert replay['information_gate_passed'] and replay['quality_variants_tested']==1 and replay['temperature']==1
    assert replay['coverage']['uniform']['loss_reduction']>.0005
    assert proof['passed'] and proof['cell_sources_identical'] and proof['internet_enabled'] is False
    assert gate['source']['kernel_version']==proof['kernel_version']
    assert gate['source']['notebook_sha256']==proof['notebook_sha256']==bundle['notebook_sha256']
    assert set(gate['control_identity'])=={'m0','m3','reverse_control','reverse_uniform','reverse_long','reverse_test'}
    assert all(v['passed'] and 0<=v['max_abs_probability_difference']<1e-5 for v in gate['control_identity'].values())
    assert all(gate['coverage'][name]['route_replayed'] and gate['coverage'][name]['mixture_replayed']
               for name in ['control','uniform','long','test'])
    runtime=gate['runtime'];assert runtime['margin']==1.2 and runtime['reserve_seconds']==600
    calculated=25000*(runtime['base_uniform_seconds']+runtime['reverse_routed_uniform_seconds'])/512*1.2
    calculated+=runtime['cold_seconds']+runtime['full_warmup_seconds']+runtime['other_seconds']+600
    assert math.isclose(calculated,runtime['projection_seconds'],rel_tol=1e-12)
    assert 0<calculated<32400 and all(math.isfinite(v) and v>=0 for v in runtime.values())
    assert bundle['hidden_cap_seconds']==32400 and bundle['internet_enabled'] is False


def candidate(root,pub,role):
    gate=read_json(pub/(role+'_qualification.json'));bundle=read_json(pub/(role+'_bundle.json'))
    proof=read_json(pub/(role+'_source_verification.json'));replay=read_json(pub/'H60_cached_replay.json')
    check_gate(role,gate,proof,bundle,replay);verify_sources(REPO,bundle['source_sha256'])
    assert sha256(pub/'profile_selection.json')==bundle['public_profile_selection_sha256']
    assert sha256(root/'notebooks'/role/'job.ipynb')==bundle['notebook_sha256']
    assert sha256(root/'notebooks'/role/'kernel-metadata.json')==bundle['metadata_sha256']
    job=root/'jobs'/role;push=read_json(job/'push.json')
    ref=push['ref'].removeprefix('/code/');version=push['version_number'];exact=ref+'/'+str(version)
    assert proof['kernel_version']==exact and read_json(job/'status.json')['status']=='COMPLETE'
    assert sha256(job/'output/submission.csv')==gate['csv_sha256']
    for name,digest in read_json(job/'retrieval.json')['artifact_sha256'].items():assert sha256(job/'output'/name)==digest
    names=[role+'_qualification.json',role+'_bundle.json',role+'_source_verification.json','H60_cached_replay.json']
    decision=dict(at=utc_now(),submit=True,role=role,kernel_version=exact,temperature=1,packing_tokens=8192,
        recipe='On fixed route: .25 Gemma original + .25 aligned Gemma swapped + .5 aligned Llama swapped; off-route base',
        parameters_selected_without_leaderboard_feedback=True,independent_local_quality=False,
        interpretation='Training-overlapping descriptive information only; official result tests target transfer',
        evidence_sha256={name:sha256(pub/name) for name in names},projection_seconds=gate['runtime']['projection_seconds'])
    return dict(role=role,reference=ref,version=version,exact=exact,notebook_sha256=bundle['notebook_sha256'],
                qualification_sha256=sha256(pub/(role+'_qualification.json')),projection_seconds=gate['runtime']['projection_seconds']),decision


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--observe-only',action='store_true')
    parser.add_argument('--prepare-decision',action='store_true')
    parser.add_argument('--role',choices=['gemma_order','gemma_order_followup'],default='gemma_order');args=parser.parse_args()
    root=ROOT/'runs/span_013';pub=REPO/'results/span_013';folder=root/'official/candidate'
    folder.mkdir(parents=True,exist_ok=True)
    with exclusive_lock(folder/'controller.lock'):
        def save(name,value):
            atomic_json(folder/name,value);atomic_json(pub/('official_'+name),value)
        def state(stage,**values):
            value=dict(at=utc_now(),stage=stage,**values);save('state.json',value);print(json.dumps(value),flush=True)
        if not (folder/'receipt.json').exists():
            assert not args.observe_only,'No accepted receipt to observe'
            assert not (folder/'submission_intent.json').exists(),'Reconcile ambiguous intent; never resubmit'
            registry=read_json(REPO/'backlog/hypotheses.json')
            assert registry['active_version']==13 and next(i for i in registry['items'] if i['id']=='H60')['status']=='running'
            source,decision=candidate(root,pub,args.role)
            if args.prepare_decision:
                atomic_json(pub/'deployment_decision.json',decision);print(json.dumps(decision));return
            frozen=read_json(pub/'deployment_decision.json')
            assert {k:v for k,v in frozen.items() if k!='at'}=={k:v for k,v in decision.items() if k!='at'}
            verify_sources(REPO,{str(Path(__file__).relative_to(REPO)):sha256(Path(__file__)),
                'scripts/submit_span_010_order.py':sha256(REPO/'scripts/submit_span_010_order.py'),
                'results/span_013/deployment_decision.json':sha256(pub/'deployment_decision.json')})
            api=api_client()
            assert str(read_call(lambda:api.kernels_status(source['exact'])).status).split('.')[-1]=='COMPLETE'
            plan=read_json(pub/'span_plan.json');fresh=public_inventory(inventory(api,plan['started_at']));q=fresh['quota']
            assert not fresh['nonterminal_jobs'] and q['reserved_seconds']==0 and not q['paid_scaling']
            assert q['refresh_time']==plan['initial_quota_refresh_time']
            charge=q['used_seconds']-plan['quota_used_at_start'];assert -.1<=charge<=6000.1 and q['remaining_seconds']>=32400
            assert datetime.now(timezone.utc).timestamp()+32400+600<datetime.fromisoformat(plan['deadline']).timestamp()
            assert len(list((root/'official').glob('*/receipt.json')))<plan['official_candidate_limit']
            message='Span013 frozen Gemma-only order averaging; original route, equal expert totals, T1'
            prior=submissions(api)
            assert all(str(s.status).split('.')[-1] in TERMINAL for s in prior),'Official scoring pending'
            assert not any(getattr(s,'description','')==message for s in prior),'Reconcile possible duplicate'
            dependencies=dataset_dependencies(api,read_json(root/'notebooks'/source['role']/'kernel-metadata.json'))
            save('admission.json',dict(at=utc_now(),inventory=fresh,dependencies=dependencies,
                submission_history_complete=True,prior_submissions=len(prior),public_gpu_charged_seconds=charge,**source))
            intent=dict(at=utc_now(),**source,quota_before=q,hidden_seconds_cap=32400,message=message,
                decision_sha256=sha256(pub/'deployment_decision.json'),controller_sha256=sha256(Path(__file__)))
            write_intent(folder/'submission_intent.json',intent);atomic_json(pub/'official_submission_intent.json',intent)
            state('submitting_one_frozen_version',kernel=source['exact'])
            reply=api.competition_submit_code(file_name='submission.csv',message=message,competition='llm-classification-finetuning',
                                            kernel=source['reference'],kernel_version=source['version'])
            receipt=fields(reply);save('receipt.json',receipt)
            assert receipt.get('ref'),'Ambiguous accepted state; preserve intent and reconcile'
            state('accepted',receipt=receipt['ref'])
        elif args.prepare_decision:
            raise RuntimeError('Already submitted; preserve frozen decision')
        api=api_client();receipt=read_json(folder/'receipt.json');plan=read_json(pub/'span_plan.json')
        while time.time()<datetime.fromisoformat(plan['deadline']).timestamp():
            try:
                matches=[fields(s) for s in submissions(api) if str(s.ref)==str(receipt['ref'])]
                if matches:
                    assert len(matches)==1;result=matches[0];save('result.json',result)
                    status=result['status'].split('.')[-1];state('monitoring',receipt=receipt['ref'],status=status)
                    if status in TERMINAL:
                        if status=='COMPLETE':
                            live=leaderboard(api);save('leaderboard.json',live)
                            state('terminal',receipt=receipt['ref'],score=float(result['public_score']),leaderboard=live)
                        else:state('terminal',receipt=receipt['ref'],status=status)
                        return
                else:state('receipt_not_yet_visible',receipt=receipt['ref'])
            except Exception as error:state('observation_retry',receipt=receipt['ref'],error_type=type(error).__name__)
            time.sleep(45)
        state('original_observation_deadline_reached',receipt=receipt['ref'])


if __name__=='__main__':main()
