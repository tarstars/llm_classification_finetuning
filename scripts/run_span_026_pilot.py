"""Prepare fixed resource accounting, then admit/observe the one H72pilot."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,read_json,sha256,utc_now
from arena.kaggle_control import run_job
from arena.span004_support import api_client,quota,read_call
from submit_span_013_order import submissions,TERMINAL


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--dry-run',action='store_true');args=parser.parse_args()
    root=ROOT/'runs/span_026';pub=REPO/'results/span_026';cfg=REPO/'configs/span_026.json'
    if args.prepare:
        assert not (pub/'span_plan.json').exists(),'Never reset phase accounting'
        cohort=read_json(pub/'pilot_preparation.json');assert cohort['prepared'] and not cohort['uses_targets']
        q=quota(api_client());assert not q['paid_scaling'] and q['reserved_seconds']==0
        inherited=read_json(REPO/'results/span_019/span_plan.json')
        assert q['refresh_time']==inherited['initial_quota_refresh_time'] and q['remaining_seconds']>=10800+32400
        assert datetime.now(timezone.utc).timestamp()+10800+32400+600<datetime.fromisoformat(inherited['deadline']).timestamp()
        started=utc_now()
        plan=dict(started_at=started,deadline=inherited['deadline'],initial_quota=q,
            quota_used_at_start=q['used_seconds'],initial_quota_refresh_time=q['refresh_time'],
            gpu_session_seconds_cap=10800,public_version_limit=2,official_candidate_limit=1,
            initial_public_cap_seconds=7200,followup_public_cap_seconds=3600,
            hidden_cap_seconds=32400,raw_probability_tolerance=1e-5,
            gross_regression_threshold=-.005,bootstrap_seed=42,bootstrap_resamples=2000,
            target_rank_maximum=19,parent_official_receipt=58103553,parent_official_score=.82987)
        atomic_json(pub/'span_plan.json',plan)
        atomic_json(cfg,dict(started_at=started,deadline=plan['deadline'],
            job_seconds_caps=dict(context_pilot=7200,context_qualification=3600)))
        print(dict(prepared=True,quota=q,deadline=plan['deadline'],new_gpu_executions=0));return
    def precondition(role):
        registry=read_json(REPO/'backlog/hypotheses.json')
        assert registry['active_version']==26 and next(v for v in registry['versions'] if v['number']==26)['focus_ids']==['H72']
        assert next(i for i in registry['items'] if i['id']=='H72')['status']=='running'
        assert role=='context_pilot' and read_json(pub/(role+'_cell_rehearsals.json'))['passed']
        cohort=read_json(pub/'pilot_preparation.json');assert cohort['prepared']
        assert sha256(root/'pilot_spec.json')==cohort['spec_sha256']
        for n,h in cohort['control_sha256'].items():assert sha256(root/'controls'/n)==h
        preservation=read_json(pub/'preservation.json')
        for n,h in {**preservation['snapshot_sha256'],**preservation['user_notebook_sha256']}.items():assert sha256(REPO/n)==h
        assert not list((root/'jobs').glob('*/push.json')) and not list((root/'jobs').glob('*/push_intent.json'))
        api=api_client()
        from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
        for slug in ['lmsys-checkpoints-0-0805','lmsys-checkpoints-3-0805','lmsys-modules-0805','lmsys-packages']:
            request=ApiGetDatasetRequest();request.owner_slug='tascj0';request.dataset_slug=slug
            with api.build_kaggle_client() as client:
                data=read_call(lambda:client.datasets.dataset_api_client.get_dataset(request))
            assert data.current_version_number==1 and not data.is_private
        history=submissions(api);assert all(str(x.status).split('.')[-1] in TERMINAL for x in history)
        fresh=read_json(pub/'jobs'/role/'admission_inventory.json')
        assert fresh['quota']['remaining_seconds']>=10800+32400
        atomic_json(pub/'pilot_admission.json',dict(at=utc_now(),admitted=True,
            all_official_terminal=True,official_receipts=len(history),public_cap_seconds=7200,
            full_controls_required=True,information_only=True,official_submission_eligible=False))
    run_job(REPO,root,pub,cfg,'context_pilot',dry_run=args.dry_run,precondition=precondition)


if __name__=='__main__':main()
