"""One H76 public qualification under the inherited deadline and quota anchor."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,read_json,sha256,utc_now,verify_sources
from arena.kaggle_control import run_job
from arena.span004_support import api_client,quota
from submit_span_013_order import submissions,TERMINAL,dataset_dependencies


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();root=ROOT/'runs/span_030';pub=REPO/'results/span_030';cfg=REPO/'configs/span_030.json'
    if args.prepare:
        assert not (pub/'span_plan.json').exists(),'Never reset accounting'
        assert read_json(pub/'quality_replay.json')['gross_regression_guard_passed']
        assert read_json(pub/'candidate_preparation.json')['prepared']
        old=read_json(REPO/'results/span_028/span_plan.json');q=quota(api_client())
        assert not q['paid_scaling'] and q['reserved_seconds']==0 and q['remaining_seconds']>=3600+32400
        assert q['refresh_time']==old['initial_quota_refresh_time']
        assert datetime.now(timezone.utc).timestamp()+3600+32400+600<datetime.fromisoformat(old['deadline']).timestamp()
        at=utc_now();plan=dict(started_at=at,deadline=old['deadline'],initial_quota=q,quota_used_at_start=q['used_seconds'],
            initial_quota_refresh_time=q['refresh_time'],gpu_session_seconds_cap=3600,public_version_limit=1,
            official_candidate_limit=1,initial_public_cap_seconds=3600,hidden_cap_seconds=32400,
            packing_tokens=8192,conversation_tokens=4096,raw_probability_tolerance=1e-5,
            recipe='normalized(.4 Gemma original + .6 aligned Llama swapped), no route/reverse, T1',
            parent_profile_public_charged_seconds=read_json(REPO/'results/span_028/closure_audit.json')['public_gpu_charged_seconds'],
            prior_weekly_used_seconds=read_json(REPO/'results/span_029/resource_close.json')['quota']['used_seconds'])
        atomic_json(pub/'span_plan.json',plan)
        atomic_json(cfg,dict(started_at=at,deadline=plan['deadline'],job_seconds_caps=dict(weight_qualification=3600)))
        print(dict(prepared=True,new_gpu_executions=0,quota=q,deadline=plan['deadline']));return
    def precondition(role):
        registry=read_json(REPO/'backlog/hypotheses.json');assert registry['active_version']==30
        assert next(v for v in registry['versions'] if v['number']==30)['focus_ids']==['H76']
        assert next(i for i in registry['items'] if i['id']=='H76')['status']=='running' and role=='weight_qualification'
        quality=read_json(pub/'quality_replay.json');assert quality['gross_regression_guard_passed'] and quality['expert_weights']==[.4,.6]
        verify_sources(REPO,quality['source_sha256'])
        assert read_json(pub/(role+'_cell_rehearsals.json'))['passed']
        tests=read_json(pub/'candidate_tests.json');assert tests['passed']
        for n,h in tests['source_sha256'].items():assert sha256(REPO/n)==h,n
        prep=read_json(pub/'candidate_preparation.json')
        for n,k in [('qualification_spec.json','qualification_spec_sha256'),('hidden_spec.json','hidden_spec_sha256')]:
            assert sha256(root/n)==prep[k]
        for n,h in prep['canonical_control_sha256'].items():assert sha256(root/'controls'/n)==h
        preserve=read_json(pub/'preservation.json')
        for n,h in {**preserve['snapshot_sha256'],**preserve['user_notebook_sha256']}.items():assert sha256(REPO/n)==h,n
        assert not list((root/'jobs').glob('*/push_intent.json')) and not list((root/'jobs').glob('*/push.json'))
        assert not list((root/'official').glob('*/submission_intent.json'))
        api=api_client();history=submissions(api);assert all(str(s.status).split('.')[-1] in TERMINAL for s in history)
        deps=dataset_dependencies(api,read_json(root/'notebooks'/role/'kernel-metadata.json'))
        fresh=read_json(pub/'jobs'/role/'admission_inventory.json')
        assert fresh['quota']['remaining_seconds']>=3600+32400 and not fresh['nonterminal_jobs']
        atomic_json(pub/'qualification_admission.json',dict(at=utc_now(),admitted=True,all_official_terminal=True,
            official_receipts=len(history),public_cap_seconds=3600,phase_cap_seconds=3600,dependencies=deps,
            expert_weights=[.4,.6],official_submission_eligible=False))
    run_job(REPO,root,pub,cfg,'weight_qualification',dry_run=args.dry_run,precondition=precondition)


if __name__=='__main__':main()
