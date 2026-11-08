"""One H74profile under the fixed v27phase and inherited deadline/quota anchor."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,read_json,sha256,utc_now
from arena.kaggle_control import run_job
from arena.span004_support import api_client,quota
from submit_span_013_order import submissions,TERMINAL,dataset_dependencies


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();root=ROOT/'runs/span_027';pub=REPO/'results/span_027';cfg=REPO/'configs/span_027.json'
    if args.prepare:
        assert not (pub/'span_plan.json').exists(),'Never reset accounting'
        assert read_json(pub/'packing_cpu_audit.json')['cpu_packing_proof_passed']
        assert read_json(pub/'profile_preparation.json')['prepared']
        old=read_json(REPO/'results/span_026/span_plan.json');q=quota(api_client())
        assert not q['paid_scaling'] and q['reserved_seconds']==0 and q['remaining_seconds']>=7200+32400
        assert q['refresh_time']==old['initial_quota_refresh_time']
        assert datetime.now(timezone.utc).timestamp()+7200+32400+600<datetime.fromisoformat(old['deadline']).timestamp()
        at=utc_now();plan=dict(started_at=at,deadline=old['deadline'],initial_quota=q,quota_used_at_start=q['used_seconds'],
            initial_quota_refresh_time=q['refresh_time'],gpu_session_seconds_cap=7200,public_version_limit=2,
            official_candidate_limit=1,initial_public_cap_seconds=3600,followup_public_cap_seconds=3600,
            hidden_cap_seconds=32400,packing_tokens=10240,conversation_tokens=4096,raw_probability_tolerance=1e-5,
            prior_span_public_charged_seconds=read_json(REPO/'results/span_026/closure_audit.json')['total_public_gpu_charged_seconds'],
            prior_weekly_used_seconds=read_json(REPO/'results/span_026/terminal_inventory.json')['quota']['used_seconds'])
        atomic_json(pub/'span_plan.json',plan)
        atomic_json(cfg,dict(started_at=at,deadline=plan['deadline'],job_seconds_caps=dict(packing_profile=3600,weight_qualification=3600)))
        print(dict(prepared=True,new_gpu_executions=0,quota=q,deadline=plan['deadline']));return
    def precondition(role):
        registry=read_json(REPO/'backlog/hypotheses.json');assert registry['active_version']==27
        assert set(next(v for v in registry['versions'] if v['number']==27)['focus_ids'])=={'H74','H73'}
        assert next(i for i in registry['items'] if i['id']=='H74')['status']=='running' and role=='packing_profile'
        assert read_json(pub/'packing_cpu_audit.json')['cpu_packing_proof_passed']
        assert read_json(pub/(role+'_cell_rehearsals.json'))['passed']
        prep=read_json(pub/'profile_preparation.json');assert sha256(root/'profile_spec.json')==prep['profile_spec_sha256']
        for n,h in prep['canonical_control_sha256'].items():assert sha256(root/'controls'/n)==h
        preserve=read_json(pub/'preservation.json')
        for n,h in {**preserve['snapshot_sha256'],**preserve['user_notebook_sha256']}.items():assert sha256(REPO/n)==h
        assert not list((root/'jobs').glob('*/push_intent.json')) and not list((root/'jobs').glob('*/push.json'))
        api=api_client();history=submissions(api);assert all(str(s.status).split('.')[-1] in TERMINAL for s in history)
        deps=dataset_dependencies(api,read_json(root/'notebooks'/role/'kernel-metadata.json'))
        fresh=read_json(pub/'jobs'/role/'admission_inventory.json')
        assert fresh['quota']['remaining_seconds']>=7200+32400 and not fresh['nonterminal_jobs']
        atomic_json(pub/'profile_admission.json',dict(at=utc_now(),admitted=True,all_official_terminal=True,
            official_receipts=len(history),public_cap_seconds=3600,phase_cap_seconds=7200,dependencies=deps,
            information_only=True,official_submission_eligible=False))
    run_job(REPO,root,pub,cfg,'packing_profile',dry_run=args.dry_run,precondition=precondition)


if __name__=='__main__':main()
