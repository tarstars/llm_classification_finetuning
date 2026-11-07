"""Prepare or run/observe one exact public pilot; never retry a GPU mutation."""
from datetime import datetime,timedelta,timezone
from pathlib import Path
import argparse
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,read_json,exclusive_lock,utc_now
from arena.kaggle_control import run_job
from arena.span004_support import api_client,quota


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--dry-run',action='store_true');args=parser.parse_args()
    root=ROOT/'runs/span_010';pub=REPO/'results/span_010';config=REPO/'configs/span_010.json'
    root.mkdir(parents=True,exist_ok=True)
    with exclusive_lock(root/'preparation.lock'):
        if not (pub/'span_plan.json').exists():
            assert args.prepare,'Prepare/freeze the phase explicitly before admission'
            fresh=quota(api_client());assert not fresh['paid_scaling'] and fresh['reserved_seconds']==0
            plan=dict(started_at=utc_now(),initial_quota_refresh_time=fresh['refresh_time'],
                quota_used_at_start=fresh['used_seconds'],gpu_session_seconds_cap=21600,
                deadline=(datetime.now(timezone.utc)+timedelta(hours=24)).isoformat(),
                target_rank_maximum=19,public_pilot_seconds_cap=5400,official_candidate_limit=2,
                initial_quota=fresh)
            atomic_json(pub/'span_plan.json',plan)
            atomic_json(config,dict(started_at=plan['started_at'],deadline=plan['deadline'],
                                   job_seconds_caps={'order_pilot':5400}))
        if args.prepare:
            print('Phase prepared; no GPU mutation');return
    def precondition(role):
        assert read_json(pub/(role+'_cell_rehearsals.json'))['passed']
        registry=read_json(REPO/'backlog/hypotheses.json');items={i['id']:i for i in registry['items']}
        version=next(v for v in registry['versions'] if v['number']==10)
        assert registry['active_version']==10 and 'H57' in version['focus_ids']
        assert items['E15']['status']=='done' and items['H57']['status']=='running'
    run_job(REPO,root,pub,config,'order_pilot',dry_run=args.dry_run,precondition=precondition)


if __name__=='__main__':main()
