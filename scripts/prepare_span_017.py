"""Freeze v17period/resources before any new cached experiment output."""
from datetime import datetime,timedelta,timezone
from pathlib import Path
import hashlib
import json
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,exclusive_lock,read_json,sha256,utc_now
from arena.span004_support import api_client,quota
from submit_span_010_order import leaderboard


def main():
    root,pub = ROOT/'runs/span_017',REPO/'results/span_017'
    root.mkdir(parents=True,exist_ok=True)
    pub.mkdir(parents=True,exist_ok=True)
    with exclusive_lock(root/'preparation.lock'):
        assert not (pub/'span_plan.json').exists(),'Preserve the original admission period'
        registry = read_json(REPO/'backlog/hypotheses.json')
        assert registry['active_version'] == 17
        assert next(v for v in registry['versions'] if v['number'] == 17)['focus_ids'] == ['E17','H62']
        assert all(next(i for i in registry['items'] if i['id'] == k)['status'] == 'running' for k in ['E17','H62'])
        assert read_json(REPO/'results/span_016/closure_audit.json')['passed']
        reference = subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
        names = subprocess.check_output(['git','ls-tree','-r','--name-only',reference,'--','backlog/versions'],cwd=REPO,text=True).splitlines()
        assert len(names) == 66
        snapshots = {}
        for name in names:
            digest = hashlib.sha256(subprocess.check_output(['git','show',reference+':'+name],cwd=REPO)).hexdigest()
            assert sha256(REPO/name) == digest
            snapshots[name] = digest
        users = read_json(REPO/'results/span_016/preservation.json')['user_notebook_sha256']
        for name,digest in users.items():
            assert sha256(REPO/name) == digest
        atomic_json(pub/'preservation.json',dict(at=utc_now(),snapshot_reference_commit=reference,
            snapshot_sha256=snapshots,user_notebook_sha256=users,
            previous_goal_archive_sha256=sha256(REPO/'docs/GOAL_V16.md')))
        api = api_client()
        q = quota(api)
        assert q['reserved_seconds'] == 0 and not q['paid_scaling'] and q['remaining_seconds'] >= 2400
        board = leaderboard(api)
        atomic_json(pub/'initial_leaderboard.json',board)
        started = utc_now()
        deadline = (datetime.now(timezone.utc)+timedelta(hours=48)).isoformat()
        plan = dict(started_at=started,deadline=deadline,initial_quota=q,quota_used_at_start=q['used_seconds'],
            initial_quota_refresh_time=q['refresh_time'],target_rank_maximum=19,
            gpu_session_seconds_cap=6000,public_version_limit=2,production_recipe_limit=1,official_candidate_limit=1,
            initial_public_cap_seconds=2400,followup_public_cap_seconds=3600,hidden_cap_seconds=32400,
            cpu_seconds_caps=dict(E17=900,H62=720),focus_ids=['E17','H62'],
            parent_official_receipt=58103553,parent_official_score=.82987,
            work_fraction=.15,row_fraction=.30,parent_work_fraction=.1,parent_row_fraction=.2,
            diagnostic_work_fraction=.15,diagnostic_row_fraction=.2,
            incremental_information_gain_minimum=.0005,temperature=1,packing_tokens=8192,context_tokens=4096,
            warmup_rows=80,raw_probability_tolerance=1e-5,E17_fitting_allowed=False,
            E17_existing_temperature=1.3161518671742531)
        atomic_json(pub/'span_plan.json',plan)
        atomic_json(REPO/'configs/span_017.json',dict(started_at=started,deadline=deadline,
            job_seconds_caps=dict(gemma_route=2400,gemma_route_followup=3600)))
        print(json.dumps(dict(prepared=True,new_gpu_executions=0,deadline=deadline,quota=q,leaderboard=board)))


if __name__ == '__main__':
    main()
