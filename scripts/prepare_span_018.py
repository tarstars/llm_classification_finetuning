"""Fix v18resources and preservation before the one teacher lookup."""
from datetime import datetime,timedelta,timezone
from pathlib import Path
import hashlib
import json
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,exclusive_lock,read_json,sha256,utc_now
from arena.span004_support import api_client,quota
from submit_span_010_order import leaderboard


def main():
    root,pub=ROOT/'runs/span_018',REPO/'results/span_018'
    root.mkdir(parents=True,exist_ok=True);pub.mkdir(parents=True,exist_ok=True)
    with exclusive_lock(root/'preparation.lock'):
        assert not (pub/'span_plan.json').exists(),'Preserve original period'
        r=read_json(REPO/'backlog/hypotheses.json')
        assert r['active_version']==18 and next(v for v in r['versions'] if v['number']==18)['focus_ids']==['H64']
        assert next(i for i in r['items'] if i['id']=='H64')['status']=='running'
        assert read_json(REPO/'results/span_017/closure_audit.json')['passed']
        reference=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
        names=subprocess.check_output(['git','ls-tree','-r','--name-only',reference,'--','backlog/versions'],cwd=REPO,text=True).splitlines()
        assert len(names)==70
        snapshots={}
        for name in names:
            digest=hashlib.sha256(subprocess.check_output(['git','show',reference+':'+name],cwd=REPO)).hexdigest()
            assert sha256(REPO/name)==digest;snapshots[name]=digest
        users=read_json(REPO/'results/span_017/preservation.json')['user_notebook_sha256']
        for name,digest in users.items():assert sha256(REPO/name)==digest
        atomic_json(pub/'preservation.json',dict(at=utc_now(),snapshot_reference_commit=reference,
            snapshot_sha256=snapshots,user_notebook_sha256=users,
            previous_goal_archive_sha256=sha256(REPO/'docs/GOAL_V17.md')))
        api=api_client();q=quota(api)
        assert not q['paid_scaling'] and q['reserved_seconds']==0 and q['remaining_seconds']>=3600
        atomic_json(pub/'initial_leaderboard.json',leaderboard(api))
        started=utc_now();deadline=(datetime.now(timezone.utc)+timedelta(hours=48)).isoformat()
        plan=dict(started_at=started,deadline=deadline,initial_quota=q,quota_used_at_start=q['used_seconds'],
            initial_quota_refresh_time=q['refresh_time'],target_rank_maximum=19,
            gpu_session_seconds_cap=18000,public_version_limit=2,production_recipe_limit=1,official_candidate_limit=1,
            initial_public_cap_seconds=3600,followup_public_cap_seconds=14400,hidden_cap_seconds=32400,
            cpu_teacher_cap_seconds=600,teacher_information_gain_minimum=.001,
            source_audit_seconds_cap=1800,source_audit_bytes_cap=50*1024**2,
            teacher_recipe='Actual H60on-route; equal four-orientation mean only off-route',
            parent_official_receipt=58103553,parent_official_score=.82987,
            temperature=1,packing_tokens=8192,context_tokens=4096,
            raw_probability_tolerance=1e-5,warmup_rows=80,CPU_gate_admits_GPU=False)
        atomic_json(pub/'span_plan.json',plan)
        atomic_json(REPO/'configs/span_018.json',dict(started_at=started,deadline=deadline,
            job_seconds_caps=dict(distillation_pilot=3600,distillation_followup=14400)))
        print(json.dumps(dict(prepared=True,new_gpu_executions=0,deadline=deadline,quota=q)))


if __name__=='__main__':
    main()
