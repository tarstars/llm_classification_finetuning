"""Freeze original v14 quota/deadline and preserve the54 historical snapshots."""
from datetime import datetime,timedelta,timezone
from pathlib import Path
import hashlib
import json
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,exclusive_lock,sha256,utc_now
from arena.span004_support import api_client,quota


def main():
    root=ROOT/'runs/span_014';pub=REPO/'results/span_014';root.mkdir(parents=True,exist_ok=True)
    with exclusive_lock(root/'preparation.lock'):
        assert not (pub/'span_plan.json').exists(),'Preserve the original period/deadline/caps'
        registry=read_json(REPO/'backlog/hypotheses.json');items={i['id']:i for i in registry['items']}
        assert registry['active_version']==14 and next(v for v in registry['versions'] if v['number']==14)['focus_ids']==['H61']
        assert items['H60']['status']=='done' and items['H61']['status']=='running'
        assert read_json(REPO/'results/span_013/closure_audit.json')['passed']
        old=read_json(REPO/'results/span_013/preservation.json');reference='8d38a42'
        snapshots={};names=subprocess.check_output(['git','ls-tree','-r','--name-only',reference,'--','backlog/versions'],cwd=REPO,text=True).splitlines()
        assert len(names)==54
        for name in names:
            digest=hashlib.sha256(subprocess.check_output(['git','show',reference+':'+name],cwd=REPO)).hexdigest()
            assert sha256(REPO/name)==digest;snapshots[name]=digest
        for name,digest in old['user_notebook_sha256'].items():assert sha256(REPO/name)==digest
        atomic_json(pub/'preservation.json',dict(at=utc_now(),snapshot_reference_commit=reference,
            snapshot_sha256=snapshots,user_notebook_sha256=old['user_notebook_sha256'],
            previous_goal_archive_sha256=sha256(REPO/'docs/GOAL_V13.md')))
        q=quota(api_client());assert q['reserved_seconds']==0 and not q['paid_scaling'] and q['remaining_seconds']>=2400
        started=utc_now();deadline=(datetime.now(timezone.utc)+timedelta(hours=48)).isoformat()
        plan=dict(started_at=started,deadline=deadline,initial_quota=q,quota_used_at_start=q['used_seconds'],
            initial_quota_refresh_time=q['refresh_time'],gpu_session_seconds_cap=6000,public_version_limit=2,
            production_recipe_limit=1,official_candidate_limit=1,target_rank_maximum=19,
            temperature=1,packing_tokens=8192,warmup_rows=80,work_fraction=.15,parent_work_fraction=.1,
            row_fraction=.2,incremental_information_gain_minimum=.0005,raw_probability_tolerance=1e-5,
            hidden_cap_seconds=32400,parent_official_receipt=58103553,parent_official_score=.82987)
        atomic_json(pub/'span_plan.json',plan)
        atomic_json(REPO/'configs/span_014.json',dict(started_at=started,deadline=deadline,
            job_seconds_caps=dict(gemma_route=2400,gemma_route_followup=3600)))
        print(json.dumps(dict(prepared=True,new_gpu_executions=0,quota=q,deadline=deadline)))


if __name__=='__main__':main()
