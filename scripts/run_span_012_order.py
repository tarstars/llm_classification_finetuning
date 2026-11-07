"""Freeze original v12resource caps; admit/observe one immutable public version."""
from datetime import datetime,timedelta,timezone
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,exclusive_lock,sha256,utc_now
from arena.kaggle_control import run_job
from arena.span004_support import api_client,quota,read_call


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--role',choices=['packing_profile','llama_order','llama_order_followup'],default='packing_profile')
    args=parser.parse_args();root=ROOT/'runs/span_012';pub=REPO/'results/span_012';config=REPO/'configs/span_012.json'
    root.mkdir(parents=True,exist_ok=True)
    if args.prepare:
        with exclusive_lock(root/'preparation.lock'):
            assert not (pub/'span_plan.json').exists(),'Never reset original quota period/deadline/caps'
            registry=read_json(REPO/'backlog/hypotheses.json')
            assert registry['active_version']==12 and next(v for v in registry['versions'] if v['number']==12)['focus_ids']==['E16','H58']
            old=read_json(REPO/'results/span_011/preservation.json');snapshots={}
            for name,digest in old['user_notebook_sha256'].items():assert sha256(REPO/name)==digest
            names=subprocess.check_output(['git','ls-tree','-r','--name-only','6762497','--','backlog/versions'],cwd=REPO,text=True).splitlines()
            assert len(names)==46
            for name in names:
                digest=hashlib.sha256(subprocess.check_output(['git','show','6762497:'+name],cwd=REPO)).hexdigest()
                assert sha256(REPO/name)==digest;snapshots[name]=digest
            atomic_json(pub/'preservation.json',dict(at=utc_now(),snapshot_sha256=snapshots,
                snapshot_reference_commit='6762497',user_notebook_sha256=old['user_notebook_sha256'],
                previous_goal_archive_sha256=sha256(REPO/'docs/GOAL_V11.md')))
            q=quota(api_client());assert q['reserved_seconds']==0 and not q['paid_scaling'] and q['remaining_seconds']>=1800
            started=utc_now();deadline=(datetime.now(timezone.utc)+timedelta(hours=48)).isoformat()
            plan=dict(started_at=started,deadline=deadline,initial_quota=q,quota_used_at_start=q['used_seconds'],
                initial_quota_refresh_time=q['refresh_time'],gpu_session_seconds_cap=7200,
                public_version_limit=3,production_recipe_limit=1,official_candidate_limit=1,
                target_rank_maximum=19,temperature=1,work_fraction=.1,row_fraction=.2,
                packing_choices=[6144,8192],packing_default=8192,packing_speed_ratio_max=.95,bracket_ratio_max=1.10,
                raw_probability_tolerance=1e-5,hidden_cap_seconds=32400,full_inventory_required_before_each_gpu=True)
            atomic_json(pub/'span_plan.json',plan)
            atomic_json(config,dict(started_at=started,deadline=deadline,
                job_seconds_caps=dict(packing_profile=1800,llama_order=1800,llama_order_followup=3600)))
            print(json.dumps(dict(prepared=True,quota=q,new_gpu_executions=0,deadline=deadline)))
        return
    def precondition(role):
        registry=read_json(REPO/'backlog/hypotheses.json');items={i['id']:i for i in registry['items']}
        assert registry['active_version']==12 and next(v for v in registry['versions'] if v['number']==12)['focus_ids']==['E16','H58']
        if role=='packing_profile':assert items['E16']['status']=='running'
        else:
            assert items['E16']['status']=='done' and items['H58']['status']=='running'
            assert read_json(pub/'E16_qualification.json')['engineering_gate_passed']
            assert read_json(pub/'H58_cached_replay.json')['information_gate_passed']
        assert read_json(pub/(role+'_cell_rehearsals.json'))['passed']
        assert len(list((root/'jobs').glob('*/push.json')))<3
        api=api_client()
        from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
        for slug in ['lmsys-checkpoints-0-0805','lmsys-checkpoints-3-0805','lmsys-modules-0805','lmsys-packages']:
            request=ApiGetDatasetRequest();request.owner_slug='tascj0';request.dataset_slug=slug
            with api.build_kaggle_client() as client:dataset=read_call(lambda:client.datasets.dataset_api_client.get_dataset(request))
            assert dataset.current_version_number==1 and not dataset.is_private,'Pinned public dependency changed'
        rows=list(read_call(lambda:api.competition_submissions('llm-classification-finetuning')) or [])
        assert all(str(r.status).split('.')[-1] in ['COMPLETE','ERROR','FAILED','INVALID','CANCELED','CANCELLED'] for r in rows),'Official scoring pending'
    run_job(REPO,root,pub,config,args.role,dry_run=args.dry_run,precondition=precondition)


if __name__=='__main__':main()
