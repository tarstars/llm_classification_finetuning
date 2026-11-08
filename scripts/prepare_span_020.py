"""Prepare the sole H66candidate without resetting the inherited resource period."""
from datetime import datetime,timezone
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,read_json,sha256,utc_now
from arena.span004_support import api_client,quota


def main():
    pub=REPO/'results/span_020';old=REPO/'results/span_019';root=ROOT/'runs/span_020'
    pub.mkdir(parents=True,exist_ok=True);root.mkdir(parents=True,exist_ok=True)
    assert not (pub/'span_plan.json').exists(),'Preserve the original preparation'
    registry=read_json(REPO/'backlog/hypotheses.json')
    assert registry['active_version']==20 and next(v for v in registry['versions'] if v['number']==20)['focus_ids']==['H66']
    assert next(i for i in registry['items'] if i['id']=='H66')['status']=='running'
    assert read_json(old/'closure_audit.json')['passed']
    memory=read_json(old/'memory_discovery.json');assert memory['memory_supports_one_bounded_candidate']
    assert memory['candidate_padded_token_cap']==12288 and memory['candidate_max_rows']==32
    names=subprocess.check_output(['git','ls-tree','-r','--name-only','HEAD','--','backlog/versions'],cwd=REPO,text=True).splitlines()
    assert len(names)==78
    snapshots={}
    for name in names:
        snapshots[name]=sha256(REPO/name)
        original=subprocess.check_output(['git','show','HEAD:'+name],cwd=REPO)
        assert (REPO/name).read_bytes()==original
    users=read_json(old/'preservation.json')['user_notebook_sha256']
    for name,digest in users.items():assert sha256(REPO/name)==digest
    assert (Path.home()/'notebooks/llm_classification_finetuning').resolve()==REPO/'notebooks'
    atomic_json(pub/'preservation.json',dict(at=utc_now(),snapshot_sha256=snapshots,
        user_notebook_sha256=users,previous_goal_archive_sha256=sha256(REPO/'docs/GOAL_V19.md')))
    for name in ['asset_manifest.json','asset_upload.json','evaluation_selection.json','token_reference.json',
                 'quality_preparation.json']:
        destination=pub/name
        if destination.exists():assert sha256(destination)==sha256(old/name)
        else:destination.write_bytes((old/name).read_bytes())
    previous=read_json(old/'parent_runtime_reference.json')
    h65=ROOT/'runs/span_019/jobs/classifier_pilot/output/mirror_experiment'
    summary=read_json(h65/'m0_summary.json');engineering=read_json(h65/'engineering.json')
    other=engineering['retained_process_seconds']-sum(c['seconds'] for c in summary['cohorts'].values())
    assert summary['cohorts']['uniform']['seconds']>=previous['uniform']['seconds']
    assert summary['cohorts']['warmup']['seconds']>=previous['warmup']['seconds']
    atomic_json(pub/'parent_runtime_reference.json',dict(at=utc_now(),source_path=str(h65/'m0_summary.json'),
        source_sha256=sha256(h65/'m0_summary.json'),cold_seconds=max(other,previous['cold_seconds']),
        uniform=summary['cohorts']['uniform'],warmup=summary['cohorts']['warmup'],
        includes_prior_import_and_other_overhead=True,prior_reference_sha256=sha256(old/'parent_runtime_reference.json'),
        engineering_sha256=sha256(h65/'engineering.json')))
    inherited=read_json(old/'span_plan.json');q=quota(api_client())
    assert q['reserved_seconds']==0 and not q['paid_scaling'] and q['remaining_seconds']>=5400+32400
    assert q['refresh_time']==inherited['initial_quota_refresh_time']
    assert datetime.now(timezone.utc).timestamp()+5400+32400+600<datetime.fromisoformat(inherited['deadline']).timestamp()
    charge=read_json(old/'jobs/classifier_pilot/controller_state.json')['charged_seconds']
    plan=dict(started_at=inherited['started_at'],deadline=inherited['deadline'],prepared_at=utc_now(),
        initial_quota=q,quota_used_at_start=inherited['quota_used_at_start'],
        initial_quota_refresh_time=inherited['initial_quota_refresh_time'],gpu_session_seconds_cap=18000,
        public_cap_seconds=5400,hidden_cap_seconds=32400,public_version_limit=2,public_versions_used_before_span=1,
        public_gpu_seconds_before_span=charge,span_public_version_limit=1,official_candidate_limit=1,
        padded_tokens_cap=12288,max_rows_per_batch=32,target_rank_maximum=19,
        parent_official_receipt=58103553,parent_official_score=.82987,raw_probability_tolerance=1e-5,
        uniform_rows=512,warmup_rows=80,source_rows=1024,new_gpu_executions=0,resource_period_reset=False)
    atomic_json(pub/'span_plan.json',plan)
    atomic_json(REPO/'configs/span_020.json',dict(started_at=plan['started_at'],deadline=plan['deadline'],
        job_seconds_caps=dict(batch_candidate=5400)))
    print(dict(prepared=True,new_gpu_executions=0,quota=q,deadline=plan['deadline']))


if __name__=='__main__':main()
