"""Prepare a bounded phase, then admit/observe one immutable calibration job."""
from datetime import datetime,timedelta,timezone
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,read_json,exclusive_lock,sha256,utc_now
from arena.kaggle_control import run_job
from arena.span004_support import api_client,inventory,public_inventory,read_call


def prepare(root,pub,config):
    assert not (pub/'span_plan.json').exists(),'Preserve the original phase/quota/deadline'
    cohort=read_json(pub/'cohort_manifest.json');assert cohort['complete'] and cohort['data_ready']
    assert cohort['cpu_gpu_token_identity'] and not cohort['uses_targets_for_selection']
    for name,digest in cohort['artifact_sha256'].items():assert sha256(root/'cohort'/name)==digest
    mapping={s:pd.read_parquet(root/'cohort'/(s+'_mapping.parquet')) for s in ['learning','confirmation']}
    for field in ['id','source_uuid','session','prompt_group','group_id']:
        assert not set(mapping['learning'][field])&set(mapping['confirmation'][field]),'Split group leakage'
    for name in ['learning','confirmation']:
        inputs=pd.read_parquet(root/'cohort'/(name+'.parquet'))
        assert inputs.columns.tolist()==['id','prompt','response_a','response_b'] and inputs.id.is_unique
        assert np.array_equal(inputs.id,mapping[name].id)
        assert (pd.to_datetime(mapping[name].timestamp,utc=True)>=pd.Timestamp('2024-08-06',tz='UTC')).all()
    pages=read_json(root/'source_audit/competition_pages.json')
    assert (datetime.now(timezone.utc)-datetime.fromisoformat(pages['at'])).total_seconds()<86400
    requirements=next(p['content'].lower() for p in pages['pages'] if p['name'].lower()=='code requirements')
    rules=next(p['content'].lower() for p in pages['pages'] if p['name'].lower()=='rules')
    assert 'freely & publicly available external data is allowed' in requirements
    assert 'hand-labeling is forbidden' in rules and 'does not have an end date' in rules
    previous=ROOT/'runs/span_010/jobs/order_pilot/output/order_completion'
    summaries={r:read_json(previous/(r+'_summary.json')) for r in ['m0','m3']}
    reference=np.load(previous/'m0_uniform.npz')['lengths'].astype(float)
    lengths=mapping['learning'].prepared_tokens.to_numpy(dtype=float)
    # Conservative planning proxy only; actual original-cohort timing is mandatory.
    def work(x):return x*np.maximum(1,x/2048)
    ratio=max(1.,float(work(lengths).mean()/work(reference).mean()))
    base=sum(s['cohorts']['uniform']['seconds'] for s in summaries.values())/512
    fixed=sum(s['cold_seconds']+sum(s['cohorts'][n]['seconds'] for n in ['warmup','control','uniform','long','test']) for s in summaries.values())
    learning_forecast=1024*base*ratio*1.5
    forecast=fixed+learning_forecast+600
    assert forecast<5400,'Prospectively stage learning before GPU admission'
    fresh=public_inventory(inventory(api_client(),utc_now()))
    q=fresh['quota'];assert not fresh['nonterminal_jobs'] and q['reserved_seconds']==0 and not q['paid_scaling']
    assert q['remaining_seconds']>=5400
    started=utc_now();plan=dict(started_at=started,initial_quota_refresh_time=q['refresh_time'],quota_used_at_start=q['used_seconds'],
        gpu_session_seconds_cap=21600,deadline=(datetime.now(timezone.utc)+timedelta(hours=48)).isoformat(),
        target_rank_maximum=19,public_pilot_seconds_cap=5400,official_candidate_limit=2,initial_quota=q,
        frozen_recipe_limit=2,learning_rows=1024,confirmation_rows=1024,temperature_bounds=[.5,2.],
        learning_forecast_seconds=forecast,learning_forecast_safety_factor=1.5,learning_work_ratio=ratio,
        original_fixed_profile_seconds=fixed,forecast_learning_inference_seconds=learning_forecast,
        learning_control_source='arseniytrushin/llm-preference-span010-order-completion/1',
        confirmed_external_data_allowed=True,official_pages_sha256=sha256(root/'source_audit/competition_pages.json'),
        source_format_dates_overlap_verified=True,cohort_manifest_sha256=sha256(pub/'cohort_manifest.json'),
        confirmation_labels_inspected=False,ready_for_gpu=True)
    atomic_json(pub/'initial_inventory.json',fresh);atomic_json(pub/'span_plan.json',plan)
    atomic_json(config,dict(started_at=started,deadline=plan['deadline'],job_seconds_caps={'calibration_learning':5400,'calibration_confirmation':7200}))
    print(json.dumps(dict(prepared=True,ready_for_gpu=True,forecast_seconds=forecast,quota=q,new_gpu_executions=0)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--dry-run',action='store_true');parser.add_argument('--stage',choices=['learning','confirmation'],default='learning')
    args=parser.parse_args();root=ROOT/'runs/span_011';pub=REPO/'results/span_011';config=REPO/'configs/span_011.json'
    if args.prepare:
        with exclusive_lock(root/'preparation.lock'):prepare(root,pub,config)
        return
    def precondition(role):
        registry=read_json(REPO/'backlog/hypotheses.json');items={i['id']:i for i in registry['items']}
        v=next(v for v in registry['versions'] if v['number']==11)
        assert registry['active_version']==11 and v['focus_ids']==['H59'] and items['H59']['status']=='running'
        assert read_json(pub/(role+'_cell_rehearsals.json'))['passed']
        assert read_json(pub/'span_plan.json')['ready_for_gpu']
        from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
        api=api_client()
        for source in ['lmsys-checkpoints-0-0805','lmsys-checkpoints-3-0805','lmsys-modules-0805','lmsys-packages']:
            request=ApiGetDatasetRequest();request.owner_slug='tascj0';request.dataset_slug=source
            with api.build_kaggle_client() as client:
                dataset=read_call(lambda:client.datasets.dataset_api_client.get_dataset(request))
            assert dataset.current_version_number==1 and not dataset.is_private,'Pinned public dependency changed'
        submissions=list(read_call(lambda:api.competition_submissions('llm-classification-finetuning')) or [])
        assert all(str(s.status).split('.')[-1] in ['COMPLETE','ERROR','FAILED','INVALID','CANCELED','CANCELLED'] for s in submissions),'Official scoring pending'
        # Any accepted pending official receipt must be observed before more research GPU.
        for receipt in (root/'official').glob('*/receipt.json'):
            result=receipt.parent/'result.json'
            assert result.exists() and read_json(result)['status'].split('.')[-1] in ['COMPLETE','ERROR','FAILED','INVALID','CANCELED','CANCELLED']
        if args.stage=='confirmation':
            frozen=read_json(pub/'frozen_temperature.json')
            assert frozen['learning_gate_passed'] and not frozen['confirmation_labels_inspected']
    run_job(REPO,root,pub,config,'calibration_'+args.stage,dry_run=args.dry_run,precondition=precondition)


if __name__=='__main__':main()
