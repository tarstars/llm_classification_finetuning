"""One compact H02 save after read-only proof the embedded save created no run."""
from pathlib import Path
import argparse
import json
import sys
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,sha256,utc_now
from arena.kaggle_control import run_job
from arena.span004_support import api_client,read_call
from arena.span031_inventory import inventory_with_resolution
from submit_span_013_order import submissions,TERMINAL


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dry-run',action='store_true');args=parser.parse_args()
    root=ROOT/'runs/span_031';pub=REPO/'results/span_031';role='size_pilot_compact'
    def precondition(actual_role):
        assert actual_role==role
        registry=read_json(REPO/'backlog/hypotheses.json');assert registry['active_version']==31
        assert next(v for v in registry['versions'] if v['number']==31)['focus_ids']==['H02']
        assert next(i for i in registry['items'] if i['id']=='H02')['status']=='running'
        plan=read_json(pub/'span_plan.json');initial=read_json(pub/'initial_span_plan.json')
        for key in ['started_at','deadline','quota_used_at_start','initial_quota_refresh_time','gpu_session_seconds_cap']:
            assert plan[key]==initial[key],key
        assert plan['gpu_session_seconds_cap']==5400 and plan['public_version_limit']==1 and plan['official_candidate_limit']==0
        cfg=read_json(REPO/'configs/span_031_compact.json')
        assert cfg['size_pilot_accepted_updates']==32 and cfg['size_pilot_nominal_cycles_cap']==48
        assert read_json(pub/(role+'_cell_rehearsals.json'))['passed']
        prep=read_json(pub/'pilot_preparation.json')
        assert sha256(root/'pilot_ids.json')==prep['sample_ids_sha256']
        assert sha256(root/'pilot_rows.csv.gz')==prep['compressed_csv_sha256']
        preserve=read_json(pub/'preservation.json')
        for name,digest in {**preserve['snapshot_sha256'],**preserve['user_notebook_sha256']}.items():assert sha256(REPO/name)==digest,name
        assert not list((root/'jobs').glob('*/push.json')),'An accepted version already exists'
        assert not (root/'jobs'/role/'push_intent.json').exists()
        assert not list((root/'official').glob('*/submission_intent.json'))
        rejection=read_json(pub/'save_reconciliation.json')
        assert rejection['save_http_status']==400 and rejection['read_http_status']==404 and not rejection['matching_owned_refs']
        fresh=read_json(pub/'jobs'/role/'admission_inventory.json')
        assert not fresh['nonterminal_jobs'] and fresh['quota']['remaining_seconds']>=5400+32400
        assert abs(fresh['quota']['used_seconds']-plan['quota_used_at_start'])<=.1,'Unexpected GPU charge before accepted pilot'
        api=api_client();old_ref=rejection['kernel']
        try:read_call(lambda:api.kernels_status(old_ref))
        except Exception as e:
            assert getattr(getattr(e,'response',None),'status_code',None)==404,'Unresolved original notebook'
        else:raise AssertionError('Original rejected notebook now has a run; observe it instead')
        matches=list(read_call(lambda:api.kernels_list(mine=True,search=old_ref.split('/')[-1],page_size=100)) or [])
        assert len(matches)<100 and all(x.ref!=old_ref for x in matches)
        history=submissions(api);assert all(str(s.status).split('.')[-1] in TERMINAL for s in history)
        asset=read_json(pub/'input_asset_ready.json');assert asset['ready'] and asset['private_verified']
        from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
        request=ApiGetDatasetRequest();request.owner_slug='arseniytrushin';request.dataset_slug='llm-span031-size-inputs'
        with api.build_kaggle_client() as client:
            current=read_call(lambda:client.datasets.dataset_api_client.get_dataset(request))
        status=json.loads(read_call(lambda:api.dataset_status('arseniytrushin/llm-span031-size-inputs',format='json')))
        assert current.is_private and current.current_version_number==status['current_version_number']==1 and status['status']=='ready'
        atomic_json(pub/'compact_admission.json',dict(at=utc_now(),admitted=True,rejected_original_has_no_run=True,
            accepted_versions_before=0,all_official_terminal=True,official_receipts=len(history),public_cap_seconds=5400,
            private_asset_ready=True,official_submissions_allowed=0,accepted_updates_per_arm=32))
    def observed_api():
        api=api_client();save=api.kernels_push
        def guarded(*a,**kw):
            try:return save(*a,**kw)
            except Exception as error:
                response=getattr(error,'response',None);detail={}
                if response is not None:
                    try:
                        value=response.json();value=value.get('error',value)
                        if isinstance(value,dict):detail={k:str(v)[:1000] for k,v in value.items() if k in ['code','message','status']}
                        elif isinstance(value,str):detail={'message':value[:1000]}
                    except (ValueError,AttributeError):pass
                record=dict(at=utc_now(),error_type=type(error).__name__,http_status=getattr(response,'status_code',None),
                    server_detail=detail,mutation_retried=False)
                atomic_json(pub/'jobs'/role/'transport_error.json',record)
                print(record,flush=True);raise
        api.kernels_push=guarded
        return api
    def scoped_inventory(api,start):
        return inventory_with_resolution(api,start,read_json(pub/'save_reconciliation.json')['kernel'],
            pub/'inventory_admission_resolution.json')
    with patch('arena.kaggle_control.api_client',observed_api),patch('arena.kaggle_control.inventory',scoped_inventory):
        run_job(REPO,root,pub,REPO/'configs/span_031_compact.json',role,dry_run=args.dry_run,precondition=precondition)


if __name__=='__main__':main()
