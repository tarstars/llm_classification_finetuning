"""One freshly admitted native joint-comparator information pilot."""
from pathlib import Path
import sys
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,sha256,atomic_json,utc_now
from arena.kaggle_control import run_job
from arena.span004_support import api_client
from arena.span031_inventory import inventory_with_resolution
from submit_span_013_order import submissions,TERMINAL


def main():
    root,pub=ROOT/'runs/span_036',REPO/'results/span_036';role='pairrm_pilot'
    def inventory(api,start):
        return inventory_with_resolution(api,start,'arseniytrushin/llm-preference-span031-size-pilot',pub/'inventory_admission_resolution.json')
    def precondition(actual):
        assert actual==role
        registry=read_json(REPO/'backlog/hypotheses.json');assert registry['active_version']==36
        assert next(v for v in registry['versions'] if v['number']==36)['focus_ids']==['H79']
        assert next(i for i in registry['items'] if i['id']=='H79')['status']=='running'
        assert read_json(pub/'preparation.json')['native_token_proof_passed']
        assert read_json(pub/'checkpoint_cpu_check.json')['passed']
        assert read_json(pub/(role+'_cell_rehearsals.json'))['passed']
        assert read_json(pub/'input_asset_ready.json')['ready']
        preserve=read_json(pub/'preservation.json')
        for n,h in {**preserve['snapshot_sha256'],**preserve['user_notebook_sha256']}.items():assert sha256(REPO/n)==h,n
        assert not list((root/'jobs').glob('*/push.json')) and not list((root/'official').glob('*/submission_intent.json'))
        fresh=read_json(pub/'jobs'/role/'admission_inventory.json')
        assert not fresh['nonterminal_jobs'] and fresh['quota']['remaining_seconds']>=5400+32400
        assert abs(fresh['quota']['used_seconds']-read_json(pub/'span_plan.json')['quota_used_at_start'])<=.1
        history=submissions(api_client());assert all(str(s.status).split('.')[-1] in TERMINAL for s in history)
        atomic_json(pub/'pilot_admission.json',dict(at=utc_now(),admitted=True,all_official_terminal=True,
            official_receipts=len(history),public_cap_seconds=2700,total_public_cap_seconds=5400,
            official_submission_eligible=False,accepted_versions_before=0))
    with patch('arena.kaggle_control.inventory',inventory):
        run_job(REPO,root,pub,REPO/'configs/span_036.json',role,precondition=precondition)


if __name__=='__main__':main()
