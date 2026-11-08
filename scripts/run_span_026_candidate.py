"""Admit one final H72 qualification after the terminal information pilot."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,read_json,sha256,utc_now
from arena.kaggle_control import run_job
from arena.span004_support import api_client
from submit_span_013_order import submissions,TERMINAL,dataset_dependencies


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dry-run',action='store_true');args=parser.parse_args()
    root=ROOT/'runs/span_026';pub=REPO/'results/span_026'
    def precondition(role):
        registry=read_json(REPO/'backlog/hypotheses.json')
        assert registry['active_version']==26 and next(v for v in registry['versions'] if v['number']==26)['focus_ids']==['H72']
        assert next(i for i in registry['items'] if i['id']=='H72')['status']=='running'
        assert role=='context_qualification' and read_json(pub/(role+'_cell_rehearsals.json'))['passed']
        gate=read_json(pub/'pilot_audit.json')
        assert gate['complete_information'] and gate['engineering_screen_passed'] and gate['gross_regression_guard_passed']
        assert read_json(root/'jobs/context_pilot/status.json')['status']=='COMPLETE'
        assert read_json(root/'jobs/context_pilot/controller_state.json')['stage']=='complete'
        bundle=read_json(pub/(role+'_bundle.json'));preparation=read_json(pub/'candidate_preparation.json')
        assert bundle['pilot_audit_sha256']==sha256(pub/'pilot_audit.json')
        assert bundle['candidate_spec_sha256']==preparation['spec_sha256']==sha256(root/'candidate_spec.json')
        assert bundle['hidden_spec_sha256']==preparation['hidden_spec_sha256']==sha256(root/'hidden_asset_spec.json')
        assert bundle['public_cap_seconds']==3600 and bundle['hidden_cap_seconds']==32400 and not bundle['public_prediction_cache_embedded']
        preservation=read_json(pub/'preservation.json')
        for n,h in {**preservation['snapshot_sha256'],**preservation['user_notebook_sha256']}.items():assert sha256(REPO/n)==h
        accepted=list((root/'jobs').glob('*/push.json'));intents=list((root/'jobs').glob('*/push_intent.json'))
        assert len(accepted)==len(intents)==1 and accepted[0].parent.name=='context_pilot'
        plan=read_json(pub/'span_plan.json')
        assert datetime.now(timezone.utc).timestamp()+3600+32400+600<datetime.fromisoformat(plan['deadline']).timestamp()
        fresh=read_json(pub/'jobs'/role/'admission_inventory.json');q=fresh['quota']
        assert q['remaining_seconds']>=3600+32400 and not fresh['nonterminal_jobs']
        assert q['used_seconds']-plan['quota_used_at_start']+3600<=10800.1
        api=api_client();history=submissions(api);assert all(str(x.status).split('.')[-1] in TERMINAL for x in history)
        deps=dataset_dependencies(api,read_json(root/'notebooks'/role/'kernel-metadata.json'))
        atomic_json(pub/'candidate_admission.json',dict(at=utc_now(),admitted=True,all_official_terminal=True,
            official_receipts=len(history),public_cap_seconds=3600,phase_cap_seconds=10800,
            pilot_audit_sha256=sha256(pub/'pilot_audit.json'),dependencies=deps,official_submission_eligible=False))
    run_job(REPO,root,pub,REPO/'configs/span_026.json','context_qualification',dry_run=args.dry_run,precondition=precondition)


if __name__=='__main__':main()
