"""Admit one bounded H02 size pilot, preserving inherited accounting and deadline."""
from pathlib import Path
import argparse
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,sha256,utc_now
from arena.kaggle_control import run_job
from arena.span004_support import api_client
from submit_span_013_order import submissions,TERMINAL


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dry-run',action='store_true');args=parser.parse_args()
    root=ROOT/'runs/span_031';pub=REPO/'results/span_031';cfg=REPO/'configs/span_031.json'
    def precondition(role):
        registry=read_json(REPO/'backlog/hypotheses.json');assert registry['active_version']==31
        assert next(v for v in registry['versions'] if v['number']==31)['focus_ids']==['H02']
        assert next(i for i in registry['items'] if i['id']=='H02')['status']=='running' and role=='size_pilot'
        config=read_json(cfg);plan=read_json(pub/'span_plan.json');initial=read_json(pub/'initial_span_plan.json')
        assert config['size_pilot_accepted_updates']==32 and config['size_pilot_nominal_cycles_cap']==48
        for key in ['started_at','deadline','quota_used_at_start','initial_quota_refresh_time','gpu_session_seconds_cap']:
            assert plan[key]==initial[key],key
        assert plan['gpu_session_seconds_cap']==5400 and plan['public_version_limit']==1 and plan['official_candidate_limit']==0
        assert plan['deadline']==read_json(REPO/'results/span_030/span_plan.json')['deadline']
        prep=read_json(pub/'pilot_preparation.json');assert prep['prepared']
        assert sha256(root/'pilot_ids.json')==prep['sample_ids_sha256']
        assert sha256(root/'pilot_rows.csv.gz')==prep['compressed_csv_sha256']
        assert read_json(pub/'size_pilot_cell_rehearsals.json')['passed']
        preservation=read_json(pub/'preservation.json')
        for name,digest in {**preservation['snapshot_sha256'],**preservation['user_notebook_sha256']}.items():assert sha256(REPO/name)==digest,name
        assert not list((root/'jobs').glob('*/push_intent.json'))
        assert not list((root/'official').glob('*/submission_intent.json'))
        fresh=read_json(pub/'jobs'/role/'admission_inventory.json')
        assert not fresh['nonterminal_jobs'] and fresh['quota']['remaining_seconds']>=5400+32400
        history=submissions(api_client());assert all(str(s.status).split('.')[-1] in TERMINAL for s in history)
        atomic_json(pub/'admission.json',dict(at=utc_now(),admitted=True,all_official_terminal=True,
            official_receipts=len(history),public_cap_seconds=5400,official_submissions_allowed=0,
            accepted_updates_per_arm=32,protocol_revision=2))
    run_job(REPO,root,pub,cfg,'size_pilot',dry_run=args.dry_run,precondition=precondition)


if __name__=='__main__':main()
