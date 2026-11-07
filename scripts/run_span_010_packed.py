"""Admit/observe only the second frozen packing recipe within the original phase."""
from pathlib import Path
import argparse
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json
from arena.kaggle_control import run_job


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dry-run',action='store_true');args=parser.parse_args()
    root=ROOT/'runs/span_010';pub=REPO/'results/span_010'
    first=read_json(pub/'H57_order_pilot_qualification.json')
    assert first['complete'] and first['all_controls_passed'] and not first['runtime_gate_passed']
    assert read_json(root/'jobs/order_pilot/status.json')['status']=='COMPLETE'
    assert read_json(pub/'order_packed_equivalence.json')['passed']
    def precondition(role):
        assert read_json(pub/(role+'_cell_rehearsals.json'))['passed']
        registry=read_json(REPO/'backlog/hypotheses.json');items={x['id']:x for x in registry['items']}
        version=next(v for v in registry['versions'] if v['number']==10)
        assert registry['active_version']==10 and 'H57' in version['focus_ids']
        assert items['E15']['status']=='done' and items['H57']['status']=='running'
        assert read_json(pub/'span_plan.json')['gpu_session_seconds_cap']==21600
    run_job(REPO,root,pub,REPO/'configs/span_010.json','order_packed',dry_run=args.dry_run,precondition=precondition)


if __name__=='__main__':main()
