"""V8 admission with a single shared qualification-or-size-pilot reserve."""
from pathlib import Path
import argparse
import sys

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json
from arena.kaggle_control import run_job


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('role');parser.add_argument('--dry-run',action='store_true');args=parser.parse_args()
    root=ROOT/'runs/span_008';pub=REPO/'results/span_008';cfg=REPO/'configs/span_008.json'
    def precondition(role):
        registry=read_json(REPO/'backlog/hypotheses.json');items={i['id']:i for i in registry['items']}
        assert registry['active_version']==8 and items['E09']['status']=='done'
        mapping={'serial_cascade':'H52','context_completion':'H51','promotion':'H50','size_pilot':'H02'}
        assert items[mapping[role]]['status']=='running'
        if role=='serial_cascade':assert read_json(REPO/'results/span_007/judge_audit.json')['case_stability']['serial']
        if role=='context_completion':assert read_json(REPO/'results/span_007/context_partial_audit.json')['passed']
        if role in ['promotion','size_pilot']:
            peer='size_pilot' if role=='promotion' else 'promotion'
            assert not (root/'jobs'/peer/'push_intent.json').exists(),'The reserve cannot fund both branches'
            assert read_json(pub/'candidate_selection.json')['eligible']==(role=='promotion')
    run_job(REPO,root,pub,cfg,args.role,args.dry_run,precondition)


if __name__=='__main__':main()
