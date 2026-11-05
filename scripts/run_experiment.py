"""Run one role from the current fixed span; status is durable across observers."""
from pathlib import Path
import argparse
import sys

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json
from arena.kaggle_control import run_job

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config',type=Path,default=REPO/'configs/span_007.json')
parser.add_argument('--role',required=True)
parser.add_argument('--dry-run',action='store_true')
args=parser.parse_args();cfg=read_json(args.config);name=f"span_{cfg['span']:03d}"
def precondition(role):
    registry=read_json(REPO/'backlog/hypotheses.json');items={r['id']:r for r in registry['items']}
    assert registry['active_version']==cfg['span'] and items['E09']['status']=='done'
    if role=='judge_cache':assert read_json(REPO/'results'/name/'E05_results.json')['affordable_stable_recipe']
    if role=='promotion':assert read_json(REPO/'results'/name/'candidate_selection.json')['eligible']
run_job(REPO,ROOT/'runs'/name,REPO/'results'/name,args.config,args.role,args.dry_run,precondition)
