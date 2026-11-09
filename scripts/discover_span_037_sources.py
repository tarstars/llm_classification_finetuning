"""Bounded public metadata discovery only; no weights, task data or model code."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import sys
import time
import requests

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,read_json,sha256,utc_now
from arena.span004_support import api_client,read_call


def main():
    started=time.monotonic();pub=REPO/'results/span_037';run=ROOT/'runs/span_037/source_audit'
    assert not (pub/'discovery.json').exists();run.mkdir(parents=True,exist_ok=True)
    queries=[('hf_classifiers',dict(search='gemma',pipeline_tag='text-classification',limit=100)),
        ('hf_lmsys',dict(search='lmsys',limit=100))]
    def hf(item):
        name,params=item;r=requests.get('https://huggingface.co/api/models',params=params,timeout=60);r.raise_for_status()
        rows=r.json();assert len(rows)<=100
        atomic_json(run/(name+'.json'),rows)
        return dict(name=name,url='https://huggingface.co/api/models',params=params,results=rows,complete=len(rows)<100)
    def kg(text):
        api=api_client();rows=read_call(lambda:api.dataset_list(search=text)) or []
        assert len(rows)<=20;values=[x.to_dict() for x in rows]
        name='kaggle_'+text.replace(' ','_');atomic_json(run/(name+'.json'),values)
        return dict(name=name,query=text,results=values,complete=False)
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending=[pool.submit(hf,q) for q in queries]+[pool.submit(kg,q) for q in ['lmsys gemma','gemma2 9b']]
        results=[p.result() for p in pending]
    previous=read_json(REPO/'results/span_036/preservation.json')
    for n,h in previous['user_notebook_sha256'].items():assert sha256(REPO/n)==h
    snapshots=sorted((REPO/'backlog/versions').glob('*'));assert len(snapshots)==146
    atomic_json(pub/'preservation.json',dict(at=utc_now(),snapshot_sha256={str(p.relative_to(REPO)):sha256(p) for p in snapshots},user_notebook_sha256=previous['user_notebook_sha256']))
    record=dict(at=utc_now(),task='H80',stage='catalog_discovery_only',queries=results,
        no_weights_downloaded=True,no_task_targets_loaded=True,new_gpu_executions=0,
        elapsed_seconds=time.monotonic()-started,method_sha256={n:sha256(REPO/n) for n in ['GOAL.md','docs/SPAN_037_PLAN.md',str(Path(__file__).relative_to(REPO))]})
    atomic_json(pub/'discovery.json',record)
    for result in results:
        print(result['name'],json.dumps([dict(id=r.get('id') or r.get('ref'),pipeline=r.get('pipeline_tag'),bytes=r.get('totalBytes'),updated=r.get('lastUpdated')) for r in result['results']]))


if __name__=='__main__':main()
