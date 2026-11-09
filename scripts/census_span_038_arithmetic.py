"""Full label-blind coverage, frozen audit sample and actual bounded CPU use."""
from collections import Counter
from datetime import datetime
from pathlib import Path
import json
import resource
import signal
import sqlite3
import sys
import time
import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,sha256,utc_now
from arena.span038_arithmetic import row_feature


def main():
    start=time.monotonic();pub,run=REPO/'results/span_038',ROOT/'runs/span_038'
    assert not (pub/'coverage.json').exists();run.mkdir(parents=True,exist_ok=True)
    def expired(*unused):raise TimeoutError('Frozen CPU preparation cap')
    signal.signal(signal.SIGALRM,expired);signal.alarm(1200)
    raw=ROOT/'raw/train.csv';assert sha256(raw)==read_json(REPO/'results/span_035/preparation.json')['raw_sha256']
    sql=ROOT/'runs/span_011/cohort/rows.sqlite';assert sql.exists()
    def parsed(value):
        data=json.loads(value) if isinstance(value,str) else list(value)
        return data if isinstance(data,list) else []
    counters={};nonzero={};artifacts={}
    def scan(name,rows):
        stats=Counter();examples=[];path=run/(name+'_coverage.jsonl')
        with path.open('x') as stream:
            for row in rows:
                p,a,b=[parsed(row[k]) for k in ['prompt','response_a','response_b']]
                f=row_feature(p,a,b);claims=f['claims_a']+f['claims_b']
                stats['rows']+=1;stats['rows_with_claims']+=bool(claims);stats['claims']+=len(claims)
                stats['false_claims']+=sum(x['error'] for x in claims);stats['nonzero_rows']+=f['feature']!=0
                stats.update({'reason_'+k:v for k,v in f.get('reasons',{}).items()})
                if claims:
                    reverse=row_feature(p,b,a);assert reverse['feature']==-f['feature']
                record=dict(id=int(row['id']),group=row.get('group_id',str(row['id'])),**f)
                stream.write(json.dumps(record,ensure_ascii=False)+'\n')
                if f['feature']!=0:examples.append(dict(**record,prompt=p,response_a=a,response_b=b))
                if stats['rows']%2000==0:
                    progress=dict(at=utc_now(),phase=name,rows=stats['rows'],nonzero=stats['nonzero_rows'],elapsed_seconds=time.monotonic()-start)
                    atomic_json(pub/'coverage_progress.json',progress);print(progress,flush=True)
                assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2<6
        counters[name]=dict(stats);nonzero[name]=examples;artifacts[path.name]=sha256(path)
    original=pd.read_csv(raw,usecols=['id','prompt','response_a','response_b']);assert len(original)==57477
    scan('original',original.to_dict('records'));del original
    cutoff=datetime.fromisoformat('2025-07-06T18:16:33+00:00')
    connection=sqlite3.connect('file:'+str(sql)+'?mode=ro',uri=True)
    def later():
        for payload,group in connection.execute('SELECT r.payload,p.group_id FROM representatives p JOIN rows r ON r.uid=p.uid'):
            row=json.loads(payload)
            assert not any(k in row for k in ['winner','private_winner','model_a','model_b'])
            if datetime.fromisoformat(row['timestamp'])>cutoff:yield dict(row,group_id=group)
    scan('later_public',later());connection.close()
    assert len({r['group'] for r in nonzero['later_public']})==len(nonzero['later_public'])
    eligible=all(counters[n]['nonzero_rows']>=128 for n in counters)
    if eligible:
        rng=np.random.default_rng(20261218);chosen=[]
        for name in ['original','later_public']:
            candidates=sorted(nonzero[name],key=lambda r:r['id'])
            for j in rng.choice(len(candidates),16,replace=False):chosen.append(dict(source=name,**candidates[j]))
        atomic_json(run/'blind_audit_cases.json',dict(seed=20261218,labels_hidden=True,predictions_hidden=True,model_identities_hidden=True,rows=chosen))
        artifacts['blind_audit_cases.json']=sha256(run/'blind_audit_cases.json')
    snapshots=sorted((REPO/'backlog/versions').glob('*'));assert len(snapshots)==150
    users=read_json(REPO/'results/span_037/preservation.json')['user_notebook_sha256']
    for n,h in users.items():assert sha256(REPO/n)==h
    atomic_json(pub/'preservation.json',dict(at=utc_now(),snapshot_sha256={str(p.relative_to(REPO)):sha256(p) for p in snapshots},user_notebook_sha256=users))
    result=dict(at=utc_now(),task='H35',stage='label_blind_coverage',counters=counters,coverage_gate_passed=eligible,
        minimum_nonzero_per_source=128,blind_audit_prepared=eligible,quality_measured=False,task_targets_loaded=False,
        source_sql_sha256=sha256(sql),raw_sha256=sha256(raw),artifacts=artifacts,
        source_sha256={n:sha256(REPO/n) for n in ['GOAL.md','docs/SPAN_038_PLAN.md','src/arena/span038_arithmetic.py','tests/test_span038_arithmetic.py',str(Path(__file__).relative_to(REPO))]},
        elapsed_seconds=time.monotonic()-start,peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        decision='blind_reliability_audit' if eligible else 'stop_insufficient_verified_coverage')
    atomic_json(pub/'coverage.json',result);print({k:v for k,v in result.items() if k not in ['source_sha256','artifacts']})


if __name__=='__main__':main()
