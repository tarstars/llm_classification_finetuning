"""Independently count saved claims/features and verify bounded span closure."""
from collections import Counter
from fractions import Fraction
from pathlib import Path
import json
import sys
import nbformat

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,sha256,utc_now
from arena.span038_arithmetic import rational


def main():
    pub,run=REPO/'results/span_038',ROOT/'runs/span_038'
    assert not (pub/'closure_audit.json').exists();r=read_json(pub/'coverage.json')
    assert r['decision']=='stop_insufficient_verified_coverage' and not r['coverage_gate_passed']
    assert not r['quality_measured'] and not r['task_targets_loaded'] and not r['blind_audit_prepared']
    assert not (run/'blind_audit_cases.json').exists()
    assert sha256(ROOT/'raw/train.csv')==r['raw_sha256']
    assert sha256(ROOT/'runs/span_011/cohort/rows.sqlite')==r['source_sql_sha256']
    for n,h in r['source_sha256'].items():assert sha256(REPO/('docs/GOAL_V38.md' if n=='GOAL.md' else n))==h,n
    for n,h in r['artifacts'].items():assert sha256(run/n)==h,n
    checks=0
    for name in ['original','later_public']:
        stats=Counter();ids=set();groups=set()
        for line in (run/(name+'_coverage.jsonl')).open():
            row=json.loads(line);assert row['id'] not in ids;ids.add(row['id']);groups.add(row['group'])
            a,b=row['claims_a'],row['claims_b'];claims=a+b
            for c in claims:
                left,right=rational(c['left']),rational(c['right'])
                assert str(left)==c['left_value'] and str(right)==c['right_value']
                assert (abs(left-right)>Fraction(c['rounding_tolerance']))==c['error'];checks+=1
            ea,eb=sum(c['error'] for c in a),sum(c['error'] for c in b)
            assert row['feature']==eb-ea
            stats['rows']+=1;stats['rows_with_claims']+=bool(claims);stats['claims']+=len(claims)
            stats['false_claims']+=ea+eb;stats['nonzero_rows']+=row['feature']!=0
            stats.update({'reason_'+k:v for k,v in row.get('reasons',{}).items()})
        assert dict(stats)==r['counters'][name]
        if name=='later_public':assert len(groups)==stats['rows']
    assert checks==804 and r['counters']['original']['nonzero_rows']==73 and r['counters']['later_public']['nonzero_rows']==1
    preserve=read_json(pub/'preservation.json');assert len(preserve['snapshot_sha256'])==150
    for n,h in {**preserve['snapshot_sha256'],**preserve['user_notebook_sha256']}.items():assert sha256(REPO/n)==h,n
    assert (Path.home()/'notebooks/llm_classification_finetuning').resolve()==REPO/'notebooks'
    resource=read_json(pub/'resource_close.json');assert resource['passed']
    assert resource['gpu_charge_seconds']==resource['public_versions']==resource['official_submissions']==0
    assert resource['result_sha256']==sha256(pub/'coverage.json')
    assert resource['leaderboard_sha256']==sha256(pub/'final_leaderboard.json')
    final=read_json(REPO/'backlog/versions/v38.final.json');v=next(x for x in final['versions'] if x['number']==38);items={x['id']:x for x in final['items']}
    assert v['focus_ids']==['H35'] and items['H35']['status']=='done'
    carry={k for k in v['item_ids'] if items[k]['status'] not in ['done','rejected']};assert len(carry)==34
    assert not any(items[k]['status']=='running' for k in v['item_ids'])
    nxt=read_json(REPO/'backlog/versions/v39.start.json');nv=next(x for x in nxt['versions'] if x['number']==39)
    assert nxt['active_version']==39 and set(nv['item_ids'])==carry|{'H81'} and not nv.get('focus_ids')
    assert len(list((REPO/'backlog/versions').glob('*')))==154
    path=REPO/'notebooks/50_arithmetic_evidence.ipynb';nb=nbformat.read(path,4)
    assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in nb.cells if c.cell_type=='code')
    for n in ['coverage','resource_close']:assert nb.metadata['span038_'+n+'_sha256']==sha256(pub/(n+'.json'))
    atomic_json(pub/'closure_audit.json',dict(at=utc_now(),passed=True,sealed_version=38,active_version=39,
        independent_numeric_claim_checks=checks,all_saved_coverage_and_feature_counts_replayed=True,
        quality_measured=False,historical_snapshots_preserved=150,snapshot_files_after_transition=154,
        carried_tasks=34,new_discoveries=['H81'],goal_achieved=False,report_sha256=sha256(REPO/'docs/SPAN_038_RESULTS.md'),
        notebook_sha256=sha256(path),source_sha256=sha256(Path(__file__))))
    print('V38 independent804claim/coverage/source/resource/preservation/closure audit passed.')


if __name__=='__main__':main()
