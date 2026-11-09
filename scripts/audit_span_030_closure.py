"""Verify reporting and preservation after the real terminal audit and v31 transition.

This local audit never submits, starts compute, or marks the persistent goal complete.
Run only after audit_span_030_outcome.py, final reporting, and backlog advance.
"""
from pathlib import Path
import sys

import nbformat

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.experiment import atomic_json,read_json,sha256,utc_now
from arena.official_outcome import assess


def main():
    pub=REPO/'results/span_030'
    output=pub/'closure_audit.json'
    assert not output.exists(),'Preserve the first completed closure audit'
    assert read_json(pub/'official_state.json')['stage']=='terminal','Official evaluation is still pending'
    outcome=read_json(pub/'outcome_audit.json')
    assert outcome['evidence_valid'] and outcome['official_terminal']
    observation=read_json(pub/'outcome_receipt_observation.json')
    receipt=read_json(pub/'official_receipt.json')
    assert sha256(pub/'outcome_receipt_observation.json')==outcome['receipt_observation_sha256']
    assert observation['history_complete'] and observation['receipt']==str(receipt['ref'])
    board=None
    if outcome['official_status']=='COMPLETE':
        board=read_json(pub/'outcome_leaderboard.json')
        assert sha256(pub/'outcome_leaderboard.json')==outcome['board_sha256']
    reassessed=assess(observation['result'],receipt,observation['observed_at'],board)
    assert all(outcome[k]==v for k,v in reassessed.items()),'Outcome and actual receipt/board disagree'
    for n,h in outcome['source_sha256'].items():assert sha256(REPO/n)==h,n
    assert outcome['actual_hidden_inference_source_verified'] and outcome['official_submissions']==1
    assert sha256(pub/'candidate_qualification.json')==outcome['qualification_sha256']
    assert sha256(pub/'deployment_decision.json')==outcome['decision_sha256']
    qualification=read_json(pub/'candidate_qualification.json')
    assert qualification['official_submission_eligible'] and qualification['expert_weights']==[.4,.6]
    for n,h in read_json(pub/'weight_qualification_bundle.json')['source_sha256'].items():
        path=REPO/('docs/GOAL_V30.md' if n=='GOAL.md' else n)
        assert sha256(path)==h,n
    plan=read_json(pub/'span_plan.json')
    resources=read_json(pub/'terminal_inventory.json')
    assert not resources['nonterminal_jobs'] and resources['quota']['reserved_seconds']==0
    assert not resources['quota']['paid_scaling']
    assert resources['quota']['refresh_time']==plan['initial_quota_refresh_time']
    assert 0<outcome['public_gpu_charged_seconds']<=plan['gpu_session_seconds_cap']==3600
    assert plan['public_version_limit']==plan['official_candidate_limit']==1
    assert plan['hidden_cap_seconds']==32400
    assert -.1<=outcome['total_quota_delta_seconds']<=3600+32400+.1
    preserve=read_json(pub/'preservation.json')
    assert len(preserve['snapshot_sha256'])==118
    for n,h in {**preserve['snapshot_sha256'],**preserve['user_notebook_sha256']}.items():
        assert sha256(REPO/n)==h,n
    assert (Path.home()/'notebooks/llm_classification_finetuning').resolve()==REPO/'notebooks'
    final=read_json(REPO/'backlog/versions/v30.final.json')
    version=next(v for v in final['versions'] if v['number']==30)
    items={i['id']:i for i in final['items']}
    assert version['focus_ids']==['H76'] and items['H76']['status']=='done'
    report='docs/SPAN_030_RESULTS.md'
    assert (REPO/report).is_file()
    assert any(h['status']=='done' and report in h.get('evidence',[]) for h in items['H76']['history'])
    assert items['H73']['status']=='deferred'
    carried={i for i in version['item_ids'] if items[i]['status'] not in ['done','rejected']}
    assert len(carried)==38 and not any(items[i]['status']=='running' for i in version['item_ids'])
    nxt=read_json(REPO/'backlog/versions/v31.start.json')
    next_version=next(v for v in nxt['versions'] if v['number']==31)
    next_items={i['id']:i for i in nxt['items']}
    assert carried<=set(next_version['item_ids'])
    new_ids=set(next_version['item_ids'])-carried
    assert all(next_items[i]['introduced_in']==31 for i in new_ids)
    assert nxt['active_version']==31
    assert not next_version.get('focus_ids',[]),'Audit closure before starting another span'
    for name in ['v30.final.json','v30.final.md','v31.start.json','v31.start.md']:
        assert (REPO/'backlog/versions'/name).is_file(),name
    assert len(list((REPO/'backlog/versions').glob('*')))==122
    notebook=REPO/'notebooks/42_llama_heavy_prior.ipynb'
    nb=nbformat.read(notebook,4)
    cells=[c for c in nb.cells if c.cell_type=='code']
    assert cells and all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in cells)
    assert nb.metadata['span030_outcome_audit_sha256']==sha256(pub/'outcome_audit.json')
    assert nb.metadata['span030_official_receipt']==str(receipt['ref'])
    rendered='\n'.join(o.get('text','') for c in cells for o in c.outputs if o.output_type=='stream')
    assert 'Verified terminal outcome:' in rendered and str(receipt['ref']) in rendered
    result=dict(at=utc_now(),passed=True,sealed_version=30,
        goal_achieved_by_span=outcome['goal_achieved'],official_receipt=str(receipt['ref']),
        official_status=outcome['official_status'],rank=outcome.get('rank'),
        candidate_score=outcome.get('candidate_score'),historical_snapshots_preserved=118,
        snapshot_files_after_transition=len(list((REPO/'backlog/versions').glob('*'))),
        carried_tasks=len(carried),new_discoveries=sorted(new_ids),
        notebook_sha256=sha256(notebook),report_sha256=sha256(REPO/report),
        goal_archive_sha256=sha256(REPO/'docs/GOAL_V30.md'),
        outcome_audit_sha256=sha256(pub/'outcome_audit.json'),
        source_sha256={str(Path(__file__).relative_to(REPO)):sha256(Path(__file__))})
    atomic_json(output,result)
    print(result)


if __name__=='__main__':main()
