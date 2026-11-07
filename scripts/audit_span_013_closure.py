"""Verify sealed v13, its actual official outcome and preserved historical sources."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

import nbformat

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,sha256,utc_now


def main():
    pub=REPO/'results/span_013';root=ROOT/'runs/span_013';preserved=read_json(pub/'preservation.json')
    for name,digest in {**preserved['snapshot_sha256'],**preserved['user_notebook_sha256']}.items():
        assert sha256(REPO/name)==digest
    assert sha256(REPO/'docs/GOAL_V13.md')==preserved['goal_archive_sha256']
    assert sha256(REPO/'docs/GOAL_V12.md')==preserved['previous_goal_archive_sha256']
    assert (Path.home()/'notebooks/llm_classification_finetuning').resolve()==REPO/'notebooks'
    role='gemma_order';bundle=read_json(pub/(role+'_bundle.json'))
    for name,digest in bundle['source_sha256'].items():
        historical=subprocess.check_output(['git','show','344c9de:'+name],cwd=REPO)
        assert hashlib.sha256(historical).hexdigest()==digest
        assert sha256(REPO/('docs/GOAL_V13.md' if name=='GOAL.md' else name))==digest
    assert sha256(root/'notebooks'/role/'job.ipynb')==bundle['notebook_sha256']
    assert sha256(root/'notebooks'/role/'kernel-metadata.json')==bundle['metadata_sha256']
    job=root/'jobs'/role;retrieval=read_json(job/'retrieval.json')
    assert read_json(job/'status.json')['status']=='COMPLETE'
    for name,digest in retrieval['artifact_sha256'].items():assert sha256(job/'output'/name)==digest
    assert len(retrieval['artifact_sha256'])==36
    gate=read_json(pub/(role+'_qualification.json'))
    assert all(gate[k] for k in ['complete','all_controls_passed','runtime_gate_passed','csv_verified','both_t4_verified'])
    assert set(gate['control_identity'])=={'m0','m3','reverse_control','reverse_uniform','reverse_long','reverse_test'}
    assert all(v['passed'] and v['max_abs_probability_difference']==0 for v in gate['control_identity'].values())
    assert gate['runtime']['margin']==1.2 and gate['runtime']['reserve_seconds']==600
    assert gate['runtime']['projection_seconds']<32400 and not gate['independent_local_quality']
    replay=read_json(pub/'H60_cached_replay.json')
    assert replay['information_gate_passed'] and replay['quality_variants_tested']==1 and replay['temperature']==1
    for name,digest in replay['source_sha256'].items():assert sha256(REPO/name)==digest
    for name,digest in replay['cache_sha256'].items():assert sha256(Path(name))==digest
    decision=read_json(pub/'deployment_decision.json')
    assert decision['submit'] and decision['temperature']==1 and decision['packing_tokens']==8192
    for name,digest in decision['evidence_sha256'].items():assert sha256(pub/name)==digest
    assert len(list((root/'jobs').glob('*/push.json')))==1 and len(list((root/'official').glob('*/receipt.json')))==1
    receipt=read_json(pub/'official_receipt.json');result=read_json(pub/'official_result.json')
    assert str(receipt['ref'])==str(result['ref'])=='58103553'
    status=result['status'].split('.')[-1]
    assert status in ['COMPLETE','ERROR','FAILED','INVALID','CANCELED','CANCELLED']
    intent=read_json(pub/'official_submission_intent.json')
    assert intent['exact']=='arseniytrushin/llm-preference-span013-gemma-order/1'
    assert intent['notebook_sha256']==bundle['notebook_sha256'] and intent['hidden_seconds_cap']==32400
    assert intent['decision_sha256']==sha256(pub/'deployment_decision.json')
    assert sha256(REPO/'scripts/submit_span_013_order.py')==intent['controller_sha256']
    admitted=read_json(pub/'official_admission.json')
    assert admitted['submission_history_complete'] and not admitted['inventory']['nonterminal_jobs']
    dependencies=admitted['dependencies']['datasets']
    assert len(dependencies)==4 and all(d['public'] and d['current_version']==1 for d in dependencies)
    terminal=read_json(pub/'terminal_inventory.json');fresh=terminal['inventory']
    assert terminal['passed'] and not fresh['nonterminal_jobs'] and not terminal['pending_official_receipts']
    assert fresh['quota']['reserved_seconds']==0 and not fresh['quota']['paid_scaling']
    charge=terminal['public_gpu_seconds'];assert abs(charge-1671.483)<.001 and charge<=6000
    leader=read_json(pub/'final_leaderboard.json')
    assert leader['complete'] and leader['team_id']==17244411
    assert leader['target_achieved']==(leader['rank']<=19)
    score=float(result['public_score']) if status=='COMPLETE' else None
    if score is not None:assert leader['score']<=min(.83053,score)+1e-8
    notebook=REPO/'notebooks/25_gemma_only_order.ipynb';nb=nbformat.read(notebook,4)
    code=[c for c in nb.cells if c.cell_type=='code'];outputs=[o for c in code for o in c.get('outputs',[])]
    assert len(code)==3 and all(c.execution_count is not None for c in code)
    assert not any(o.output_type=='error' for o in outputs)
    assert sum('image/png' in o.get('data',{}) for o in outputs)==2
    if score is not None:assert f'Official result: {score:.5f}' in notebook.read_text()
    registry=read_json(REPO/'backlog/hypotheses.json');versions={v['number']:v for v in registry['versions']}
    final=read_json(REPO/'backlog/versions/v13.final.json');old={i['id']:i for i in final['items']}
    assert versions[13]['state']=='closed' and versions[13]['focus_ids']==['H60'] and old['H60']['status']=='done'
    carried={k for k in versions[13]['item_ids'] if old[k]['status'] not in ['done','rejected']}
    started=read_json(REPO/'backlog/versions/v14.start.json');successor=next(v for v in started['versions'] if v['number']==14)
    new={i['id'] for i in started['items'] if i['introduced_in']==14}
    assert len(carried)==39 and set(successor['item_ids'])==carried|new
    record=dict(at=utc_now(),passed=True,sealed_version=13,goal_achieved_by_span=leader['target_achieved'],
        official_receipt=58103553,official_status=status,official_score=score,best_score=leader['score'],leaderboard=leader,
        public_gpu_seconds=charge,official_submissions=1,artifact_files_verified=36,
        prior_snapshots_preserved=len(preserved['snapshot_sha256']),user_notebooks_preserved=True,
        source_reference_commit='344c9de',notebook_code_cells=3,notebook_figures=2,notebook_errors=0,
        notebook_sha256=sha256(notebook),unfinished_carryovers=39,new_discoveries=sorted(new),successor_members=len(successor['item_ids']))
    atomic_json(pub/'closure_audit.json',record)
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
