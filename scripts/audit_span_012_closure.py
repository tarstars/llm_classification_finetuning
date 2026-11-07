"""Verify sealed v12 evidence without depending on a later span's root goal."""
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
    pub=REPO/'results/span_012';root=ROOT/'runs/span_012';preserved=read_json(pub/'preservation.json')
    for name,digest in {**preserved['snapshot_sha256'],**preserved['user_notebook_sha256']}.items():
        assert sha256(REPO/name)==digest
    assert sha256(REPO/'docs/GOAL_V12.md')==preserved['goal_archive_sha256']
    assert sha256(REPO/'docs/GOAL_V11.md')==preserved['previous_goal_archive_sha256']
    assert (Path.home()/'notebooks/llm_classification_finetuning').resolve()==REPO/'notebooks'
    bundle=read_json(pub/'packing_profile_bundle.json')
    for name,digest in bundle['source_sha256'].items():
        historical=subprocess.check_output(['git','show',preserved['source_reference_commit']+':'+name],cwd=REPO)
        assert hashlib.sha256(historical).hexdigest()==digest
        assert sha256(REPO/('docs/GOAL_V12.md' if name=='GOAL.md' else name))==digest
    assert sha256(root/'notebooks/packing_profile/job.ipynb')==bundle['notebook_sha256']
    assert sha256(root/'notebooks/packing_profile/kernel-metadata.json')==bundle['metadata_sha256']
    job=root/'jobs/packing_profile';receipt=read_json(job/'retrieval.json')
    assert read_json(job/'status.json')['status']=='COMPLETE'
    for name,digest in receipt['artifact_sha256'].items():assert sha256(job/'output'/name)==digest
    assert len(receipt['artifact_sha256'])==21
    q=read_json(pub/'E16_qualification.json');s=q['selection']
    assert q['complete'] and q['engineering_gate_passed'] and q['selected_packing']==8192
    assert q['hardware_telemetry_passed'] and q['both_t4_verified'] and not q['targets_used']
    assert q['base_only_projection_seconds']<32400 and s['bracket_stable']
    assert not s['candidate_numerically_eligible'] and not s['candidate_at_least_5pct_faster']
    for model,values in q['raw_control_differences'].items():
        assert values['profile_8192_first']==values['profile_8192_last']==0
        assert values['profile_6144']>=1e-5
    replay=read_json(pub/'H58_cached_replay.json')
    assert not replay['information_gate_passed'] and not replay['independent_local_quality']
    assert replay['coverage']['uniform']['loss_reduction']<=.0005 and replay['quality_variants_tested']==1
    assert replay['new_gpu_executions']==0 and replay['temperature']==1
    for name,digest in replay['source_sha256'].items():assert sha256(REPO/name)==digest
    for name,digest in replay['cache_sha256'].items():assert sha256(Path(name))==digest
    assert len(list((root/'jobs').glob('*/push.json')))==1
    assert not list((root/'official').glob('*/receipt.json'))
    terminal=read_json(pub/'terminal_inventory.json');fresh=terminal['inventory']
    assert terminal['passed'] and not fresh['nonterminal_jobs'] and not terminal['pending_official_receipts']
    assert fresh['quota']['reserved_seconds']==0 and not fresh['quota']['paid_scaling']
    charge=terminal['public_gpu_seconds'];assert abs(charge-1299.905)<.001 and charge<=7200
    notebook=REPO/'notebooks/24_packing_and_llama_order.ipynb';nb=nbformat.read(notebook,4)
    code=[c for c in nb.cells if c.cell_type=='code'];outputs=[o for c in code for o in c.get('outputs',[])]
    assert len(code)==3 and all(c.execution_count is not None for c in code)
    assert not any(o.output_type=='error' for o in outputs)
    assert sum('image/png' in o.get('data',{}) for o in outputs)==1
    assert 'H58 stopped' in notebook.read_text()
    registry=read_json(REPO/'backlog/hypotheses.json');versions={v['number']:v for v in registry['versions']}
    final=read_json(REPO/'backlog/versions/v12.final.json');old={i['id']:i for i in final['items']}
    assert versions[12]['state']=='closed' and versions[12]['focus_ids']==['E16','H58']
    assert old['E16']['status']==old['H58']['status']=='done'
    carried={k for k in versions[12]['item_ids'] if old[k]['status'] not in ['done','rejected']}
    started=read_json(REPO/'backlog/versions/v13.start.json');successor=next(v for v in started['versions'] if v['number']==13)
    assert len(carried)==39 and set(successor['item_ids'])==carried|{'H60'}
    leader=read_json(pub/'final_leaderboard.json')
    assert leader['complete'] and leader['rank']==30 and leader['score']==.83053 and not leader['target_achieved']
    record=dict(at=utc_now(),passed=True,sealed_version=12,goal_achieved_by_span=False,
        outcome='Retain8192; Llama-only order information gate failed',public_gpu_seconds=charge,
        official_submissions=0,artifact_files_verified=21,prior_snapshots_preserved=len(preserved['snapshot_sha256']),
        user_notebooks_preserved=True,source_reference_commit=preserved['source_reference_commit'],
        notebook_code_cells=3,notebook_figures=1,notebook_axes=2,notebook_errors=0,notebook_sha256=sha256(notebook),
        unfinished_carryovers=39,new_discoveries=['H60'],successor_members=40,leaderboard=leader)
    atomic_json(pub/'closure_audit.json',record)
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
