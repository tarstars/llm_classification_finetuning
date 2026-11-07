"""Audit the completed rank30 span without another GPU execution."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys

import nbformat

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,sha256,verify_sources,utc_now


def main():
    pub=REPO/'results/span_009';run=ROOT/'runs/span_009'
    baseline=read_json(pub/'preclosure_preservation.json')
    for name,digest in baseline['snapshot_sha256'].items():assert sha256(REPO/name)==digest
    for name,digest in baseline['user_notebook_sha256'].items():assert sha256(REPO/name)==digest
    assert sha256(REPO/'GOAL.md')==sha256(REPO/'docs/GOAL_V9.md')==baseline['goal_archive_sha256']
    assert (Path.home()/'notebooks/llm_classification_finetuning').is_symlink()
    assert (Path.home()/'notebooks/llm_classification_finetuning').resolve()==REPO/'notebooks'
    registry=read_json(REPO/'backlog/hypotheses.json');items={x['id']:x for x in registry['items']}
    versions={x['number']:x for x in registry['versions']}
    assert registry['active_version']==10 and versions[9]['state']=='closed'
    assert versions[9]['focus_ids']==['E14','H54'] and all(items[k]['status']=='done' for k in versions[9]['focus_ids'])
    final=read_json(REPO/'backlog/versions/v9.final.json');old={x['id']:x for x in final['items']}
    carry={k for k in versions[9]['item_ids'] if old[k]['status'] not in ['done','rejected']}
    discoveries={'E15','H55','H56'}
    assert len(carry)==36 and set(versions[10]['item_ids'])==carry|discoveries
    assert not versions[10]['focus_ids'] and all(items[k]['status']=='planned' for k in discoveries)
    assert not versions[11]['item_ids'] and versions[11]['state']=='draft'
    bundle=read_json(pub/'public_ensemble_resume_bundle.json');verify_sources(REPO,bundle['source_sha256'])
    folder=run/'notebooks/public_ensemble_resume'
    assert sha256(folder/'job.ipynb')==bundle['notebook_sha256']
    assert sha256(folder/'kernel-metadata.json')==bundle['metadata_sha256']
    candidate=read_json(pub/'deployed_candidate.json');official=read_json(pub/'official_result.json')
    proof=read_json(pub/'final_leaderboard_verification.json');live=proof['leaderboard']
    assert str(official['ref'])=='58068582' and official['status'].endswith('COMPLETE')
    assert float(official['public_score'])==candidate['official_log_loss']==0.83053
    assert live['complete'] and live['rank']<=30 and live['team_id']==17244411 and live['score']==0.83053
    assert candidate['kernel_version'].endswith('/4') and candidate['notebook_sha256']==bundle['notebook_sha256']
    intent=read_json(pub/'official_submission_intent.json')
    assert intent['version']==4 and intent['notebook_sha256']==bundle['notebook_sha256']
    assert len(list((run/'official').glob('*/receipt.json')))==1
    terminal=read_json(pub/'terminal_audit.json');assert terminal['passed']
    assert terminal['public_gpu_seconds']<=terminal['public_cap_seconds']==2400
    assert sum(j['retrieved_files_verified'] for j in terminal['jobs'])==48
    assert not terminal['fresh_inventory']['nonterminal_jobs']
    assert not terminal['fresh_inventory']['quota']['paid_scaling']
    processes=subprocess.check_output(['ps','-eo','args='],text=True).splitlines()
    controllers=[p for p in processes if re.match(r'^\S*python[0-9.]* (?:\S*/)?(?:run_span_009_public|submit_span_009_public)\.py(?: |$)',p)]
    assert not controllers,'An owned v9 controller is still live'
    notebook=REPO/'notebooks/21_backlog_v9_public_ensemble.ipynb';nb=nbformat.read(notebook,4)
    code=[c for c in nb.cells if c.cell_type=='code'];assert len(code)==5 and all(c.execution_count is not None for c in code)
    outputs=[o for c in code for o in c.get('outputs',[])]
    assert not any(o.output_type=='error' for o in outputs)
    figures=sum('image/png' in o.get('data',{}) for o in outputs);assert figures==2
    text='\n'.join(str(o.get('text',o.get('data',{}).get('text/plain',''))) for o in outputs)
    assert '0.83053' in text and "'rank': 30" in text
    forbidden={'prompt','response_a','response_b','access_token','api_key','authorization','cookie','password'}
    def check(value):
        if isinstance(value,dict):
            assert not forbidden.intersection(k.lower() for k in value)
            for child in value.values():check(child)
        elif isinstance(value,list):
            for child in value:check(child)
        elif isinstance(value,str):assert 'X-Goog-Signature=' not in value and 'X-Amz-Signature=' not in value
    files=list(pub.rglob('*.json'))
    for path in files:check(read_json(path))
    validation=read_json(pub/'qualification_validation.json')
    validation.update(official_status='COMPLETE',goal_achieved=True,leaderboard_rank=live['rank'])
    validation['notebook'].update(code_cells=5,errors=0,sha256=sha256(notebook))
    atomic_json(pub/'qualification_validation.json',validation)
    record=dict(at=utc_now(),passed=True,objective='move to 30 place in leaderboard',
        official_ref=58068582,official_score=0.83053,leaderboard=live,
        prior_snapshots_preserved=len(baseline['snapshot_sha256']),user_notebooks_preserved=True,
        exact_goal_archive=True,notebook_link_valid=True,notebook_code_cells=5,
        notebook_figures=figures,notebook_errors=0,notebook_sha256=sha256(notebook),
        public_gpu_seconds=terminal['public_gpu_seconds'],public_gpu_cap_seconds=2400,
        verified_artifact_files=48,no_owned_live_controllers=True,
        current_frozen_sources_unchanged=True,privacy_json_files_checked=len(files),
        sealed_version=9,active_version=10,active_members=39,unfinished_carryovers=36,
        new_discoveries=sorted(discoveries),successor_focus=[],next_draft=11,
        next_draft_empty=True,goal_archive_sha256=sha256(REPO/'docs/GOAL_V9.md'),
        closing_commit='The Git commit containing this audit; verify remote main=HEAD after push.')
    atomic_json(pub/'closure_audit.json',record)
    print(json.dumps({k:record[k] for k in ['passed','official_score','prior_snapshots_preserved',
        'active_members','notebook_code_cells','notebook_figures','notebook_errors','public_gpu_seconds']}))


if __name__=='__main__':main()
