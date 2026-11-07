"""Verify the sealed negative-result span without changing its historical source."""
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
from arena.experiment import read_json,atomic_json,sha256,utc_now


def main():
    pub=REPO/'results/span_010';root=ROOT/'runs/span_010'
    baseline=read_json(pub/'preservation.json')
    for name,digest in {**baseline['snapshot_sha256'],**baseline['user_notebook_sha256']}.items():
        assert sha256(REPO/name)==digest
    assert sha256(REPO/'docs/GOAL_V9.md')==baseline['previous_goal_archive_sha256']
    assert (Path.home()/'notebooks/llm_classification_finetuning').resolve()==REPO/'notebooks'
    goal=sha256(REPO/'docs/GOAL_V10.md')
    for role in ['order_pilot','order_packed']:
        bundle=read_json(pub/(role+'_bundle.json'))
        for name,digest in bundle['source_sha256'].items():
            historic=subprocess.check_output(['git','show','25d71c9:'+name],cwd=REPO)
            assert hashlib.sha256(historic).hexdigest()==digest
            assert sha256(REPO/('docs/GOAL_V10.md' if name=='GOAL.md' else name))==digest
        assert bundle['source_sha256']['GOAL.md']==goal
        assert sha256(root/'notebooks'/role/'job.ipynb')==bundle['notebook_sha256']
        assert sha256(root/'notebooks'/role/'kernel-metadata.json')==bundle['metadata_sha256']
        q=read_json(pub/('H57_'+role+'_qualification.json'))
        assert q['role']==role and q['complete'] and not q['runtime_gate_passed']
        assert q['runtime_projection_seconds']>=32400 and q['source']['passed']
        assert q['csv_verified'] and q['both_t4_verified']
    first=read_json(pub/'H57_order_pilot_qualification.json');second=read_json(pub/'H57_order_packed_qualification.json')
    assert first['all_controls_passed'] and not second['all_controls_passed']
    terminal=read_json(pub/'terminal_audit.json');assert terminal['passed']
    assert len(terminal['jobs'])==2 and sum(x['files_verified'] for x in terminal['jobs'])==100
    assert terminal['public_gpu_seconds']<=21600 and terminal['official_submissions']==0
    assert not terminal['fresh_inventory']['nonterminal_jobs'] and not terminal['fresh_inventory']['quota']['paid_scaling']
    assert not list((root/'official').glob('*/receipt.json')) and not list((root/'official').glob('*/submission_intent.json'))
    assert not terminal['leaderboard']['target_achieved'] and terminal['leaderboard']['rank']>19
    registry=read_json(REPO/'backlog/hypotheses.json');versions={v['number']:v for v in registry['versions']}
    final=read_json(REPO/'backlog/versions/v10.final.json');old={x['id']:x for x in final['items']}
    assert registry['active_version']==11 and versions[10]['state']=='closed'
    assert versions[10]['focus_ids']==['E15','H57'] and all(old[k]['status']=='done' for k in ['E15','H57'])
    carried={k for k in versions[10]['item_ids'] if old[k]['status'] not in ['done','rejected']}
    assert len(carried)==38 and set(versions[11]['item_ids'])==carried|{'E16','H58','H59'}
    assert versions[11]['focus_ids']==['H59'] and versions[12]['state']=='draft' and not versions[12]['item_ids']
    nbpath=REPO/'notebooks/22_backlog_v10_order_completion.ipynb';nb=nbformat.read(nbpath,4)
    cells=[c for c in nb.cells if c.cell_type=='code'];outputs=[o for c in cells for o in c.get('outputs',[])]
    assert len(cells)==4 and all(c.execution_count is not None for c in cells)
    assert not any(o.output_type=='error' for o in outputs)
    figures=sum('image/png' in o.get('data',{}) for o in outputs);assert figures==2
    processes=subprocess.check_output(['ps','-eo','args='],text=True).splitlines()
    assert not any(re.match(r'^\S*python[0-9.]* (?:\S*/)?(?:run_span_010_order|run_span_010_packed|submit_span_010_order)\.py(?: |$)',s) for s in processes)
    result=dict(at=utc_now(),passed=True,span_outcome='Two completed but unqualified recipes; no submission',
        goal_achieved=False,goal_objective='move above 20th place in leaderboard',leaderboard=terminal['leaderboard'],
        prior_snapshots_preserved=len(baseline['snapshot_sha256']),user_notebooks_preserved=True,
        exact_goal_archive_sha256=goal,historical_source_commit='25d71c9',source_maps_verified=2,
        public_gpu_seconds=terminal['public_gpu_seconds'],public_cap_seconds=21600,artifact_files_verified=100,
        official_submissions=0,notebook_code_cells=4,notebook_figures=figures,notebook_errors=0,notebook_sha256=sha256(nbpath),
        sealed_version=10,active_version=11,active_members=41,unfinished_carryovers=38,
        new_discoveries=['E16','H58','H59'],next_focus=['H59'],next_draft=12,next_draft_empty=True)
    atomic_json(pub/'closure_audit.json',result)
    print(json.dumps({k:result[k] for k in ['passed','goal_achieved','public_gpu_seconds','active_members','notebook_errors']}))


if __name__=='__main__':main()
