"""Recheck sealed v11 evidence against its archived contract and historical source."""
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
    pub=REPO/'results/span_011';root=ROOT/'runs/span_011';preserved=read_json(pub/'preservation.json')
    for name,digest in {**preserved['snapshot_sha256'],**preserved['user_notebook_sha256']}.items():
        assert sha256(REPO/name)==digest
    assert sha256(REPO/'docs/GOAL_V11.md')==preserved['goal_archive_sha256']
    assert sha256(REPO/'docs/GOAL_V10.md')==preserved['previous_goal_archive_sha256']
    assert (Path.home()/'notebooks/llm_classification_finetuning').resolve()==REPO/'notebooks'
    files=0;charge=0.
    for stage in ['learning','confirmation']:
        role='calibration_'+stage;bundle=read_json(pub/(role+'_bundle.json'))
        for name,digest in bundle['source_sha256'].items():
            historical=subprocess.check_output(['git','show',preserved['source_reference_commit']+':'+name],cwd=REPO)
            assert hashlib.sha256(historical).hexdigest()==digest
            actual=REPO/('docs/GOAL_V11.md' if name=='GOAL.md' else name)
            assert sha256(actual)==digest
        assert sha256(root/'notebooks'/role/'job.ipynb')==bundle['notebook_sha256']
        assert sha256(root/'notebooks'/role/'kernel-metadata.json')==bundle['metadata_sha256']
        job=root/'jobs'/role;q=read_json(pub/(role+'_qualification.json'))
        assert read_json(job/'status.json')['status']=='COMPLETE'
        assert all(q[k] for k in ['complete','all_controls_passed','runtime_gate_passed','csv_verified','both_t4_verified'])
        assert all(v['max_abs_probability_difference']==0 for v in q['control_identity'].values())
        assert q['runtime']['projection_seconds']<32400
        receipt=read_json(job/'retrieval.json')
        for name,digest in receipt['artifact_sha256'].items():assert sha256(job/'output'/name)==digest
        files+=len(receipt['artifact_sha256'])
        charge+=read_json(job/'quota.json')['used_seconds']-read_json(job/'push_intent.json')['quota_before']['used_seconds']
    assert files==47 and abs(charge-4087.202)<.001 and charge<=21600
    decision=read_json(pub/'deployment_decision.json')
    for name,digest in decision['evidence_sha256'].items():assert sha256(pub/name)==digest
    frozen=read_json(pub/'frozen_temperature.json');metrics=read_json(pub/'confirmation_metrics.json')
    assert metrics['information_gate_passed'] and not metrics['temperature_fit_on_confirmation']
    assert metrics['temperature']==frozen['temperature']==1.3161518671742531
    assert metrics['comparison']['paired_group95'][0]>0 and metrics['comparison']['loss_reduction']>=.001
    result=read_json(pub/'official_result.json');leader=read_json(pub/'official_leaderboard.json')
    assert str(result['ref'])=='58091536' and result['status'].endswith('COMPLETE') and float(result['public_score'])==.84797
    assert leader['complete'] and leader['rank']==30 and leader['teams']==313 and leader['score']==.83053 and not leader['target_achieved']
    assert len(list((root/'official').glob('*/receipt.json')))==1
    terminal=read_json(pub/'terminal_inventory.json');fresh=terminal['inventory']
    assert terminal['passed'] and not fresh['nonterminal_jobs'] and not fresh['quota']['paid_scaling'] and fresh['quota']['reserved_seconds']==0
    notebook=REPO/'notebooks/23_independent_confidence_calibration.ipynb';nb=nbformat.read(notebook,4)
    code=[c for c in nb.cells if c.cell_type=='code'];outputs=[o for c in code for o in c.get('outputs',[])]
    assert len(code)==4 and all(c.execution_count is not None for c in code)
    assert not any(o.output_type=='error' for o in outputs)
    assert sum('image/png' in o.get('data',{}) for o in outputs)==2
    assert '.84797' in notebook.read_text() and 'did not improve' in notebook.read_text()
    registry=read_json(REPO/'backlog/hypotheses.json');versions={v['number']:v for v in registry['versions']}
    final=read_json(REPO/'backlog/versions/v11.final.json');old={i['id']:i for i in final['items']}
    assert versions[11]['state']=='closed' and versions[11]['focus_ids']==['H59'] and old['H59']['status']=='done'
    carried={k for k in versions[11]['item_ids'] if old[k]['status'] not in ['done','rejected']}
    started=read_json(REPO/'backlog/versions/v12.start.json')
    next_version=next(v for v in started['versions'] if v['number']==12)
    assert len(carried)==40 and set(next_version['item_ids'])==carried|{'E17'}
    assert registry['active_version']>=12
    processes=subprocess.check_output(['ps','-eo','args='],text=True).splitlines()
    assert not any(re.match(r'^\S*python[0-9.]* (?:\S*/)?(?:run_span_011_calibration|submit_span_011_calibration)\.py(?: |$)',s) for s in processes)
    record=dict(at=utc_now(),passed=True,sealed_version=11,span_outcome='Confirmed later-vote gain failed official transfer',
        goal_achieved_by_span=False,official_score=.84797,best_score=.83053,official_regression=.01744,
        leaderboard=leader,public_gpu_seconds=charge,public_cap_seconds=21600,official_submissions=1,
        artifact_files_verified=files,prior_snapshots_preserved=len(preserved['snapshot_sha256']),
        user_notebooks_preserved=True,goal_archive_sha256=preserved['goal_archive_sha256'],
        source_reference_commit=preserved['source_reference_commit'],source_maps_verified=2,
        notebook_code_cells=4,notebook_figures=2,notebook_errors=0,notebook_sha256=sha256(notebook),
        unfinished_carryovers=40,new_discoveries=['E17'],successor_members=41)
    atomic_json(pub/'closure_audit.json',record)
    print(json.dumps({k:record[k] for k in ['passed','goal_achieved_by_span','official_score','public_gpu_seconds','unfinished_carryovers']}))


if __name__=='__main__':main()
