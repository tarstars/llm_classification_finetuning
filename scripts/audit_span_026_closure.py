"""Verify the stopped H72 span after terminal artifacts and backlog transition."""
from pathlib import Path
import sys
import nbformat

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,read_json,sha256,utc_now


def verify_sources(hashes,cpu=False):
    for name,digest in hashes.items():
        mapped=name
        if name=='GOAL.md':mapped='docs/SPAN_026_CPU_GOAL.md' if cpu else 'docs/GOAL_V26.md'
        if cpu and name=='docs/SPAN_026_PLAN.md':mapped='docs/SPAN_026_CPU_PLAN.md'
        assert sha256(REPO/mapped)==digest,mapped


def main():
    pub=REPO/'results/span_026';root=ROOT/'runs/span_026'
    preserve=read_json(pub/'preservation.json');assert len(preserve['snapshot_sha256'])==102
    for n,h in {**preserve['snapshot_sha256'],**preserve['user_notebook_sha256']}.items():assert sha256(REPO/n)==h,n
    coverage=read_json(pub/'coverage_audit.json');verify_sources(coverage['source_sha256'],cpu=True)
    for n,h in {**coverage['input_sha256'],**coverage['output_sha256']}.items():assert sha256(ROOT/n)==h,n
    assert coverage['rows']==57477 and coverage['changed_rows']==357 and coverage['cpu_feasibility_passed']
    assert coverage['exact_original_lengths_reproduced'] and coverage['canonical_control_tokens_exact']
    assert coverage['candidate_max_tokens']<coverage['model_position_limit']
    selector=read_json(pub/'selector_audit.json');verify_sources(selector['source_sha256'])
    assert selector['selector_matches_all_native_changed_inputs'] and selector['false_positive_rows']==selector['false_negative_rows']==0
    prep=read_json(pub/'pilot_preparation.json');candidate_prep=read_json(pub/'candidate_preparation.json')
    assert candidate_prep['aggregate_cpu_source_seconds']<1200
    assert sha256(root/'pilot_spec.json')==prep['spec_sha256']
    assert sha256(root/'candidate_spec.json')==candidate_prep['spec_sha256']
    assert sha256(root/'hidden_asset_spec.json')==candidate_prep['hidden_spec_sha256']
    for n,h in prep['control_sha256'].items():assert sha256(root/'controls'/n)==h
    pilot=read_json(pub/'pilot_audit.json');verify_sources(pilot['source_sha256'])
    assert pilot['complete_information'] and pilot['engineering_screen_passed'] and pilot['gross_regression_guard_passed']
    assert pilot['weighted_lower_gain']<0 and not pilot['local_positive_gain_required']
    assert pilot['original_training_overlap'] and not pilot['independent_cv'] and not pilot['hidden_score_forecast']
    assert all(v['max_abs_probability_difference']==0 for v in pilot['controls'].values())
    assert sha256(root/'context_information_analysis.npz')==pilot['analysis_sha256']
    assert sha256(root/'jobs/context_pilot/retrieval.json')==pilot['retrieval_sha256']
    assert not (pub/'candidate_qualification.json').exists(),'Preserve the original frozen audit failure'
    qualification=read_json(pub/'candidate_qualification_replay.json');verify_sources(qualification['source_sha256'])
    replay=qualification['validation_replay']
    assert not replay['original_frozen_audit_passed'] and not replay['allows_official_submission']
    assert replay['canonical_raw_probability_tolerance_unchanged']==1e-5
    assert sha256(REPO/'scripts/replay_span_026_rejected_audit.py')==replay['companion_sha256']
    assert sha256(pub/'frozen_qualification_audit_failure.json')==replay['original_failure_sha256']
    assert qualification['complete'] and qualification['all_controls_passed'] and qualification['tokens_exact']
    assert not qualification['runtime_gate_passed'] and not qualification['official_submission_eligible']
    assert not qualification['labels_read'] and qualification['csv_verified'] and qualification['both_t4_verified']
    assert qualification['pilot_audit_sha256']==sha256(pub/'pilot_audit.json')
    floor=read_json(pub/'qualification_runtime_floor_replay.json')
    assert floor['passed'] and not floor['uses_targets'] and not floor['official_submission_eligible']
    assert sha256(REPO/'scripts/verify_span_026_runtime_floor.py')==floor['script_sha256']
    assert sha256(REPO/'src/arena/span026_qualification.py')==floor['helper_sha256']
    assert sha256(pub/'qualification_runtime_lower_bound.json')==floor['lower_bound_source_sha256']
    assert qualification['runtime']['projection_seconds']>=floor['minimum_projection_seconds']>=32400
    charges={};files={}
    for role,cap in [('context_pilot',7200),('context_qualification',3600)]:
        job=root/'jobs'/role;bundle=read_json(pub/(role+'_bundle.json'));verify_sources(bundle['source_sha256'])
        assert sha256(root/'notebooks'/role/'job.ipynb')==bundle['notebook_sha256']
        assert sha256(root/'notebooks'/role/'kernel-metadata.json')==bundle['metadata_sha256']
        proof=read_json(pub/'jobs'/role/'source_verification.json');assert proof['passed'] and proof['notebook_sha256']==bundle['notebook_sha256']
        assert read_json(pub/(role+'_cell_rehearsals.json'))['passed']
        state=read_json(job/'controller_state.json');assert state['stage']=='complete' and state['status']=='COMPLETE'
        assert read_json(job/'status.json')['status']=='COMPLETE'
        session=read_json(job/'output/session_summary.json');assert session['exit_code']==0 and session['seconds']<=cap
        charges[role]=state['charged_seconds'];assert 0<charges[role]<=cap+.1
        receipt=read_json(job/'retrieval.json');files[role]=receipt['files']
        for n,h in receipt['artifact_sha256'].items():assert sha256(job/'output'/n)==h,n
    assert sum(charges.values())<=10800 and len(list((root/'jobs').glob('*/push.json')))==2
    assert not list((root/'official').glob('*/submission_intent.json')) and not list((root/'official').glob('*/receipt.json'))
    inventory=read_json(pub/'terminal_inventory.json');assert not inventory['nonterminal_jobs']
    assert inventory['quota']['reserved_seconds']==0 and not inventory['quota']['paid_scaling']
    assert inventory['quota']['refresh_time']==read_json(pub/'span_plan.json')['initial_quota_refresh_time']
    assert read_json(pub/'terminal_submission_check.json')['all_official_terminal']
    board=read_json(pub/'final_leaderboard.json');assert board['complete'] and board['score']==.82987 and not board['target_achieved']
    final=read_json(REPO/'backlog/versions/v26.final.json');v=next(v for v in final['versions'] if v['number']==26)
    items={i['id']:i for i in final['items']};assert v['focus_ids']==['H72'] and items['H72']['status']=='done'
    assert any(h['status']=='done' and 'docs/SPAN_026_RESULTS.md' in h.get('evidence',[]) for h in items['H72']['history'])
    assert (REPO/'docs/SPAN_026_RESULTS.md').exists()
    carry={i for i in v['item_ids'] if items[i]['status'] not in ['done','rejected']};assert len(carry)==37
    nxt=read_json(REPO/'backlog/versions/v27.start.json');nv=next(v for v in nxt['versions'] if v['number']==27)
    assert set(nv['item_ids'])==carry|{'H73','H74'}
    notebook=REPO/'notebooks/38_context_coverage.ipynb';nb=nbformat.read(notebook,4)
    code=[c for c in nb.cells if c.cell_type=='code'];assert len(code)==2
    assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in code)
    assert (Path.home()/'notebooks/llm_classification_finetuning').resolve()==REPO/'notebooks'
    result=dict(at=utc_now(),passed=True,sealed_version=26,goal_achieved_by_span=False,
        public_gpu_charged_seconds=charges,total_public_gpu_charged_seconds=sum(charges.values()),
        artifact_files_verified=files,official_submissions=0,runtime_gate_passed=False,
        actual_projection_seconds=qualification['runtime']['projection_seconds'],
        historical_snapshots_preserved=102,snapshot_files_after_transition=len(list((REPO/'backlog/versions').glob('*'))),
        carried_tasks=37,new_discoveries=['H73','H74'],original_frozen_audit_passed=False,
        rejected_only_replay_complete=True,notebook_sha256=sha256(notebook),goal_archive_sha256=sha256(REPO/'docs/GOAL_V26.md'))
    atomic_json(pub/'closure_audit.json',result);print(result)


if __name__=='__main__':main()
