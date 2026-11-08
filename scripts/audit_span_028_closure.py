"""Verify exact-but-infeasible E18 and the immutable transition to v29."""
from pathlib import Path
import math
import sys
import nbformat

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json, read_json, sha256, utc_now


def main():
    pub=REPO/'results/span_028'; root=ROOT/'runs/span_028'; job=root/'jobs/prefetch_profile'
    preserve=read_json(pub/'preservation.json'); assert len(preserve['snapshot_sha256'])==110
    for n,h in {**preserve['snapshot_sha256'],**preserve['user_notebook_sha256']}.items():
        assert sha256(REPO/n)==h,n
    bundle=read_json(pub/'prefetch_profile_bundle.json'); cpu=read_json(pub/'cpu_audit.json')
    for mapping in [bundle['source_sha256'],cpu['source_sha256']]:
        for n,h in mapping.items():
            assert sha256(REPO/('docs/GOAL_V28.md' if n=='GOAL.md' else n))==h,n
    for n,h in cpu['output_sha256'].items():assert sha256(ROOT/n)==h,n
    for n,h in cpu['native_source_sha256'].items():assert sha256(ROOT/n)==h,n
    assert cpu['passed'] and cpu['elapsed_cpu_seconds']<=300
    assert sha256(root/'notebooks/prefetch_profile/job.ipynb')==bundle['notebook_sha256']
    assert sha256(root/'notebooks/prefetch_profile/kernel-metadata.json')==bundle['metadata_sha256']
    assert read_json(pub/'prefetch_profile_cell_rehearsals.json')['passed']
    assert read_json(pub/'jobs/prefetch_profile/source_verification.json')['passed']
    audit=read_json(pub/'profile_audit.json')
    assert audit['profile_complete'] and audit['all_controls_passed']
    assert all(x==0 for x in audit['canonical_max_abs'].values()) and len(audit['canonical_max_abs'])==15
    assert audit['host_tensors_and_microbatches_exact'] and audit['original_routes_exact']
    assert audit['mixtures_replayed'] and audit['csv_verified'] and audit['memory_guard_passed']
    assert not audit['engineering_gate_passed'] and not audit['runtime_gate_passed']
    assert not audit['official_submission_eligible'] and not audit['labels_read'] and not audit['quality_measured']
    receipt=read_json(job/'retrieval.json')
    for n,h in receipt['artifact_sha256'].items():assert sha256(job/'output'/n)==h,n
    state=read_json(job/'controller_state.json');assert state['stage']=='complete' and state['status']=='COMPLETE'
    assert read_json(job/'status.json')['status']=='COMPLETE'
    assert read_json(job/'output/session_summary.json')['exit_code']==0
    assert 0<state['charged_seconds']<3600
    assert len(list((root/'jobs').glob('*/push.json')))==1
    assert not list((root/'official').glob('*/submission_intent.json')) and not list((root/'official').glob('*/receipt.json'))
    runtime=audit['runtime']; floor=read_json(pub/'runtime_reference.json')['overhead_floor']
    out=job/'output/prefetch_profile'; roles=['m0','m3','m0_reverse']
    summaries={r:read_json(out/(r+'_summary.json')) for r in roles}
    uniform={r:summaries[r]['cohorts']['routed_uniform' if r=='m0_reverse' else 'uniform']['seconds'] for r in roles}
    cold=sum(max(floor['cold_by_role'][r],summaries[r]['cold_seconds']) for r in roles)
    warm=sum(max(floor['warm_by_role'][r],summaries[r]['cohorts']['warmup']['seconds']) for r in roles)
    calculated=sum(uniform.values())*25000/512*1.2+cold+warm+runtime['other_seconds']+600
    assert math.isclose(calculated,runtime['projection_seconds'],rel_tol=1e-12) and calculated>32400
    assert runtime['margin']==1.2 and runtime['reserve_seconds']==600
    description=read_json(pub/'runtime_description.json')
    assert description['source_sha256']==sha256(REPO/'scripts/analyze_span_028_runtime.py')
    assert description['audit_sha256']==sha256(pub/'profile_audit.json')
    for n,h in description['input_sha256'].items():assert sha256(ROOT/n)==h,n
    feasibility=description['two_role_feasibility']
    assert feasibility['projection_seconds']<32400 and not feasibility['candidate_qualified']
    assert not feasibility['official_submission_eligible'] and not feasibility['quality_measured']
    assert not description['labels_read'] and description['new_gpu_executions']==0
    inventory=read_json(pub/'terminal_inventory.json'); assert not inventory['nonterminal_jobs']
    q=inventory['quota']; plan=read_json(pub/'span_plan.json')
    assert q['reserved_seconds']==0 and not q['paid_scaling'] and q['refresh_time']==plan['initial_quota_refresh_time']
    assert abs(q['used_seconds']-plan['quota_used_at_start']-state['charged_seconds'])<.1
    submissions=read_json(pub/'terminal_submission_check.json')
    assert submissions['all_official_terminal'] and submissions['official_receipts']==7
    board=read_json(pub/'final_leaderboard.json'); assert board['complete'] and not board['target_achieved']
    final=read_json(REPO/'backlog/versions/v28.final.json'); version=next(v for v in final['versions'] if v['number']==28)
    items={i['id']:i for i in final['items']}; assert set(version['focus_ids'])=={'E18','H73'}
    for name,status in [('E18','done'),('H73','deferred')]:
        assert items[name]['status']==status
        assert any(h['status']==status and 'docs/SPAN_028_RESULTS.md' in h.get('evidence',[]) for h in items[name]['history'])
    carried={i for i in version['item_ids'] if items[i]['status'] not in ['done','rejected']}; assert len(carried)==38
    nxt=read_json(REPO/'backlog/versions/v29.start.json'); nv=next(v for v in nxt['versions'] if v['number']==29)
    assert set(nv['item_ids'])==carried|{'H75'}
    path=REPO/'notebooks/40_native_cpu_prefetch.ipynb'; nb=nbformat.read(path,4)
    cells=[c for c in nb.cells if c.cell_type=='code']; assert len(cells)==1
    assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in cells)
    assert (Path.home()/'notebooks/llm_classification_finetuning').resolve()==REPO/'notebooks'
    result=dict(at=utc_now(),passed=True,sealed_version=28,goal_achieved_by_span=False,
        engineering_gate_passed=False,all_raw_probabilities_exact=True,actual_projection_seconds=calculated,
        public_gpu_charged_seconds=state['charged_seconds'],official_submissions=0,
        artifact_files_verified=receipt['files'],historical_snapshots_preserved=110,
        snapshot_files_after_transition=len(list((REPO/'backlog/versions').glob('*'))),
        carried_tasks=38,new_discoveries=['H75'],notebook_sha256=sha256(path),
        goal_archive_sha256=sha256(REPO/'docs/GOAL_V28.md'))
    atomic_json(pub/'closure_audit.json',result); print(result)


if __name__=='__main__':main()
