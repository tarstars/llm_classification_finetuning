"""Verify the rejected H74 span and preserved transition; never admits a model."""
from pathlib import Path
import sys

import nbformat

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT
from arena.experiment import atomic_json, read_json, sha256, utc_now


def main():
    pub = REPO / 'results/span_027'
    root = ROOT / 'runs/span_027'
    job = root / 'jobs/packing_profile'
    preserve = read_json(pub / 'preservation.json')
    assert len(preserve['snapshot_sha256']) == 106
    for n, h in {**preserve['snapshot_sha256'], **preserve['user_notebook_sha256']}.items():
        assert sha256(REPO / n) == h, n
    bundle = read_json(pub / 'packing_profile_bundle.json')
    for n, h in bundle['source_sha256'].items():
        path = REPO / ('docs/GOAL_V27.md' if n == 'GOAL.md' else n)
        assert sha256(path) == h, n
    assert sha256(root / 'notebooks/packing_profile/job.ipynb') == bundle['notebook_sha256']
    assert sha256(root / 'notebooks/packing_profile/kernel-metadata.json') == bundle['metadata_sha256']
    assert read_json(pub / 'packing_profile_cell_rehearsals.json')['passed']
    assert read_json(pub / 'jobs/packing_profile/source_verification.json')['passed']
    cpu = read_json(pub / 'packing_cpu_audit.json')
    assert cpu['cpu_packing_proof_passed'] and cpu['elapsed_cpu_seconds'] <= 600
    audit = read_json(pub / 'profile_audit.json')
    assert not audit['profile_complete'] and not audit['engineering_gate_passed']
    assert not audit['labels_read'] and not audit['quality_measured']
    assert not audit['official_submission_eligible']
    assert audit['failed_guard_independently_replayed']
    failure = audit['guard_failure']
    assert failure['role'] == 'm3' and failure['cohort'] == 'control'
    assert failure['checks']['canonical_max_abs'] > audit['original_probability_tolerance'] == 1e-5
    for key in ['ids_match', 'lengths_match', 'token_hash_match', 'checkpoint_match', 'memory_passed']:
        assert failure['checks'][key]
    receipt = read_json(job / 'retrieval.json')
    for n, h in receipt['artifact_sha256'].items():
        assert sha256(job / 'output' / n) == h, n
    state = read_json(job / 'controller_state.json')
    assert state['stage'] == 'complete' and state['status'] == 'COMPLETE'
    assert read_json(job / 'status.json')['status'] == 'COMPLETE'
    session = read_json(job / 'output/session_summary.json')
    assert session['exit_code'] == 0 and session['seconds'] <= 3600
    assert 0 < state['charged_seconds'] < 3600
    assert len(list((root / 'jobs').glob('*/push.json'))) == 1
    assert not list((root / 'official').glob('*/submission_intent.json'))
    assert not list((root / 'official').glob('*/receipt.json'))
    description = read_json(pub / 'failure_description.json')
    assert description['script_sha256'] == sha256(REPO / 'scripts/analyze_span_027_failure.py')
    assert description['original_audit_sha256'] == sha256(pub / 'profile_audit.json')
    assert not description['labels_read'] and not description['quality_measured']
    assert description['new_gpu_executions'] == 0
    for n, h in description['input_sha256'].items():
        assert sha256(ROOT / n) == h, n
    assert description['cohorts']['m0_uniform']['rows_above_original_tolerance'] == 14
    assert description['cohorts']['m0_long']['rows_above_original_tolerance'] == 4
    assert description['cohorts']['m3_control']['rows_above_original_tolerance'] == 12
    inventory = read_json(pub / 'terminal_inventory.json')
    assert not inventory['nonterminal_jobs']
    q = inventory['quota']; plan = read_json(pub / 'span_plan.json')
    assert q['reserved_seconds'] == 0 and not q['paid_scaling']
    assert q['refresh_time'] == plan['initial_quota_refresh_time']
    assert abs(q['used_seconds'] - plan['quota_used_at_start'] - state['charged_seconds']) < .1
    submissions = read_json(pub / 'terminal_submission_check.json')
    assert submissions['all_official_terminal'] and submissions['official_receipts'] == 7
    board = read_json(pub / 'final_leaderboard.json')
    assert board['complete'] and board['score'] == .82987 and not board['target_achieved']
    final = read_json(REPO / 'backlog/versions/v27.final.json')
    version = next(v for v in final['versions'] if v['number'] == 27)
    items = {i['id']: i for i in final['items']}
    assert set(version['focus_ids']) == {'H74', 'H73'}
    assert items['H74']['status'] == 'done' and items['H73']['status'] == 'deferred'
    for name, status in [('H74', 'done'), ('H73', 'deferred')]:
        assert any(h['status'] == status and 'docs/SPAN_027_RESULTS.md' in h.get('evidence', [])
                   for h in items[name]['history'])
    carried = {i for i in version['item_ids'] if items[i]['status'] not in ['done', 'rejected']}
    assert len(carried) == 38
    nxt = read_json(REPO / 'backlog/versions/v28.start.json')
    next_version = next(v for v in nxt['versions'] if v['number'] == 28)
    assert set(next_version['item_ids']) == carried | {'E18'}
    nb_path = REPO / 'notebooks/39_native_packing.ipynb'
    nb = nbformat.read(nb_path, 4)
    cells = [c for c in nb.cells if c.cell_type == 'code']
    assert len(cells) == 2 and all(c.execution_count is not None for c in cells)
    assert not any(o.output_type == 'error' for c in cells for o in c.outputs)
    assert (Path.home() / 'notebooks/llm_classification_finetuning').resolve() == REPO / 'notebooks'
    result = dict(at=utc_now(), passed=True, sealed_version=27, goal_achieved_by_span=False,
                  engineering_gate_passed=False, official_submissions=0,
                  public_gpu_charged_seconds=state['charged_seconds'],
                  artifact_files_verified=receipt['files'], historical_snapshots_preserved=106,
                  snapshot_files_after_transition=len(list((REPO / 'backlog/versions').glob('*'))),
                  carried_tasks=38, new_discoveries=['E18'], notebook_sha256=sha256(nb_path),
                  goal_archive_sha256=sha256(REPO / 'docs/GOAL_V27.md'),
                  failure_description_sha256=sha256(pub / 'failure_description.json'))
    atomic_json(pub / 'closure_audit.json', result)
    print(result)


if __name__ == '__main__':
    main()
