"""Read-only terminal resources, complete owned-job inventory and official history."""
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT
from arena.experiment import atomic_json, read_json, sha256, utc_now
from arena.span004_support import api_client, public_inventory
from arena.span031_inventory import inventory_with_resolution
from submit_span_013_order import submissions, TERMINAL
from submit_span_010_order import fields


def main():
    pub = REPO / 'results/span_031'
    run = ROOT / 'runs/span_031'
    output = pub / 'resource_close.json'
    assert not output.exists(), 'Preserve the first terminal observation'
    plan = read_json(pub / 'span_plan.json')
    state = read_json(run / 'jobs/size_pilot_compact/controller_state.json')
    audit = read_json(pub / 'H02_results.json')
    assert state['stage'] in {'complete', 'failed'} and state['status'] in TERMINAL
    assert audit['service_status'] == state['status']
    assert audit['charged_gpu_seconds'] == state['charged_seconds']
    assert 0 < state['charged_seconds'] <= plan['gpu_session_seconds_cap'] == 5400
    assert plan['public_version_limit'] == 1 and plan['official_candidate_limit'] == 0
    pushes = list((run / 'jobs').glob('*/push.json'))
    assert pushes == [run / 'jobs/size_pilot_compact/push.json']
    assert not list((run / 'official').glob('*/submission_intent.json'))
    api = api_client()
    rejected = read_json(pub / 'save_reconciliation.json')['kernel']
    fresh = public_inventory(inventory_with_resolution(
        api, plan['started_at'], rejected, pub / 'inventory_terminal_resolution.json'))
    atomic_json(pub / 'terminal_inventory.json', fresh)
    assert not fresh['nonterminal_jobs']
    quota = fresh['quota']
    assert quota['reserved_seconds'] == 0 and not quota['paid_scaling']
    assert quota['refresh_time'] == plan['initial_quota_refresh_time']
    delta = quota['used_seconds'] - plan['quota_used_at_start']
    assert -.1 <= delta <= 5400.1 and abs(delta - state['charged_seconds']) <= .1
    assert quota['remaining_seconds'] >= 32400
    history = submissions(api)
    rows = [fields(s) for s in history]
    assert all(str(s.status).split('.')[-1] in TERMINAL for s in history)
    assert len(rows) == read_json(pub / 'compact_admission.json')['official_receipts']
    assert all(str(s.date)[:19] < plan['started_at'][:19] for s in history)
    atomic_json(pub / 'terminal_official_history.json', dict(
        observed_at=utc_now(), complete=True, count=len(rows), all_terminal=True,
        new_official_submissions=0, submissions=rows))
    subprocess.run([sys.executable, str(REPO / 'scripts/observe_leaderboard.py'),
        '--output', 'results/span_031/final_leaderboard.json'], cwd=REPO,
        check=True, timeout=180)
    result = dict(at=utc_now(), passed=True, accepted_public_versions=1,
        official_submissions=0, all_owned_jobs_terminal=True,
        public_gpu_charged_seconds=state['charged_seconds'],
        quota_delta_seconds=delta, final_quota=quota,
        inventory_sha256=sha256(pub / 'terminal_inventory.json'),
        official_history_sha256=sha256(pub / 'terminal_official_history.json'),
        leaderboard_sha256=sha256(pub / 'final_leaderboard.json'),
        pilot_audit_sha256=sha256(pub / 'H02_results.json'),
        source_sha256=sha256(Path(__file__)))
    atomic_json(output, result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
