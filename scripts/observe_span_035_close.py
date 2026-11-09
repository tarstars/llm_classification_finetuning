"""Terminal full inventory, quota, official history and board after the stopped pilot."""
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT
from arena.experiment import read_json, atomic_json, sha256, utc_now
from arena.span004_support import api_client, public_inventory
from arena.span031_inventory import inventory_with_resolution
from submit_span_013_order import submissions, TERMINAL
from submit_span_010_order import fields


def main():
    pub, run = REPO / 'results/span_035', ROOT / 'runs/span_035'
    assert not (pub / 'resource_close.json').exists(), 'Preserve the first terminal resource close'
    result = read_json(pub / 'H27_results.json'); plan = read_json(pub / 'span_plan.json')
    assert result['complete'] and not result['qualification_justified']
    state = read_json(run / 'jobs/recent_pilot/controller_state.json')
    assert state['stage'] == 'complete' and state['status'] == 'COMPLETE'
    assert state['charged_seconds'] == result['charged_gpu_seconds'] and 0 < state['charged_seconds'] <= 5400
    assert list((run / 'jobs').glob('*/push.json')) == [run / 'jobs/recent_pilot/push.json']
    assert not list((run / 'official').glob('*/submission_intent.json'))
    api = api_client()
    fresh = public_inventory(inventory_with_resolution(api, plan['started_at'],
        'arseniytrushin/llm-preference-span031-size-pilot', pub / 'inventory_terminal_resolution.json'))
    atomic_json(pub / 'terminal_inventory.json', fresh)
    q = fresh['quota']; assert not fresh['nonterminal_jobs']
    assert q['reserved_seconds'] == 0 and not q['paid_scaling']
    assert q['refresh_time'] == plan['initial_quota_refresh_time'] and q['remaining_seconds'] >= 32400
    delta = q['used_seconds'] - plan['quota_used_at_start']
    assert abs(delta - state['charged_seconds']) <= .1 and delta <= 5400.1
    history = submissions(api); assert len(history) == 8 and all(str(s.status).split('.')[-1] in TERMINAL for s in history)
    atomic_json(pub / 'terminal_official_history.json', dict(at=utc_now(), complete=True,
        count=len(history), all_terminal=True, new_official_submissions=0, submissions=[fields(s) for s in history]))
    subprocess.run([sys.executable, str(REPO / 'scripts/observe_leaderboard.py'),
        '--output', 'results/span_035/final_leaderboard.json'], cwd=REPO, check=True, timeout=180)
    record = dict(at=utc_now(), passed=True, accepted_public_versions=1, official_submissions=0,
        all_owned_jobs_terminal=True, public_gpu_charged_seconds=state['charged_seconds'],
        quota_delta_seconds=delta, quota=q, inventory_sha256=sha256(pub / 'terminal_inventory.json'),
        history_sha256=sha256(pub / 'terminal_official_history.json'),
        leaderboard_sha256=sha256(pub / 'final_leaderboard.json'), result_sha256=sha256(pub / 'H27_results.json'),
        public_allocation_seconds=9000, public_pilot_cap_seconds=5400,
        qualification_versions=0, qualifications_performed=0, source_sha256=sha256(Path(__file__)))
    atomic_json(pub / 'resource_close.json', record); print(record)


if __name__ == '__main__': main()
