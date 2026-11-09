"""Read-only quota, old pilot status, complete official history and board."""
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.experiment import atomic_json, read_json, sha256, utc_now
from arena.span004_support import api_client, quota, read_call
from submit_span_013_order import submissions, TERMINAL


def main():
    pub = REPO / 'results/span_033'
    assert not (pub / 'resource_close.json').exists()
    assert read_json(pub / 'preparation.json')['next_action'] == 'stop_insufficient_cached_coverage'
    plan = read_json(pub / 'span_plan.json')
    api = api_client()
    current = quota(api)
    status = str(read_call(lambda: api.kernels_status('arseniytrushin/llm-preference-span031-size-compact/1')).status).split('.')[-1]
    history = submissions(api)
    assert status == 'COMPLETE' and all(str(s.status).split('.')[-1] in TERMINAL for s in history)
    assert len(history) == 8
    assert current['used_seconds'] == plan['quota_used_at_start']
    assert current['reserved_seconds'] == 0 and not current['paid_scaling']
    assert current['refresh_time'] == plan['initial_quota_refresh_time']
    subprocess.run([sys.executable, str(REPO / 'scripts/observe_leaderboard.py'),
        '--output', 'results/span_033/final_leaderboard.json'], cwd=REPO, check=True, timeout=180)
    result = dict(at=utc_now(), passed=True, quota=current, gpu_charge_seconds=0,
        public_versions=0, official_submissions=0, official_history_complete=True,
        all_official_terminal=True, official_count=len(history), last_pilot_status=status,
        full_owned_inventory_refreshed=False,
        inventory_limit='CPU-only span; full owned catalog last verified in v31. Fresh admission required before any future GPU job.',
        leaderboard_sha256=sha256(pub / 'final_leaderboard.json'), source_sha256=sha256(Path(__file__)))
    atomic_json(pub / 'resource_close.json', result)
    print(result)


if __name__ == '__main__':
    main()
