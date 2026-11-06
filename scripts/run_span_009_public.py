"""One durable GPU mutation; subsequent invocations observe the exact receipt."""
from pathlib import Path
import argparse
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT
from arena.experiment import atomic_json, read_json, utc_now, exclusive_lock
from arena.kaggle_control import run_job
from arena.span004_support import api_client, quota


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--role', choices=['public_ensemble', 'public_ensemble_runtime_fix'], default='public_ensemble')
    args = parser.parse_args()
    root = ROOT / 'runs/span_009'
    pub = REPO / 'results/span_009'
    config = REPO / 'configs/span_009.json'
    with exclusive_lock(root / 'preparation.lock'):
        if not (pub / 'span_plan.json').exists():
            fresh = quota(api_client())
            assert not fresh['paid_scaling']
            plan = dict(started_at=utc_now(), initial_quota_refresh_time=fresh['refresh_time'],
                quota_used_at_start=fresh['used_seconds'], gpu_session_seconds_cap=2400,
                deadline='2026-11-07T00:00:00+00:00', public_phase='before_observed_refresh')
            atomic_json(pub / 'span_plan.json', plan)
            atomic_json(config, dict(started_at=plan['started_at'], deadline=plan['deadline'],
                job_seconds_caps={'public_ensemble': 2400}))
        if args.role == 'public_ensemble_runtime_fix':
            assert read_json(root/'jobs/public_ensemble/status.json')['status'] == 'ERROR'
            cfg = read_json(config)
            cfg['job_seconds_caps']['public_ensemble_runtime_fix'] = 2200
            atomic_json(config, cfg)
    def precondition(role):
        rehearsal = pub / (role+'_cell_rehearsals.json') if role != 'public_ensemble' else pub/'cell_rehearsals.json'
        assert read_json(rehearsal)['passed']
        assert read_json(pub / (role+'_bundle.json'))['internet_enabled'] is False
    run_job(REPO, root, pub, config, args.role, dry_run=args.dry_run, precondition=precondition)


if __name__ == '__main__':
    main()
