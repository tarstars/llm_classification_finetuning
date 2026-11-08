"""Fix v16 resource period, audit deadline and historical preservation once."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib
import json
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT
from arena.experiment import atomic_json, exclusive_lock, read_json, sha256, utc_now
from arena.span004_support import api_client, quota


def main():
    root = ROOT / 'runs/span_016'
    pub = REPO / 'results/span_016'
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(root / 'preparation.lock'):
        assert not (pub / 'span_plan.json').exists(), 'Preserve original resources/deadlines'
        data = read_json(REPO / 'backlog/hypotheses.json')
        version = next(v for v in data['versions'] if v['number'] == 16)
        assert data['active_version'] == 16 and version['focus_ids'] == ['H56']
        assert next(i for i in data['items'] if i['id'] == 'H56')['status'] == 'running'
        assert read_json(REPO / 'results/span_015/closure_audit.json')['passed']
        reference = '9dfb891'
        names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', reference,
            '--', 'backlog/versions'], cwd=REPO, text=True).splitlines()
        assert len(names) == 62
        snapshots = {}
        for name in names:
            digest = hashlib.sha256(subprocess.check_output(['git', 'show', reference + ':' + name], cwd=REPO)).hexdigest()
            assert sha256(REPO / name) == digest
            snapshots[name] = digest
        old = read_json(REPO / 'results/span_015/preservation.json')
        for name, digest in old['user_notebook_sha256'].items():
            assert sha256(REPO / name) == digest
        atomic_json(pub / 'preservation.json', dict(at=utc_now(), snapshot_reference_commit=reference,
            snapshot_sha256=snapshots, user_notebook_sha256=old['user_notebook_sha256'],
            previous_goal_archive_sha256=sha256(REPO / 'docs/GOAL_V15.md')))
        q = quota(api_client())
        assert q['reserved_seconds'] == 0 and not q['paid_scaling'] and q['remaining_seconds'] >= 3600
        start = datetime.now(timezone.utc)
        plan = dict(started_at=start.isoformat(), deadline=(start + timedelta(hours=48)).isoformat(),
            initial_quota=q, quota_used_at_start=q['used_seconds'],
            initial_quota_refresh_time=q['refresh_time'], gpu_session_seconds_cap=18000,
            public_version_limit=2, production_recipe_limit=1, official_candidate_limit=1,
            source_audit_deadline=(start + timedelta(hours=1)).isoformat(),
            source_audit_bytes_cap=50 * 1024**2, source_audit_admits_gpu=False,
            target_rank_maximum=19, parent_official_receipt=58103553, parent_official_score=.82987,
            hidden_cap_seconds=32400, raw_probability_tolerance=1e-5,
            calibration_parameters=3, pilot_rows=1024, seed=42,
            models={
                '8B': dict(repository='Skywork/Skywork-Reward-V2-Qwen3-8B', revision='6f19fdefb933293d4898bdb59a96f7223d998659'),
                '0.6B': dict(repository='Skywork/Skywork-Reward-V2-Qwen3-0.6B', revision='8c14a4e9e6321deaf572544339b16b8d6bbe8886')})
        atomic_json(pub / 'span_plan.json', plan)
        atomic_json(REPO / 'configs/span_016.json', dict(started_at=plan['started_at'],
            deadline=plan['deadline'], job_seconds_caps=dict(reward_pilot=3600, reward_followup=14400)))
        print(json.dumps(dict(prepared=True, new_gpu_executions=0, quota=q,
            source_audit_deadline=plan['source_audit_deadline'])))


if __name__ == '__main__':
    main()
