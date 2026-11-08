"""Freeze H65's source-first contract and resource period without GPU admission."""
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
    pub, root = REPO / 'results/span_019', ROOT / 'runs/span_019'
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(root / 'preparation.lock'):
        assert not (pub / 'span_plan.json').exists(), 'Preserve original deadline and caps'
        registry = read_json(REPO / 'backlog/hypotheses.json')
        assert registry['active_version'] == 19
        assert next(v for v in registry['versions'] if v['number'] == 19)['focus_ids'] == ['H65']
        assert next(i for i in registry['items'] if i['id'] == 'H65')['status'] == 'running'
        assert read_json(REPO / 'results/span_018/closure_audit.json')['passed']
        reference = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
        names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', reference,
            '--', 'backlog/versions'], cwd=REPO, text=True).splitlines()
        assert len(names) == 74
        snapshots = {}
        for name in names:
            digest = hashlib.sha256(subprocess.check_output(['git', 'show', reference + ':' + name], cwd=REPO)).hexdigest()
            assert sha256(REPO / name) == digest
            snapshots[name] = digest
        users = read_json(REPO / 'results/span_018/preservation.json')['user_notebook_sha256']
        for name, digest in users.items():
            assert sha256(REPO / name) == digest
        atomic_json(pub / 'preservation.json', dict(at=utc_now(), snapshot_reference_commit=reference,
            snapshot_sha256=snapshots, user_notebook_sha256=users,
            previous_goal_archive_sha256=sha256(REPO / 'docs/GOAL_V18.md')))
        q = quota(api_client())
        assert q['reserved_seconds'] == 0 and not q['paid_scaling']
        now = datetime.now(timezone.utc)
        plan = dict(started_at=now.isoformat(), deadline=(now + timedelta(hours=48)).isoformat(),
            source_audit_deadline=(now + timedelta(hours=1)).isoformat(), source_audit_bytes_cap=50*1024**2,
            initial_quota=q, quota_used_at_start=q['used_seconds'], initial_quota_refresh_time=q['refresh_time'],
            repository='wath5/kgl_lmsys_pref_classif', revision='cd24909db06583ecf00e65b6919c6cc1531c9885',
            gpu_session_seconds_cap=18000, initial_public_cap_seconds=3600, followup_public_cap_seconds=14400,
            public_version_limit=2, production_recipe_limit=1, official_candidate_limit=1,
            hidden_cap_seconds=32400, source_stage_admits_gpu=False, source_stage_admits_full_weights=False,
            target_rank_maximum=19, parent_official_receipt=58103553, parent_official_score=.82987,
            raw_probability_tolerance=1e-5, warmup_rows=80)
        atomic_json(pub / 'span_plan.json', plan)
        atomic_json(REPO / 'configs/span_019.json', dict(started_at=plan['started_at'], deadline=plan['deadline'],
            job_seconds_caps=dict(classifier_pilot=3600, classifier_followup=14400)))
        print(json.dumps(dict(prepared=True, new_gpu_executions=0,
            deadline=plan['deadline'], source_audit_deadline=plan['source_audit_deadline'], quota=q)))


if __name__ == '__main__':
    main()
