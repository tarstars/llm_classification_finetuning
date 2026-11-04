"""Admit one immutable v4 diagnostic execution and monitor its exact version."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT, json_write
from arena.span004_support import api_client, inventory, quota, read_call, TERMINAL, public_inventory


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job', choices=['repeatability'])
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    heavy, aggregate = ROOT / 'runs/span_004', REPO / 'results/span_004'
    directory, public = heavy / 'jobs' / args.job, aggregate / 'jobs' / args.job
    directory.mkdir(parents=True, exist_ok=True)
    public.mkdir(parents=True, exist_ok=True)
    lock = (directory / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config, plan = read(REPO / 'configs/span_004.json'), read(heavy / 'span_plan.json')
    folder = heavy / 'notebooks' / args.job
    bundle, metadata = read(aggregate / (args.job + '_bundle.json')), read(folder / 'kernel-metadata.json')
    cap = config['job_seconds_caps'][args.job]
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest(folder / metadata['code_file']) == bundle['notebook_sha256']
    assert digest(REPO / 'configs/span_004.json') == bundle['config_sha256']
    assert cap <= plan['per_job_seconds_cap'] and metadata['enable_internet'] is False
    contract = read(aggregate / 'premeasurement_contract.json')
    for source, sha in contract['source_sha256'].items():
        assert digest(REPO / source) == sha, source
        committed = subprocess.check_output(['git', 'show', 'HEAD:' + source], cwd=REPO)
        assert hashlib.sha256(committed).hexdigest() == sha, 'Commit frozen sources before GPU launch'
    deadline = datetime.fromisoformat(plan['deadline']).timestamp()
    def save(name, value):
        for target in [directory, public]:
            saved = public_inventory(value) if name == 'admission_inventory.json' and target == public else value
            json_write(target / name, saved)
    def state(stage, **values):
        value = {'stage': stage, 'utc': datetime.now(timezone.utc).isoformat(), 'job': args.job, **values}
        save('controller_state.json', value)
        print(json.dumps(value), flush=True)
    api = api_client()
    if not (directory / 'push.json').exists():
        assert not (directory / 'push_intent.json').exists(), 'Reconcile an ambiguous push before any retry'
        admission = (heavy / 'admission.lock').open('a')
        fcntl.flock(admission, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state('checking_live_inventory')
        observed = inventory(api, plan['started_at'])
        q = observed['quota']
        assert not observed['nonterminal_jobs'] and q['reserved_seconds'] == 0 and not q['paid_scaling']
        dependencies = read(aggregate / 'dependency_inventory.json')
        assert dependencies['pinned_model_version_available']
        assert set(metadata['kernel_sources']) == {row['source'] for row in dependencies['parents']}
        assert all(row['status'] == 'COMPLETE' for row in dependencies['parents'])
        assert (datetime.now(timezone.utc) - datetime.fromisoformat(dependencies['observed_at'])).total_seconds() < 3600
        assert q['refresh_time'] == plan['initial_quota_refresh_time']
        spent = q['used_seconds'] - plan['quota_used_at_start']
        assert spent >= 0 and spent + cap <= plan['gpu_session_seconds_cap']
        assert q['remaining_seconds'] >= cap and time.time() + cap + 300 < deadline
        intent = {'utc': datetime.now(timezone.utc).isoformat(), 'kernel': metadata['id'],
                  'max_session_seconds': cap, 'quota_before': q, 'span_charged_before': spent,
                  'notebook_sha256': bundle['notebook_sha256'], 'config_sha256': bundle['config_sha256']}
        save('admission_inventory.json', observed)
        if args.dry_run:
            state('dry_run_admissible', intent=intent)
            return
        save('push_intent.json', intent)
        state('pushing_one_version')
        # No retry on this mutation: a timeout leaves a persisted reconciliation fence.
        pushed = api.kernels_push(str(folder), timeout=str(cap), acc='NvidiaTeslaT4')
        result = {key: getattr(pushed, key) for key in ['ref', 'version_number', 'url', 'error',
                  'invalid_model_sources', 'invalid_dataset_sources'] if hasattr(pushed, key)}
        save('push.json', result)
        assert result.get('version_number') and not result.get('error'), 'Failed push; preserve response and reconcile'
        assert not result.get('invalid_model_sources') and not result.get('invalid_dataset_sources')
        fcntl.flock(admission, fcntl.LOCK_UN)
    pushed = read(directory / 'push.json')
    exact = pushed['ref'].removeprefix('/code/') + '/' + str(pushed['version_number'])
    state('monitoring_exact_version', kernel_version=exact)
    while time.time() < deadline:
        try:
            response = read_call(lambda: api.kernels_status(exact))
        except Exception as error:
            state('observation_retry', kernel_version=exact, error_type=type(error).__name__)
            time.sleep(30)
            continue
        status = str(response.status).split('.')[-1]
        save('status.json', {'status': str(response.status), 'kernel_version': exact})
        state('waiting_for_kernel', kernel_version=exact, status=status)
        if status in TERMINAL:
            read_call(lambda: api.kernels_output(exact, path=str(directory / 'output'),
                file_pattern=r'\.json$|\.jsonl$|\.npz$|\.log$', force=True, quiet=True, page_size=100))
            q = quota(api)
            save('quota.json', q)
            charged = q['used_seconds'] - read(directory / 'push_intent.json')['quota_before']['used_seconds']
            state('complete' if status == 'COMPLETE' else 'failed', kernel_version=exact,
                  status=status, observed_charge_seconds=charged)
            if status != 'COMPLETE':
                raise SystemExit(1)
            return
        time.sleep(45)
    state('original_deadline_reached', kernel_version=exact)
    raise TimeoutError('Inspect/stop the exact job; never infer termination or duplicate it')


if __name__ == '__main__':
    main()
