"""Launch once and monitor an explicitly budgeted span-001 Kaggle job."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT, json_write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['reward','joint'])
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--tag', default='')
    parser.add_argument('--recovery-note', default='')
    args = parser.parse_args()
    root = ROOT / 'runs/span_001'
    plan = json.loads((root / 'span_plan.json').read_text())
    config = json.loads((REPO / 'configs/span_001.json').read_text())
    folder = REPO / f'kaggle/span_001_{args.stage}'
    metadata = json.loads((folder / 'kernel-metadata.json').read_text())
    job = args.stage + ('-' + args.tag if args.tag else '')
    if args.tag and not args.recovery_note:
        raise ValueError('A new recovery job needs its failure analysis in --recovery-note')
    directory = root / 'jobs' / job
    directory.mkdir(parents=True, exist_ok=True)
    aggregate = REPO / 'results/span_001/jobs' / job
    aggregate.mkdir(parents=True, exist_ok=True)
    lock = (root / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cap = config['reward_max_session_seconds' if args.stage == 'reward' else 'training_max_session_seconds']
    deadline = datetime.fromisoformat(plan['deadline']).timestamp()

    def state(stage, **details):
        value = {'stage': stage, 'utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
                 'job': job, **details}
        json_write(root / 'pipeline_state.json', value)
        json_write(aggregate / 'controller_state.json', value)
        print(json.dumps(value), flush=True)

    def action(name, **kwargs):
        command = [sys.executable, 'scripts/kaggle_job.py', name, '--attempt', 'span_001',
                   '--run-directory', str(root), '--folder', str(folder), '--job', job]
        for key, value in kwargs.items():
            command.extend(['--'+key.replace('_','-'), str(value)])
        result = subprocess.run(command, cwd=REPO, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=120)
        print(result.stdout[-5000:], flush=True)
        if result.returncode:
            raise RuntimeError(f'{name} failed with status {result.returncode}')
        for filename in ('push.json','status.json','quota.json'):
            if (directory / filename).exists():
                (aggregate / filename).write_bytes((directory / filename).read_bytes())

    if not (directory / 'push.json').exists():
        action('quota')
        quota = json.loads((directory / 'quota.json').read_text())
        spent = max(0., quota['used_seconds'] - plan['quota_used_at_start'])
        remaining = plan['gpu_session_seconds_cap'] - spent - quota['reserved_seconds']
        assert quota['remaining_seconds'] >= cap, 'Insufficient verified Kaggle quota'
        assert remaining >= cap, 'Job ceiling would exceed the recorded span budget'
        assert time.time() + cap < deadline, 'Job ceiling would exceed the original goal deadline'
        intent = {'utc': datetime.now(timezone.utc).isoformat(), 'kernel': metadata['id'],
                  'max_session_seconds': cap, 'span_budget_remaining_before_reservation': remaining,
                  'quota_before': quota, 'recovery_note': args.recovery_note,
                  'notebook_sha256': hashlib.sha256((folder / metadata['code_file']).read_bytes()).hexdigest()}
        if args.dry_run:
            print(json.dumps({'admissible': True, 'intent': intent}, indent=2))
            return
        if (directory / 'push_intent.json').exists():
            raise RuntimeError('Unresolved launch intent: reconcile the exact remote kernel before any retry')
        json_write(directory / 'push_intent.json', intent)
        json_write(aggregate / 'push_intent.json', intent)
        (directory / 'notebook_snapshot.ipynb').write_bytes((folder / metadata['code_file']).read_bytes())
        state('starting_kernel')
        action('push', accelerator='NvidiaTeslaT4', timeout=cap)
    elif args.dry_run:
        print('A recorded job already exists; resume monitoring that version instead of pushing again.')
        return
    push = json.loads((directory / 'push.json').read_text())
    assert not push.get('error') and not push.get('invalid_model_sources') and not push.get('invalid_dataset_sources'), push
    assert push.get('version_number'), 'No immutable Kaggle version returned'
    remote_kernel = push.get('ref', '').removeprefix('/code/') or metadata['id']
    state('monitoring', kernel=remote_kernel, version=push['version_number'])
    while time.time() < deadline:
        try:
            action('status')
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            state('observation_retry', error=str(error), version=push['version_number'])
            time.sleep(15)
            continue
        status = json.loads((directory / 'status.json').read_text())
        current = status.get('status', '').split('.')[-1]
        state('waiting_for_kernel', status=current, version=push['version_number'])
        if current in ('COMPLETE','ERROR','CANCELLED','CANCELED','DEAD'):
            # Collect partial predictions and diagnostics even after a failed run.
            action('output', pattern=r'\.json$|\.jsonl$|\.npz$|\.log$|\.ipynb$|adapter_model\.safetensors$')
            action('quota')
            if current != 'COMPLETE':
                state('failed', status=current, version=push['version_number'])
                raise RuntimeError(f'GPU job is terminal: {current}; no automatic retry')
            state('complete', version=push['version_number'])
            return
        time.sleep(45)
    state('monitoring_deadline', version=push['version_number'])
    raise TimeoutError('Goal deadline reached; inspect/stop the recorded live job, never duplicate it')


if __name__ == '__main__':
    main()
