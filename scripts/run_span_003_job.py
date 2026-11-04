"""Launch one v3 job with persistent admission, immutable version and bounded accounting."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job', choices=['classifier_smoke','classifier_predict','native_profile','native_profile_retry','cascade_followup'])
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = ROOT / 'runs/span_003'
    folder = root / 'notebooks' / args.job
    config = json.loads((REPO / 'configs/span_003.json').read_text())
    plan = json.loads((root / 'span_plan.json').read_text())
    metadata = json.loads((folder / 'kernel-metadata.json').read_text())
    directory = root / 'jobs' / args.job
    aggregate = REPO / 'results/span_003/jobs' / args.job
    directory.mkdir(parents=True, exist_ok=True)
    aggregate.mkdir(parents=True, exist_ok=True)
    lock = (directory / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cap = config['job_seconds_caps'][args.job.replace('-', '_')]
    deadline = datetime.fromisoformat(plan['deadline']).timestamp()

    def state(stage, **details):
        value = {'stage':stage,'utc':datetime.now(timezone.utc).isoformat(), 'pid':__import__('os').getpid(),
                 'job':args.job, **details}
        json_write(directory / 'controller_state.json', value)
        json_write(aggregate / 'controller_state.json', value)
        print(json.dumps(value), flush=True)

    def action(name, **kwargs):
        command = [sys.executable, 'scripts/kaggle_job.py', name, '--run-directory',str(root),
                   '--attempt','span_003','--folder',str(folder),'--job',args.job]
        for key, value in kwargs.items():
            command += ['--'+key.replace('_','-'), str(value)]
        result = subprocess.run(command, cwd=REPO, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=180 if name != 'output' else 900)
        print(result.stdout[-3500:], flush=True)
        if result.returncode:
            raise RuntimeError(f'{name} returned {result.returncode}')
        for filename in ('push.json','status.json','quota.json'):
            if (directory / filename).exists():
                (aggregate / filename).write_bytes((directory / filename).read_bytes())

    if not (directory / 'push.json').exists():
        admission = (root / 'admission.lock').open('a')
        fcntl.flock(admission, fcntl.LOCK_EX | fcntl.LOCK_NB)
        action('quota')
        quota = json.loads((directory / 'quota.json').read_text())
        spent = max(0., quota['used_seconds'] - plan['quota_used_at_start'])
        remaining = plan['gpu_session_seconds_cap'] - spent - quota['reserved_seconds']
        assert quota['remaining_seconds'] >= cap and remaining >= cap, 'Insufficient span/quota allowance'
        assert time.time()+cap+300 < deadline, 'Ceiling exceeds original deadline'
        intent = {'utc':datetime.now(timezone.utc).isoformat(),'kernel':metadata['id'],
            'max_session_seconds':cap,'quota_before':quota,'span_remaining_before_reservation':remaining,
            'notebook_sha256':hashlib.sha256((folder/metadata['code_file']).read_bytes()).hexdigest()}
        if (directory/'draft_push.json').exists():
            verification=json.loads((directory/'draft_verification.json').read_text())
            assert verification['source_matches'] and verification['notebook_sha256']==intent['notebook_sha256']
            intent['expected_execution_version']=verification['next_execution_version']
            intent['prior_quick_save_version']=verification['draft_version']
        if args.dry_run:
            print(json.dumps({'admissible':True,'intent':intent},indent=2))
            return
        assert not (directory / 'push_intent.json').exists(), 'Unresolved push: reconcile exact remote version before retry'
        json_write(directory / 'push_intent.json', intent)
        json_write(aggregate / 'push_intent.json', intent)
        state('starting_kernel')
        action('push', accelerator='NvidiaTeslaT4', timeout=cap)
        fcntl.flock(admission, fcntl.LOCK_UN)
    push = json.loads((directory / 'push.json').read_text())
    assert push.get('version_number') and not push.get('error'), push
    assert not push.get('invalid_model_sources') and not push.get('invalid_dataset_sources'), push
    state('monitoring', version=push['version_number'], kernel=push.get('ref'))
    while time.time() < deadline:
        try:
            action('status')
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            state('observation_retry', error=str(error), version=push['version_number'])
            time.sleep(30)
            continue
        status = json.loads((directory / 'status.json').read_text())['status'].split('.')[-1]
        state('waiting_for_kernel', status=status, version=push['version_number'])
        if status in ('COMPLETE','ERROR','CANCELLED','CANCELED','DEAD'):
            action('output', pattern=r'\.json$|\.jsonl$|\.npz$|\.log$|\.csv$|adapter_model\.safetensors$|exposure_rank.*')
            action('quota')
            state('complete' if status == 'COMPLETE' else 'failed', status=status, version=push['version_number'])
            if status != 'COMPLETE':
                raise RuntimeError('Terminal failed job; retain partial evidence and diagnose before any new version')
            return
        time.sleep(45)
    state('monitoring_deadline', version=push['version_number'])
    raise TimeoutError('Deadline reached; inspect/stop this exact live job, never duplicate it')


if __name__ == '__main__':
    main()
