"""Resume attempt 003 through offline evaluation, submission, and recorded results.

Expensive jobs are started at most once. Failures stop the pipeline, retain logs,
and never silently replace the requested judge with another trained classifier.
"""
import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT

p = argparse.ArgumentParser()
p.add_argument('--publish-results', action='store_true')
a = p.parse_args()
root = ROOT / 'runs/attempt_003'
root.mkdir(parents=True, exist_ok=True)
config = json.loads((REPO/'configs/attempt_003_judge.json').read_text())
runtime_job = config.get('runtime_job', 'runtime_probe')
runtime_folder = config.get('runtime_folder', 'kaggle/attempt_003_probe')
lock = (root / 'pipeline.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
state_path = root / 'pipeline_state.json'


def state(stage, **details):
    value = {'stage': stage, 'utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(), **details}
    state_path.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(value), flush=True)


def command(*arguments, retries=0):
    for attempt in range(retries + 1):
        result = subprocess.run([sys.executable, *map(str, arguments)], cwd=REPO, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(result.stdout, flush=True)
        if result.returncode == 0: return
        if attempt < retries: time.sleep(15)
    raise RuntimeError(f'Command failed: {arguments}')


def action(name, folder, job, **kwargs):
    args = ['scripts/kaggle_job.py', name, '--attempt', '003', '--folder', folder, '--job', job]
    for key, value in kwargs.items(): args.extend(['--' + key.replace('_', '-'), str(value)])
    command(*args, retries=2 if name in ['status', 'output', 'submissions', 'quota'] else 0)


def finish(folder, job, timeout):
    directory = root / 'jobs' / job
    directory.mkdir(parents=True, exist_ok=True)
    if not (directory / 'push.json').exists():
        intent = directory / 'push_intent.json'
        if intent.exists():
            raise RuntimeError(f'Unresolved push attempt for {job}; inspect Kaggle before launching a duplicate')
        intent.write_text(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), 'folder': folder}) + '\n')
        state('starting_kernel', job=job)
        action('push', folder, job, accelerator='NvidiaTeslaT4', timeout=timeout)
    push = json.loads((directory / 'push.json').read_text())
    if push.get('error') or push.get('invalid_model_sources') or push.get('invalid_dataset_sources'):
        raise RuntimeError(f'Kaggle rejected required sources for {job}: {push}')
    deadline = time.monotonic() + timeout + 1800
    while time.monotonic() < deadline:
        action('status', folder, job)
        status = json.loads((directory / 'status.json').read_text())
        current = status.get('status', '').split('.')[-1]
        state('waiting_for_kernel', job=job, status=current)
        if current in ['COMPLETE', 'ERROR', 'CANCELLED', 'CANCELED', 'DEAD']:
            action('output', folder, job, pattern=r'(?:probe_summary|smoke|runtime|judge_summary|judge_results|prediction_summary)\.json$|judge_predictions\.npz$|\.log$|submission\.csv$')
            if current != 'COMPLETE': raise RuntimeError(f'{job} failed: {status}')
            return
        time.sleep(45)
    raise TimeoutError(f'Monitoring deadline exceeded for {job}; no duplicate job launched')


def refresh():
    command('scripts/collect_attempt_003.py')
    if not a.publish_results: return
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=REPO, text=True).strip()
    assert branch == 'main', 'Automatic publication requires main'
    paths = ['results/attempt_003', 'notebooks/08_prompted_judge.ipynb', 'ATTEMPT_003_RESULTS.md']
    for path in ['kaggle/attempt_003_dev', 'kaggle/attempt_003_dev-ctx1024', 'kaggle/attempt_003_predict']:
        if (REPO / path).exists(): paths.append(path)
    subprocess.run(['git', 'add', '--', *paths], cwd=REPO, check=True)
    if subprocess.run(['git', 'diff', '--cached', '--quiet', '--', *paths], cwd=REPO).returncode:
        subprocess.run(['git', 'commit', '-m', 'Record prompted-judge experiment progress and results', '--', *paths], cwd=REPO, check=True)
        subprocess.run(['git', 'push', 'origin', 'main'], cwd=REPO, check=True)


try:
    state('started')
    finish(runtime_folder, runtime_job, 1800 if runtime_job == 'runtime_binary' else 3600)
    probes = json.loads((root / 'jobs' / runtime_job / 'output/probe_summary.json').read_text())
    available = []
    for probe in probes:
        if 'response' not in probe: continue
        ids = [probe['label_tokens'][label]['tokens'][0] for label in ['A', 'B', 'C']]
        assert len(set(ids)) == 3
        if probe['response']['content'].strip() != 'A':
            raise RuntimeError('Arithmetic smoke test preferred the wrong answer; inspect template/probability handling')
        available.append('gemma31' if '31b' in probe['model_source'] else 'gemma26')
    assert available, 'No model passed the loading/probability probe'
    available=[key for key in available if key in config.get('evaluate_models',available)]
    assert available, 'No configured evaluation model passed setup'
    state('runtime_verified', models=available)
    refresh()
    if not (root/'judge_selection.json').exists():
        command('scripts/build_judge_notebook.py', 'dev', '--models', *available)
        finish('kaggle/attempt_003_dev', 'development', 10800)
        state('analyzing_development')
        command('scripts/analyze_judge.py')
        state('development_estimate_available',max_length=config['max_length'])
        refresh()
        development=json.loads((root/'development.json').read_text())
        if any(row['runtime_feasible'] for row in development['rows']):
            command('scripts/analyze_judge.py','--freeze')
        else:
            # Predeclared, runtime-triggered fallback. Never overwrite the original
            # comparison or silently reuse probabilities from different packing.
            state('shorter_context_for_runtime',max_length=1024)
            for directory in [root,REPO/'results/attempt_003']:
                shutil.copy2(directory/'development.json',directory/'development_2048.json')
            fallback={**config,'max_length':1024}
            path=root/'context1024_config.json'
            path.write_text(json.dumps(fallback,indent=2)+'\n')
            command('scripts/build_judge_notebook.py','dev','--models',*available,
                    '--config',path,'--tag','ctx1024')
            finish('kaggle/attempt_003_dev-ctx1024','development_ctx1024',10800)
            command('scripts/analyze_judge.py','--job','development_ctx1024',
                    '--folder','kaggle/attempt_003_dev-ctx1024','--config',path,'--freeze')
    selection = json.loads((root / 'judge_selection.json').read_text())
    state('judge_frozen', model=selection['model_key'], both_orders=selection['both_orders'],
          development_log_loss=selection['development']['metrics']['log_loss'])
    refresh()
    command('scripts/build_judge_notebook.py', 'predict')
    finish('kaggle/attempt_003_predict', 'prediction', 32400)
    state('submission_file_verified')
    refresh()
    prediction = root / 'jobs/prediction'
    if not (prediction / 'submission.json').exists():
        intent = prediction / 'submission_intent.json'
        if intent.exists(): raise RuntimeError('Unresolved submission request; inspect accepted submissions before retry')
        intent.write_text(json.dumps({'utc': datetime.now(timezone.utc).isoformat()}) + '\n')
        action('submit', 'kaggle/attempt_003_predict', 'prediction',
               message='Attempt 003: frozen prompted Gemma judge, no fine-tuning or classifier blend')
    accepted = json.loads((prediction / 'submission.json').read_text())
    if not accepted.get('ref'): raise RuntimeError(f'Submission not accepted: {accepted}')
    deadline = time.monotonic() + 12 * 3600
    while time.monotonic() < deadline:
        action('submissions', 'kaggle/attempt_003_predict', 'prediction')
        records = json.loads((prediction / 'submissions.json').read_text())
        record = next((s for s in records if str(s.get('ref')) == str(accepted['ref'])), None)
        if record is None: raise RuntimeError('Accepted submission missing from Kaggle response')
        status = record.get('status', '')
        state('waiting_for_submission', submission=record)
        if status.endswith('COMPLETE'):
            assert record.get('public_score') not in [None, ''], 'Official score is missing'
            state('complete', submission=record)
            refresh()
            break
        if any(token in status for token in ['ERROR', 'FAILED', 'CANCEL']):
            raise RuntimeError(f'Hidden-test evaluation failed: {record}')
        time.sleep(45)
    else:
        raise TimeoutError('Hidden-test scoring exceeded monitoring deadline')
except BaseException as error:
    state('failed', error=str(error))
    traceback.print_exc()
    try: refresh()
    except Exception: traceback.print_exc()
    raise
