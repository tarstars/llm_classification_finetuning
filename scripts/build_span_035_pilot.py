"""Freeze one pilot notebook and exercise its actual staging/pipeline/final cells."""
from pathlib import Path
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

import nbformat as nbf
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT
from arena.experiment import read_json, atomic_json, sha256
from arena.notebook_worker import worker_cell
from build_span_030_candidate import FAKE_ENGINE


def main():
    run, pub, role = ROOT / 'runs/span_035', REPO / 'results/span_035', 'recent_pilot'
    assert read_json(pub / 'pilot_preparation.json')['prepared']
    assert not (run / 'jobs' / role / 'push_intent.json').exists()
    folder = run / 'notebooks' / role; folder.mkdir(parents=True, exist_ok=True)
    old, new = (REPO / 'src/arena/span030_engine.py').read_text(), (REPO / 'src/arena/span035_engine.py').read_text()
    def forward(text): return text[text.index('with CpuPrefetch(loader)'):text.index('seconds = time.monotonic() - begin')]
    assert forward(old) == forward(new), 'The tested native forward may not change'
    proof = read_json(REPO / 'results/span_028/cpu_audit.json')
    native_hashes = {'/'.join(Path(n).parts[-2:]): h for n, h in proof['native_source_sha256'].items()}
    setup = f'''from pathlib import Path
import os,sys,time,json,shutil,subprocess,hashlib,base64
STARTED=time.time();WORK=Path('/kaggle/working');DEADLINE=STARTED+5400-120
FLAG=os.environ.get('KAGGLE_IS_COMPETITION_RERUN','').strip().lower()
assert FLAG in ['', 'false', '0'],'Information pilot only; no competition deployment'
# MODE_END
assert sys.version_info[:2]==(3,10),'Pinned author image not honored'
RAW=next(p for p in [Path('/kaggle/input/llm-classification-finetuning'),Path('/kaggle/input/competitions/llm-classification-finetuning')] if (p/'train.csv').exists())
GPU_INFO=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True).strip().splitlines()
assert len(GPU_INFO)==2 and all('T4' in x for x in GPU_INFO)
(WORK/'hardware.json').write_text(json.dumps(dict(gpus=GPU_INFO,python=sys.version,mode='information_pilot')))
ASSETS=WORK/'assets';ASSETS.mkdir(exist_ok=True)
for slug in ['lmsys-checkpoints-0-0805','lmsys-checkpoints-3-0805','lmsys-modules-0805','lmsys-packages']:
 source=next(p for p in [Path('/kaggle/input')/slug,Path('/kaggle/input/datasets/tascj0')/slug] if p.exists())
 (ASSETS/slug).symlink_to(source,target_is_directory=True)
for name,digest in {native_hashes!r}.items():assert hashlib.sha256((ASSETS/'lmsys-modules-0805'/name).read_bytes()).hexdigest()==digest
shutil.copytree(ASSETS/'lmsys-modules-0805',WORK/'human_pref')
for wheel in ['triton-2.2.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl','xformers-0.0.24042abc8.d20240802-cp310-cp310-linux_x86_64.whl']:
 subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',str(ASSETS/'lmsys-packages'/wheel)],check=True,timeout=120)
os.chdir(WORK)
'''
    files = {'recent_engine.py': 'src/arena/span035_engine.py', 'recent_checks.py': 'src/arena/span035_checks.py',
        'recent_policy.py': 'src/arena/span035_recent.py', 'recent_pilot.py': 'src/arena/span035_pilot.py',
        'cpu_prefetch.py': 'src/arena/cpu_prefetch.py'}
    stage = ''.join(f'(WORK/{name!r}).write_text({(REPO/path).read_text()!r})\n' for name, path in files.items())
    stage += f"SPEC=WORK/'pilot_spec.json';SPEC.write_text({(run/'pilot_spec.json').read_text()!r})\n(WORK/'controls').mkdir(exist_ok=True)\n"
    control = run / 'controls/m0.npz'
    stage += f"(WORK/'controls/m0.npz').write_bytes(base64.b64decode({base64.b64encode(control.read_bytes()).decode()!r}))\n"
    stage += '''command=[sys.executable,'-u',str(WORK/'recent_pilot.py'),'--raw',str(RAW),'--assets',str(ASSETS),'--out',str(WORK/'recent_audit'),'--engine',str(WORK/'recent_engine.py'),'--spec',str(SPEC)]
env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
'''
    final = worker_cell(role, '')
    nb = nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell('''# H27 complete recent-turn information pilot

One fixed input policy on312affected rows. Native control remains unchanged;
recent inputs match CPU proofs. No preference targets enter inference and no
submission is produced. Source votes overlap checkpoint training, so this is
an exploratory information screen. The two-pass candidate requires a separate
actual qualification and frozen official trial after its information gate.'''),
        nbf.v4.new_code_cell(setup), nbf.v4.new_code_cell(stage), nbf.v4.new_code_cell(final)])
    nb.metadata.update(kernelspec=dict(name='python3', display_name='Python 3', language='python'), language_info=dict(name='python', version='3.10.13'))
    for c in nb.cells:
        if c.cell_type == 'code': compile(c.source, 'span035-serialized-cell', 'exec')
    nbf.write(nb, folder / 'job.ipynb')
    actual = nbf.read(folder / 'job.ipynb', 4); checks = []
    for flag in ['', 'false', '0']:
        with patch.dict(os.environ, {'KAGGLE_IS_COMPETITION_RERUN': flag}): exec(actual.cells[1].source.split('# MODE_END')[0], {})
    for flag in ['true', '1', 'unexpected']:
        with patch.dict(os.environ, {'KAGGLE_IS_COMPETITION_RERUN': flag}):
            try: exec(actual.cells[1].source.split('# MODE_END')[0], {})
            except AssertionError: pass
            else: raise AssertionError('Information pilot accepted a deployment/unknown mode')
    import pandas as pd
    for stopped in [False, True]:
        with tempfile.TemporaryDirectory(prefix='span035-staging-') as tmp:
            work = Path(tmp); raw = work / 'raw'; raw.mkdir()
            scope = dict(WORK=work, RAW=raw, ASSETS=work / 'assets', sys=sys, os=os, json=json, base64=base64)
            exec(compile(actual.cells[2].source, 'actual-staging', 'exec'), scope)
            subprocess.run([sys.executable, '-c', 'import recent_pilot,recent_engine,recent_checks,recent_policy,cpu_prefetch'], cwd=work, check=True, timeout=30)
            spec = read_json(scope['SPEC']); ids = sorted(set(sum(spec['frame_ids'].values(), [])))
            pd.DataFrame(dict(id=ids, prompt=['["p"]'] * len(ids), response_a=['["a"]'] * len(ids),
                response_b=['["b"]'] * len(ids), winner_model_a='must_not_read')).to_csv(raw / 'train.csv', index=False)
            spec['train_csv_sha256'] = sha256(raw / 'train.csv'); scope['SPEC'].write_text(json.dumps(spec))
            fake = work / 'cpu_engine.py'
            fake.write_text('import os\n' + FAKE_ENGINE + "\nif os.environ.get('FAKE_GUARD_STOP')=='1':(out/'engineering_stop.json').write_text('{}')\n")
            scope['command'][scope['command'].index('--engine') + 1] = str(fake)
            scope['env']['FAKE_GUARD_STOP'] = '1' if stopped else '0'
            scope.update(STARTED=time.time(), DEADLINE=time.time() + 30)
            exec(compile(actual.cells[3].source, 'actual-final', 'exec'), scope)
            done = read_json(work / 'recent_audit/pilot_complete.json')
            assert done['complete'] != stopped and done['stopped_for_guard'] == stopped
            assert not (work / 'submission.csv').exists()
            checks.append(dict(name='actual_staging_pipeline_final', stopped_for_guard=stopped, passed=True,
                fake_model_forward=True, session=read_json(work / 'session_summary.json')))
    for name, program, limit in [('failure', 'raise SystemExit(7)', 10), ('watchdog', 'import time;time.sleep(10)', .2)]:
        with tempfile.TemporaryDirectory(prefix='span035-worker-') as tmp:
            scope = dict(WORK=Path(tmp), STARTED=time.time(), DEADLINE=time.time() + limit, command=[sys.executable, '-c', program], env=None)
            try: exec(actual.cells[3].source, scope)
            except AssertionError: pass
            else: raise AssertionError('Failed subprocess accepted')
            record = read_json(Path(tmp) / 'session_summary.json'); assert record['exit_code'] != 0
            checks.append(dict(name=name, passed=True, session=record))
    tiny = nbf.v4.new_notebook(metadata=nb.metadata, cells=[nbf.v4.new_code_cell(
        'from pathlib import Path\nimport sys,time,tempfile\ntmp=tempfile.TemporaryDirectory(prefix="span035-jupyter-")\nWORK=Path(tmp.name);STARTED=time.time();DEADLINE=STARTED+10\ncommand=[sys.executable,"-c","print(123)"];env=None'), nbf.v4.new_code_cell(actual.cells[3].source)])
    executed = NotebookClient(tiny, timeout=30, kernel_name='python3').execute()
    assert all(c.execution_count is not None and not any(o.output_type == 'error' for o in c.get('outputs', [])) for c in executed.cells)
    metadata = read_json(ROOT / 'runs/span_030/notebooks/weight_qualification/kernel-metadata.json')
    metadata.update(id='arseniytrushin/llm-preference-span035-recent-pilot', title='LLM Preference Span035 Recent Pilot')
    atomic_json(folder / 'kernel-metadata.json', metadata)
    size = len(json.dumps(dict(source=(folder / 'job.ipynb').read_text())).encode())
    assert size < 900000
    names = list(files.values()) + ['GOAL.md', 'docs/SPAN_035_PLAN.md', 'docs/SPAN_035_RUNTIME_REVISION.md',
        'configs/span_035.json', 'results/span_035/span_plan.json', 'results/span_035/binding_revision.json',
        'results/span_035/pre_gpu_design_revision.json', 'results/span_035/preparation.json', 'results/span_035/pilot_preparation.json',
        'scripts/prepare_span_035_recent.py', 'scripts/prepare_span_035_pilot.py', 'scripts/build_span_035_pilot.py',
        'scripts/run_span_035_pilot.py', 'scripts/verify_span_035_pilot.py', 'scripts/audit_span_035_pilot.py',
        'src/arena/notebook_worker.py', 'src/arena/kaggle_control.py', 'src/arena/span031_inventory.py',
        'src/arena/span004_support.py', 'src/arena/span_heads.py', 'tests/test_span035_recent.py']
    atomic_json(pub / (role + '_bundle.json'), dict(role=role, source_sha256={n: sha256(REPO / n) for n in names},
        notebook_sha256=sha256(folder / 'job.ipynb'), metadata_sha256=sha256(folder / 'kernel-metadata.json'),
        pilot_spec_sha256=sha256(run / 'pilot_spec.json'), public_cap_seconds=5400,
        information_only=True, official_submission_eligible=False, sdk_payload_bytes=size,
        native_forward_source_unchanged=True))
    atomic_json(pub / (role + '_cell_rehearsals.json'), dict(passed=True, rehearsals=checks,
        actual_serialized_staging_pipeline_final_passed=True, standalone_imports_passed=True,
        mode_guards_passed=True, notebook_engine_startup_passed=True, forward_source_unchanged=True, new_gpu_executions=0))
    print(dict(folder=str(folder), sdk_payload_bytes=size, passed=True))


if __name__ == '__main__': main()
