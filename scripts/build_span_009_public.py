"""Build the frozen offline public ensemble with bounded qualification/rerun."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import tempfile
import time
from types import SimpleNamespace

import nbformat as nbf
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT
from arena.experiment import atomic_json, sha256
from arena.notebook_worker import worker_cell
from arena.span009_public import profile_indices, TEXT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', choices=['public_ensemble', 'public_ensemble_runtime_fix', 'public_ensemble_pipeline_fix'], default='public_ensemble')
    args = parser.parse_args()
    role = args.role
    cap = {'public_ensemble': 2400, 'public_ensemble_runtime_fix': 2200,
           'public_ensemble_pipeline_fix': 1000}[role]
    root = ROOT / 'runs/span_009'
    folder = root / 'notebooks' / role
    assert not (root / 'jobs' / role / 'push_intent.json').exists(), 'Never rebuild an accepted source'
    pub = REPO / 'results/span_009'
    folder.mkdir(parents=True, exist_ok=True)
    visible = sha256(ROOT / 'raw/test.csv')
    driver = REPO / 'src/arena/span009_public.py'
    setup = f'''from pathlib import Path
import os,sys,time,json,shutil,subprocess,hashlib
STARTED=time.time();WORK=Path('/kaggle/working')
RAW=next(p for p in [Path('/kaggle/input/llm-classification-finetuning'),Path('/kaggle/input/competitions/llm-classification-finetuning')] if (p/'test.csv').exists())
IS_RERUN=os.environ.get('KAGGLE_IS_COMPETITION_RERUN','').strip().lower() in ['true','1'] or hashlib.sha256((RAW/'test.csv').read_bytes()).hexdigest()!={visible!r}
DEADLINE=STARTED+(32400 if IS_RERUN else {cap})-120
print('Runtime',sys.version,'hidden',IS_RERUN,flush=True)
assert sys.version_info[:2]==(3,10),'Pinned author image was not honored; preserve failure and inspect runtime'
ASSETS=WORK/'assets';ASSETS.mkdir(exist_ok=True)
for slug in ['lmsys-checkpoints-0-0805','lmsys-checkpoints-3-0805','lmsys-modules-0805','lmsys-packages']:
    source=next(p for p in [Path('/kaggle/input')/slug,Path('/kaggle/input/datasets/tascj0')/slug] if p.exists())
    (ASSETS/slug).symlink_to(source,target_is_directory=True)
shutil.copytree(ASSETS/'lmsys-modules-0805',WORK/'human_pref')
for wheel in ['triton-2.2.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl','xformers-0.0.24042abc8.d20240802-cp310-cp310-linux_x86_64.whl']:
    subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',str(ASSETS/'lmsys-packages'/wheel)],check=True,timeout=120)
(WORK/'public_ensemble.py').write_text({driver.read_text()!r})
os.chdir(WORK)
command=[sys.executable,'-u',str(WORK/'public_ensemble.py'),'--raw',str(RAW),'--assets',str(ASSETS),'--out',str(WORK/'public_ensemble')]+([] if IS_RERUN else ['--qualify'])
env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
'''
    nb = nbf.v4.new_notebook(cells=[
        nbf.v4.new_markdown_cell('# V9 — frozen public supervised ensemble\n\nAdapted from tascj0/lmsys-0805 (Apache-2.0 source). Gemma2-9B original order and Llama3-8B swapped order, equal weights. Local profile rows may overlap model training; no independent local quality claim. Both public qualification and hidden rerun use the same inference function.'),
        nbf.v4.new_code_cell(setup),
        nbf.v4.new_code_cell(worker_cell(role, ''))])
    nb.metadata.update(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                       language_info=dict(name='python', version='3.10.14'))
    for cell in nb.cells:
        if cell.cell_type == 'code':
            compile(cell.source, 'span009-serialized-cell', 'exec')
    nbf.write(nb, folder / 'job.ipynb')
    metadata = dict(id='arseniytrushin/llm-preference-span009-public-ensemble',
        title='LLM Preference Span009 Public Ensemble', code_file='job.ipynb', language='python',
        kernel_type='notebook', is_private=True, enable_gpu=True, enable_tpu=False, enable_internet=False,
        dataset_sources=['tascj0/' + s + '/1' for s in ['lmsys-checkpoints-0-0805',
            'lmsys-checkpoints-3-0805', 'lmsys-modules-0805', 'lmsys-packages']],
        model_sources=[], kernel_sources=[], competition_sources=['llm-classification-finetuning'],
        docker_image='gcr.io/kaggle-private-byod/python@sha256:e3f9bf91d1b7b36bf77554565d27c5733585fbe5e2b7f0832e068824e7accc3c',
        docker_image_pinning_type='original', machine_shape='NvidiaTeslaT4')
    atomic_json(folder / 'kernel-metadata.json', metadata)
    actual = nbf.read(folder / 'job.ipynb', 4).cells[-1].source
    mode_source = '\n'.join(line for line in setup.splitlines() if line.startswith(('IS_RERUN=', 'DEADLINE=')))
    mode_checks = []
    for flag, observed_hash, expected in [('', visible, False), ('', 'changed', True), ('true', visible, True)]:
        scope = dict(os=SimpleNamespace(environ={'KAGGLE_IS_COMPETITION_RERUN': flag}),
            hashlib=SimpleNamespace(sha256=lambda data: SimpleNamespace(hexdigest=lambda: observed_hash)),
            RAW=ROOT/'raw', STARTED=0)
        exec(compile(mode_source, 'serialized-mode-switch', 'exec'), scope)
        assert scope['IS_RERUN'] == expected
        assert scope['DEADLINE'] == (32400 if expected else cap)-120
        mode_checks.append(dict(flag=flag, expected_hidden=expected, passed=True))
    rehearsals = []
    for name, program, limit, success in [('success', 'print("tiny worker complete")', 10, True),
            ('failure', 'raise SystemExit(7)', 10, False), ('watchdog', 'import time;time.sleep(10)', .2, False)]:
        with tempfile.TemporaryDirectory(prefix='span009-cell-') as tmp:
            scope = dict(WORK=Path(tmp), STARTED=time.time(), DEADLINE=time.time()+limit,
                         command=[sys.executable, '-c', program], env=None)
            failed = False
            try:
                exec(compile(actual, 'final-serialized-notebook-cell', 'exec'), scope)
            except AssertionError:
                failed = True
            record = json.loads((Path(tmp) / 'session_summary.json').read_text())
            assert failed != success and (record['exit_code'] == 0) == success
            rehearsals.append(dict(name=name, passed=True, summary=record))
    train = pd.read_csv(ROOT / 'raw/train.csv', usecols=['id', *TEXT])
    uniform, long = profile_indices(train)
    atomic_json(pub / 'profile_selection.json', dict(seed=42, uses_targets=False,
        uniform_ids=train.iloc[uniform].id.tolist(), long_ids=train.iloc[long].id.tolist(),
        train_csv_sha256=sha256(ROOT / 'raw/train.csv'), local_quality_is_independent=False))
    files = ['src/arena/span009_public.py', 'src/arena/notebook_worker.py',
             'src/arena/kaggle_control.py', 'scripts/build_span_009_public.py',
             'scripts/run_span_009_public.py', 'docs/SPAN_009_PLAN.md',
             'docs/SPAN_009_PUBLIC_MODEL_PROTOCOL.md', 'GOAL.md']
    # Exercise notebook-engine startup and the actual serialized final cell together.
    from nbclient import NotebookClient
    tiny = nbf.v4.new_notebook(metadata=nb.metadata, cells=[nbf.v4.new_code_cell(
        'from pathlib import Path\nimport sys,time,tempfile\n'
        'rehearsal_directory=tempfile.TemporaryDirectory(prefix="span009-startup-")\n'
        'WORK=Path(rehearsal_directory.name);STARTED=time.time();DEADLINE=STARTED+10\n'
        'command=[sys.executable,"-c","print(123)"];env=None'),
        nbf.v4.new_code_cell(actual)])
    executed = NotebookClient(tiny, timeout=30, kernel_name=nb.metadata.kernelspec.name).execute()
    assert all(c.execution_count is not None for c in executed.cells)
    assert not any(o.output_type == 'error' for c in executed.cells for o in c.get('outputs', []))
    atomic_json(pub / (role + '_bundle.json'), dict(role=role,
        source_sha256={f: sha256(REPO / f) for f in files},
        notebook_sha256=sha256(folder / 'job.ipynb'), metadata_sha256=sha256(folder / 'kernel-metadata.json'),
        public_cap_seconds=cap, hidden_cap_seconds=32400, internet_enabled=False,
        visible_test_sha256=visible, author_repository_commit='83cd93d50b9283c18711e8c63e4e1c6399c7b9ce',
        author_notebook_sha256=sha256(root/'research/tascj0__lmsys-0805/lmsys-0805.ipynb'),
        public_profile_selection_sha256=sha256(pub/'profile_selection.json')))
    atomic_json(pub / (role + '_cell_rehearsals.json'), dict(passed=True, rehearsals=rehearsals, mode_checks=mode_checks,
        notebook_engine_startup_passed=True, kernelspec=nb.metadata.kernelspec.name,
        last_cell_sha256=hashlib.sha256(actual.encode()).hexdigest(), new_gpu_executions=0))
    print(folder)


if __name__ == '__main__':
    main()
