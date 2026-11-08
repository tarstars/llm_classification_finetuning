"""Freeze the real offline pilot and rehearse its serialized worker before weights."""
from pathlib import Path
import ast
import json
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import nbformat as nbf
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT
from arena.experiment import atomic_json, read_json, sha256
from arena.notebook_worker import worker_cell


def main():
    root, pub = ROOT / 'runs/span_016', REPO / 'results/span_016'
    role = 'reward_pilot'
    assert not (root / 'jobs' / role / 'push_intent.json').exists()
    assert read_json(pub / 'source_audit.json')['source_stage_passed']
    assert read_json(pub / 'pinned_runtime_cpu_validation.json')['passed']
    manifest = read_json(pub / 'asset_manifest.json')
    inputs = read_json(pub / 'matched_input_manifest.json')
    folder = root / 'notebooks' / role
    folder.mkdir(parents=True, exist_ok=True)
    profile = read_json(REPO / 'results/span_013/profile_selection.json')
    atomic_json(pub / 'profile_selection.json', profile)
    visible = sha256(ROOT / 'raw/test.csv')
    files = {'reward_pipeline.py': 'src/arena/span016_pipeline.py',
        'reward_inference.py': 'src/arena/span016_infer.py',
        'reward_mapping.py': 'src/arena/span016_reward.py',
        'frozen_inference.py': 'src/arena/span010_order.py',
        'gemma_order.py': 'src/arena/span013_order.py'}
    setup = f'''from pathlib import Path
import os,sys,time,json,shutil,subprocess,hashlib
STARTED=time.time();WORK=Path('/kaggle/working')
RAW=next(p for p in [Path('/kaggle/input/llm-classification-finetuning'),Path('/kaggle/input/competitions/llm-classification-finetuning')] if (p/'test.csv').exists())
IS_RERUN=os.environ.get('KAGGLE_IS_COMPETITION_RERUN','').strip().lower() in ['true','1'] or hashlib.sha256((RAW/'test.csv').read_bytes()).hexdigest()!={visible!r}
DEADLINE=STARTED+(32400 if IS_RERUN else 3600)-120
assert sys.version_info[:2]==(3,10),'Pinned author image not honored'
GPU_INFO=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True).strip().splitlines()
assert len(GPU_INFO)==2 and all('T4' in name for name in GPU_INFO)
(WORK/'hardware.json').write_text(json.dumps(dict(gpus=GPU_INFO,python=sys.version)))
ASSETS=WORK/'assets';ASSETS.mkdir(exist_ok=True)
for slug in ['lmsys-checkpoints-0-0805','lmsys-checkpoints-3-0805','lmsys-modules-0805','lmsys-packages']:
    source=next(p for p in [Path('/kaggle/input')/slug,Path('/kaggle/input/datasets/tascj0')/slug] if p.exists())
    (ASSETS/slug).symlink_to(source,target_is_directory=True)
shutil.copytree(ASSETS/'lmsys-modules-0805',WORK/'human_pref')
for wheel in ['triton-2.2.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl','xformers-0.0.24042abc8.d20240802-cp310-cp310-linux_x86_64.whl']:
    subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',str(ASSETS/'lmsys-packages'/wheel)],check=True,timeout=120)
REWARD_INPUT=next(p for p in [Path('/kaggle/input/llm-span016-reward-assets'),Path('/kaggle/input/datasets/arseniytrushin/llm-span016-reward-assets')] if p.exists())
EXPECTED={manifest['entries']!r}
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
REWARD=Path('/tmp/span016-reward-assets');REWARD.mkdir()
for size in ['8B','0.6B']:(REWARD/size).mkdir()
verified={{}}
for item in EXPECTED:
    path=REWARD_INPUT/item['filename']
    assert path.stat().st_size==item['bytes'] and digest(path)==item['sha256'],'Asset mismatch'
    verified[item['filename']]=item['sha256']
    if 'model' in item:(REWARD/item['model']/item['model_filename']).symlink_to(path)
MATCHED_INPUT=next(p for p in [Path('/kaggle/input/llm-span016-lengthmatched-inputs'),Path('/kaggle/input/datasets/arseniytrushin/llm-span016-lengthmatched-inputs')] if p.exists())
for item in {inputs['entries']!r}:
    path=MATCHED_INPUT/item['filename']
    assert path.stat().st_size==item['bytes'] and digest(path)==item['sha256'],'Matched input mismatch'
    verified['matched_inputs/'+item['filename']]=item['sha256']
    (REWARD/item['filename']).symlink_to(path)
(WORK/'asset_verification.json').write_text(json.dumps(dict(passed=True,files=verified)))
MODERN=Path('/tmp/span016-qwen-runtime');MODERN.mkdir(exist_ok=True)
subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps','--target',str(MODERN)]+[str(REWARD_INPUT/item['filename']) for item in EXPECTED if item.get('wheel')],check=True,timeout=180)
'''
    for destination, source in files.items():
        setup += f"(WORK/{destination!r}).write_text({(REPO/source).read_text()!r})\n"
    setup += f"(WORK/'profile_selection.json').write_text({json.dumps(profile)!r})\n"
    setup += '''os.chdir(WORK)
command=[sys.executable,'-u',str(WORK/'reward_pipeline.py'),'--raw',str(RAW),'--assets',str(ASSETS),'--reward-assets',str(REWARD),'--modern-runtime',str(MODERN),'--out',str(WORK/'reward_experiment'),'--selection',str(WORK/'profile_selection.json')]+([] if IS_RERUN else ['--qualify','--information-only'])
env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
'''
    nb = nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(
        '# H56: frozen native reward expert\n\nOne fixed Gemma/reward replacement. Later-vote calibration is recomputed from known external training data. Hidden mode performs real inference and never reads hidden votes or public prediction caches.'),
        nbf.v4.new_code_cell(setup), nbf.v4.new_code_cell(worker_cell(role, ''))])
    nb.metadata.update(kernelspec=dict(display_name='Python 3', language='python', name='python3'))
    for cell in nb.cells:
        if cell.cell_type == 'code':
            compile(cell.source, 'serialized-span016', 'exec')
    nbf.write(nb, folder / 'job.ipynb')
    metadata = dict(id='arseniytrushin/llm-preference-span016-reward-pilot',
        title='LLM Preference Span016 Reward Pilot', code_file='job.ipynb', language='python',
        kernel_type='notebook', is_private=True, enable_gpu=True, enable_tpu=False, enable_internet=False,
        dataset_sources=['tascj0/' + s + '/1' for s in ['lmsys-checkpoints-0-0805',
            'lmsys-checkpoints-3-0805', 'lmsys-modules-0805', 'lmsys-packages']] + [manifest['dataset'], inputs['dataset']],
        model_sources=[], kernel_sources=[], competition_sources=['llm-classification-finetuning'],
        docker_image='gcr.io/kaggle-private-byod/python@sha256:e3f9bf91d1b7b36bf77554565d27c5733585fbe5e2b7f0832e068824e7accc3c',
        docker_image_pinning_type='original', machine_shape='NvidiaTeslaT4')
    atomic_json(folder / 'kernel-metadata.json', metadata)
    actual = nbf.read(folder / 'job.ipynb', 4).cells[-1].source
    override = next(node for node in ast.parse(setup).body if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name) and node.target.id == 'item'
        and isinstance(node.iter, ast.List))
    with tempfile.TemporaryDirectory(prefix='span016-override-') as tmp:
        destination = Path(tmp)
        scope = dict(MATCHED_INPUT=root/'matched_assets/upload', REWARD=destination,
            verified={}, digest=sha256)
        exec(compile(ast.Module(body=[override],type_ignores=[]),'actual-input-override','exec'),scope)
        for name in ['source.parquet','labels.npz']:
            assert (destination/name).resolve() == (root/'matched_assets/upload'/name).resolve()
        assert len(scope['verified']) == 2
        assert sha256(destination/'source.parquet') != sha256(root/'cohort/source.parquet')
    checks = []
    for name, program, limit, success in [('success', 'print(123)', 10, True),
        ('failure', 'raise SystemExit(7)', 10, False), ('watchdog', 'import time;time.sleep(10)', .2, False)]:
        with tempfile.TemporaryDirectory(prefix='span016-cell-') as tmp:
            scope = dict(WORK=Path(tmp), STARTED=time.time(), DEADLINE=time.time()+limit,
                command=[sys.executable, '-c', program], env=None)
            failed = False
            try:
                exec(compile(actual, 'actual-serialized-worker', 'exec'), scope)
            except AssertionError:
                failed = True
            result = read_json(Path(tmp) / 'session_summary.json')
            assert failed != success and (result['exit_code'] == 0) == success
            checks.append(dict(name=name, passed=True, summary=result))
    tiny = nbf.v4.new_notebook(cells=[nbf.v4.new_code_cell('from pathlib import Path\nimport sys,time,tempfile\n'
        'tmp=tempfile.TemporaryDirectory();WORK=Path(tmp.name);STARTED=time.time();DEADLINE=STARTED+10\n'
        'command=[sys.executable,"-c","print(123)"];env=None'), nbf.v4.new_code_cell(actual)])
    executed = NotebookClient(tiny, timeout=30, kernel_name='python3').execute()
    assert all(c.execution_count is not None for c in executed.cells)
    assert not any(o.output_type == 'error' for c in executed.cells for o in c.get('outputs', []))
    mode = '\n'.join(line for line in setup.splitlines() if line.startswith(('IS_RERUN=', 'DEADLINE=', 'command=')))
    mode_checks = []
    for flag, observed, expected in [('', visible, False), ('', 'changed', True), ('true', visible, True)]:
        scope = dict(os=SimpleNamespace(environ={'KAGGLE_IS_COMPETITION_RERUN': flag}),
            hashlib=SimpleNamespace(sha256=lambda data:SimpleNamespace(hexdigest=lambda:observed)),
            RAW=ROOT/'raw', WORK=Path('/tmp'), ASSETS=Path('/tmp/assets'), REWARD=Path('/tmp/reward'),
            MODERN=Path('/tmp/modern'), STARTED=0, sys=sys)
        exec(compile(mode, 'actual-serialized-branch', 'exec'), scope)
        assert scope['IS_RERUN'] == expected and ('--qualify' in scope['command']) != expected
        assert ('--information-only' in scope['command']) != expected
        assert scope['DEADLINE'] == (32400 if expected else 3600)-120
        mode_checks.append(dict(hidden=expected, passed=True))
    # Exercise the actual orchestrator through CSV/final bookkeeping, including
    # no competition train.csv in hidden mode and child failure propagation.
    tested = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests',
        '-p', 'test_span016_pipeline.py', '-v'], cwd=REPO,
        env={**__import__('os').environ, 'PYTHONPATH': str(REPO/'src')}, capture_output=True, text=True)
    assert tested.returncode == 0, tested.stdout + tested.stderr
    atomic_json(pub / (role + '_cell_rehearsals.json'), dict(passed=True, checks=checks,
        notebook_startup_passed=True, mode_checks=mode_checks, actual_pipeline_cpu_tests_passed=True,
        actual_matched_input_override_passed=True,
        actual_pipeline_tests=5, new_gpu_executions=0))
    source_names = ['GOAL.md', 'docs/SPAN_016_PLAN.md', 'docs/SPAN_016_PILOT_PROTOCOL.md',
        'docs/SPAN_016_PUBLIC_STAGING.md',
        'docs/SPAN_016_COHORT_REFINEMENT.md',
        'scripts/build_span_016_pilot.py', 'src/arena/notebook_worker.py', *files.values(),
        'tests/test_span016_pipeline.py', 'tests/test_span016_reward.py',
        'results/span_016/asset_manifest.json', 'results/span_016/matched_cohort_manifest.json',
        'results/span_016/matched_input_manifest.json']
    atomic_json(pub / (role + '_bundle.json'), dict(role=role,
        source_sha256={n: sha256(REPO/n) for n in source_names},
        notebook_sha256=sha256(folder/'job.ipynb'), metadata_sha256=sha256(folder/'kernel-metadata.json'),
        public_cap_seconds=3600, hidden_cap_seconds=32400, internet_enabled=False,
        public_stage='information_only', official_submission_eligible=False,
        public_profile_selection_sha256=sha256(pub/'profile_selection.json'),
        asset_manifest_sha256=sha256(pub/'asset_manifest.json'),
        matched_input_manifest_sha256=sha256(pub/'matched_input_manifest.json')))
    print(folder)


if __name__ == '__main__':
    main()
