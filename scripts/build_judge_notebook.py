"""Create offline development/profile/submission notebooks for the pure prompted judge."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import nbformat as nbf
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT

p = argparse.ArgumentParser()
p.add_argument('mode', choices=['profile', 'dev', 'predict'])
p.add_argument('--models', nargs='+', default=['gemma31', 'gemma26'])
p.add_argument('--config', type=Path, default=REPO / 'configs/attempt_003_judge.json')
p.add_argument('--selection', type=Path, default=ROOT / 'runs/attempt_003/judge_selection.json')
p.add_argument('--tag', default='')
p.add_argument('--single-order', action='store_true')
a = p.parse_args()
config = json.loads(a.config.read_text())
root = ROOT / 'runs/attempt_003'
root.mkdir(parents=True, exist_ok=True)
ids = np.load(ROOT / 'runs/attempt_002/jobs/qwen4b1024/output/joint_run/dev_predictions.npz')['ids']
rng = np.random.default_rng(config['seed'])
chosen = rng.choice(len(ids), config['sample_rows'], replace=False)
sample = {'seed': config['seed'], 'population_rows': len(ids), 'rows': len(chosen),
          'sampling': 'Uniform development rows selected before judge predictions; order is seeded random',
          'ids': ids[chosen].tolist()}
sample_path = root / 'dev_sample.json'
if sample_path.exists(): assert json.loads(sample_path.read_text()) == sample
else: sample_path.write_text(json.dumps(sample, indent=2) + '\n')
push = json.loads((root / 'jobs/runtime_probe/push.json').read_text())
runtime_ref = push['ref'].removeprefix('/code/')
owner, runtime_slug = runtime_ref.split('/')
selection = json.loads(a.selection.read_text()) if a.mode == 'predict' else None
models = [selection['model_key']] if selection else a.models
both_orders = selection['both_orders'] if selection else not a.single_order
if selection:
    assert hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest() == selection['config_sha256']
    assert hashlib.sha256((REPO/'src/arena/judge.py').read_bytes()).hexdigest() == selection['judge_source_sha256']
    assert hashlib.sha256((REPO/'src/arena/judge_runtime.py').read_bytes()).hexdigest() == selection['runtime_source_sha256']
suffix = ('-' + a.tag) if a.tag else ''
dest = REPO / f'kaggle/attempt_003_{a.mode}{suffix}'
dest.mkdir(parents=True, exist_ok=True)
cells = [nbf.v4.new_markdown_cell(f'''# Attempt 003: prompted judge — {a.mode}

Frozen public model weights, a fixed neutral rubric, and three scored labels.
No fine-tuning or classifier ensemble. Internet is disabled. Both answer orders:
{both_orders}. Development IDs and runtime/model versions are explicitly pinned.
'''), nbf.v4.new_code_cell(f'''import json,os,sys,shutil,hashlib
from pathlib import Path
WORK=Path('/kaggle/working')
MODULES=WORK/'bundled_code/arena';MODULES.mkdir(parents=True,exist_ok=True)
(MODULES/'__init__.py').write_text('')
sys.path.insert(0,str(MODULES.parent))
CONFIG={config!r}
SAMPLE={sample!r}
SELECTION={selection!r}
RUNTIME=next(p for p in [Path('/kaggle/input/{runtime_slug}'),Path('/kaggle/input/notebooks/{owner}/{runtime_slug}')]
    if (p/'runtime/runtime.json').exists())
assert json.loads((RUNTIME/'runtime/runtime.json').read_text())['revision']==CONFIG['llama_revision']
BIN=WORK/'runtime_bin';shutil.copytree(RUNTIME/'runtime/bin',BIN,dirs_exist_ok=True)
EXE=BIN/'llama-server';EXE.chmod(0o755)
RAW=next(p for p in [Path('/kaggle/input/llm-classification-finetuning'),
    Path('/kaggle/input/competitions/llm-classification-finetuning')] if (p/'test.csv').exists())
''')]
for name in ['data', 'judge_runtime', 'judge']:
    source = (REPO / f'src/arena/{name}.py').read_text()
    cells.append(nbf.v4.new_code_cell(f"(MODULES/{(name+'.py')!r}).write_text({source!r})"))
cells.append(nbf.v4.new_code_cell(f'''from arena.judge import run
summaries=[]
for model_key in {models!r}:
    summary=run(CONFIG,EXE,RAW,WORK/model_key,model_key,{a.mode!r},
        sample=SAMPLE,selection=SELECTION,both_orders={both_orders!r})
    summaries.append(summary)
if {a.mode!r}=='predict':
    shutil.copy2(WORK/{models[0]!r}/'submission.csv',WORK/'submission.csv')
    (WORK/'prediction_summary.json').write_text(json.dumps(summaries[0],indent=2))
(WORK/'judge_results.json').write_text(json.dumps(summaries,indent=2))
summaries
'''))
for cell in cells:
    if cell.cell_type == 'code': compile(cell.source, 'judge-notebook-cell', 'exec')
notebook = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}})
nbf.write(notebook, dest / 'prompted_judge.ipynb')
metadata = {'id': f'{owner}/llm-preference-judge-{a.mode}-003{suffix}',
            'title': f'LLM Preference Judge {a.mode.title()} 003' + ((' ' + a.tag) if a.tag else ''),
            'code_file': 'prompted_judge.ipynb', 'language': 'python', 'kernel_type': 'notebook',
            'is_private': True, 'enable_gpu': True, 'enable_internet': False,
            'competition_sources': ['llm-classification-finetuning'], 'dataset_sources': [],
            'kernel_sources': [f"{runtime_ref}/{push['version_number']}"],
            'model_sources': [config['models'][key] for key in models]}
(dest / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
print(dest)
