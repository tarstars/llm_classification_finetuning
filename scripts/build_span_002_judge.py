"""Freeze extra dev IDs and package the existing judge unchanged for H37 expansion."""
import hashlib
import json
from pathlib import Path
import sys

import nbformat as nbf
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT, json_write

root = ROOT / 'runs/span_002'
config = json.loads((REPO / 'configs/span_002.json').read_text())
result = json.loads((root / 'blend_results.json').read_text())['cohorts']['cached_judge']
assert result['blend_gain_vs_classifier']['gain'] >= config['cheap_useful_gain']
assert result['blend_gain_vs_classifier']['interval_95'][1] >= config['cheap_useful_gain']
baseline = np.load(ROOT / 'runs/attempt_002/jobs/qwen4b1024/output/joint_run/dev_predictions.npz')
cached = np.load(ROOT / 'runs/attempt_003/jobs/development_ctx1024/output/gemma26/judge_predictions.npz')
remaining = baseline['ids'][~np.isin(baseline['ids'], cached['ids'])]
chosen = np.random.default_rng(config['seed']).choice(remaining, config['ensemble_expanded_rows']-len(cached['ids']), replace=False)
sample = {'seed': config['seed'], 'population_rows': len(remaining), 'rows': len(chosen),
          'sampling': 'Uniform additional dev rows, excluding cached judge IDs, fixed before extra judge outputs.',
          'ids': chosen.tolist()}
sample_path = root / 'ensemble_expansion_ids.json'
if sample_path.exists():
    assert json.loads(sample_path.read_text()) == sample
else:
    json_write(sample_path, sample)
selection = json.loads((ROOT / 'runs/attempt_003/judge_selection.json').read_text())
judge_config = selection['config']
assert judge_config['max_length'] == 1024 and not selection['both_orders']
for name, key in [('judge.py', 'judge_source_sha256'), ('judge_runtime.py', 'runtime_source_sha256')]:
    assert hashlib.sha256((REPO / 'src/arena' / name).read_bytes()).hexdigest() == selection[key]
assert hashlib.sha256(json.dumps(judge_config, sort_keys=True).encode()).hexdigest() == selection['config_sha256']
owner, slug, version = selection['runtime_kernel'].split('/')
dest = root / 'notebooks/ensemble-expansion'
dest.mkdir(parents=True, exist_ok=True)
cells = [nbf.v4.new_markdown_cell('# Span 002 — H37 fixed judge coverage expansion\n\nExisting 1024-token single-order Gemma judge, unchanged prompt/config/runtime. No model or blend fitting in this notebook.'),
    nbf.v4.new_code_cell(f'''import json,os,sys,shutil,time
from pathlib import Path
SESSION_STARTED=time.time()
WORK=Path('/kaggle/working')
MODULES=WORK/'bundled_code/arena';MODULES.mkdir(parents=True,exist_ok=True)
(MODULES/'__init__.py').write_text('')
sys.path.insert(0,str(MODULES.parent))
CONFIG={judge_config!r}
SAMPLE={sample!r}
RUNTIME=next(p for p in [Path('/kaggle/input/{slug}'),Path('/kaggle/input/notebooks/{owner}/{slug}')]
    if (p/'runtime/runtime.json').exists())
assert json.loads((RUNTIME/'runtime/runtime.json').read_text())['revision']==CONFIG['llama_revision']
BIN=WORK/'runtime_bin';shutil.copytree(RUNTIME/'runtime/bin',BIN,dirs_exist_ok=True)
EXE=BIN/'llama-server';EXE.chmod(0o755)
RAW=next(p for p in [Path('/kaggle/input/llm-classification-finetuning'),
    Path('/kaggle/input/competitions/llm-classification-finetuning')] if (p/'train.csv').exists())
''')]
for name in ['data', 'judge_runtime', 'judge']:
    source = (REPO / f'src/arena/{name}.py').read_text()
    cells.append(nbf.v4.new_code_cell(f"(MODULES/{(name+'.py')!r}).write_text({source!r})"))
cells.append(nbf.v4.new_code_cell('''from arena.judge import run
summary=run(CONFIG,EXE,RAW,WORK/'gemma26','gemma26','dev',sample=SAMPLE,both_orders=False)
(WORK/'session_summary.json').write_text(json.dumps({'session_seconds':time.time()-SESSION_STARTED,'rows':summary['rows']},indent=2))
summary
'''))
for cell in cells:
    if cell.cell_type == 'code': compile(cell.source, 'span002-judge-cell', 'exec')
nbf.write(nbf.v4.new_notebook(cells=cells, metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}}), dest / 'job.ipynb')
metadata = {'id':'arseniytrushin/llm-preference-span-002-judge-expansion',
    'title':'LLM Preference Span 002 Judge Expansion','code_file':'job.ipynb',
    'language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':False,
    'competition_sources':['llm-classification-finetuning'],'dataset_sources':[],
    'kernel_sources':[selection['runtime_kernel']], 'model_sources':[selection['model_source']]}
json_write(dest / 'kernel-metadata.json', metadata)
old_summary = json.loads((ROOT / 'runs/attempt_003/jobs/development_ctx1024/output/gemma26/judge_summary.json').read_text())
projection = old_summary['seconds']/2/len(cached['ids'])*len(chosen)*1.2 + 600
assert old_summary['both_orders'], 'Historical timing division assumes a paired pass'
assert projection < config['job_seconds_caps']['ensemble_expansion'], 'Expansion exceeds predeclared allocation'
json_write(REPO / 'results/span_002/ensemble_expansion_plan.json', {
    'additional_rows':len(chosen),'total_rows':len(chosen)+len(cached['ids']),
    'sample_sha256':hashlib.sha256(sample_path.read_bytes()).hexdigest(),
    'notebook_sha256':hashlib.sha256((dest/'job.ipynb').read_bytes()).hexdigest(),
    'model_source':selection['model_source'],'config_sha256':selection['config_sha256'],
    'projected_seconds_with_margin':projection,'cap_seconds':config['job_seconds_caps']['ensemble_expansion'],
    'timing_note':'Half the historical two-order measured pass for single-order scoring, plus 20% and 600 seconds; projection only.'})
print(dest)
