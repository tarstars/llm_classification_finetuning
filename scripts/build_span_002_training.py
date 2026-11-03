"""Bundle v2 accepted-update probes and stateful training into offline notebooks."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
import zlib

import nbformat as nbf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT, json_write

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('job', choices=['instrumentation','exposure'])
args = parser.parse_args()
root = ROOT / 'runs/span_002'
selection = json.loads((ROOT / 'runs/attempt_002/model_selection.json').read_text())
owner, slug = selection['kernel'].split('/')
dest = root / 'notebooks' / args.job
dest.mkdir(parents=True, exist_ok=True)
description = ('E01 actual CUDA overflow and resume probe. Tiny model; no competition training.'
               if args.job == 'instrumentation' else
               'H03 stateful continuation from checkpoint 800 toward one epoch. Fixed final endpoint, direct accepted-exposure accounting.')
cells = [nbf.v4.new_markdown_cell('# Span 002 — ' + description),
    nbf.v4.new_code_cell(f'''import os,sys,subprocess,time,json,hashlib
from pathlib import Path
SESSION_STARTED=time.time()
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
WORK=Path('/kaggle/working')
TRAIN=next(p for p in [Path('/kaggle/input/{slug}'),Path('/kaggle/input/notebooks/{owner}/{slug}')]
    if (p/'runtime_requirements.json').exists())
runtime=json.loads((TRAIN/'runtime_requirements.json').read_text())
subprocess.run([sys.executable,'-m','pip','install','-q','--no-index','--no-deps',
    '--find-links',str(TRAIN/'runtime_wheels'),*runtime],check=True)
MODULES=WORK/'bundled_code'
(MODULES/'arena').mkdir(parents=True,exist_ok=True)
(MODULES/'arena/__init__.py').write_text('')
''')]
sources = {}
for name in ['data','joint_data','joint_train','training_exposure',
             'exposure_probe' if args.job == 'instrumentation' else 'exposure_train']:
    path = REPO / f'src/arena/{name}.py'
    sources[str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
    cells.append(nbf.v4.new_code_cell(f"(MODULES/'arena/{name}.py').write_text({path.read_text()!r})"))
if args.job == 'exposure':
    assert json.loads((REPO/'results/span_002/instrumentation_results.json').read_text())['passed']
    span = json.loads((REPO/'configs/span_002.json').read_text())
    split = (ROOT/'runs/attempt_002/split_ids.json').read_bytes()
    encoded = base64.b64encode(zlib.compress(split,9)).decode()
    source = selection['config']['model_source']
    cells.append(nbf.v4.new_code_cell(f'''import base64,zlib
(WORK/'base_config.json').write_text({json.dumps(selection['config'])!r})
(WORK/'span_config.json').write_text({json.dumps(span)!r})
(WORK/'split_ids.json').write_bytes(zlib.decompress(base64.b64decode({encoded!r})))
RAW=next(p for p in [Path('/kaggle/input/llm-classification-finetuning'),
    Path('/kaggle/input/competitions/llm-classification-finetuning')] if (p/'train.csv').exists())
MODEL=next(p for p in [Path('/kaggle/input/{'/'.join(source.split('/')[1:])}'),
    Path('/kaggle/input/models/{source}'),TRAIN/'base_model'] if (p/'config.json').exists())
'''))
    command_tail = "'--module','arena.exposure_train','--raw',str(RAW),'--model',str(MODEL),\n    '--parent',str(TRAIN),'--split',str(WORK/'split_ids.json'),'--base-config',str(WORK/'base_config.json'),\n    '--config',str(WORK/'span_config.json'),'--out',str(WORK/'exposure'),\n    '--deadline',str(SESSION_STARTED+21600)"
    postcheck = "summary=json.loads((WORK/'exposure/summary.json').read_text());assert summary['complete'];summary"
else:
    command_tail = "'--module','arena.exposure_probe','--out',str(WORK/'probe')"
    postcheck = "summaries=[json.loads((WORK/f'probe/probe_rank{rank}.json').read_text()) for rank in (0,1)];assert all(row['passed'] for row in summaries);summaries"
cells.append(nbf.v4.new_code_cell('''import torch,importlib.metadata
assert torch.cuda.device_count()==2
software={name:importlib.metadata.version(name) for name in ['torch','transformers','accelerate','peft']}
(WORK/'software.json').write_text(json.dumps(software,indent=2))
print('SOFTWARE',software,flush=True)
environment=os.environ.copy();environment['PYTHONPATH']=str(MODULES);environment['PYTHONUNBUFFERED']='1'
command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
    COMMAND_TAIL]
with (WORK/'console.log').open('w') as log:
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=environment)
    for line in process.stdout:
        print(line,end='',flush=True);log.write(line);log.flush()
    status=process.wait()
(WORK/'session_summary.json').write_text(json.dumps({'exit_code':status,'session_seconds':time.time()-SESSION_STARTED},indent=2))
assert status==0,'Stage failed; retain logs and diagnose before another execution'
POSTCHECK
'''.replace('COMMAND_TAIL',command_tail).replace('POSTCHECK',postcheck)))
for cell in cells:
    if cell.cell_type == 'code': compile(cell.source, 'span002-training-cell', 'exec')
nbf.write(nbf.v4.new_notebook(cells=cells, metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}}), dest / 'job.ipynb')
metadata = {'id':'arseniytrushin/llm-preference-span-002-'+('exposure-probe' if args.job=='instrumentation' else 'training-exposure'),
    'title':'LLM Preference Span 002 '+args.job.title(),'code_file':'job.ipynb',
    'language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':False,
    'competition_sources':[] if args.job=='instrumentation' else ['llm-classification-finetuning'],'dataset_sources':[],
    'kernel_sources':[f"{selection['kernel']}/{selection['version_number']}"],
    'model_sources':[] if args.job=='instrumentation' else [selection['config']['model_source']]}
json_write(dest / 'kernel-metadata.json', metadata)
json_write(REPO / f'results/span_002/{args.job}_bundle.json', {
    'notebook_sha256':hashlib.sha256((dest/'job.ipynb').read_bytes()).hexdigest(),
    'source_sha256':sources,'parent_kernel':metadata['kernel_sources'][0],
    'cap_seconds':900 if args.job=='instrumentation' else 21600})
print(dest)
