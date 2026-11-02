"""Package a fixed, inference-only development diagnostic on the frozen Kaggle model."""
import base64,json,sys,zlib
from pathlib import Path
import nbformat as nbf
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,json_write
selection=json.loads((ROOT/'runs/attempt_002/model_selection.json').read_text())
ids=np.load(ROOT/'runs/attempt_002/jobs/qwen4b1024/output/joint_run/dev_predictions.npz')['ids']
rng=np.random.default_rng(20261107)
sample={'seed':20261107,'sampling':'Uniform random development rows, selected without looking at outcomes',
        'ids':ids[np.sort(rng.choice(len(ids),512,replace=False))].tolist()}
json_write(ROOT/'eda/attempt_002/model_diagnostic_sample.json',sample)
owner,slug=selection['kernel'].split('/')
dest=REPO/'kaggle/attempt_002_eda';dest.mkdir(parents=True,exist_ok=True)
split=base64.b64encode(zlib.compress((ROOT/'runs/attempt_002/split_ids.json').read_bytes(),9)).decode()
source=selection['config']['model_source']
cells=[nbf.v4.new_markdown_cell('''# Frozen Qwen3-4B model diagnostics

Inference only on 512 uniformly sampled development examples. Measure raw response
order sensitivity and the effect of 2,048-token inputs with the same frozen model
and temperature. Reuse predictions where token sequences are unchanged. No
confirmation data is evaluated and no competition submission is produced.
'''),nbf.v4.new_code_cell(f'''import os,sys,json,shutil,subprocess,base64,zlib
from pathlib import Path
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['HF_HUB_OFFLINE']='1'
WORK=Path('/kaggle/working')
TRAIN=next(p for p in [Path('/kaggle/input/{slug}'),Path('/kaggle/input/notebooks/{owner}/{slug}')]
    if (p/'joint_run/adapter/adapter_model.safetensors').exists())
runtime=json.loads((TRAIN/'runtime_requirements.json').read_text())
subprocess.run([sys.executable,'-m','pip','install','-q','--no-index','--no-deps','--find-links',str(TRAIN/'runtime_wheels'),*runtime],check=True)
MODULES=WORK/'bundled_code';shutil.copytree(TRAIN/'bundled_code',MODULES,dirs_exist_ok=True)
(WORK/'selection.json').write_text(json.dumps({selection!r}))
(WORK/'sample.json').write_text(json.dumps({sample!r}))
(WORK/'split.json').write_bytes(zlib.decompress(base64.b64decode({split!r})))
''')]
for name in ['joint_infer','eda_infer']:
    cells.append(nbf.v4.new_code_cell(f"(MODULES/'arena/{name}.py').write_text({(REPO/f'src/arena/{name}.py').read_text()!r})"))
cells.append(nbf.v4.new_code_cell(f'''import torch
assert torch.cuda.is_available()
RAW=next(p for p in [Path('/kaggle/input/llm-classification-finetuning'),Path('/kaggle/input/competitions/llm-classification-finetuning')] if (p/'train.csv').exists())
MODEL=next(p for p in [Path('/kaggle/input/{'/'.join(source.split('/')[1:])}'),Path('/kaggle/input/models/{source}')]
    if (p/'config.json').exists())
environment=os.environ.copy();environment['PYTHONPATH']=str(MODULES);environment['PYTHONUNBUFFERED']='1'
command=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={{torch.cuda.device_count()}}',
    '--module','arena.eda_infer','--raw',str(RAW),'--model',str(MODEL),'--train',str(TRAIN),
    '--selection',str(WORK/'selection.json'),'--sample',str(WORK/'sample.json'),
    '--split',str(WORK/'split.json'),'--out',str(WORK)]
with (WORK/'eda_console.log').open('w') as log:
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=environment)
    for line in process.stdout:print(line,end='',flush=True);log.write(line);log.flush()
    code=process.wait()
if code:raise RuntimeError(f'Frozen diagnostics failed: {{code}}')
json.loads((WORK/'eda_model_summary.json').read_text())
'''))
notebook=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})
for i,c in enumerate(cells):
    if c.cell_type=='code':compile(c.source,f'eda-cell{i}','exec')
nbf.write(notebook,dest/'model_diagnostics.ipynb')
metadata={'id':f'{owner}/llm-preference-qwen3-4b-eda-002','title':'LLM Preference Qwen3 4B EDA 002',
    'code_file':'model_diagnostics.ipynb','language':'python','kernel_type':'notebook','is_private':True,
    'enable_gpu':True,'enable_internet':False,'competition_sources':['llm-classification-finetuning'],
    'dataset_sources':[],'kernel_sources':[f"{selection['kernel']}/{selection['version_number']}"],
    'model_sources':[selection['config']['model_source']]}
(dest/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
print('Built',dest)
