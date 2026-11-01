"""Build confirmation or submission notebooks using a frozen adapter and local wheels."""
import argparse,base64,json,sys,zlib
from pathlib import Path
import nbformat as nbf
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT

parser=argparse.ArgumentParser()
parser.add_argument('mode',choices=['confirm','predict'])
args=parser.parse_args()
selection=json.loads((ROOT/'runs/attempt_002/model_selection.json').read_text())
parent=selection['kernel'];owner,slug=parent.split('/')
dest=REPO/f'kaggle/attempt_002_{args.mode}'
dest.mkdir(parents=True,exist_ok=True)
cells=[nbf.v4.new_markdown_cell(f'''# Attempt 002: frozen joint preference model — {args.mode}

Load the selected QLoRA adapter and base checkpoint locally. Package versions are
restored from wheels produced by training. Internet is disabled. The adapter digest,
preprocessing digest, split assignments, and temperature are fixed before this run.
'''),nbf.v4.new_code_cell(f'''import os,sys,json,subprocess,shutil,base64,zlib
from pathlib import Path
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
WORK=Path('/kaggle/working')
parents=[Path('/kaggle/input/{slug}'),Path('/kaggle/input/notebooks/{owner}/{slug}')]
TRAIN=next(p for p in parents if (p/'joint_run/adapter/adapter_model.safetensors').exists())
runtime=json.loads((TRAIN/'runtime_requirements.json').read_text())
subprocess.run([sys.executable,'-m','pip','install','-q','--no-index','--no-deps',
    '--find-links',str(TRAIN/'runtime_wheels'),*runtime],check=True)
MODULES=WORK/'bundled_code'
shutil.copytree(TRAIN/'bundled_code',MODULES,dirs_exist_ok=True)
SELECTION={selection!r}
(WORK/'model_selection.json').write_text(json.dumps(SELECTION,indent=2))
''')]
source=(REPO/'src/arena/joint_infer.py').read_text()
cells.append(nbf.v4.new_code_cell(f"(MODULES/'arena/joint_infer.py').write_text({source!r})"))
if args.mode=='confirm':
    split=base64.b64encode(zlib.compress((ROOT/'runs/attempt_002/split_ids.json').read_bytes(),9)).decode()
    cells.append(nbf.v4.new_code_cell(f"(WORK/'split_ids.json').write_bytes(zlib.decompress(base64.b64decode({split!r})))"))
cells.append(nbf.v4.new_code_cell(f'''import torch
assert torch.cuda.is_available()
RAW=next(p for p in [Path('/kaggle/input/llm-classification-finetuning'),
    Path('/kaggle/input/competitions/llm-classification-finetuning')] if (p/'test.csv').exists())
candidates=[Path('/kaggle/input/qwen-3/transformers/8b/1'),
    Path('/kaggle/input/models/qwen-lm/qwen-3/transformers/8b/1'),TRAIN/'base_model']
MODEL=next(p for p in candidates if (p/'config.json').exists())
environment=os.environ.copy();environment['PYTHONPATH']=str(MODULES);environment['PYTHONUNBUFFERED']='1'
command=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={{torch.cuda.device_count()}}',
    '--module','arena.joint_infer','--raw',str(RAW),'--model',str(MODEL),
    '--adapter',str(TRAIN/'joint_run/adapter'),'--config',str(TRAIN/'joint_run/training_config.json'),
    '--summary',str(TRAIN/'joint_run/summary.json'),'--selection',str(WORK/'model_selection.json'),
    '--out',str(WORK),'--mode',{args.mode!r}]
command += {['--split','/kaggle/working/split_ids.json'] if args.mode=='confirm' else []!r}
with (WORK/'inference_console.log').open('w') as log:
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=environment)
    for line in process.stdout: print(line,end='',flush=True);log.write(line);log.flush()
    code=process.wait()
if code:raise RuntimeError(f'Inference failed with exit code {{code}}')
'''))
notebook=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})
nbf.write(notebook,dest/'joint_inference.ipynb')
metadata={'id':f'{owner}/llm-preference-qwen3-8b-{args.mode}-002',
    'title':f'LLM Preference Qwen3 8B {args.mode.title()} 002','code_file':'joint_inference.ipynb',
    'language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':False,
    'competition_sources':['llm-classification-finetuning'],'dataset_sources':[],
    'kernel_sources':[f"{parent}/{selection['version_number']}"],'model_sources':[selection['config']['model_source']]}
(dest/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
print('Built',dest)
