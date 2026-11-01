"""Package the tested training implementation for a private Kaggle GPU run."""
import argparse
import base64
import hashlib
import json
import sys
import zlib
from pathlib import Path
import nbformat as nbf

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT

parser=argparse.ArgumentParser()
parser.add_argument('stage',choices=['pilot','train'])
parser.add_argument('--pilot-steps',type=int,default=20)
parser.add_argument('--max-length',type=int)
parser.add_argument('--train-batch-size',type=int)
parser.add_argument('--epochs',type=float)
args=parser.parse_args()
config=json.loads((REPO/'configs/attempt_002.json').read_text())
for key in ['max_length','train_batch_size','epochs']:
    value=getattr(args,key)
    if value is not None:config[key]=value
if args.stage=='pilot':config['max_train_seconds']=1800
split_bytes=(ROOT/'runs/attempt_002/split_ids.json').read_bytes()
split_encoded=base64.b64encode(zlib.compress(split_bytes,9)).decode()
dest=REPO/f'kaggle/attempt_002_{args.stage}'
dest.mkdir(parents=True,exist_ok=True)
cells=[nbf.v4.new_markdown_cell(f'''# Attempt 002: Qwen3-8B joint preference classifier ({args.stage})

Read the question and both responses jointly, train a three-class head and LoRA adapters,
and compare with training-prior and majority baselines. The base model is the general
Qwen3-8B checkpoint, not a model fine-tuned on this competition. Gemma is not required.

All split assignments are fixed by explicit IDs from a separate prompt-grouped partition.
Training augments response swaps; evaluation averages both orientations. Model selection
uses development data. The confirmation partition is not evaluated in this notebook.

This training notebook uses internet for package setup and a model-download fallback.
The final competition inference notebook will have internet disabled.
'''),nbf.v4.new_code_cell('''import os, sys, json, subprocess, base64, zlib, hashlib
from pathlib import Path
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['WANDB_DISABLED']='true'
WORK=Path('/kaggle/working')
MODULES=WORK/'bundled_code'
(MODULES/'arena').mkdir(parents=True,exist_ok=True)
(MODULES/'arena/__init__.py').write_text('')
os.environ['HF_HOME']=str(WORK/'hf_metadata_cache')
import torch
assert torch.cuda.is_available(), 'A GPU must be allocated for QLoRA'
print('CUDA devices:', [(torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i).total_memory) for i in range(torch.cuda.device_count())])
import importlib.metadata
before={d.metadata['Name'].lower():d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
subprocess.run([sys.executable,'-m','pip','install','-q','--disable-pip-version-check',
    'transformers==5.22.1','peft==0.21.2','accelerate==1.15.0','bitsandbytes==0.50.2'],check=True)
after={d.metadata['Name'].lower():d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
changed={n:v for n,v in after.items() if before.get(n)!=v and not n.startswith(('torch','nvidia-'))}
changed.update({n:after[n] for n in ['transformers','peft','accelerate','bitsandbytes']})
runtime=[f'{n}=={v}' for n,v in sorted(changed.items())]
(WORK/'runtime_requirements.json').write_text(json.dumps(runtime,indent=2))
subprocess.run([sys.executable,'-m','pip','download','-q','--no-deps','--dest',str(WORK/'runtime_wheels'),*runtime],check=True)
''')]
for name in ['data','joint_data','joint_train']:
    source=(REPO/f'src/arena/{name}.py').read_text()
    cells.append(nbf.v4.new_code_cell(f"(MODULES/'arena/{name}.py').write_text({source!r})\nprint('Bundled arena/{name}.py')"))
cells.append(nbf.v4.new_code_cell(f"CONFIG={config!r}\nSPLIT_BYTES=zlib.decompress(base64.b64decode({split_encoded!r}))\nassert hashlib.sha256(SPLIT_BYTES).hexdigest()=={hashlib.sha256(split_bytes).hexdigest()!r}\n(WORK/'split_ids.json').write_bytes(SPLIT_BYTES)\n(WORK/'training_config.json').write_text(json.dumps(CONFIG,indent=2))\nprint(CONFIG)"))
cells.append(nbf.v4.new_code_cell('''RAW=next(p for p in [Path('/kaggle/input/llm-classification-finetuning'),
    Path('/kaggle/input/competitions/llm-classification-finetuning')] if (p/'train.csv').exists())
MODEL_CANDIDATES=[Path('/kaggle/input/qwen-3/transformers/8b/1'),
    Path('/kaggle/input/models/qwen-lm/qwen-3/transformers/8b/1')]
MODEL=next((p for p in MODEL_CANDIDATES if (p/'config.json').exists()),None)
if MODEL is None:
    from huggingface_hub import HfApi, snapshot_download
    revision=HfApi().model_info(CONFIG['model_id']).sha
    MODEL=WORK/'base_model'
    snapshot_download(CONFIG['model_id'],revision=revision,local_dir=MODEL,
        allow_patterns=['*.json','*.safetensors','*.model','merges.txt','vocab.json'])
    CONFIG['hf_revision']=revision
    (WORK/'training_config.json').write_text(json.dumps(CONFIG,indent=2))
print('Pinned base model:', MODEL)
assert json.loads((MODEL/'config.json').read_text())['model_type']=='qwen3'
'''))
cells.append(nbf.v4.new_code_cell(f'''environment=os.environ.copy()
environment['PYTHONPATH']=str(MODULES)
environment['PYTHONUNBUFFERED']='1'
processes=torch.cuda.device_count()
command=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={{processes}}',
    '--module','arena.joint_train','--raw',str(RAW),'--model',str(MODEL),
    '--split',str(WORK/'split_ids.json'),'--config',str(WORK/'training_config.json'),
    '--out',str(WORK/'joint_run')]
command += {['--pilot-steps',str(args.pilot_steps)] if args.stage=='pilot' else []!r}
with (WORK/'training_console.log').open('w') as log:
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=environment)
    for line in process.stdout:
        print(line,end='',flush=True);log.write(line);log.flush()
    return_code=process.wait()
if return_code:
    raise RuntimeError(f'Joint training exited with status {{return_code}}; see training_console.log')
summary=json.loads((WORK/'joint_run/summary.json').read_text())
summary
'''))
notebook=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})
nbf.write(notebook,dest/'joint_training.ipynb')
owner=json.loads((REPO/'kaggle/kernel-metadata.json').read_text())['id'].split('/')[0]
metadata={'id':f'{owner}/llm-preference-qwen3-8b-{args.stage}-002',
    'title':f'LLM Preference Qwen3 8B {args.stage.title()} 002','code_file':'joint_training.ipynb',
    'language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':True,
    'competition_sources':['llm-classification-finetuning'],'dataset_sources':[],
    'kernel_sources':[],'model_sources':[config['model_source']]}
(dest/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
print('Built',dest,'notebook bytes',(dest/'joint_training.ipynb').stat().st_size)
