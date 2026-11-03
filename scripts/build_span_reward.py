"""Package the frozen H01 pilot as a private, bounded Kaggle notebook."""
import base64
import hashlib
import json
from pathlib import Path
import sys
import zlib

import nbformat as nbf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT

config = json.loads((REPO / 'configs/span_001.json').read_text())
samples = (ROOT / 'runs/span_001/sample_ids.json').read_bytes()
inventory = json.loads((ROOT / 'runs/span_001/reward_candidate.json').read_text())
assert config['reward_revision'] == inventory['revision']
dest = REPO / 'kaggle/span_001_reward'
dest.mkdir(parents=True, exist_ok=True)
cells = [nbf.v4.new_markdown_cell('''# Backlog v1 / H01 — frozen preference reward pilot

Pinned public Skywork Qwen3-4B reward weights, fixed train/development IDs,
baseline-like retained content, no model fine-tuning. Fit/evaluate probability heads
separately. All real text stays in private Kaggle/heavy artifacts.
See docs/SPAN_001_PROTOCOL.md for the predeclared controls and limits.
'''), nbf.v4.new_code_cell(f'''import os, sys, subprocess, time, json, base64, zlib
from pathlib import Path
SESSION_STARTED=time.time()
DEADLINE=SESSION_STARTED+{config['reward_max_session_seconds']}
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['HF_HOME']='/kaggle/temp/span_hf'
WORK=Path('/kaggle/working')
MODULES=WORK/'bundled_code'
(MODULES/'arena').mkdir(parents=True,exist_ok=True)
(MODULES/'arena/__init__.py').write_text('')
subprocess.run([sys.executable,'-m','pip','install','-q','--disable-pip-version-check',
    'transformers==5.22.1','peft==0.21.2','accelerate==1.15.0','bitsandbytes==0.50.2'],check=True)
import torch
assert torch.cuda.device_count()==2
print('GPUS',[(torch.cuda.get_device_name(i),torch.cuda.get_device_properties(i).total_memory) for i in range(2)])
(WORK/'config.json').write_text({json.dumps(config)!r})
(WORK/'sample_ids.json').write_bytes(zlib.decompress(base64.b64decode({base64.b64encode(zlib.compress(samples,9)).decode()!r})))
''')]
for name in ('data.py','joint_data.py','span_pack.py','span_reward.py'):
    code = (REPO / 'src/arena' / name).read_text()
    cells.append(nbf.v4.new_code_cell(f"(MODULES/'arena/{name}').write_text({code!r})"))
cells.append(nbf.v4.new_code_cell(f'''from huggingface_hub import snapshot_download
MODEL=snapshot_download({config['reward_model']!r},revision={config['reward_revision']!r},
    cache_dir='/kaggle/temp/reward_model_cache',ignore_patterns=['*.bin','*.h5','*.msgpack'])
RAW=next(p for p in [Path('/kaggle/input/llm-classification-finetuning'),
    Path('/kaggle/input/competitions/llm-classification-finetuning')] if (p/'train.csv').exists())
from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
assert tokenizer.chat_template, 'Pinned checkpoint must provide its conversation template'
print('MODEL',MODEL,'TEMPLATE_PRESENT',bool(tokenizer.chat_template))
environment=os.environ.copy();environment['PYTHONPATH']=str(MODULES);environment['PYTHONUNBUFFERED']='1'
command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
    '--module','arena.span_reward','--raw',str(RAW),'--model',MODEL,
    '--samples',str(WORK/'sample_ids.json'),'--config',str(WORK/'config.json'),
    '--out',str(WORK/'reward'),'--deadline',str(DEADLINE)]
with (WORK/'reward_console.log').open('w') as log:
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
        text=True,bufsize=1,env=environment)
    for line in process.stdout:
        print(line,end='',flush=True);log.write(line);log.flush()
    status=process.wait()
if status:raise RuntimeError(f'Reward pilot failed with status {{status}}; partial outputs retained')
summary={{'complete':True,'model_id':{config['reward_model']!r},'revision':{config['reward_revision']!r},
         'session_seconds':time.time()-SESSION_STARTED,'world_size':2}}
(WORK/'reward_summary.json').write_text(json.dumps(summary,indent=2))
summary
'''))
notebook = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {
    'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}})
nbf.write(notebook, dest / 'reward_pilot.ipynb')
metadata = {'id': 'arseniytrushin/llm-preference-span-001-frozen-reward-pilot',
    'title': 'LLM Preference Span 001 Frozen Reward Pilot', 'code_file': 'reward_pilot.ipynb',
    'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
    'enable_gpu': True, 'enable_internet': True,
    'competition_sources': ['llm-classification-finetuning'], 'dataset_sources': [],
    'kernel_sources': [], 'model_sources': []}
(dest / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
print('Built', dest, 'sha256', hashlib.sha256((dest/'reward_pilot.ipynb').read_bytes()).hexdigest())
