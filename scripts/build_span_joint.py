"""Build the offline pooled-feature / matched-adapter pilot notebook for span 001."""
import base64
import hashlib
import json
from pathlib import Path
import sys
import zlib

import nbformat as nbf

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT

config=json.loads((REPO/'configs/span_001.json').read_text())
selection=json.loads((ROOT/'runs/attempt_002/model_selection.json').read_text())
samples=(ROOT/'runs/span_001/sample_ids.json').read_bytes()
dest=REPO/'kaggle/span_001_joint'
dest.mkdir(parents=True,exist_ok=True)
parent=selection['kernel'];owner,slug=parent.split('/')
source=selection['config']['model_source']
cells=[nbf.v4.new_markdown_cell('''# Backlog v1 — H04 / H09 / H11 controlled pilots

Frozen attempt-002 weights plus fresh adapters, four prespecified short training
arms, and frozen pooled answer features. Uses fixed IDs and offline pinned inputs.
See docs/SPAN_001_PROTOCOL.md. A complete kernel can contain explicitly recorded
failed/skipped stages; inspect every stage artifact before resolving a task.
'''),nbf.v4.new_code_cell(f'''import os,sys,subprocess,time,json,base64,zlib,hashlib
from pathlib import Path
SESSION_STARTED=time.time()
DEADLINE=SESSION_STARTED+{config['training_max_session_seconds']}
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
WORK=Path('/kaggle/working')
parents=[Path('/kaggle/input/{slug}'),Path('/kaggle/input/notebooks/{owner}/{slug}')]
TRAIN=next(p for p in parents if (p/'joint_run/adapter/adapter_model.safetensors').exists())
ADAPTER=TRAIN/'joint_run/adapter'
assert hashlib.sha256((ADAPTER/'adapter_model.safetensors').read_bytes()).hexdigest()=={selection['adapter_sha256']!r}
runtime=json.loads((TRAIN/'runtime_requirements.json').read_text())
subprocess.run([sys.executable,'-m','pip','install','-q','--no-index','--no-deps',
    '--find-links',str(TRAIN/'runtime_wheels'),*runtime],check=True)
MODULES=WORK/'bundled_code'
(MODULES/'arena').mkdir(parents=True,exist_ok=True)
(MODULES/'arena/__init__.py').write_text('')
(WORK/'span_config.json').write_text({json.dumps(config)!r})
(WORK/'base_config.json').write_text({json.dumps(selection['config'])!r})
(WORK/'sample_ids.json').write_bytes(zlib.decompress(base64.b64decode({base64.b64encode(zlib.compress(samples,9)).decode()!r})))
''')]
for name in ['data.py','joint_data.py','joint_train.py','span_pack.py','span_adapters.py','span_joint.py']:
    code=(REPO/'src/arena'/name).read_text()
    cells.append(nbf.v4.new_code_cell(f"(MODULES/'arena/{name}').write_text({code!r})"))
cells.append(nbf.v4.new_code_cell(f'''import torch
assert torch.cuda.device_count()==2
print('GPUS',[(torch.cuda.get_device_name(i),torch.cuda.get_device_properties(i).total_memory) for i in range(2)])
assert hashlib.sha256((MODULES/'arena/joint_data.py').read_bytes()).hexdigest()=={selection['config']['preprocessing_sha256']!r}
RAW=next(p for p in [Path('/kaggle/input/llm-classification-finetuning'),
    Path('/kaggle/input/competitions/llm-classification-finetuning')] if (p/'train.csv').exists())
MODEL=next(p for p in [Path('/kaggle/input/{'/'.join(source.split('/')[1:])}'),
    Path('/kaggle/input/models/{source}'),TRAIN/'base_model'] if (p/'config.json').exists())
environment=os.environ.copy();environment['PYTHONPATH']=str(MODULES);environment['PYTHONUNBUFFERED']='1'
stages=[]
def execute(mode,name,stage_deadline,arm=None):
    directory=WORK/'joint'/name
    directory.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
        '--module','arena.span_joint',mode,'--raw',str(RAW),'--model',str(MODEL),
        '--adapter',str(ADAPTER),'--samples',str(WORK/'sample_ids.json'),
        '--config',str(WORK/'span_config.json'),'--base-config',str(WORK/'base_config.json'),
        '--out',str(directory),'--deadline',str(stage_deadline)]
    if arm:command+=['--arm',arm]
    started=time.time()
    with (directory/'console.log').open('w') as log:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            text=True,bufsize=1,env=environment)
        for line in process.stdout:
            print(line,end='',flush=True);log.write(line);log.flush()
        status=process.wait()
    result={{'stage':name,'exit_code':status,'elapsed_seconds':time.time()-started,'complete':status==0}}
    stages.append(result)
    (directory/'stage_status.json').write_text(json.dumps(result,indent=2))
    print('SPAN_STAGE',json.dumps(result),flush=True)
    return status==0
feature_deadline=min(DEADLINE-4*3600,time.time()+5400)
execute('features','features',feature_deadline)
for arm in {config['arms']!r}:
    if time.time()>DEADLINE-1200:
        stages.append({{'stage':arm,'complete':False,'reason':'shared session budget exhausted'}})
        continue
    execute('train',arm,DEADLINE,arm=arm)
summary={{'stages':stages,'all_complete':all(s['complete'] for s in stages),
         'session_seconds':time.time()-SESSION_STARTED,'base_adapter_sha256':{selection['adapter_sha256']!r}}}
(WORK/'joint_summary.json').write_text(json.dumps(summary,indent=2))
if not any(s['complete'] for s in stages):raise RuntimeError('No pilot stage completed; inspect retained logs')
summary
'''))
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})
nbf.write(nb,dest/'joint_pilot.ipynb')
metadata={'id':'arseniytrushin/llm-preference-span-001-joint-pilots',
    'title':'LLM Preference Span 001 Joint Pilots','code_file':'joint_pilot.ipynb',
    'language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':False,
    'competition_sources':['llm-classification-finetuning'],'dataset_sources':[],
    'kernel_sources':[f"{parent}/{selection['version_number']}"],'model_sources':[source]}
(dest/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
print('Built',dest,'sha256',hashlib.sha256((dest/'joint_pilot.ipynb').read_bytes()).hexdigest())
