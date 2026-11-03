"""One diagnosed H37 recovery: prove CUDA backend availability and screen throughput."""
import hashlib
import json
from pathlib import Path
import sys

import nbformat as nbf
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,json_write

root=ROOT/'runs/span_002'
prior=root/'jobs/ensemble-expansion'
assert json.loads((prior/'status.json').read_text())['status'].endswith('ERROR')
nb=nbf.read(root/'notebooks/ensemble-expansion/job.ipynb',as_version=4)
dest=root/'notebooks/ensemble-recovery'
dest.mkdir(parents=True,exist_ok=True)
nb.cells[0]=nbf.v4.new_markdown_cell('''# Span 002 — diagnosed H37 recovery

Previous immutable expansion job ended with a 600-second local-server timeout.
Check actual CUDA backend availability, repair missing CUDA-12 libraries only if
needed, and verify the unchanged judge on 32 cached rows. Continue the fixed extra
sample only if numerical identity and measured runtime fit the remaining ceiling.
''')
diagnostic=nbf.v4.new_code_cell('''import subprocess,importlib.metadata
from arena import judge_runtime
DEADLINE=SESSION_STARTED+4500
hardware={'nvidia_smi':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv'],text=True)}
environment=judge_runtime.runtime_environment(EXE)
backend=next(BIN.glob('libggml-cuda.so*'))
dependency=subprocess.run(['ldd',str(backend)],env=environment,capture_output=True,text=True)
hardware['ldd_before']=dependency.stdout+dependency.stderr
devices=subprocess.run([str(EXE),'--list-devices'],env=environment,capture_output=True,text=True,timeout=60)
hardware['devices_before']=devices.stdout+devices.stderr
hardware['cuda12_repair']=False
if 'CUDA0' not in hardware['devices_before']:
    target=WORK/'cuda12'
    subprocess.run([sys.executable,'-m','pip','install','-q','--disable-pip-version-check','--no-deps',
        '--target',str(target),'nvidia-cublas-cu12==12.8.4.1','nvidia-cuda-runtime-cu12==12.8.90'],check=True,timeout=600)
    paths=[str(target/'nvidia/cublas/lib'),str(target/'nvidia/cuda_runtime/lib')]
    original_environment=judge_runtime.runtime_environment
    def repaired_environment(executable):
        result=original_environment(executable)
        result['LD_LIBRARY_PATH']=':'.join(paths)+':'+result.get('LD_LIBRARY_PATH','')
        return result
    judge_runtime.runtime_environment=repaired_environment
    hardware['cuda12_repair']=True
    hardware['repair_packages']=['nvidia-cublas-cu12==12.8.4.1','nvidia-cuda-runtime-cu12==12.8.90']
environment=judge_runtime.runtime_environment(EXE)
devices=subprocess.run([str(EXE),'--list-devices'],env=environment,capture_output=True,text=True,timeout=60)
hardware['devices_after']=devices.stdout+devices.stderr
(WORK/'hardware_diagnostic.json').write_text(json.dumps(hardware,indent=2))
print('HARDWARE_DIAGNOSTIC',json.dumps(hardware),flush=True)
assert 'CUDA0' in hardware['devices_after'] and 'CUDA1' in hardware['devices_after'],'CUDA backend still unavailable: stop before full scoring'
''')
cached=np.load(ROOT/'runs/attempt_003/jobs/development_ctx1024/output/gemma26/judge_predictions.npz')
profile={'ids':cached['ids'][:32].tolist()}
raw=cached['original'][:32].tolist()
nb.cells.insert(len(nb.cells)-1,diagnostic)
nb.cells[-1]=nbf.v4.new_code_cell(f'''from arena.judge import run
import numpy as np
profile=run(CONFIG,EXE,RAW,WORK/'profile','gemma26','profile',sample={profile!r},both_orders=False)
observed=np.load(WORK/'profile/judge_predictions.npz')['original']
drift=np.abs(observed-np.asarray({raw!r}))
gate={{'profile_rows':32,'profile_seconds':profile['seconds'],
    'mean_probability_difference':float(drift.mean()),'max_probability_difference':float(drift.max()),
    'mean_limit':0.01,'max_limit':0.1,
    'projected_extra_seconds_with_margin':profile['seconds']/32*len(SAMPLE['ids'])*1.2+600,
    'remaining_session_seconds':DEADLINE-time.time()}}
gate['identity_passed']=bool(drift.mean()<.01 and drift.max()<.1)
gate['throughput_passed']=gate['projected_extra_seconds_with_margin']<gate['remaining_session_seconds']
(WORK/'recovery_gate.json').write_text(json.dumps(gate,indent=2))
print('RECOVERY_GATE',json.dumps(gate),flush=True)
if gate['identity_passed'] and gate['throughput_passed']:
    summary=run(CONFIG,EXE,RAW,WORK/'gemma26','gemma26','dev',sample=SAMPLE,both_orders=False)
    result={{'expanded':True,'rows':summary['rows']}}
else:
    result={{'expanded':False,'reason':'Predeclared numerical/runtime screen failed; no full scoring'}}
(WORK/'session_summary.json').write_text(json.dumps({{'session_seconds':time.time()-SESSION_STARTED,**result}},indent=2))
result
''')
for cell in nb.cells:
    if cell.cell_type=='code':compile(cell.source,'span002-judge-recovery-cell','exec')
nbf.write(nb,dest/'job.ipynb')
metadata=json.loads((root/'notebooks/ensemble-expansion/kernel-metadata.json').read_text())
metadata.update(id='arseniytrushin/llm-preference-span-002-judge-recovery',
    title='LLM Preference Span 002 Judge Recovery',enable_internet=True)
json_write(dest/'kernel-metadata.json',metadata)
json_write(REPO/'results/span_002/recovery_decision.json',{
    'failed_job':'arseniytrushin/llm-preference-span-002-judge-expansion/1',
    'terminal_status':'ERROR','completed_rows':1,'failure':'600-second local completion timeout',
    'hypothesis':'CUDA-12 native backend may be unavailable in the new CUDA-13 runtime; unproven until ldd/device diagnostics.',
    'maximum_new_session_seconds':4500,'allocation':'Original 75-minute recovery reserve; no increase in total cap.',
    'model_prompt_packing_changed':False,'sample_changed':False,
    'gates':'Require two actual CUDA devices, cached-row numerical identity, and measured projection fitting remaining ceiling.',
    'notebook_sha256':hashlib.sha256((dest/'job.ipynb').read_bytes()).hexdigest(),
    'internet_use':'Pinned NVIDIA CUDA-12 runtime packages only if device discovery proves unavailable; no dataset text sent to package services.'})
print(dest)
