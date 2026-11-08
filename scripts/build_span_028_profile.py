"""Freeze the information-only E18notebook and rehearse its actual final cell."""
from pathlib import Path
import base64
import json
import subprocess
import sys
import tempfile
import time

import nbformat as nbf
from nbclient import NotebookClient

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,read_json,sha256
from arena.notebook_worker import worker_cell


def main():
    role='prefetch_profile';root=ROOT/'runs/span_028';pub=REPO/'results/span_028'
    assert read_json(pub/'cpu_audit.json')['passed']
    assert read_json(pub/'profile_preparation.json')['prepared'] and (pub/'span_plan.json').exists()
    forward=read_json(pub/'forward_loop_rehearsal.json');assert forward['passed']
    for name,digest in forward['source_sha256'].items():assert sha256(REPO/name)==digest
    assert not (root/'jobs'/role/'push_intent.json').exists()
    folder=root/'notebooks'/role;folder.mkdir(parents=True,exist_ok=True)
    source_spec=(root/'profile_spec.json').read_text();proof=read_json(pub/'cpu_audit.json')
    module_hashes={'data/dataset.py':proof['native_source_sha256']['runs/span_009/research/lmsys-modules-0805/data/dataset.py'],'data/collators.py':proof['native_source_sha256']['runs/span_009/research/lmsys-modules-0805/data/collators.py'],'data/processors.py':proof['native_source_sha256']['runs/span_009/research/lmsys-modules-0805/data/processors.py']}
    setup=f'''from pathlib import Path
import os,sys,time,json,shutil,subprocess,hashlib,base64
STARTED=time.time();WORK=Path('/kaggle/working');DEADLINE=STARTED+3600-120
assert os.environ.get('KAGGLE_IS_COMPETITION_RERUN','').strip().lower() not in ['true','1'],'Information-only CPU prefetch profile'
assert sys.version_info[:2]==(3,10),'Pinned author image not honored'
RAW=next(p for p in [Path('/kaggle/input/llm-classification-finetuning'),Path('/kaggle/input/competitions/llm-classification-finetuning')] if (p/'train.csv').exists())
GPU_INFO=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True).strip().splitlines()
assert len(GPU_INFO)==2 and all('T4' in x for x in GPU_INFO)
(WORK/'hardware.json').write_text(json.dumps(dict(gpus=GPU_INFO,python=sys.version)))
ASSETS=WORK/'assets';ASSETS.mkdir(exist_ok=True)
for slug in ['lmsys-checkpoints-0-0805','lmsys-checkpoints-3-0805','lmsys-modules-0805','lmsys-packages']:
 source=next(p for p in [Path('/kaggle/input')/slug,Path('/kaggle/input/datasets/tascj0')/slug] if p.exists())
 (ASSETS/slug).symlink_to(source,target_is_directory=True)
for name,digest in {module_hashes!r}.items():assert hashlib.sha256((ASSETS/'lmsys-modules-0805'/name).read_bytes()).hexdigest()==digest
shutil.copytree(ASSETS/'lmsys-modules-0805',WORK/'human_pref')
for wheel in ['triton-2.2.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl','xformers-0.0.24042abc8.d20240802-cp310-cp310-linux_x86_64.whl']:
 subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',str(ASSETS/'lmsys-packages'/wheel)],check=True,timeout=120)
(WORK/'prefetch_spec.json').write_text({source_spec!r})
(WORK/'controls').mkdir(exist_ok=True)
os.chdir(WORK)
command=[sys.executable,'-u',str(WORK/'prefetch_profile.py'),'--raw',str(RAW),'--assets',str(ASSETS),'--out',str(WORK/'prefetch_profile'),'--engine',str(WORK/'prefetch_engine.py'),'--spec',str(WORK/'prefetch_spec.json')]
env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
'''
    files={'prefetch_profile.py':'src/arena/span028_profile.py','prefetch_engine.py':'src/arena/span028_engine.py',
        'prefetch_checks.py':'src/arena/span028_checks.py','frozen_inference.py':'src/arena/span010_order.py','gemma_order.py':'src/arena/span013_order.py','cpu_prefetch.py':'src/arena/cpu_prefetch.py'}
    for name,path in files.items():setup+=f"(WORK/{name!r}).write_text({(REPO/path).read_text()!r})\n"
    for path in sorted((root/'controls').glob('*.npz')):
        setup+=f"(WORK/'controls'/{path.name!r}).write_bytes(base64.b64decode({base64.b64encode(path.read_bytes()).decode()!r}))\n"
    with tempfile.TemporaryDirectory(prefix='span028-import-') as tmp:
        for name,path in files.items():(Path(tmp)/name).write_bytes((REPO/path).read_bytes())
        subprocess.run([sys.executable,'-c','import prefetch_profile,prefetch_engine,prefetch_checks'],cwd=tmp,check=True,timeout=30)
    final=worker_cell(role,'')
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(
        '# E18 fixed native CPU prefetch information profile\n\nOne FIFO CPU producer and queue2; original8192microbatches,4kindividual inputs, expert weights and H60route unchanged. Strict canonical/model/memory guards stop early. No quality claim, labels or official deployment.'),
        nbf.v4.new_code_cell(setup),nbf.v4.new_code_cell(final)])
    nb.metadata.update(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python',version='3.10.13'))
    for c in nb.cells:
        if c.cell_type=='code':compile(c.source,'span028-actual-cell','exec')
    nbf.write(nb,folder/'job.ipynb')
    metadata=read_json(ROOT/'runs/span_027/notebooks/packing_profile/kernel-metadata.json')
    metadata.update(id='arseniytrushin/llm-preference-span028-native-prefetch',title='LLM Preference Span028 Native CPU Prefetch')
    atomic_json(folder/'kernel-metadata.json',metadata)
    checks=[]
    for name,program,limit,success in [('success','print(123)',10,True),('failure','raise SystemExit(7)',10,False),('watchdog','import time;time.sleep(10)',.2,False)]:
        with tempfile.TemporaryDirectory(prefix='span028-cell-') as tmp:
            scope=dict(WORK=Path(tmp),STARTED=time.time(),DEADLINE=time.time()+limit,command=[sys.executable,'-c',program],env=None)
            failed=False
            try:exec(compile(final,'actual-final-cell','exec'),scope)
            except AssertionError:failed=True
            record=read_json(Path(tmp)/'session_summary.json');assert failed!=success and (record['exit_code']==0)==success
            checks.append(dict(name=name,passed=True,summary=record))
    guard=next(line for line in setup.splitlines() if line.startswith("assert os.environ.get('KAGGLE_IS_COMPETITION_RERUN'"))
    guard_checks=[]
    from types import SimpleNamespace
    for flag,allowed in [('',True),('false',True),('true',False),('1',False)]:
        failed=False
        try:exec(guard,dict(os=SimpleNamespace(environ={'KAGGLE_IS_COMPETITION_RERUN':flag})))
        except AssertionError:failed=True
        assert failed!=allowed;guard_checks.append(dict(flag=flag,allowed=allowed,passed=True))
    tiny=nbf.v4.new_notebook(metadata=nb.metadata,cells=[nbf.v4.new_code_cell(
        'from pathlib import Path\nimport sys,time,tempfile\n'
        'tmp=tempfile.TemporaryDirectory(prefix="span028-startup-")\nWORK=Path(tmp.name);STARTED=time.time();DEADLINE=STARTED+10\n'
        'command=[sys.executable,"-c","print(123)"];env=None'),nbf.v4.new_code_cell(final)])
    executed=NotebookClient(tiny,timeout=30,kernel_name='python3').execute()
    assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.get('outputs',[])) for c in executed.cells)
    names=list(files.values())+['GOAL.md','docs/SPAN_028_PLAN.md','docs/SPAN_028_CPU_PROOF.md',
        'configs/span_028.json','results/span_028/span_plan.json','results/span_028/cpu_audit.json',
        'results/span_028/profile_preparation.json','results/span_028/runtime_reference.json','src/arena/span028_runtime.py',
        'src/arena/kaggle_control.py','src/arena/notebook_worker.py','scripts/audit_span_028_cpu.py',
        'scripts/prepare_span_028_profile.py','src/arena/cpu_prefetch.py', 'tests/test_cpu_prefetch.py','scripts/build_span_028_profile.py',
        'scripts/run_span_028_profile.py','scripts/audit_span_028_profile.py','tests/test_span028_profile.py',
        'tests/test_span028_forward_rehearsal.py','results/span_028/forward_loop_rehearsal.json',
        'results/span_013/jobs/gemma_order/retrieval.json']
    atomic_json(pub/(role+'_bundle.json'),dict(role=role,source_sha256={n:sha256(REPO/n) for n in names},
        notebook_sha256=sha256(folder/'job.ipynb'),metadata_sha256=sha256(folder/'kernel-metadata.json'),
        public_cap_seconds=3600,information_only=True,official_submission_eligible=False,
        profile_spec_sha256=sha256(root/'profile_spec.json'),packing_tokens=8192,execution_mode='cpu_prefetch',queue_depth=2,conversation_tokens=4096))
    atomic_json(pub/(role+'_cell_rehearsals.json'),dict(passed=True,rehearsals=checks,information_guard_checks=guard_checks,
        standalone_imports_passed=True,notebook_engine_startup_passed=True,new_gpu_executions=0))
    print(folder)


if __name__=='__main__':main()
