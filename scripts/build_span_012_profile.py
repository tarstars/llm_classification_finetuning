"""Freeze/rehearse the engineering-only packing notebook; never submit it."""
from pathlib import Path
import hashlib
import sys
import tempfile
import time
from types import SimpleNamespace

import nbformat as nbf
from nbclient import NotebookClient

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,read_json,sha256
from arena.notebook_worker import worker_cell


def main():
    role='packing_profile';root=ROOT/'runs/span_012';pub=REPO/'results/span_012'
    registry=read_json(REPO/'backlog/hypotheses.json');assert registry['active_version']==12
    assert next(v for v in registry['versions'] if v['number']==12)['focus_ids']==['E16','H58']
    assert not (root/'jobs'/role/'push_intent.json').exists(),'Never rewrite accepted source'
    plan=read_json(pub/'span_plan.json');assert plan['gpu_session_seconds_cap']==7200
    folder=root/'notebooks'/role;folder.mkdir(parents=True,exist_ok=True)
    driver=REPO/'src/arena/span012_profile.py';visible=sha256(ROOT/'raw/test.csv')
    setup=f'''from pathlib import Path
import os,sys,time,json,shutil,subprocess,hashlib
STARTED=time.time();WORK=Path('/kaggle/working')
RAW=next(p for p in [Path('/kaggle/input/llm-classification-finetuning'),Path('/kaggle/input/competitions/llm-classification-finetuning')] if (p/'test.csv').exists())
IS_RERUN=os.environ.get('KAGGLE_IS_COMPETITION_RERUN','').strip().lower() in ['true','1'] or hashlib.sha256((RAW/'test.csv').read_bytes()).hexdigest()!={visible!r}
assert not IS_RERUN,'Engineering-only profile cannot be an official submission'
DEADLINE=STARTED+1800-120
assert sys.version_info[:2]==(3,10),'Pinned author image not honored'
GPU_INFO=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True).strip().splitlines()
assert len(GPU_INFO)==2 and all('T4' in name for name in GPU_INFO),'Expected two T4 devices'
(WORK/'hardware.json').write_text(json.dumps(dict(gpus=GPU_INFO,python=sys.version)))
ASSETS=WORK/'assets';ASSETS.mkdir(exist_ok=True)
for slug in ['lmsys-checkpoints-0-0805','lmsys-checkpoints-3-0805','lmsys-modules-0805','lmsys-packages']:
    source=next(p for p in [Path('/kaggle/input')/slug,Path('/kaggle/input/datasets/tascj0')/slug] if p.exists())
    (ASSETS/slug).symlink_to(source,target_is_directory=True)
shutil.copytree(ASSETS/'lmsys-modules-0805',WORK/'human_pref')
for wheel in ['triton-2.2.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl','xformers-0.0.24042abc8.d20240802-cp310-cp310-linux_x86_64.whl']:
    subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',str(ASSETS/'lmsys-packages'/wheel)],check=True,timeout=120)
(WORK/'packing_profile.py').write_text({driver.read_text()!r})
os.chdir(WORK)
command=[sys.executable,'-u',str(WORK/'packing_profile.py'),'--mode','profile','--raw',str(RAW),'--assets',str(ASSETS),'--out',str(WORK/'packing_profile')]
env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
'''
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(
        '# E16: matched packing profile\n\nEngineering only, no submission. Same256rows and loaded experts, full80row warmups,8192/6144/8192bracket, no target fields. Strict raw controls, GPU telemetry and timing determine engineering eligibility.'),
        nbf.v4.new_code_cell(setup),nbf.v4.new_code_cell(worker_cell(role,''))])
    nb.metadata.update(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python',version='3.10.13'))
    for c in nb.cells:
        if c.cell_type=='code':compile(c.source,'span012-profile-cell','exec')
    nbf.write(nb,folder/'job.ipynb')
    metadata=dict(id='arseniytrushin/llm-preference-span012-packing-profile',title='LLM Preference Span012 Packing Profile',
        code_file='job.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_tpu=False,
        enable_internet=False,dataset_sources=['tascj0/'+s+'/1' for s in ['lmsys-checkpoints-0-0805','lmsys-checkpoints-3-0805','lmsys-modules-0805','lmsys-packages']],
        model_sources=[],kernel_sources=[],competition_sources=['llm-classification-finetuning'],
        docker_image='gcr.io/kaggle-private-byod/python@sha256:e3f9bf91d1b7b36bf77554565d27c5733585fbe5e2b7f0832e068824e7accc3c',
        docker_image_pinning_type='original',machine_shape='NvidiaTeslaT4')
    atomic_json(folder/'kernel-metadata.json',metadata)
    actual=nbf.read(folder/'job.ipynb',4).cells[-1].source;checks=[]
    for name,program,limit,success in [('success','print(123)',10,True),('failure','raise SystemExit(7)',10,False),('watchdog','import time;time.sleep(10)',.2,False)]:
        with tempfile.TemporaryDirectory(prefix='span012-cell-') as tmp:
            scope=dict(WORK=Path(tmp),STARTED=time.time(),DEADLINE=time.time()+limit,command=[sys.executable,'-c',program],env=None);failed=False
            try:exec(compile(actual,'actual-profile-worker-cell','exec'),scope)
            except AssertionError:failed=True
            record=read_json(Path(tmp)/'session_summary.json');assert failed!=success and (record['exit_code']==0)==success
            checks.append(dict(name=name,passed=True,summary=record))
    mode_source='\n'.join(line for line in setup.splitlines() if line.startswith(('IS_RERUN=','assert not IS_RERUN','DEADLINE=')))
    modes=[]
    for flag,observed_hash,denied in [('',visible,False),('','changed',True),('true',visible,True)]:
        scope=dict(os=SimpleNamespace(environ={'KAGGLE_IS_COMPETITION_RERUN':flag}),
            hashlib=SimpleNamespace(sha256=lambda data:SimpleNamespace(hexdigest=lambda:observed_hash)),RAW=ROOT/'raw',STARTED=0)
        failed=False
        try:exec(compile(mode_source,'actual-engineering-mode-guard','exec'),scope)
        except AssertionError:failed=True
        assert failed==denied
        modes.append(dict(hidden=denied,denied=failed,passed=True))
    tiny=nbf.v4.new_notebook(metadata=nb.metadata,cells=[nbf.v4.new_code_cell(
        'from pathlib import Path\nimport sys,time,tempfile\ntmp=tempfile.TemporaryDirectory(prefix="span012-startup-")\n'
        'WORK=Path(tmp.name);STARTED=time.time();DEADLINE=STARTED+10\ncommand=[sys.executable,"-c","print(123)"];env=None'),nbf.v4.new_code_cell(actual)])
    executed=NotebookClient(tiny,timeout=30,kernel_name='python3').execute()
    assert all(c.execution_count is not None for c in executed.cells)
    assert not any(o.output_type=='error' for c in executed.cells for o in c.get('outputs',[]))
    sources=['GOAL.md','docs/SPAN_012_PLAN.md','configs/span_012.json','results/span_012/span_plan.json',
        'src/arena/span012_profile.py','src/arena/notebook_worker.py','src/arena/kaggle_control.py',
        'scripts/build_span_012_profile.py','scripts/run_span_012_order.py','tests/test_span012_profile.py']
    atomic_json(pub/(role+'_bundle.json'),dict(role=role,source_sha256={p:sha256(REPO/p) for p in sources},
        notebook_sha256=sha256(folder/'job.ipynb'),metadata_sha256=sha256(folder/'kernel-metadata.json'),
        public_cap_seconds=1800,engineering_only=True,hidden_submissions_forbidden=True,visible_test_sha256=visible))
    atomic_json(pub/(role+'_cell_rehearsals.json'),dict(passed=True,rehearsals=checks,mode_checks=modes,
        notebook_engine_startup_passed=True,last_cell_sha256=hashlib.sha256(actual.encode()).hexdigest(),new_gpu_executions=0))
    print(folder)


if __name__=='__main__':main()
