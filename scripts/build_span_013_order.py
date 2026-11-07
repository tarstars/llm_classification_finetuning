"""Freeze an offline order-completion notebook and rehearse its actual cells."""
from pathlib import Path
import json
import sys
import tempfile
import time
from types import SimpleNamespace

import nbformat as nbf
import numpy as np
import pandas as pd
from nbclient import NotebookClient

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,read_json,sha256
from arena.notebook_worker import worker_cell
from arena.span010_order import profile_indices,TEXT


def main():
    role='gemma_order';root=ROOT/'runs/span_013';pub=REPO/'results/span_013'
    registry=read_json(REPO/'backlog/hypotheses.json')
    assert registry['active_version']==13
    assert next(v for v in registry['versions'] if v['number']==13)['focus_ids']==['H60']
    assert read_json(pub/'H60_cached_replay.json')['information_gate_passed']
    assert not (root/'jobs'/role/'push_intent.json').exists(),'Never rewrite accepted source'
    folder=root/'notebooks'/role;folder.mkdir(parents=True,exist_ok=True)
    visible=sha256(ROOT/'raw/test.csv');driver=REPO/'src/arena/span013_pipeline.py'
    setup=f'''from pathlib import Path
import os,sys,time,json,shutil,subprocess,hashlib
STARTED=time.time();WORK=Path('/kaggle/working')
RAW=next(p for p in [Path('/kaggle/input/llm-classification-finetuning'),Path('/kaggle/input/competitions/llm-classification-finetuning')] if (p/'test.csv').exists())
IS_RERUN=os.environ.get('KAGGLE_IS_COMPETITION_RERUN','').strip().lower() in ['true','1'] or hashlib.sha256((RAW/'test.csv').read_bytes()).hexdigest()!={visible!r}
DEADLINE=STARTED+(32400 if IS_RERUN else 2400)-120
print('Runtime',sys.version,'hidden',IS_RERUN,flush=True)
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
(WORK/'order_completion.py').write_text({driver.read_text()!r})
(WORK/'frozen_inference.py').write_text({(REPO/'src/arena/span010_order.py').read_text()!r})
(WORK/'gemma_order.py').write_text({(REPO/'src/arena/span013_order.py').read_text()!r})
os.chdir(WORK)
command=[sys.executable,'-u',str(WORK/'order_completion.py'),'--raw',str(RAW),'--assets',str(ASSETS),'--out',str(WORK/'order_completion')]+([] if IS_RERUN else ['--qualify'])
env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
'''
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(
        '# V13: fixed Gemma-only answer-order averaging\n\nFrozen tascj supervised classifiers, original recipe off-route; Gemma orientations averaged with equal total expert weights on a label-free disagreement route. Local scores overlap training. Hidden mode performs actual inference with no public prediction cache.'),
        nbf.v4.new_code_cell(setup),nbf.v4.new_code_cell(worker_cell(role,''))])
    nb.metadata.update(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
                       language_info=dict(name='python',version='3.10.13'))
    for c in nb.cells:
        if c.cell_type=='code':compile(c.source,'span013-serialized-cell','exec')
    nbf.write(nb,folder/'job.ipynb')
    metadata=dict(id='arseniytrushin/llm-preference-span013-gemma-order',
        title='LLM Preference Span013 Gemma Order',code_file='job.ipynb',language='python',
        kernel_type='notebook',is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=False,
        dataset_sources=['tascj0/'+s+'/1' for s in ['lmsys-checkpoints-0-0805','lmsys-checkpoints-3-0805',
            'lmsys-modules-0805','lmsys-packages']],model_sources=[],kernel_sources=[],
        competition_sources=['llm-classification-finetuning'],
        docker_image='gcr.io/kaggle-private-byod/python@sha256:e3f9bf91d1b7b36bf77554565d27c5733585fbe5e2b7f0832e068824e7accc3c',
        docker_image_pinning_type='original',machine_shape='NvidiaTeslaT4')
    atomic_json(folder/'kernel-metadata.json',metadata)
    actual=nbf.read(folder/'job.ipynb',4).cells[-1].source
    rehearsals=[]
    for name,program,limit,success in [('success','print(123)',10,True),
            ('failure','raise SystemExit(7)',10,False),('watchdog','import time;time.sleep(10)',.2,False)]:
        with tempfile.TemporaryDirectory(prefix='span013-cell-') as tmp:
            scope=dict(WORK=Path(tmp),STARTED=time.time(),DEADLINE=time.time()+limit,
                       command=[sys.executable,'-c',program],env=None)
            failed=False
            try:exec(compile(actual,'actual-serialized-last-cell','exec'),scope)
            except AssertionError:failed=True
            record=read_json(Path(tmp)/'session_summary.json')
            assert failed!=success and (record['exit_code']==0)==success
            rehearsals.append(dict(name=name,passed=True,summary=record))
    mode_source='\n'.join(line for line in setup.splitlines() if line.startswith(('IS_RERUN=','DEADLINE=','command=')))
    mode_checks=[]
    for flag,observed_hash,expected in [('',visible,False),('','changed',True),('true',visible,True)]:
        scope=dict(os=SimpleNamespace(environ={'KAGGLE_IS_COMPETITION_RERUN':flag}),
            hashlib=SimpleNamespace(sha256=lambda data:SimpleNamespace(hexdigest=lambda:observed_hash)),
            RAW=ROOT/'raw',WORK=Path('/tmp'),STARTED=0,ASSETS=Path('/tmp/assets'),sys=sys)
        exec(compile(mode_source,'serialized-mode-branch','exec'),scope)
        assert scope['IS_RERUN']==expected
        assert ('--qualify' in scope['command'])!=expected
        assert scope['DEADLINE']==(32400 if expected else 2400)-120
        mode_checks.append(dict(hidden=expected,passed=True))
    tiny=nbf.v4.new_notebook(metadata=nb.metadata,cells=[nbf.v4.new_code_cell(
        'from pathlib import Path\nimport sys,time,tempfile\n'
        'tmp=tempfile.TemporaryDirectory(prefix="span010-startup-")\n'
        'WORK=Path(tmp.name);STARTED=time.time();DEADLINE=STARTED+10\n'
        'command=[sys.executable,"-c","print(123)"];env=None'),nbf.v4.new_code_cell(actual)])
    executed=NotebookClient(tiny,timeout=30,kernel_name='python3').execute()
    assert all(c.execution_count is not None for c in executed.cells)
    assert not any(o.output_type=='error' for c in executed.cells for o in c.get('outputs',[]))
    train=pd.read_csv(ROOT/'raw/train.csv',usecols=['id',*TEXT]);control,long=profile_indices(train)
    available=np.setdiff1d(np.arange(len(train)),np.r_[control,long])
    uniform=np.random.default_rng(43).choice(available,512,replace=False)
    atomic_json(pub/'profile_selection.json',dict(seed=43,uses_targets=False,
        control_ids=train.iloc[control].id.tolist(),uniform_ids=train.iloc[uniform].id.tolist(),
        long_ids=train.iloc[long].id.tolist(),train_csv_sha256=sha256(ROOT/'raw/train.csv'),local_quality_is_independent=False))
    sources=['GOAL.md','docs/SPAN_013_PLAN.md','configs/span_013.json','results/span_013/span_plan.json',
        'results/span_013/H60_cached_replay.json','src/arena/span010_order.py','src/arena/span013_order.py',
        'src/arena/span013_pipeline.py','src/arena/notebook_worker.py','src/arena/kaggle_control.py',
        'scripts/build_span_013_order.py','scripts/run_span_013_order.py','tests/test_span013_order.py']
    atomic_json(pub/(role+'_bundle.json'),dict(role=role,source_sha256={f:sha256(REPO/f) for f in sources},
        notebook_sha256=sha256(folder/'job.ipynb'),metadata_sha256=sha256(folder/'kernel-metadata.json'),
        public_cap_seconds=2400,hidden_cap_seconds=32400,internet_enabled=False,visible_test_sha256=visible,
        work_fraction=.1,row_fraction=.2,public_profile_selection_sha256=sha256(pub/'profile_selection.json')))
    atomic_json(pub/(role+'_cell_rehearsals.json'),dict(passed=True,rehearsals=rehearsals,
        mode_checks=mode_checks,notebook_engine_startup_passed=True,new_gpu_executions=0))
    print(folder)


if __name__=='__main__':main()
