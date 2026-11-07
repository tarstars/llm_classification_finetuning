"""Freeze the two-expert notebook and rehearse its serialized worker/branches."""
from pathlib import Path
import argparse
import hashlib
import json
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
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['learning','confirmation'],default='learning')
    args=parser.parse_args();stage=args.stage;role='calibration_'+stage
    root=ROOT/'runs/span_011';pub=REPO/'results/span_011'
    assert read_json(REPO/'backlog/hypotheses.json')['active_version']==11
    assert not (root/'jobs'/role/'push_intent.json').exists(),'Never rewrite accepted source'
    cohort=read_json(pub/'cohort_manifest.json');assets=read_json(pub/(stage+'_asset_upload.json'))
    assert cohort['complete'] and assets['ready'] and assets['private_verified'] and not assets['labels_included']
    temperature=1.
    if stage=='confirmation':
        frozen=read_json(pub/'frozen_temperature.json');assert frozen['learning_gate_passed'] and not frozen['confirmation_labels_inspected']
        temperature=frozen['temperature']
    folder=root/'notebooks'/role;folder.mkdir(parents=True,exist_ok=True)
    visible=sha256(ROOT/'raw/test.csv');driver=REPO/'src/arena/span011_calibration.py'
    cap=5400 if stage=='learning' else 7200
    cohort_sha=cohort['artifact_sha256'][stage+'.parquet']
    dataset_slug='llm-span011-'+stage+'-inputs'
    setup=f'''from pathlib import Path
import os,sys,time,json,shutil,subprocess,hashlib
STARTED=time.time();WORK=Path('/kaggle/working')
RAW=next(p for p in [Path('/kaggle/input/llm-classification-finetuning'),Path('/kaggle/input/competitions/llm-classification-finetuning')] if (p/'test.csv').exists())
IS_RERUN=os.environ.get('KAGGLE_IS_COMPETITION_RERUN','').strip().lower() in ['true','1'] or hashlib.sha256((RAW/'test.csv').read_bytes()).hexdigest()!={visible!r}
DEADLINE=STARTED+(32400 if IS_RERUN else {cap})-120
print('Runtime',sys.version,'hidden',IS_RERUN,flush=True)
assert sys.version_info[:2]==(3,10),'Pinned author image not honored'
ASSETS=WORK/'assets';ASSETS.mkdir(exist_ok=True)
for slug in ['lmsys-checkpoints-0-0805','lmsys-checkpoints-3-0805','lmsys-modules-0805','lmsys-packages']:
    source=next(p for p in [Path('/kaggle/input')/slug,Path('/kaggle/input/datasets/tascj0')/slug] if p.exists())
    (ASSETS/slug).symlink_to(source,target_is_directory=True)
shutil.copytree(ASSETS/'lmsys-modules-0805',WORK/'human_pref')
for wheel in ['triton-2.2.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl','xformers-0.0.24042abc8.d20240802-cp310-cp310-linux_x86_64.whl']:
    subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',str(ASSETS/'lmsys-packages'/wheel)],check=True,timeout=120)
(WORK/'calibration.py').write_text({driver.read_text()!r})
os.chdir(WORK)
command=[sys.executable,'-u',str(WORK/'calibration.py'),'--raw',str(RAW),'--assets',str(ASSETS),'--out',str(WORK/'calibration'),'--temperature',{str(temperature)!r}]
if IS_RERUN:
    command+=['--stage','deploy']
else:
    INPUT=next(p/{(stage+'.parquet')!r} for p in [Path('/kaggle/input')/{dataset_slug!r},Path('/kaggle/input/datasets/arseniytrushin')/{dataset_slug!r}] if (p/{(stage+'.parquet')!r}).exists())
    command+=['--stage',{stage!r},'--cohort',str(INPUT),'--cohort-sha256',{cohort_sha!r}]
env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
'''
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(
        '# V11: independent-source confidence calibration\n\nUnchanged tascj Gemma2-9B original and ArmoRM/Llama3-8B swapped, equal weights, 8192 packing. Source votes and identities are absent. Hidden mode performs actual inference and reads no calibration cohort or public prediction cache.'),
        nbf.v4.new_code_cell(setup),nbf.v4.new_code_cell(worker_cell(role,''))])
    nb.metadata.update(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python',version='3.10.13'))
    for cell in nb.cells:
        if cell.cell_type=='code':compile(cell.source,'span011-serialized-cell','exec')
    nbf.write(nb,folder/'job.ipynb')
    metadata=dict(id='arseniytrushin/llm-preference-span011-'+stage+'-calibration',
        title='LLM Preference Span011 '+stage.title()+' Calibration',code_file='job.ipynb',language='python',
        kernel_type='notebook',is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=False,
        dataset_sources=['tascj0/'+slug+'/1' for slug in ['lmsys-checkpoints-0-0805','lmsys-checkpoints-3-0805',
            'lmsys-modules-0805','lmsys-packages']]+[assets['dataset']],model_sources=[],kernel_sources=[],
        competition_sources=['llm-classification-finetuning'],
        docker_image='gcr.io/kaggle-private-byod/python@sha256:e3f9bf91d1b7b36bf77554565d27c5733585fbe5e2b7f0832e068824e7accc3c',
        docker_image_pinning_type='original',machine_shape='NvidiaTeslaT4')
    atomic_json(folder/'kernel-metadata.json',metadata)
    actual=nbf.read(folder/'job.ipynb',4).cells[-1].source;checks=[]
    for name,program,limit,success in [('success','print(123)',10,True),('failure','raise SystemExit(7)',10,False),
                                        ('watchdog','import time;time.sleep(10)',.2,False)]:
        with tempfile.TemporaryDirectory(prefix='span011-cell-') as tmp:
            scope=dict(WORK=Path(tmp),STARTED=time.time(),DEADLINE=time.time()+limit,
                       command=[sys.executable,'-c',program],env=None);failed=False
            try:exec(compile(actual,'actual-serialized-last-cell','exec'),scope)
            except AssertionError:failed=True
            record=read_json(Path(tmp)/'session_summary.json')
            assert failed!=success and (record['exit_code']==0)==success
            checks.append(dict(name=name,passed=True,summary=record))
    # Execute the exact serialized mode and input branch; hidden must not resolve INPUT.
    lines=setup.splitlines();branch=setup.split('if IS_RERUN:\n',1)[1].split('\nenv=',1)[0]
    mode_source='\n'.join(line for line in lines if line.startswith(('IS_RERUN=','DEADLINE=','command=')))
    mode_source+='\nif IS_RERUN:\n'+branch
    modes=[]
    for flag,observed_hash,expected in [('',visible,False),('','changed',True),('true',visible,True)]:
        with tempfile.TemporaryDirectory(prefix='span011-branch-') as tmp:
            class FakePath:
                def __new__(cls,value):
                    return Path(tmp) if value.startswith('/kaggle/input') else Path(value)
            (Path(tmp)/dataset_slug).mkdir()
            (Path(tmp)/dataset_slug/(stage+'.parquet')).write_bytes(b'fixture')
            scope=dict(os=SimpleNamespace(environ={'KAGGLE_IS_COMPETITION_RERUN':flag}),
                hashlib=SimpleNamespace(sha256=lambda data:SimpleNamespace(hexdigest=lambda:observed_hash)),
                RAW=ROOT/'raw',WORK=Path('/tmp'),STARTED=0,ASSETS=Path('/tmp/assets'),sys=sys,Path=FakePath)
            exec(compile(mode_source,'serialized-mode-input-branch','exec'),scope)
            assert scope['IS_RERUN']==expected and ('--cohort' in scope['command'])!=expected
            assert ('INPUT' in scope)!=expected
            assert scope['DEADLINE']==(32400 if expected else cap)-120
            modes.append(dict(hidden=expected,passed=True))
    tiny=nbf.v4.new_notebook(metadata=nb.metadata,cells=[nbf.v4.new_code_cell(
        'from pathlib import Path\nimport sys,time,tempfile\n'
        'tmp=tempfile.TemporaryDirectory(prefix="span011-startup-")\n'
        'WORK=Path(tmp.name);STARTED=time.time();DEADLINE=STARTED+10\n'
        'command=[sys.executable,"-c","print(123)"];env=None'),nbf.v4.new_code_cell(actual)])
    executed=NotebookClient(tiny,timeout=30,kernel_name='python3').execute()
    assert all(cell.execution_count is not None for cell in executed.cells)
    assert not any(o.output_type=='error' for c in executed.cells for o in c.get('outputs',[]))
    sources=['GOAL.md','docs/SPAN_011_PLAN.md','src/arena/span011_calibration.py','src/arena/notebook_worker.py',
        'src/arena/kaggle_control.py','scripts/build_span_011_calibration.py','scripts/run_span_011_calibration.py',
        'scripts/prepare_span_011_assets.py','tests/test_span011_calibration.py','tests/test_span011_data.py',
        'src/arena/span011_data.py','scripts/prepare_span_011_cohort.py']
    atomic_json(pub/(role+'_bundle.json'),dict(role=role,source_sha256={s:sha256(REPO/s) for s in sources},
        notebook_sha256=sha256(folder/'job.ipynb'),metadata_sha256=sha256(folder/'kernel-metadata.json'),
        public_cap_seconds=cap,hidden_cap_seconds=32400,visible_test_sha256=visible,
        cohort_sha256=cohort_sha,temperature=temperature,internet_enabled=False,stage=stage,
        frozen_temperature_sha256=sha256(pub/'frozen_temperature.json') if stage=='confirmation' else None))
    atomic_json(pub/(role+'_cell_rehearsals.json'),dict(passed=True,rehearsals=checks,mode_checks=modes,
        notebook_engine_startup_passed=True,last_cell_sha256=hashlib.sha256(actual.encode()).hexdigest(),new_gpu_executions=0))
    print(folder)


if __name__=='__main__':main()
