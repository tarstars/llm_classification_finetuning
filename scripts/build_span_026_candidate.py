"""Freeze actual candidate qualification and the identical hidden inference path."""
from pathlib import Path
import json
import subprocess
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
    role='context_qualification';root=ROOT/'runs/span_026';pub=REPO/'results/span_026'
    pilot=read_json(pub/'pilot_audit.json')
    assert pilot['complete_information'] and pilot['engineering_screen_passed'] and pilot['gross_regression_guard_passed']
    preparation=read_json(pub/'candidate_preparation.json');assert preparation['prepared']
    assert sha256(root/'candidate_spec.json')==preparation['spec_sha256']
    assert sha256(root/'hidden_asset_spec.json')==preparation['hidden_spec_sha256']
    assert not (root/'jobs'/role/'push_intent.json').exists(),'Never rewrite accepted source'
    folder=root/'notebooks'/role;folder.mkdir(parents=True,exist_ok=True)
    visible=sha256(ROOT/'raw/test.csv')
    files={'context_candidate.py':'src/arena/span026_candidate.py',
        'candidate_engine.py':'src/arena/span026_candidate_engine.py','context_engine.py':'src/arena/span026_engine.py',
        'context_checks.py':'src/arena/span026_checks.py','context_runtime.py':'src/arena/span026_runtime.py',
        'context_coverage.py':'src/arena/span026_context.py','context_pilot.py':'src/arena/span026_pilot.py',
        'frozen_inference.py':'src/arena/span010_order.py','gemma_order.py':'src/arena/span013_order.py'}
    setup=f'''from pathlib import Path
import os,sys,time,json,shutil,subprocess,hashlib
STARTED=time.time();WORK=Path('/kaggle/working')
RAW=next(p for p in [Path('/kaggle/input/llm-classification-finetuning'),Path('/kaggle/input/competitions/llm-classification-finetuning')] if (p/'test.csv').exists())
IS_RERUN=os.environ.get('KAGGLE_IS_COMPETITION_RERUN','').strip().lower() in ['true','1'] or hashlib.sha256((RAW/'test.csv').read_bytes()).hexdigest()!={visible!r}
DEADLINE=STARTED+(32400 if IS_RERUN else 3600)-120
assert sys.version_info[:2]==(3,10),'Pinned author image not honored'
GPU_INFO=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True).strip().splitlines()
assert len(GPU_INFO)==2 and all('T4' in s for s in GPU_INFO)
(WORK/'hardware.json').write_text(json.dumps(dict(gpus=GPU_INFO,python=sys.version)))
ASSETS=WORK/'assets';ASSETS.mkdir(exist_ok=True)
for slug in ['lmsys-checkpoints-0-0805','lmsys-checkpoints-3-0805','lmsys-modules-0805','lmsys-packages']:
 source=next(p for p in [Path('/kaggle/input')/slug,Path('/kaggle/input/datasets/tascj0')/slug] if p.exists())
 (ASSETS/slug).symlink_to(source,target_is_directory=True)
shutil.copytree(ASSETS/'lmsys-modules-0805',WORK/'human_pref')
for wheel in ['triton-2.2.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl','xformers-0.0.24042abc8.d20240802-cp310-cp310-linux_x86_64.whl']:
 subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',str(ASSETS/'lmsys-packages'/wheel)],check=True,timeout=120)
SPEC_TEXT={((root/'hidden_asset_spec.json').read_text())!r} if IS_RERUN else {((root/'candidate_spec.json').read_text())!r}
(WORK/'candidate_spec.json').write_text(SPEC_TEXT)
os.chdir(WORK)
command=[sys.executable,'-u',str(WORK/'context_candidate.py'),'--raw',str(RAW),'--assets',str(ASSETS),'--out',str(WORK/'context_candidate'),'--engine',str(WORK/'candidate_engine.py'),'--spec',str(WORK/'candidate_spec.json')]+([] if IS_RERUN else ['--qualify'])
env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
'''
    for name,source in files.items():setup+=f"(WORK/{name!r}).write_text({(REPO/source).read_text()!r})\n"
    with tempfile.TemporaryDirectory(prefix='span026-candidate-import-') as tmp:
        for name,source in files.items():(Path(tmp)/name).write_bytes((REPO/source).read_bytes())
        subprocess.run([sys.executable,'-c','import context_candidate,candidate_engine'],cwd=tmp,check=True,timeout=30)
    final=worker_cell(role,'')
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(
        '# H72 fixed context candidate\n\nOriginal H60 route and weights. An extra original-Gemma6144 pass replaces that component only on actually truncated inputs. No public prediction cache; hidden mode performs actual text inference. Public local scores overlap training.'),
        nbf.v4.new_code_cell(setup),nbf.v4.new_code_cell(final)])
    nb.metadata.update(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python',version='3.10.13'))
    for c in nb.cells:
        if c.cell_type=='code':compile(c.source,'actual-context-candidate-cell','exec')
    nbf.write(nb,folder/'job.ipynb')
    metadata=read_json(root/'notebooks/context_pilot/kernel-metadata.json')
    metadata.update(id='arseniytrushin/llm-preference-span026-context-candidate',title='LLM Preference Span026 Context Candidate')
    atomic_json(folder/'kernel-metadata.json',metadata)
    checks=[]
    for name,program,limit,success in [('success','print(123)',10,True),('failure','raise SystemExit(7)',10,False),('watchdog','import time;time.sleep(10)',.2,False)]:
        with tempfile.TemporaryDirectory(prefix='span026-candidate-cell-') as tmp:
            scope=dict(WORK=Path(tmp),STARTED=time.time(),DEADLINE=time.time()+limit,command=[sys.executable,'-c',program],env=None)
            failed=False
            try:exec(compile(final,'actual-final-cell','exec'),scope)
            except AssertionError:failed=True
            record=read_json(Path(tmp)/'session_summary.json')
            assert failed!=success and (record['exit_code']==0)==success
            checks.append(dict(name=name,passed=True,summary=record))
    modes='\n'.join(line for line in setup.splitlines() if line.startswith(('IS_RERUN=','DEADLINE=','command=','SPEC_TEXT=')))
    mode_checks=[]
    for flag,observed_hash,hidden in [('',visible,False),('','changed',True),('true',visible,True)]:
        scope=dict(os=SimpleNamespace(environ={'KAGGLE_IS_COMPETITION_RERUN':flag}),
            hashlib=SimpleNamespace(sha256=lambda data:SimpleNamespace(hexdigest=lambda:observed_hash)),
            RAW=ROOT/'raw',WORK=Path('/tmp'),STARTED=0,ASSETS=Path('/tmp/assets'),sys=sys)
        exec(compile(modes,'actual-mode-branch','exec'),scope)
        assert scope['IS_RERUN']==hidden and ('--qualify' in scope['command'])!=hidden
        assert scope['DEADLINE']==(32400 if hidden else 3600)-120
        selected_spec=json.loads(scope['SPEC_TEXT']);assert selected_spec['qualify']!=hidden
        if hidden:assert set(selected_spec)=={'qualify','uses_targets','checkpoint_references'}
        mode_checks.append(dict(hidden=hidden,passed=True))
    tiny=nbf.v4.new_notebook(metadata=nb.metadata,cells=[nbf.v4.new_code_cell(
        'from pathlib import Path\nimport sys,time,tempfile\n'
        'tmp=tempfile.TemporaryDirectory(prefix="span026-candidate-startup-")\n'
        'WORK=Path(tmp.name);STARTED=time.time();DEADLINE=STARTED+10\n'
        'command=[sys.executable,"-c","print(123)"];env=None'),nbf.v4.new_code_cell(final)])
    executed=NotebookClient(tiny,timeout=30,kernel_name='python3').execute()
    assert all(c.execution_count is not None for c in executed.cells)
    assert not any(o.output_type=='error' for c in executed.cells for o in c.get('outputs',[]))
    sources=list(files.values())+['GOAL.md','docs/SPAN_026_PLAN.md','docs/SPAN_026_EVALUATION_AMENDMENT.md',
        'docs/SPAN_026_GPU_PROTOCOL.md','docs/SPAN_026_QUALIFICATION.md','configs/span_026.json',
        'results/span_026/span_plan.json','results/span_026/pilot_audit.json','results/span_026/candidate_preparation.json',
        'results/span_026/pilot_preparation.json','results/span_026/context_pilot_bundle.json',
        'results/span_026/jobs/context_pilot/retrieval.json',
        'results/span_013/gemma_order_qualification.json','results/span_013/jobs/gemma_order/retrieval.json',
        'src/arena/span026_qualification.py','src/arena/notebook_worker.py','src/arena/kaggle_control.py',
        'scripts/prepare_span_026_candidate.py','scripts/build_span_026_candidate.py','scripts/run_span_026_candidate.py',
        'scripts/audit_span_026_candidate.py','scripts/submit_span_026_candidate.py',
        'tests/test_span026_candidate.py','tests/test_span026_submission.py']
    atomic_json(pub/(role+'_bundle.json'),dict(role=role,source_sha256={n:sha256(REPO/n) for n in sources},
        notebook_sha256=sha256(folder/'job.ipynb'),metadata_sha256=sha256(folder/'kernel-metadata.json'),
        public_cap_seconds=3600,hidden_cap_seconds=32400,pilot_audit_sha256=sha256(pub/'pilot_audit.json'),
        candidate_spec_sha256=sha256(root/'candidate_spec.json'),hidden_spec_sha256=sha256(root/'hidden_asset_spec.json'),
        visible_test_sha256=visible,public_prediction_cache_embedded=False,information_only=False))
    atomic_json(pub/(role+'_cell_rehearsals.json'),dict(passed=True,rehearsals=checks,mode_checks=mode_checks,
        standalone_imports_passed=True,notebook_engine_startup_passed=True,new_gpu_executions=0))
    print(folder)


if __name__=='__main__':main()
