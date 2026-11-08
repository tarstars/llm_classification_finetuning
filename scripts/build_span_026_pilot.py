"""Build one information-only notebook and exercise its actual final cell."""
from pathlib import Path
import json
import base64
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
    role='context_pilot';root=ROOT/'runs/span_026';pub=REPO/'results/span_026'
    assert read_json(pub/'pilot_preparation.json')['prepared']
    assert not (root/'jobs'/role/'push_intent.json').exists()
    folder=root/'notebooks'/role;folder.mkdir(parents=True,exist_ok=True)
    engine=REPO/'src/arena/span010_order.py'
    assert sha256(engine)==read_json(REPO/'results/span_013/gemma_order_bundle.json')['source_sha256']['src/arena/span010_order.py']
    selection=(root/'pilot_spec.json').read_text()
    setup=f'''from pathlib import Path
import os,sys,time,json,shutil,subprocess
STARTED=time.time();WORK=Path('/kaggle/working');DEADLINE=STARTED+7200-120
assert os.environ.get('KAGGLE_IS_COMPETITION_RERUN','').strip().lower() not in ['true','1'],'Information only; not a competition deployment'
assert sys.version_info[:2]==(3,10),'Pinned author image not honored'
RAW=next(p for p in [Path('/kaggle/input/llm-classification-finetuning'),Path('/kaggle/input/competitions/llm-classification-finetuning')] if (p/'train.csv').exists())
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
(WORK/'frozen_inference.py').write_text({engine.read_text()!r})
(WORK/'context_pilot.py').write_text({(REPO/'src/arena/span026_pilot.py').read_text()!r})
(WORK/'pilot_spec.json').write_text({selection!r})
os.chdir(WORK)
command=[sys.executable,'-u',str(WORK/'context_pilot.py'),'--raw',str(RAW),'--assets',str(ASSETS),'--out',str(WORK/'context_audit'),'--engine',str(WORK/'context_engine.py'),'--spec',str(WORK/'pilot_spec.json')]
env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
'''
    files={'context_engine.py':'src/arena/span026_engine.py','context_checks.py':'src/arena/span026_checks.py',
        'context_runtime.py':'src/arena/span026_runtime.py','context_coverage.py':'src/arena/span026_context.py'}
    setup+="\nimport base64\n(WORK/'controls').mkdir(exist_ok=True)\n"
    for name,source in files.items():setup+=f"(WORK/{name!r}).write_text({(REPO/source).read_text()!r})\n"
    for path in sorted((root/'controls').glob('*.npz')):
        setup+=f"(WORK/'controls'/{path.name!r}).write_bytes(base64.b64decode({base64.b64encode(path.read_bytes()).decode()!r}))\n"
    # Exercise imports of exactly the standalone file names shipped to Kaggle.
    with tempfile.TemporaryDirectory(prefix='span026-import-') as tmp:
        for name,source in {**files,'frozen_inference.py':'src/arena/span010_order.py',
                'context_pilot.py':'src/arena/span026_pilot.py'}.items():
            (Path(tmp)/name).write_bytes((REPO/source).read_bytes())
        subprocess.run([sys.executable,'-c','import context_engine,context_checks,context_runtime,context_pilot'],cwd=tmp,check=True,timeout=30)
    final=worker_cell(role,'')
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(
        '# H72 context information pilot\n\nOne native6k original-Gemma pass on357 changed rows, keeping H60route/components. Early engineering stop before larger diagnostic scoring. Training-overlapping original votes; no target labels enter inference. Not a competition deployment.'),
        nbf.v4.new_code_cell(setup),nbf.v4.new_code_cell(final)])
    nb.metadata.update(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        language_info=dict(name='python',version='3.10.13'))
    for cell in nb.cells:
        if cell.cell_type=='code':compile(cell.source,'span026-actual-cell','exec')
    nbf.write(nb,folder/'job.ipynb')
    image='gcr.io/kaggle-private-byod/python@sha256:e3f9bf91d1b7b36bf77554565d27c5733585fbe5e2b7f0832e068824e7accc3c'
    metadata=dict(id='arseniytrushin/llm-preference-span026-context-audit',title='LLM Preference Span026 Context Audit',
        code_file='job.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,
        enable_tpu=False,enable_internet=False,dataset_sources=['tascj0/'+s+'/1' for s in [
            'lmsys-checkpoints-0-0805','lmsys-checkpoints-3-0805','lmsys-modules-0805','lmsys-packages']],
        model_sources=[],kernel_sources=[],competition_sources=['llm-classification-finetuning'],
        docker_image=image,docker_image_pinning_type='original',machine_shape='NvidiaTeslaT4')
    atomic_json(folder/'kernel-metadata.json',metadata)
    rehearsals=[]
    for name,program,limit,success in [('success','print(123)',10,True),
            ('failure','raise SystemExit(7)',10,False),('watchdog','import time;time.sleep(10)',.2,False)]:
        with tempfile.TemporaryDirectory(prefix='span026-cell-') as tmp:
            scope=dict(WORK=Path(tmp),STARTED=time.time(),DEADLINE=time.time()+limit,
                command=[sys.executable,'-c',program],env=None)
            failed=False
            try:exec(compile(final,'actual-final-cell','exec'),scope)
            except AssertionError:failed=True
            record=read_json(Path(tmp)/'session_summary.json')
            assert failed!=success and (record['exit_code']==0)==success
            rehearsals.append(dict(name=name,passed=True,summary=record))
    tiny=nbf.v4.new_notebook(metadata=nb.metadata,cells=[nbf.v4.new_code_cell(
        'from pathlib import Path\nimport sys,time,tempfile\n'
        'tmp=tempfile.TemporaryDirectory(prefix="span026-startup-")\n'
        'WORK=Path(tmp.name);STARTED=time.time();DEADLINE=STARTED+10\n'
        'command=[sys.executable,"-c","print(123)"];env=None'),nbf.v4.new_code_cell(final)])
    executed=NotebookClient(tiny,timeout=30,kernel_name='python3').execute()
    assert all(c.execution_count is not None for c in executed.cells)
    assert not any(o.output_type=='error' for c in executed.cells for o in c.get('outputs',[]))
    names=['GOAL.md','docs/SPAN_026_PLAN.md','docs/SPAN_026_EVALUATION_AMENDMENT.md',
        'docs/SPAN_026_GPU_PROTOCOL.md','configs/span_026.json','results/span_026/span_plan.json',
        'results/span_026/pilot_preparation.json','results/span_026/coverage_audit.json','results/span_026/selector_audit.json',
        'src/arena/span010_order.py','src/arena/span026_engine.py','src/arena/span026_checks.py',
        'src/arena/span026_runtime.py','src/arena/span026_context.py','src/arena/span026_pilot.py','src/arena/span026_quality.py',
        'src/arena/span019_quality.py','src/arena/notebook_worker.py','src/arena/kaggle_control.py',
        'scripts/prepare_span_026_pilot.py','scripts/build_span_026_pilot.py','scripts/run_span_026_pilot.py',
        'scripts/audit_span_026_pilot.py','tests/test_span026_context.py','tests/test_span026_runtime.py',
        'tests/test_span026_pilot.py','tests/test_span026_quality.py']
    atomic_json(pub/(role+'_bundle.json'),dict(role=role,source_sha256={n:sha256(REPO/n) for n in names},
        notebook_sha256=sha256(folder/'job.ipynb'),metadata_sha256=sha256(folder/'kernel-metadata.json'),
        public_cap_seconds=7200,information_only=True,official_submission_eligible=False,
        pilot_spec_sha256=sha256(root/'pilot_spec.json'),frozen_parent_engine_sha256=sha256(engine)))
    atomic_json(pub/(role+'_cell_rehearsals.json'),dict(passed=True,rehearsals=rehearsals,
        notebook_engine_startup_passed=True,standalone_worker_imports_passed=True,new_gpu_executions=0))
    print(folder)


if __name__=='__main__':main()
