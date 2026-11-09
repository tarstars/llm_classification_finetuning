"""Build/rehearse the actual two-device native information notebook."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch
import nbformat as nbf
from nbclient import NotebookClient

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,sha256
from arena.notebook_worker import worker_cell

FAKE_WORKER='''import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser()
for k in ['assets','out','orientation','device']:p.add_argument('--'+k)
a=p.parse_args();o=Path(a.out)
(o/('worker_'+a.orientation+'.json')).write_text(json.dumps(dict(passed=True,rows=1024)))
np.savez_compressed(o/(a.orientation+'.npz'),ids=np.arange(1024),logits=np.zeros(1024))
'''


def rehearse(actual):
    checks=[]
    for flag in ['', 'false','0']:
        with patch.dict(os.environ,{'KAGGLE_IS_COMPETITION_RERUN':flag}):exec(actual.cells[1].source.split('# MODE_END')[0],{})
    for flag in ['true','1','unexpected']:
        with patch.dict(os.environ,{'KAGGLE_IS_COMPETITION_RERUN':flag}):
            try:exec(actual.cells[1].source.split('# MODE_END')[0],{})
            except AssertionError:pass
            else:raise AssertionError('Deployment mode allowed')
    with tempfile.TemporaryDirectory(prefix='span036-stage-') as tmp:
        work=Path(tmp);scope=dict(WORK=work,ASSETS=work/'assets',os=os,sys=sys)
        exec(actual.cells[2].source,scope)
        subprocess.run([sys.executable,'-c','import pairrm_native,pairrm_worker,pairrm_pilot'],cwd=work,check=True,timeout=30)
        (work/'pairrm_worker.py').write_text(FAKE_WORKER)
        scope.update(STARTED=time.time(),DEADLINE=time.time()+30)
        exec(actual.cells[3].source,scope)
        assert read_json(work/'pairrm_audit/pilot_complete.json')['complete']
        assert not (work/'submission.csv').exists()
        checks.append(dict(name='actual_serialized_staging_two_process_pipeline_final',passed=True,fake_cpu_forward=True))
    for name,program,limit in [('failure','raise SystemExit(7)',10),('watchdog','import time;time.sleep(10)',.2)]:
        with tempfile.TemporaryDirectory(prefix='span036-worker-') as tmp:
            scope=dict(WORK=Path(tmp),STARTED=time.time(),DEADLINE=time.time()+limit,command=[sys.executable,'-c',program],env=None)
            try:exec(actual.cells[3].source,scope)
            except AssertionError:pass
            else:raise AssertionError('Failed worker accepted')
            assert read_json(Path(tmp)/'session_summary.json')['exit_code']!=0
            checks.append(dict(name=name,passed=True))
    tiny=nbf.v4.new_notebook(metadata=actual.metadata,cells=[nbf.v4.new_code_cell('from pathlib import Path\nimport sys,time,tempfile\ntmp=tempfile.TemporaryDirectory(prefix="span036-jupyter-")\nWORK=Path(tmp.name);STARTED=time.time();DEADLINE=STARTED+10\ncommand=[sys.executable,"-c","print(123)"];env=None'),nbf.v4.new_code_cell(actual.cells[3].source)])
    executed=NotebookClient(tiny,timeout=30,kernel_name='python3').execute()
    assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.get('outputs',[])) for c in executed.cells)
    return checks


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--rehearse-only",action="store_true");args=parser.parse_args()
    run,pub=ROOT/'runs/span_036',REPO/'results/span_036';role='pairrm_pilot'
    if not args.rehearse_only:assert read_json(pub/'input_asset_ready.json')['ready']
    assert read_json(pub/'checkpoint_cpu_check.json')['passed']
    assert not (run/'jobs'/role/'push_intent.json').exists()
    folder=run/'notebooks'/role;folder.mkdir(parents=True,exist_ok=True)
    files={'pairrm_native.py':'src/arena/span036_pairrm.py','pairrm_worker.py':'src/arena/span036_worker.py','pairrm_pilot.py':'src/arena/span036_pilot.py'}
    expected=read_json(pub/'asset_preparation.json')['files_sha256']
    expected={n:h for n,h in expected.items() if n not in ['dataset-metadata.json','README.md']}
    setup=f'''from pathlib import Path
import os,sys,time,json,hashlib,subprocess
STARTED=time.time();WORK=Path('/kaggle/working');DEADLINE=STARTED+2700-120
FLAG=os.environ.get('KAGGLE_IS_COMPETITION_RERUN','').strip().lower()
assert FLAG in ['', 'false', '0'],'Information pilot only'
# MODE_END
assert sys.version_info[:2]==(3,10)
ASSETS=next(p for p in [Path('/kaggle/input/llm-span036-pairrm-inputs'),Path('/kaggle/input/datasets/arseniytrushin/llm-span036-pairrm-inputs')] if (p/'model.safetensors').exists())
for name,digest in {expected!r}.items():assert hashlib.sha256((ASSETS/name).read_bytes()).hexdigest()==digest,name
gpus=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True).strip().splitlines()
assert len(gpus)==2 and all('T4' in x for x in gpus)
(WORK/'hardware.json').write_text(json.dumps(dict(gpus=gpus,python=sys.version,mode='information_only')))
os.chdir(WORK)
'''
    stage=''.join(f'(WORK/{name!r}).write_text({(REPO/path).read_text()!r})\n' for name,path in files.items())
    stage+='''command=[sys.executable,'-u',str(WORK/'pairrm_pilot.py'),'--assets',str(ASSETS),'--out',str(WORK/'pairrm_audit')]
env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
'''
    final=worker_cell(role,'')
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell('''# H79 joint PairRM information pilot

One pinned native comparator, both answer orders on all1024fixed later rows.
No targets, hidden inference or submission. Literal authored model/formatting;
strict full checkpoint and CPU token references. One T4perorientation, repeat
canonical checks and memory guards. Source information must precede separately
qualified incumbent transfer; no official eligibility follows from this notebook.'''),nbf.v4.new_code_cell(setup),nbf.v4.new_code_cell(stage),nbf.v4.new_code_cell(final)])
    nb.metadata.update(kernelspec=dict(name='python3',display_name='Python 3',language='python'),language_info=dict(name='python',version='3.10.13'))
    for c in nb.cells:
        if c.cell_type=='code':compile(c.source,'span036-serialized','exec')
    path=folder/'job.ipynb'
    if not args.rehearse_only and path.exists():
        actual=nbf.read(path,4)
        assert [(c.cell_type,c.source) for c in actual.cells]==[(c.cell_type,c.source) for c in nb.cells]
        assert actual.metadata==nb.metadata
    else:
        nbf.write(nb,path);actual=nbf.read(path,4)
    receipt=pub/(role+'_cell_rehearsals.json')
    if args.rehearse_only:
        checks=rehearse(actual)
        atomic_json(receipt,dict(passed=True,rehearsals=checks,notebook_sha256=sha256(folder/'job.ipynb'),
            actual_serialized_staging_pipeline_final_passed=True,standalone_imports_passed=True,
            mode_guards_passed=True,notebook_engine_startup_passed=True,new_gpu_executions=0))
        print(dict(passed=True,stage='rehearsals_only'));return
    checked=read_json(receipt)
    assert checked['passed'] and checked['notebook_sha256']==sha256(folder/'job.ipynb'),'Rehearse these exact cells first'
    metadata=read_json(ROOT/'runs/span_035/notebooks/recent_pilot/kernel-metadata.json')
    metadata.update(id='arseniytrushin/llm-preference-span036-pairrm-pilot',title='LLM Preference Span036 PairRM Pilot',dataset_sources=['arseniytrushin/llm-span036-pairrm-inputs/1'],competition_sources=[])
    atomic_json(folder/'kernel-metadata.json',metadata)
    names=list(files.values())+['GOAL.md','docs/SPAN_036_PLAN.md','docs/SPAN_036_INFORMATION_CONTRACT.md','configs/span_036.json',
        'scripts/build_span_036_pilot.py','scripts/run_span_036_pilot.py','scripts/verify_span_036_pilot.py','scripts/analyze_span_036_pairrm.py',
        'scripts/check_span_036_checkpoint.py','scripts/prepare_span_036_assets.py','scripts/audit_span_036_sources.py',
        'src/arena/span034_residual.py','src/arena/span_heads.py','src/arena/notebook_worker.py','src/arena/kaggle_control.py','src/arena/span031_inventory.py',
        'results/span_036/preparation.json','results/span_036/asset_preparation.json','results/span_036/input_asset_ready.json','results/span_036/checkpoint_cpu_check.json']
    size=len(json.dumps(dict(source=(folder/'job.ipynb').read_text())).encode());assert size<900000
    atomic_json(pub/(role+'_bundle.json'),dict(role=role,source_sha256={n:sha256(REPO/n) for n in names},notebook_sha256=sha256(folder/'job.ipynb'),metadata_sha256=sha256(folder/'kernel-metadata.json'),public_cap_seconds=2700,information_only=True,official_submission_eligible=False,sdk_payload_bytes=size))
    print(dict(passed=True,sdk_payload_bytes=size,folder=str(folder)))


if __name__=='__main__':main()
