"""Build and rehearse the single offline H02 notebook; never launch it."""
from pathlib import Path
import base64
import gzip
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
from arena.experiment import atomic_json,read_json,sha256
from arena.notebook_worker import worker_cell


def main():
    root=ROOT/'runs/span_031';pub=REPO/'results/span_031';role='size_pilot'
    assert not (root/'jobs'/role/'push_intent.json').exists()
    assert not (pub/(role+'_bundle.json')).exists(),'Preserve the first frozen build'
    cfg=read_json(REPO/'configs/span_031.json');prep=read_json(pub/'pilot_preparation.json')
    assert prep['prepared'] and prep['protocol_revision']==2
    assert sha256(root/'pilot_rows.csv.gz')==prep['compressed_csv_sha256']
    parents=read_json(root/'parent_sources.json');owner,slug=parents['runtime'].rsplit('/',1)[0].split('/')
    adapter_owner,adapter_slug=parents['tokenizer'].rsplit('/',1)[0].split('/')
    payload=base64.b64encode((root/'pilot_rows.csv.gz').read_bytes()).decode()
    setup=f'''import os,sys,time,json,base64,gzip,subprocess,hashlib
from pathlib import Path
STARTED=time.time();DEADLINE=STARTED+5400-120
WORK=Path(os.environ.get('ARENA_WORK_ROOT','/kaggle/working'));INPUT=Path(os.environ.get('ARENA_INPUT_ROOT','/kaggle/input'))
WORK.mkdir(parents=True,exist_ok=True)
os.environ.update(TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
TRAIN=next(p for p in [INPUT/{slug!r},INPUT/'notebooks'/{owner!r}/{slug!r}] if (p/'runtime_requirements.json').exists())
PARENT=next(p for p in [INPUT/{adapter_slug!r},INPUT/'notebooks'/{adapter_owner!r}/{adapter_slug!r}] if (p/'exposure/adapter/tokenizer.json').exists())
runtime=json.loads((TRAIN/'runtime_requirements.json').read_text())
subprocess.run([sys.executable,'-m','pip','install','-q','--no-index','--no-deps','--find-links',str(TRAIN/'runtime_wheels'),*runtime],check=True,timeout=240)
MODULES=WORK/'bundled_code';(MODULES/'arena').mkdir(parents=True,exist_ok=True);(MODULES/'arena/__init__.py').write_text('')
(WORK/'config.json').write_text({json.dumps(cfg)!r})
(WORK/'base_config.json').write_text({(root/'base_config.json').read_text()!r})
(WORK/'selection.json').write_text({(root/'pilot_ids.json').read_text()!r})
packed=base64.b64decode({payload!r});assert hashlib.sha256(packed).hexdigest()=={prep['compressed_csv_sha256']!r}
RAW=WORK/'pilot_raw';RAW.mkdir(exist_ok=True);(RAW/'train.csv').write_bytes(gzip.decompress(packed))
assert (RAW/'train.csv').stat().st_size=={prep['csv_bytes']}
'''
    modules=['data','joint_data','joint_train','training_exposure','experiment','telemetry','span006_inputs','span031_size','span031_size_session']
    stage=''.join(f"(MODULES/'arena/{n}.py').write_text({(REPO/('src/arena/'+n+'.py')).read_text()!r})\n" for n in modules)
    stage+=f"(MODULES/'test_size_control.py').write_text({(REPO/'tests/test_span031_size.py').read_text()!r})\n"
    stage+='''def locate(size):
 return next(p for p in [INPUT/'models/qwen-lm/qwen-3/transformers'/size/'1',INPUT/'qwen-3/transformers'/size/'1'] if (p/'config.json').exists())
env=os.environ.copy();env.update(PYTHONPATH=str(MODULES),PYTHONUNBUFFERED='1')
command=[sys.executable,'-m','arena.span031_size_session','--raw',str(RAW),'--small-model',str(locate('4b')),'--large-model',str(locate('8b')),
 '--tokenizer-reference',str(PARENT/'exposure/adapter'),'--selection',str(WORK/'selection.json'),'--config',str(WORK/'config.json'),
 '--base-config',str(WORK/'base_config.json'),'--out',str(WORK/'size_pilot'),'--deadline',str(DEADLINE)]
'''
    runtime_check="subprocess.run([sys.executable,str(MODULES/'test_size_control.py')],env=env,check=True,timeout=90)\n"
    final=worker_cell(role,'')
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell('''# V31: fresh Qwen3 4B / 8B capacity pilot

Same fixed1024training/512development rows,32accepted updates per arm.
Limited early-learning and throughput study; no full-training or official-score
forecast. No confirmation votes. One5400-second offline two-T4 version.'''),
        nbf.v4.new_code_cell(setup),nbf.v4.new_code_cell(stage),nbf.v4.new_code_cell(runtime_check),nbf.v4.new_code_cell(final)])
    nb.metadata['kernelspec']=dict(name='python3',display_name='Python 3',language='python')
    for cell in nb.cells:
        if cell.cell_type=='code':compile(cell.source,'span031-serialized-cell','exec')
    folder=root/'notebooks'/role;folder.mkdir(parents=True,exist_ok=True)
    nbf.write(nb,folder/'job.ipynb');assert (folder/'job.ipynb').stat().st_size<16*1024**2
    actual=nbf.read(folder/'job.ipynb',4);checks=[]
    with tempfile.TemporaryDirectory(prefix='span031-staging-') as tmp:
        work=Path(tmp)/'work';inputs=Path(tmp)/'inputs'
        train=inputs/slug;train.mkdir(parents=True);(train/'runtime_requirements.json').write_text('[]')
        adapter=inputs/adapter_slug/'exposure/adapter';adapter.mkdir(parents=True);(adapter/'tokenizer.json').write_text('{}')
        for size in ['4b','8b']:
            model=inputs/'models/qwen-lm/qwen-3/transformers'/size/'1';model.mkdir(parents=True);(model/'config.json').write_text('{}')
        scope={}
        with patch.dict(os.environ,{'ARENA_WORK_ROOT':str(work),'ARENA_INPUT_ROOT':str(inputs)}),patch('subprocess.run'):
            exec(actual.cells[1].source,scope)
        exec(actual.cells[2].source,scope)
        assert read_json(work/'selection.json')==read_json(root/'pilot_ids.json')
        assert (work/'pilot_raw/train.csv').read_bytes()==gzip.decompress((root/'pilot_rows.csv.gz').read_bytes())
        # Real imports and actual Trainer control run against the staged modules.
        exec(actual.cells[3].source,scope)
        scope['command']=[sys.executable,'-c','print("CPU staging-to-final bookkeeping")']
        scope['STARTED']=time.time();scope['DEADLINE']=time.time()+15
        exec(actual.cells[4].source,scope)
        record=read_json(work/'session_summary.json');assert record['exit_code']==0 and record['role']==role
        checks.append(dict(name='actual_staging_imports_trainer_final',passed=True,worker_standin=True,session=record))
    for name,program,limit in [('failure','raise SystemExit(7)',10),('watchdog','import time;time.sleep(10)',.2)]:
        with tempfile.TemporaryDirectory(prefix='span031-stop-') as tmp:
            scope=dict(WORK=Path(tmp),STARTED=time.time(),DEADLINE=time.time()+limit,command=[sys.executable,'-c',program],env=None)
            try:exec(actual.cells[4].source,scope)
            except AssertionError:pass
            else:raise AssertionError('Failed worker was accepted')
            assert read_json(Path(tmp)/'session_summary.json')['exit_code']!=0
            checks.append(dict(name=name,passed=True))
    tiny=nbf.v4.new_notebook(metadata=nb.metadata,cells=[nbf.v4.new_code_cell(
        'from pathlib import Path\nimport sys,time,tempfile\ntmp=tempfile.TemporaryDirectory()\nWORK=Path(tmp.name);STARTED=time.time();DEADLINE=STARTED+10\ncommand=[sys.executable,"-c","print(123)"];env=None'),nbf.v4.new_code_cell(final)])
    executed=NotebookClient(tiny,timeout=30,kernel_name='python3').execute()
    assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in executed.cells)
    metadata=dict(id='arseniytrushin/llm-preference-span031-size-pilot',title='LLM Preference Span031 Size Pilot',code_file='job.ipynb',
        language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_internet=False,
        competition_sources=[],dataset_sources=[],kernel_sources=list(parents.values()),
        model_sources=['qwen-lm/qwen-3/transformers/4b/1','qwen-lm/qwen-3/transformers/8b/1'])
    atomic_json(folder/'kernel-metadata.json',metadata)
    names=['src/arena/'+n+'.py' for n in modules]+['tests/test_span031_size.py','src/arena/notebook_worker.py','src/arena/kaggle_control.py',
        'GOAL.md','docs/SPAN_031_PLAN.md','docs/SPAN_031_EXPOSURE_REVISION.md','configs/span_031.json','results/span_031/span_plan.json',
        'results/span_031/protocol_revision.json','results/span_031/preparation.json','results/span_031/pilot_preparation.json',
        'scripts/prepare_span_031_size.py','scripts/prepare_span_031_pilot.py','scripts/build_span_031_size.py',
        'scripts/run_span_031_size.py','scripts/audit_span_031_size.py','scripts/verify_span_031_source.py',
        'src/arena/exposure_validation.py','src/arena/span_heads.py','src/arena/span004_support.py','scripts/submit_span_013_order.py']
    atomic_json(pub/(role+'_bundle.json'),dict(role=role,source_sha256={n:sha256(REPO/n) for n in names},
        notebook_sha256=sha256(folder/'job.ipynb'),metadata_sha256=sha256(folder/'kernel-metadata.json'),
        cap_seconds=5400,internet_enabled=False,protocol_revision=2))
    atomic_json(pub/(role+'_cell_rehearsals.json'),dict(passed=True,rehearsals=checks,
        notebook_engine_startup_passed=True,actual_serialized_staging_trainer_final_passed=True,new_gpu_executions=0))
    print(folder)


if __name__=='__main__':main()
