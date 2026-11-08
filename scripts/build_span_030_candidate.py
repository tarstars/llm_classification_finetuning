"""Freeze the actual public/hidden notebook and rehearse serialized staging/worker cells."""
from pathlib import Path
import base64
import hashlib
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

FAKE_ENGINE='''import argparse
from pathlib import Path
import numpy as np
import pandas as pd
p=argparse.ArgumentParser()
for k in ['raw','assets','out','spec','role','cohorts']:p.add_argument('--'+k,required=True)
a=p.parse_args();out=Path(a.out);assert a.role in ['m0','m3']
for name in a.cohorts.split(','):
 f=pd.read_parquet(out/(name+'.parquet'));assert set(f.columns)=={'id','prompt','response_a','response_b'}
 probs=[.7,.1,.2] if a.role=='m0' else [.2,.6,.2]
 np.savez(out/(a.role+'_'+name+'.npz'),ids=f.id.to_numpy(),probabilities=np.tile(probs,(len(f),1)),lengths=np.ones(len(f),dtype=int))
'''


def main():
    root=ROOT/'runs/span_030';pub=REPO/'results/span_030';role='weight_qualification'
    assert read_json(pub/'quality_replay.json')['gross_regression_guard_passed']
    assert read_json(pub/'candidate_preparation.json')['prepared'] and (pub/'span_plan.json').exists()
    tests=read_json(pub/'candidate_tests.json');assert tests['passed']
    for n,h in tests['source_sha256'].items():assert sha256(REPO/n)==h,n
    assert not (root/'jobs'/role/'push_intent.json').exists()
    folder=root/'notebooks'/role;folder.mkdir(parents=True,exist_ok=True)
    proof=read_json(REPO/'results/span_028/cpu_audit.json')
    native_hashes={'/'.join(Path(n).parts[-2:]):h for n,h in proof['native_source_sha256'].items()}
    setup=f'''from pathlib import Path
import os,sys,time,json,shutil,subprocess,hashlib,base64
STARTED=time.time();WORK=Path('/kaggle/working')
RERUN_FLAG=os.environ.get('KAGGLE_IS_COMPETITION_RERUN','').strip().lower()
assert RERUN_FLAG in ['', 'false', '0', 'true', '1'],'Unknown rerun flag'
IS_HIDDEN=RERUN_FLAG in ['true','1'];LIMIT=32400 if IS_HIDDEN else 3600;DEADLINE=STARTED+LIMIT-120
# MODE_END
assert sys.version_info[:2]==(3,10),'Pinned author image not honored'
RAW=next(p for p in [Path('/kaggle/input/llm-classification-finetuning'),Path('/kaggle/input/competitions/llm-classification-finetuning')] if (p/'test.csv').exists())
GPU_INFO=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True).strip().splitlines()
assert len(GPU_INFO)==2 and all('T4' in x for x in GPU_INFO)
(WORK/'hardware.json').write_text(json.dumps(dict(gpus=GPU_INFO,python=sys.version,mode='hidden' if IS_HIDDEN else 'qualification')))
ASSETS=WORK/'assets';ASSETS.mkdir(exist_ok=True)
for slug in ['lmsys-checkpoints-0-0805','lmsys-checkpoints-3-0805','lmsys-modules-0805','lmsys-packages']:
 source=next(p for p in [Path('/kaggle/input')/slug,Path('/kaggle/input/datasets/tascj0')/slug] if p.exists())
 (ASSETS/slug).symlink_to(source,target_is_directory=True)
for name,digest in {native_hashes!r}.items():assert hashlib.sha256((ASSETS/'lmsys-modules-0805'/name).read_bytes()).hexdigest()==digest
shutil.copytree(ASSETS/'lmsys-modules-0805',WORK/'human_pref')
for wheel in ['triton-2.2.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl','xformers-0.0.24042abc8.d20240802-cp310-cp310-linux_x86_64.whl']:
 subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',str(ASSETS/'lmsys-packages'/wheel)],check=True,timeout=120)
os.chdir(WORK)
'''
    files={'two_expert_pipeline.py':'src/arena/span030_pipeline.py','two_expert_engine.py':'src/arena/span030_engine.py',
        'two_expert_checks.py':'src/arena/span030_checks.py','fixed_mixture.py':'src/arena/span030_mixture.py',
        'cpu_prefetch.py':'src/arena/cpu_prefetch.py','frozen_inference.py':'src/arena/span010_order.py'}
    stage=''
    for n,p in files.items():stage+=f'(WORK/{n!r}).write_text({(REPO/p).read_text()!r})\n'
    stage+=f'''if IS_HIDDEN:
 SPEC=WORK/'hidden_spec.json';SPEC.write_text({(root/'hidden_spec.json').read_text()!r});MODE='--hidden'
else:
 SPEC=WORK/'qualification_spec.json';SPEC.write_text({(root/'qualification_spec.json').read_text()!r});MODE='--qualify'
 (WORK/'controls').mkdir(exist_ok=True)
'''
    for p in sorted((root/'controls').glob('*.npz')):
        stage+=f" (WORK/'controls'/{p.name!r}).write_bytes(base64.b64decode({base64.b64encode(p.read_bytes()).decode()!r}))\n"
    stage+='''command=[sys.executable,'-u',str(WORK/'two_expert_pipeline.py'),'--raw',str(RAW),'--assets',str(ASSETS),'--out',str(WORK/'two_expert'),'--engine',str(WORK/'two_expert_engine.py'),'--spec',str(SPEC),MODE]
env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
'''
    final=worker_cell(role,'')
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell('''# H76 fixed 40/60 two-expert candidate

Original native8192CPU-prefetch inference, no reverse/route, T1. Public mode
qualifies fixed cohorts; competition rerun performs actual hidden text inference.
Public gold controls are materialized only in qualification mode. Training-
overlapping local evidence is not a hidden-score forecast.'''),
        nbf.v4.new_code_cell(setup),nbf.v4.new_code_cell(stage),nbf.v4.new_code_cell(final)])
    nb.metadata.update(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python',version='3.10.13'))
    for c in nb.cells:
        if c.cell_type=='code':compile(c.source,'span030-actual-cell','exec')
    nbf.write(nb,folder/'job.ipynb')
    actual=nbf.read(folder/'job.ipynb',4);mode_source=actual.cells[1].source.split('# MODE_END')[0]
    checks=[]
    for flag,hidden in [('',False),('false',False),('0',False),('true',True),('1',True),(' TRUE ',True)]:
        with patch.dict(os.environ,{'KAGGLE_IS_COMPETITION_RERUN':flag}):
            scope={};exec(mode_source,scope)
        assert scope['IS_HIDDEN']==hidden and scope['LIMIT']==(32400 if hidden else 3600)
        checks.append(dict(name='mode_flag',flag=flag,hidden=hidden,passed=True))
    with patch.dict(os.environ,{'KAGGLE_IS_COMPETITION_RERUN':'unexpected'}):
        try:exec(mode_source,{})
        except AssertionError:pass
        else:raise AssertionError('Invalid mode accepted')
    import numpy as np
    import pandas as pd
    for hidden in [False,True]:
        with tempfile.TemporaryDirectory(prefix='span030-mode-') as tmp:
            work=Path(tmp);raw=work/'raw';raw.mkdir()
            scope=dict(WORK=work,RAW=raw,ASSETS=work/'assets',IS_HIDDEN=hidden,sys=sys,base64=base64,json=json,os=os)
            exec(compile(actual.cells[2].source,'actual-staging-cell','exec'),scope)
            spec_path=scope['SPEC'];spec=read_json(spec_path)
            assert spec['mode']==('hidden' if hidden else 'qualification')
            assert (work/'controls').exists()!=hidden
            subprocess.run([sys.executable,'-c','import two_expert_pipeline,two_expert_engine,two_expert_checks,fixed_mixture'],cwd=work,check=True,timeout=30)
            if hidden:
                assert set(spec)=={'mode','uses_targets','packing_tokens','checkpoint_references'}
                ids=list(range(100000,100101))
            else:
                train_ids=sorted({i for n,values in spec['frame_ids'].items() if n!='test' for i in values})
                frame=pd.DataFrame(dict(id=train_ids,prompt=['["p"]']*len(train_ids),response_a=['["a"]']*len(train_ids),response_b=['["b"]']*len(train_ids),winner_model_a=1))
                frame.to_csv(raw/'train.csv',index=False)
                spec['train_csv_sha256']=sha256(raw/'train.csv');spec_path.write_text(json.dumps(spec));ids=spec['frame_ids']['test']
            pd.DataFrame(dict(id=ids,prompt=['["p"]']*len(ids),response_a=['["a"]']*len(ids),response_b=['["b"]']*len(ids))).to_csv(raw/'test.csv',index=False)
            cpu_engine=work/'cpu_engine.py';cpu_engine.write_text(FAKE_ENGINE)
            scope['command'][scope['command'].index('--engine')+1]=str(cpu_engine)
            scope.update(STARTED=time.time(),DEADLINE=time.time()+30)
            exec(compile(actual.cells[3].source,'actual-final-cell','exec'),scope)
            done=read_json(work/'two_expert/inference_complete.json');assert done['complete'] and done['mode']==spec['mode']
            csv=pd.read_csv(work/'submission.csv');assert csv.id.tolist()==ids
            np.testing.assert_allclose(csv.iloc[:,1:],np.tile([.64,.16,.2],(len(ids),1)))
            if hidden:assert not (raw/'train.csv').exists() and not (work/'controls').exists()
            checks.append(dict(name='actual_staging_pipeline_final_cell',hidden=hidden,rows=len(ids),passed=True,
                fake_engine=True,session=read_json(work/'session_summary.json')))
    for name,program,limit in [('failure','raise SystemExit(7)',10),('watchdog','import time;time.sleep(10)',.2)]:
        with tempfile.TemporaryDirectory(prefix='span030-failure-') as tmp:
            scope=dict(WORK=Path(tmp),STARTED=time.time(),DEADLINE=time.time()+limit,command=[sys.executable,'-c',program],env=None)
            try:exec(compile(actual.cells[3].source,'actual-final-cell','exec'),scope)
            except AssertionError:pass
            else:raise AssertionError('Failed worker was accepted')
            record=read_json(Path(tmp)/'session_summary.json');assert record['exit_code']!=0
            checks.append(dict(name=name,passed=True,session=record))
    tiny=nbf.v4.new_notebook(metadata=nb.metadata,cells=[nbf.v4.new_code_cell(
        'from pathlib import Path\nimport sys,time,tempfile\ntmp=tempfile.TemporaryDirectory(prefix="span030-startup-")\nWORK=Path(tmp.name);STARTED=time.time();DEADLINE=STARTED+10\ncommand=[sys.executable,"-c","print(123)"];env=None'),nbf.v4.new_code_cell(actual.cells[3].source)])
    executed=NotebookClient(tiny,timeout=30,kernel_name='python3').execute()
    assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.get('outputs',[])) for c in executed.cells)
    metadata=read_json(ROOT/'runs/span_028/notebooks/prefetch_profile/kernel-metadata.json')
    metadata.update(id='arseniytrushin/llm-preference-span030-llama-prior',title='LLM Preference Span030 Llama Prior')
    assert not metadata.get('kernel_sources',[]) and metadata['enable_internet'] is False
    atomic_json(folder/'kernel-metadata.json',metadata)
    names=list(files.values())+['GOAL.md','docs/SPAN_030_PLAN.md','docs/SPAN_030_CPU_RESULTS.md',
        'configs/span_030.json','results/span_030/span_plan.json','results/span_030/quality_replay.json',
        'results/span_030/candidate_preparation.json','results/span_030/runtime_reference.json','results/span_030/candidate_tests.json',
        'src/arena/span030_runtime.py','src/arena/kaggle_control.py','src/arena/notebook_worker.py',
        'scripts/prepare_span_030_candidate.py','scripts/build_span_030_candidate.py','scripts/run_span_030_candidate.py',
        'scripts/audit_span_030_candidate.py','scripts/submit_span_030_candidate.py','scripts/observe_leaderboard.py',
        'src/arena/leaderboard.py','scripts/submit_span_013_order.py','scripts/submit_span_010_order.py',
        'tests/test_span030_mixture.py','tests/test_span030_pipeline.py','tests/test_span030_forward_rehearsal.py']
    atomic_json(pub/(role+'_bundle.json'),dict(role=role,source_sha256={n:sha256(REPO/n) for n in names},
        notebook_sha256=sha256(folder/'job.ipynb'),metadata_sha256=sha256(folder/'kernel-metadata.json'),
        public_cap_seconds=3600,hidden_cap_seconds=32400,expert_weights=[.4,.6],packing_tokens=8192,
        supports_actual_hidden_inference=True,internet_enabled=False,quality_replay_sha256=sha256(pub/'quality_replay.json')))
    atomic_json(pub/(role+'_cell_rehearsals.json'),dict(passed=True,rehearsals=checks,standalone_imports_passed=True,
        actual_serialized_public_hidden_pipeline_passed=True,public_gold_not_materialized_hidden=True,
        notebook_engine_startup_passed=True,new_gpu_executions=0))
    print(folder)


if __name__=='__main__':main()
