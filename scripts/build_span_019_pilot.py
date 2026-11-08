"""Build the frozen native pilot and rehearse its actual serialized worker."""
from pathlib import Path
import argparse
import json
import os
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
    parser=argparse.ArgumentParser();parser.add_argument('--native-runtime',default='/tmp/llm_span019_runtime')
    args=parser.parse_args();role='classifier_pilot';root=ROOT/'runs/span_019';pub=REPO/'results/span_019'
    registry=read_json(REPO/'backlog/hypotheses.json')
    assert registry['active_version']==19 and next(v for v in registry['versions'] if v['number']==19)['focus_ids']==['H65']
    assert not (root/'jobs'/role/'push_intent.json').exists(),'Never rewrite accepted source'
    manifest=read_json(pub/'asset_manifest.json');selection=read_json(pub/'evaluation_selection.json')
    assert read_json(pub/'worker_cpu_validation.json')['passed']
    assert len(selection['uniform_ids'])==512 and len(selection['control_ids'])==256
    folder=root/'notebooks'/role;folder.mkdir(parents=True,exist_ok=True)
    visible=sha256(ROOT/'raw/test.csv')
    modules=['span019_pipeline','span019_infer','span019_model','span019_input','span019_batch','gptq8','span010_order']
    setup=f'''from pathlib import Path
import os,sys,time,json,shutil,subprocess,hashlib
STARTED=time.time();WORK=Path('/kaggle/working')
RAW=next(p for p in [Path('/kaggle/input/llm-classification-finetuning'),Path('/kaggle/input/competitions/llm-classification-finetuning')] if (p/'test.csv').exists())
IS_RERUN=os.environ.get('KAGGLE_IS_COMPETITION_RERUN','').strip().lower() in ['true','1'] or hashlib.sha256((RAW/'test.csv').read_bytes()).hexdigest()!={visible!r}
DEADLINE=STARTED+(32400 if IS_RERUN else 3600)-120
assert sys.version_info[:2]==(3,10),'Pinned image not honored'
GPU_INFO=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True).strip().splitlines()
assert len(GPU_INFO)==2 and all('T4' in name for name in GPU_INFO)
(WORK/'hardware.json').write_text(json.dumps(dict(gpus=GPU_INFO,python=sys.version)))
ASSETS=WORK/'assets';ASSETS.mkdir(exist_ok=True)
for slug in ['lmsys-checkpoints-0-0805','lmsys-checkpoints-3-0805','lmsys-modules-0805','lmsys-packages']:
 source=next(p for p in [Path('/kaggle/input')/slug,Path('/kaggle/input/datasets/tascj0')/slug] if p.exists())
 (ASSETS/slug).symlink_to(source,target_is_directory=True)
shutil.copytree(ASSETS/'lmsys-modules-0805',WORK/'human_pref')
for wheel in ['triton-2.2.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl','xformers-0.0.24042abc8.d20240802-cp310-cp310-linux_x86_64.whl']:
 subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',str(ASSETS/'lmsys-packages'/wheel)],check=True,timeout=120)
ORIGIN=next(p for p in [Path('/kaggle/input/llm-span019-native-classifier'),Path('/kaggle/input/datasets/arseniytrushin/llm-span019-native-classifier')] if p.exists())
EXPECTED={manifest['entries']!r}
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
verified={{}}
for item in EXPECTED:
 path=ORIGIN/item['filename']
 assert path.stat().st_size==item['bytes'] and digest(path)==item['sha256'],'Pinned asset mismatch'
 verified[item['filename']]=item['sha256']
(WORK/'asset_verification.json').write_text(json.dumps(dict(passed=True,files=verified)))
PACKAGE=WORK/'arena';PACKAGE.mkdir();(PACKAGE/'__init__.py').write_text('')
'''
    for module in modules:
        setup+=f"(PACKAGE/{(module+'.py')!r}).write_text({(REPO/('src/arena/'+module+'.py')).read_text()!r})\n"
    setup+=f"(WORK/'selection.json').write_text({json.dumps(selection)!r})\n"
    setup+=f'''os.chdir(WORK)
command=[sys.executable,'-u','-m','arena.span019_pipeline','--raw',str(RAW),'--assets',str(ASSETS),'--origin',str(ORIGIN),'--native-source',str(ORIGIN/'native_qlinear.py'),'--head-sha256',{manifest['classifier_head_sha256']!r},'--out',str(WORK/'mirror_experiment'),'--selection',str(WORK/'selection.json')]+([] if IS_RERUN else ['--qualify','--engineering-only'])
env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',PYTHONPATH=str(WORK))
(WORK/'setup_summary.json').write_text(json.dumps(dict(seconds=time.time()-STARTED,hidden=IS_RERUN,verified_files=len(verified))))
'''
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(
        '# H65 native classifier engineering pilot\n\nTwo identical native GPTQ8 replicas score the mirror expert on separate T4s. The fixed candidate equally combines it with retained Gemma. No winner labels or model identity inputs. This initial public run cannot qualify an official submission. Hidden mode performs actual test inference.'),
        nbf.v4.new_code_cell(setup),nbf.v4.new_code_cell(worker_cell(role,''))])
    nb.metadata['kernelspec']=dict(display_name='Python 3',language='python',name='python3')
    for cell in nb.cells:
        if cell.cell_type=='code':compile(cell.source,'span019-serialized','exec')
    nbf.write(nb,folder/'job.ipynb')
    metadata=dict(id='arseniytrushin/llm-preference-span019-classifier-pilot',
        title='LLM Preference Span019 Native Classifier',code_file='job.ipynb',language='python',
        kernel_type='notebook',is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=False,
        dataset_sources=['tascj0/'+s+'/1' for s in ['lmsys-checkpoints-0-0805','lmsys-checkpoints-3-0805',
            'lmsys-modules-0805','lmsys-packages']]+[manifest['dataset']],model_sources=[],kernel_sources=[],
        competition_sources=['llm-classification-finetuning'],
        docker_image='gcr.io/kaggle-private-byod/python@sha256:e3f9bf91d1b7b36bf77554565d27c5733585fbe5e2b7f0832e068824e7accc3c',
        docker_image_pinning_type='original',machine_shape='NvidiaTeslaT4')
    atomic_json(folder/'kernel-metadata.json',metadata)
    actual=nbf.read(folder/'job.ipynb',4).cells[-1].source
    rehearsals=[]
    for name,program,limit,success in [('success','print(123)',10,True),
        ('failure','raise SystemExit(7)',10,False),('watchdog','import time;time.sleep(10)',.2,False)]:
        with tempfile.TemporaryDirectory(prefix='span019-worker-') as tmp:
            scope=dict(WORK=Path(tmp),STARTED=time.time(),DEADLINE=time.time()+limit,
                command=[sys.executable,'-c',program],env=None)
            failed=False
            try:exec(compile(actual,'actual-span019-worker','exec'),scope)
            except AssertionError:failed=True
            result=read_json(Path(tmp)/'session_summary.json')
            assert failed!=success and (result['exit_code']==0)==success
            rehearsals.append(dict(name=name,passed=True,summary=result))
    mode='\n'.join(line for line in setup.splitlines() if line.startswith(('IS_RERUN=','DEADLINE=','command=')))
    modes=[]
    for flag,observed,hidden in [('',visible,False),('','changed',True),('true',visible,True)]:
        scope=dict(os=SimpleNamespace(environ={'KAGGLE_IS_COMPETITION_RERUN':flag}),
            hashlib=SimpleNamespace(sha256=lambda data:SimpleNamespace(hexdigest=lambda:observed)),
            RAW=ROOT/'raw',WORK=Path('/tmp'),ASSETS=Path('/tmp/assets'),ORIGIN=Path('/tmp/origin'),sys=sys,STARTED=0)
        exec(compile(mode,'actual-span019-branch','exec'),scope)
        assert scope['IS_RERUN']==hidden and ('--qualify' in scope['command'])!=hidden
        assert ('--engineering-only' in scope['command'])!=hidden
        assert scope['DEADLINE']==(32400 if hidden else 3600)-120
        modes.append(dict(hidden=hidden,passed=True))
    tiny=nbf.v4.new_notebook(cells=[nbf.v4.new_code_cell('from pathlib import Path\nimport sys,time,tempfile\n'
        'tmp=tempfile.TemporaryDirectory();WORK=Path(tmp.name);STARTED=time.time();DEADLINE=STARTED+10\n'
        'command=[sys.executable,"-c","print(123)"];env=None'),nbf.v4.new_code_cell(actual)])
    executed=NotebookClient(tiny,timeout=30,kernel_name='python3').execute()
    assert all(c.execution_count is not None for c in executed.cells)
    assert not any(o.output_type=='error' for c in executed.cells for o in c.get('outputs',[]))
    checks=[]
    for name,pinned in [('test_span019_pipeline.py',False),('test_span019_native_model.py',True)]:
        env={**os.environ,'PYTHONPATH':(args.native_runtime+os.pathsep if pinned else '')+str(REPO/'src'),
            'HF_HOME':'/tmp/llm_span019_hf','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
            'ARENA_NATIVE_QLINEAR_PATH':str(root/'source_audit/native/qlinear_cuda_v0.7.1.py')}
        result=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p',name,'-v'],
            cwd=REPO,env=env,capture_output=True,text=True)
        assert result.returncode==0 and 'skipped' not in result.stderr,result.stdout+result.stderr
        checks.append(dict(test_file=name,passed=True,output=result.stdout+result.stderr))
    atomic_json(pub/(role+'_cell_rehearsals.json'),dict(passed=True,checks=checks,worker=rehearsals,
        mode_checks=modes,notebook_startup_passed=True,actual_pipeline_cpu_tests_passed=True,
        real_model_gpu_exercised=False,new_gpu_executions=0))
    names=['GOAL.md','docs/SPAN_019_PLAN.md','docs/SPAN_019_ASSET_PROTOCOL.md','docs/SPAN_019_GPU_PROTOCOL.md',
        'configs/span_019.json','results/span_019/span_plan.json','results/span_019/asset_manifest.json',
        'results/span_019/evaluation_selection.json','results/span_019/worker_cpu_validation.json',
        'src/arena/notebook_worker.py','src/arena/kaggle_control.py','scripts/build_span_019_pilot.py',
        'scripts/run_span_019_pilot.py',*['src/arena/'+m+'.py' for m in modules],
        'tests/test_span019_pipeline.py','tests/test_span019_native_model.py']
    atomic_json(pub/(role+'_bundle.json'),dict(role=role,source_sha256={n:sha256(REPO/n) for n in names},
        notebook_sha256=sha256(folder/'job.ipynb'),metadata_sha256=sha256(folder/'kernel-metadata.json'),
        public_cap_seconds=3600,hidden_cap_seconds=32400,public_stage='engineering_only',
        official_submission_eligible=False,internet_enabled=False,visible_test_sha256=visible,
        asset_manifest_sha256=sha256(pub/'asset_manifest.json')))
    print(folder)


if __name__=='__main__':main()
