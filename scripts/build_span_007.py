"""Create one immutable, Internet-disabled v7 classifier notebook from shared code."""
from pathlib import Path
import argparse
import base64
import json
import sys
import zlib
import nbformat as nbf

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('role',choices=['continuation','context','promotion']);args=parser.parse_args();role=args.role
    root=ROOT/'runs/span_007';pub=REPO/'results/span_007';cfg=read_json(REPO/'configs/span_007.json')
    assert not (root/'jobs'/role/'push_intent.json').exists(),'Launched source remains immutable'
    inputs=read_json(pub/'artifact_inventory.json');selection=read_json(root/'sample_ids.json')
    old=read_json(inputs['input_artifacts']['training_selection']['path']);base=old['config']
    reference=__import__('numpy').load(inputs['input_artifacts']['classifier']['path'])
    lookup={int(value):i for i,value in enumerate(reference['ids'])};idx=[lookup[int(value)] for value in selection['selection']]
    p=reference['original'][idx];y=reference['labels'][idx]
    selection['selection_baseline_original_loss']=float(-__import__('numpy').log(__import__('numpy').maximum(p[__import__('numpy').arange(len(y)),y],1e-12)).mean())
    manifest=read_json(REPO/'results/span_002/research_candidate.json');manifest['tokenizer_sha256']=sha256(Path(inputs['input_artifacts']['adapter']['path']).parent/'tokenizer.json')
    cap=cfg['job_seconds_caps'][role];folder=root/'notebooks'/role;folder.mkdir(parents=True,exist_ok=True)
    train_owner,train_slug=old['kernel'].split('/');export_owner,export_slug=manifest['source_kernel'].split('/')
    model_source=base['model_source'];module='span007_train' if role=='continuation' else 'span007_'+role
    sources=['data','joint_data','joint_train','joint_infer','training_exposure','span006_inputs','experiment','telemetry',module]
    cells=[nbf.v4.new_markdown_cell('# Span007 — '+role),nbf.v4.new_code_cell(f'''import os,sys,time,json,base64,zlib,subprocess,signal,threading,hashlib
from pathlib import Path
STARTED=time.time();DEADLINE=STARTED+{cap}-120;WORK=Path('/kaggle/working')
os.environ.update(TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
TRAIN=next(p for p in [Path('/kaggle/input/{train_slug}'),Path('/kaggle/input/notebooks/{train_owner}/{train_slug}')] if (p/'runtime_requirements.json').exists())
PARENT=next(p for p in [Path('/kaggle/input/{export_slug}'),Path('/kaggle/input/notebooks/{export_owner}/{export_slug}')] if (p/'exposure/adapter/adapter_model.safetensors').exists())
runtime=json.loads((TRAIN/'runtime_requirements.json').read_text())
subprocess.run([sys.executable,'-m','pip','install','-q','--no-index','--no-deps','--find-links',str(TRAIN/'runtime_wheels'),*runtime],check=True,timeout=240)
MODULES=WORK/'bundled_code';(MODULES/'arena').mkdir(parents=True,exist_ok=True);(MODULES/'arena/__init__.py').write_text('')
(WORK/'config.json').write_text(json.dumps({cfg!r}))
(WORK/'base_config.json').write_text(json.dumps({base!r}))
(WORK/'manifest.json').write_text(json.dumps({manifest!r}))
(WORK/'selection.json').write_text(json.dumps({selection!r}))
(WORK/'reference.npz').write_bytes(base64.b64decode({base64.b64encode((root/'smoke_reference.npz').read_bytes()).decode()!r}))
(WORK/'split_ids.json').write_bytes(zlib.decompress(base64.b64decode({base64.b64encode(zlib.compress(Path(inputs['input_artifacts']['split']['path']).read_bytes(),9)).decode()!r})))
''')]
    hashes={}
    for name in sources:
        path=REPO/f'src/arena/{name}.py';hashes[str(path.relative_to(REPO))]=sha256(path)
        cells.append(nbf.v4.new_code_cell(f"(MODULES/'arena/{name}.py').write_text({path.read_text()!r})"))
    tail=("'--parent',str(PARENT),'--split',str(WORK/'split_ids.json'),'--base-config',str(WORK/'base_config.json')," if role=='continuation' else "'--adapter',str(PARENT/'exposure/adapter'),'--manifest',str(WORK/'manifest.json'),")
    cells.append(nbf.v4.new_code_cell(f'''import torch
assert torch.cuda.device_count()==2
RAW=next(p for p in [Path('/kaggle/input/llm-classification-finetuning'),Path('/kaggle/input/competitions/llm-classification-finetuning')] if (p/'train.csv').exists())
MODEL=next(p for p in [Path('/kaggle/input/{'/'.join(model_source.split('/')[1:])}'),Path('/kaggle/input/models/{model_source}'),TRAIN/'base_model'] if (p/'config.json').exists())
assert hashlib.sha256((MODEL/'config.json').read_bytes()).hexdigest()=='8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a'
env=os.environ.copy();env.update(PYTHONPATH=str(MODULES),PYTHONUNBUFFERED='1')
command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','--module','arena.{module}',
 '--raw',str(RAW),'--model',str(MODEL),{tail}
 '--config',str(WORK/'config.json'),'--selection',str(WORK/'selection.json'),'--reference',str(WORK/'reference.npz'),
 '--out',str(WORK/'{role}'),'--deadline',str(DEADLINE)]
process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=env,start_new_session=True)
def stop_owned_group():
    if process.poll() is None:
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
timer=threading.Timer(max(1,DEADLINE-time.time()),stop_owned_group);timer.daemon=True;timer.start()
try:
    with (WORK/'worker_console.log').open('w') as log:
        for line in process.stdout:print(line,end='',flush=True);log.write(line);log.flush()
    status=process.wait(timeout=max(1,DEADLINE-time.time()))
finally:timer.cancel();stop_owned_group()
(WORK/'session_summary.json').write_text(json.dumps({{'seconds':time.time()-STARTED,'exit_code':status,'role':role}},indent=2))
assert status==0,'Preserve failed execution; never automatically repeat a mutation'
'''))
    for cell in cells:
        if cell.cell_type=='code':compile(cell.source,'span007-'+role,'exec')
    nbf.write(nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}}),folder/'job.ipynb')
    metadata={'id':'arseniytrushin/llm-preference-span-007-'+role,'title':'LLM Preference Span 007 '+role.title(),
              'code_file':'job.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':False,
              'competition_sources':['llm-classification-finetuning'],'dataset_sources':[],
              'kernel_sources':[old['kernel']+'/'+str(old['version_number']),manifest['source_kernel']+'/'+str(manifest['version_number'])],
              'model_sources':[model_source]}
    atomic_json(folder/'kernel-metadata.json',metadata)
    for name in ['scripts/build_span_007.py','scripts/run_experiment.py','src/arena/kaggle_control.py','src/arena/span007_analysis.py',
                 'GOAL.md','configs/span_007.json','docs/SPAN_007_PLAN.md']:
        hashes[name]=sha256(REPO/name)
    atomic_json(pub/(role+'_bundle.json'),{'role':role,'source_sha256':hashes,'notebook_sha256':sha256(folder/'job.ipynb'),
        'metadata_sha256':sha256(folder/'kernel-metadata.json'),'cap_seconds':cap,'internet_enabled':False})
    print(folder)


if __name__=='__main__':main()
