"""Bundle one label-free current-H03 confirmation inference after recipe freeze."""
from pathlib import Path
import base64,hashlib,json,sys
import nbformat as nbf

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,json_write


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    root=ROOT/'runs/span_006';pub=REPO/'results/span_006';folder=root/'notebooks/confirmation';folder.mkdir(parents=True,exist_ok=True)
    assert not (root/'jobs/confirmation/push_intent.json').exists(),'Launched source is immutable'
    cfg=json.loads((REPO/'configs/span_006.json').read_text());inputs=json.loads((pub/'artifact_inventory.json').read_text());recipe=json.loads((pub/'frozen_recipe.json').read_text())
    assert not recipe['confirmation_votes_read'] and not (root/'reveal_intent.json').exists()
    manifest=json.loads((REPO/'results/span_002/research_candidate.json').read_text());parent=json.loads(Path(inputs['input_artifacts']['training_selection']['path']).read_text())
    adapter=Path(inputs['input_artifacts']['adapter']['path']).parent;manifest['tokenizer_sha256']=sha(adapter/'tokenizer.json')
    selection=json.loads((root/'sample_ids.json').read_text());selection.update(confirmation_text_digest=inputs['confirmation_text_digest'],frozen_recipe_sha256=sha(pub/'frozen_recipe.json'))
    cap=cfg['job_seconds_caps']['confirmation'];owner,train_slug=parent['kernel'].split('/');exowner,export_slug=manifest['source_kernel'].split('/');model_source=manifest['config']['model_source']
    cells=[nbf.v4.new_markdown_cell('# Span006 — frozen strength transfer: label-free H03 inference'),nbf.v4.new_code_cell(f'''import os,sys,time,json,subprocess,base64,hashlib,threading,signal
from pathlib import Path
STARTED=time.time();DEADLINE=STARTED+{cap}-120;WORK=Path('/kaggle/working')
os.environ.update(TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
TRAIN=next(p for p in [Path('/kaggle/input/{train_slug}'),Path('/kaggle/input/notebooks/{owner}/{train_slug}')] if (p/'runtime_requirements.json').exists())
EXPORT=next(p for p in [Path('/kaggle/input/{export_slug}'),Path('/kaggle/input/notebooks/{exowner}/{export_slug}')] if (p/'exposure/adapter/adapter_model.safetensors').exists())
runtime=json.loads((TRAIN/'runtime_requirements.json').read_text())
subprocess.run([sys.executable,'-m','pip','install','-q','--no-index','--no-deps','--find-links',str(TRAIN/'runtime_wheels'),*runtime],check=True,timeout=240)
MODULES=WORK/'bundled_code';(MODULES/'arena').mkdir(parents=True,exist_ok=True);(MODULES/'arena/__init__.py').write_text('')
(WORK/'manifest.json').write_text(json.dumps({manifest!r},indent=2))
(WORK/'selection.json').write_text(json.dumps({selection!r},indent=2))
(WORK/'reference.npz').write_bytes(base64.b64decode({base64.b64encode((root/'smoke_reference.npz').read_bytes()).decode()!r}))
''')]
    sources={}
    for name in ['data','joint_data','joint_train','joint_infer','span006_inputs','span006_export']:
        p=REPO/f'src/arena/{name}.py';sources[str(p.relative_to(REPO))]=sha(p)
        cells.append(nbf.v4.new_code_cell(f"(MODULES/'arena/{name}.py').write_text({p.read_text()!r})"))
    cells.append(nbf.v4.new_code_cell(f'''import torch
assert torch.cuda.device_count()==2
RAW=next(p for p in [Path('/kaggle/input/llm-classification-finetuning'),Path('/kaggle/input/competitions/llm-classification-finetuning')] if (p/'train.csv').exists())
MODEL=next(p for p in [Path('/kaggle/input/{'/'.join(model_source.split('/')[1:])}'),Path('/kaggle/input/models/{model_source}'),TRAIN/'base_model'] if (p/'config.json').exists())
assert hashlib.sha256((MODEL/'config.json').read_bytes()).hexdigest()=='8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a'
env=os.environ.copy();env.update(PYTHONPATH=str(MODULES),PYTHONUNBUFFERED='1')
command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','--module','arena.span006_export',
 '--raw',str(RAW),'--model',str(MODEL),'--adapter',str(EXPORT/'exposure/adapter'),'--manifest',str(WORK/'manifest.json'),
 '--selection',str(WORK/'selection.json'),'--reference',str(WORK/'reference.npz'),'--out',str(WORK/'confirmation'),'--deadline',str(DEADLINE)]
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
assert status==0,'Preserve failed source/output; no duplicate execution or target reveal'
summary=json.loads((WORK/'confirmation/confirmation_summary.json').read_text());assert summary['complete']
(WORK/'session_summary.json').write_text(json.dumps({{'seconds':time.time()-STARTED,'complete':True,'rows':summary['rows'],'confirmation_votes_read':False}},indent=2))
'''))
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})
    for c in cells:
        if c.cell_type=='code':compile(c.source,'span006-confirmation','exec')
    nbf.write(nb,folder/'job.ipynb')
    metadata={'id':'arseniytrushin/llm-preference-span-006-confirmation','title':'LLM Preference Span 006 Confirmation','code_file':'job.ipynb','language':'python','kernel_type':'notebook',
              'is_private':True,'enable_gpu':True,'enable_internet':False,'competition_sources':['llm-classification-finetuning'],'dataset_sources':[],
              'kernel_sources':[parent['kernel']+'/'+str(parent['version_number']),manifest['source_kernel']+'/'+str(manifest['version_number'])],'model_sources':[model_source]}
    json_write(folder/'kernel-metadata.json',metadata)
    sources.update({p:sha(REPO/p) for p in ['scripts/build_span_006_confirmation.py','scripts/run_span_006_job.py','scripts/evaluate_span_006.py','GOAL.md','configs/span_006.json','docs/SPAN_006_PROTOCOL.md','results/span_006/frozen_recipe.json']})
    bundle={'role':'confirmation','notebook_sha256':sha(folder/'job.ipynb'),'metadata_sha256':sha(folder/'kernel-metadata.json'),'source_sha256':sources,
            'frozen_recipe_sha256':sha(pub/'frozen_recipe.json'),'config_sha256':sha(REPO/'configs/span_006.json'),'session_seconds_cap':cap,
            'parents':metadata['kernel_sources'],'model_source':model_source,'internet_enabled':False,'labels_read':False}
    json_write(pub/'confirmation_bundle.json',bundle);print(folder)


if __name__=='__main__':main()
