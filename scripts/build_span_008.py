"""Build immutable v8 input-only classifier/native notebooks and rehearse their epilogues."""
from pathlib import Path
import argparse
import base64
import json
import sys
import tempfile
import time
import nbformat as nbf
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,sha256
from arena.notebook_worker import worker_cell


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('role',choices=['serial_cascade','context_completion']);args=parser.parse_args();role=args.role
    root=ROOT/'runs/span_008';pub=REPO/'results/span_008';cfg=read_json(REPO/'configs/span_008.json');inv=read_json(pub/'artifact_inventory.json')
    assert not (root/'jobs'/role/'push_intent.json').exists()
    old=read_json(inv['input_artifacts']['training_selection']['path']);manifest=read_json(inv['input_artifacts']['classifier_manifest']['path'])
    manifest.update(tokenizer_sha256=sha256(Path(inv['input_artifacts']['adapter']['path']).parent/'tokenizer.json'))
    for key in ['context_text_sha256','context_token_sha256','context_parent_receipt_sha256']:manifest[key]=inv[key]
    train_owner,train_slug=old['kernel'].split('/');export_owner,export_slug=manifest['source_kernel'].split('/');cap=cfg['job_seconds_caps'][role]
    cells=[nbf.v4.new_markdown_cell('# Span008 — '+role),nbf.v4.new_code_cell(f'''import os,sys,time,json,base64,subprocess,signal
from pathlib import Path
STARTED=time.time();DEADLINE=STARTED+{cap}-120;WORK=Path('/kaggle/working')
os.environ.update(TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
TRAIN=next(p for p in [Path('/kaggle/input/{train_slug}'),Path('/kaggle/input/notebooks/{train_owner}/{train_slug}')] if (p/'runtime_requirements.json').exists())
PARENT=next(p for p in [Path('/kaggle/input/{export_slug}'),Path('/kaggle/input/notebooks/{export_owner}/{export_slug}')] if (p/'exposure/adapter/adapter_model.safetensors').exists())
runtime=json.loads((TRAIN/'runtime_requirements.json').read_text())
subprocess.run([sys.executable,'-m','pip','install','-q','--no-index','--no-deps','--find-links',str(TRAIN/'runtime_wheels'),*runtime],check=True,timeout=240)
MODULES=WORK/'bundled_code/arena';MODULES.mkdir(parents=True,exist_ok=True);(MODULES/'__init__.py').write_text('')
sys.path.insert(0,str(MODULES.parent))
''')]
    parents=[old['kernel']+'/'+str(old['version_number']),manifest['source_kernel']+'/'+str(manifest['version_number'])];models=[manifest['config']['model_source']]
    if role=='serial_cascade':
        native_folder=ROOT/'runs/span_007/notebooks/judge_stability';native=nbf.read(native_folder/'job.ipynb',4)
        # Accepted native setup is copied verbatim except for the original whole-session clock.
        for index in range(1,6):
            source=native.cells[index].source
            if index==1:source=source.replace('SESSION_STARTED=time.time()','SESSION_STARTED=STARTED').replace('DEADLINE=SESSION_STARTED+2400-120',f'DEADLINE=SESSION_STARTED+{cap}-120')
            cells.append(nbf.v4.new_code_cell(source))
        meta=read_json(native_folder/'kernel-metadata.json');parents+=meta['kernel_sources'];models+=meta['model_sources']
        cells.append(nbf.v4.new_code_cell("(WORK/'native_config.json').write_text(json.dumps(CONFIG))\nsignal.alarm(0)"))
    sources=['data','joint_data','joint_train','joint_infer','training_exposure','span006_inputs','experiment','telemetry','prediction','prediction_chunks',
             'span008_classifier']
    if role=='serial_cascade':sources+=['judge','judge_runtime','span_repeatability','span008_serial']
    hashes={}
    for name in sources:
        path=REPO/f'src/arena/{name}.py';hashes[str(path.relative_to(REPO))]=sha256(path)
        cells.append(nbf.v4.new_code_cell(f"(MODULES/{(name+'.py')!r}).write_text({path.read_text()!r})"))
    preparation=f"(WORK/'config.json').write_text({json.dumps(cfg)!r})\n(WORK/'manifest.json').write_text({json.dumps(manifest)!r})\n"
    files=['sample_ids.json','smoke_reference.npz','classifier_profile_reference.npz','unchanged_profile_reference.npz','context_overlap_reference.npz','judge_reference.npz','judge_reference_tokens.json']
    preparation+="REFERENCES=WORK/'references';REFERENCES.mkdir(exist_ok=True)\n"
    for name in files:preparation+=f"(REFERENCES/{name!r}).write_bytes(base64.b64decode({base64.b64encode((root/name).read_bytes()).decode()!r}))\n"
    preparation+='''RAW=next(p for p in [Path('/kaggle/input/llm-classification-finetuning'),Path('/kaggle/input/competitions/llm-classification-finetuning')] if (p/'train.csv').exists())
MODEL=next(p for p in [Path('/kaggle/input/qwen-3/transformers/4b/1'),Path('/kaggle/input/models/qwen-lm/qwen-3/transformers/4b/1'),TRAIN/'base_model'] if (p/'config.json').exists())
env=os.environ.copy();env.update(PYTHONPATH=str(MODULES.parent),PYTHONUNBUFFERED='1')
'''
    module='span008_serial' if role=='serial_cascade' else 'span008_classifier'
    command="command=[sys.executable,"+("'--module'," if False else '')
    if role=='serial_cascade':command+="'-m','arena.span008_serial',"
    else:command+="'-m','torch.distributed.run','--standalone','--nproc_per_node=2','--module','arena.span008_classifier',"
    command+="'--raw',str(RAW),'--model',str(MODEL),'--adapter',str(PARENT/'exposure/adapter'),'--manifest',str(WORK/'manifest.json'),'--selection',str(REFERENCES/'sample_ids.json'),'--config',str(WORK/'config.json'),'--references',str(REFERENCES),'--out',str(WORK/"+repr(role)+"),'--deadline',str(DEADLINE),"
    if role=='serial_cascade':command+="'--native-config',str(WORK/'native_config.json'),'--executable',str(EXE),'--session-started',str(STARTED)]\n"
    else:command+="'--mode','context']\n"
    cells.append(nbf.v4.new_code_cell(preparation+command));cells.append(nbf.v4.new_code_cell(worker_cell(role,'')))
    for cell in cells:
        if cell.cell_type=='code':compile(cell.source,'span008-'+role,'exec')
    folder=root/'notebooks'/role;folder.mkdir(parents=True,exist_ok=True)
    nbf.write(nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}}),folder/'job.ipynb')
    metadata={'id':'arseniytrushin/llm-preference-span-008-'+role.replace('_','-'),'title':'LLM Preference Span008 '+role.replace('_',' ').title(),
        'code_file':'job.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':False,
        'competition_sources':['llm-classification-finetuning'],'dataset_sources':[],'kernel_sources':list(dict.fromkeys(parents)),'model_sources':models}
    atomic_json(folder/'kernel-metadata.json',metadata)
    actual=nbf.read(folder/'job.ipynb',4).cells[-1].source
    with tempfile.TemporaryDirectory(prefix='span008-cell-') as tmp:
        scope={'WORK':Path(tmp),'STARTED':time.time(),'DEADLINE':time.time()+10,'command':[sys.executable,'-c','print("whole-cell rehearsal")'],'env':None}
        exec(compile(actual,'actual-serialized-final-cell','exec'),scope)
        record=read_json(Path(tmp)/'session_summary.json');assert record['exit_code']==0 and record['role']==role
    for name in ['scripts/build_span_008.py','scripts/run_span_008_job.py','src/arena/kaggle_control.py','src/arena/notebook_worker.py','GOAL.md','configs/span_008.json','docs/SPAN_008_PLAN.md']:hashes[name]=sha256(REPO/name)
    atomic_json(pub/(role+'_bundle.json'),{'role':role,'source_sha256':hashes,'notebook_sha256':sha256(folder/'job.ipynb'),'metadata_sha256':sha256(folder/'kernel-metadata.json'),'cap_seconds':cap,'internet_enabled':False})
    atomic_json(pub/(role+'_cell_rehearsal.json'),{'passed':True,'serialized_last_cell_sha256':__import__('hashlib').sha256(actual.encode()).hexdigest(),'result':record,'new_gpu_executions':0})
    print(folder)


if __name__=='__main__':main()
