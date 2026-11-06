"""Prepare, but do not launch, the one conditional matched-size pilot."""
from pathlib import Path
import base64,json,sys,tempfile,time
import numpy as np
import pandas as pd
import nbformat as nbf
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,CLASSES,TEXT_COLUMNS
from arena.experiment import read_json,atomic_json,sha256
from arena.notebook_worker import worker_cell


def main():
    root=ROOT/'runs/span_008';pub=REPO/'results/span_008';cfg=read_json(REPO/'configs/span_008.json');inv=read_json(pub/'artifact_inventory.json');role='size_pilot'
    assert not (root/'jobs'/role/'push_intent.json').exists()
    teacher=read_json(pub/'H15_audit.json');assert len(teacher['qwen8_availability'])==1
    rng=np.random.default_rng(cfg['seed']);columns=['id']+TEXT_COLUMNS+CLASSES
    frames={name:pd.read_parquet(inv['input_artifacts'][name+'.parquet']['path'],columns=columns) for name in ['train','dev']}
    selected={name:frames[name].iloc[rng.choice(len(frames[name]),cfg['size_pilot_'+name+'_rows'],replace=False)].copy() for name in frames}
    sel={name:frame.id.tolist() for name,frame in selected.items()};atomic_json(root/'size_sample_ids.json',sel)
    data=pd.concat(selected.values(),ignore_index=True)
    for col in TEXT_COLUMNS:data[col]=data[col].map(lambda text:json.dumps([text]))
    csv=data.to_csv(index=False).encode();old=read_json(inv['input_artifacts']['training_selection']['path']);manifest=read_json(inv['input_artifacts']['classifier_manifest']['path']);base=manifest['config']
    owner,slug=old['kernel'].split('/');adapter_owner,adapter_slug=manifest['source_kernel'].split('/');cap=cfg['job_seconds_caps'][role]
    cells=[nbf.v4.new_markdown_cell('# Span008 — matched size pilot'),nbf.v4.new_code_cell(f'''import os,sys,time,json,base64,subprocess
from pathlib import Path
STARTED=time.time();DEADLINE=STARTED+{cap}-120;WORK=Path('/kaggle/working')
os.environ.update(TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
TRAIN=next(p for p in [Path('/kaggle/input/{slug}'),Path('/kaggle/input/notebooks/{owner}/{slug}')] if (p/'runtime_requirements.json').exists())
PARENT=next(p for p in [Path('/kaggle/input/{adapter_slug}'),Path('/kaggle/input/notebooks/{adapter_owner}/{adapter_slug}')] if (p/'exposure/adapter/tokenizer.json').exists())
runtime=json.loads((TRAIN/'runtime_requirements.json').read_text())
subprocess.run([sys.executable,'-m','pip','install','-q','--no-index','--no-deps','--find-links',str(TRAIN/'runtime_wheels'),*runtime],check=True,timeout=240)
MODULES=WORK/'bundled_code';(MODULES/'arena').mkdir(parents=True,exist_ok=True);(MODULES/'arena/__init__.py').write_text('')
(WORK/'config.json').write_text({json.dumps(cfg)!r});(WORK/'base_config.json').write_text({json.dumps(base)!r});(WORK/'selection.json').write_text({json.dumps(sel)!r})
RAW=WORK/'pilot_raw';RAW.mkdir(exist_ok=True);(RAW/'train.csv').write_bytes(base64.b64decode({base64.b64encode(csv).decode()!r}))
''')];hashes={}
    for name in ['data','joint_data','joint_train','training_exposure','experiment','telemetry','span006_inputs','span008_size','span008_size_session']:
        path=REPO/f'src/arena/{name}.py';hashes[str(path.relative_to(REPO))]=sha256(path);cells.append(nbf.v4.new_code_cell(f"(MODULES/'arena/{name}.py').write_text({path.read_text()!r})"))
    cells.append(nbf.v4.new_code_cell('''def locate(size):
    return next(p for p in [Path('/kaggle/input/models/qwen-lm/qwen-3/transformers')/size/'1',Path('/kaggle/input/qwen-3/transformers')/size/'1'] if (p/'config.json').exists())
env=os.environ.copy();env.update(PYTHONPATH=str(MODULES),PYTHONUNBUFFERED='1')
command=[sys.executable,'-m','arena.span008_size_session','--raw',str(RAW),'--small-model',str(locate('4b')),'--large-model',str(locate('8b')),
    '--tokenizer-reference',str(PARENT/'exposure/adapter'),'--selection',str(WORK/'selection.json'),'--config',str(WORK/'config.json'),
    '--base-config',str(WORK/'base_config.json'),'--out',str(WORK/'size_pilot'),'--deadline',str(DEADLINE)]
'''));cells.append(nbf.v4.new_code_cell(worker_cell(role,'')))
    for cell in cells:
        if cell.cell_type=='code':compile(cell.source,'span008-size','exec')
    folder=root/'notebooks'/role;folder.mkdir(parents=True,exist_ok=True);nbf.write(nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}}),folder/'job.ipynb')
    metadata={'id':'arseniytrushin/llm-preference-span008-size-pilot','title':'LLM Preference Span008 Size Pilot','code_file':'job.ipynb','language':'python','kernel_type':'notebook',
        'is_private':True,'enable_gpu':True,'enable_internet':False,'competition_sources':[],'dataset_sources':[],
        'kernel_sources':[old['kernel']+'/'+str(old['version_number']),manifest['source_kernel']+'/'+str(manifest['version_number'])],
        'model_sources':['qwen-lm/qwen-3/transformers/4b/1','qwen-lm/qwen-3/transformers/8b/1']}
    atomic_json(folder/'kernel-metadata.json',metadata)
    actual=nbf.read(folder/'job.ipynb',4).cells[-1].source
    with tempfile.TemporaryDirectory(prefix='span008-size-cell-') as tmp:
        scope={'WORK':Path(tmp),'STARTED':time.time(),'DEADLINE':time.time()+10,'command':[sys.executable,'-c','print("size pilot full-cell rehearsal")'],'env':None}
        exec(compile(actual,'actual-serialized-final-cell','exec'),scope);record=read_json(Path(tmp)/'session_summary.json');assert record['role']==role and record['exit_code']==0
    for n in ['scripts/build_span_008_size.py','scripts/run_span_008_job.py','src/arena/notebook_worker.py','src/arena/kaggle_control.py','GOAL.md','configs/span_008.json','docs/SPAN_008_PLAN.md']:hashes[n]=sha256(REPO/n)
    atomic_json(pub/'size_pilot_bundle.json',{'role':role,'source_sha256':hashes,'notebook_sha256':sha256(folder/'job.ipynb'),'metadata_sha256':sha256(folder/'kernel-metadata.json'),'cap_seconds':cap,'internet_enabled':False})
    atomic_json(pub/'size_pilot_cell_rehearsal.json',{'passed':True,'serialized_last_cell_sha256':__import__('hashlib').sha256(actual.encode()).hexdigest(),'result':record,'new_gpu_executions':0,
        'sample_rows':{k:len(v) for k,v in sel.items()},'sample_ids_sha256':sha256(root/'size_sample_ids.json'),'confirmation_targets_loaded':False,'raw_competition_csv_attached':False})
    print(folder)


if __name__=='__main__':main()
