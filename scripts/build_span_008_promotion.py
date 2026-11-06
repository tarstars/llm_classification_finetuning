"""Build one exact notebook for capped qualification and its competition rerun."""
from pathlib import Path
import base64,json,sys,tempfile,time
from types import SimpleNamespace
import numpy as np
import nbformat as nbf
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,sha256
from arena.notebook_worker import worker_cell


def main():
    root=ROOT/'runs/span_008';pub=REPO/'results/span_008';cfg=read_json(REPO/'configs/span_008.json');chosen=read_json(pub/'candidate_selection.json');assert chosen['eligible'] and chosen['candidate_kind']=='context_completion'
    assert read_json(pub/'chunk_audit.json')['passed'];assert not (root/'jobs/promotion/push_intent.json').exists() and not (root/'confirmation_reveal_intent.json').exists()
    inv=read_json(pub/'artifact_inventory.json');manifest=read_json(inv['input_artifacts']['classifier_manifest']['path']);manifest.update(tokenizer_sha256=sha256(Path(inv['input_artifacts']['adapter']['path']).parent/'tokenizer.json'),candidate_selection_sha256=sha256(pub/'candidate_selection.json'))
    ids=np.load(inv['input_artifacts']['confirmation_reference']['path'])['ids'].tolist();visible=sha256(ROOT/'raw/test.csv')
    nb=nbf.read(root/'notebooks/context_completion/job.ipynb',4);nb.cells[0]=nbf.v4.new_markdown_cell('# Span008 — frozen conditional context qualification and submission')
    old='STARTED=time.time();DEADLINE=STARTED+1800-120;WORK=Path(\'/kaggle/working\')'
    replacement=f'''STARTED=time.time();WORK=Path('/kaggle/working')
RAW=next(p for p in [Path('/kaggle/input/llm-classification-finetuning'),Path('/kaggle/input/competitions/llm-classification-finetuning')] if (p/'test.csv').exists())
import hashlib
IS_RERUN=os.environ.get('KAGGLE_IS_COMPETITION_RERUN','').strip().lower() in ['true','1'] or hashlib.sha256((RAW/'test.csv').read_bytes()).hexdigest()!={visible!r}
DEADLINE=STARTED+(25200 if IS_RERUN else 5400)-120'''
    assert old in nb.cells[1].source;nb.cells[1].source=nb.cells[1].source.replace(old,replacement)
    # Retain the proven offline setup, pinned weights, and exact original smoke reference.
    prep=nb.cells[-2].source.split('command=[',1)[0]
    prep+=f"(WORK/'manifest.json').write_text({json.dumps(manifest)!r})\n(WORK/'confirmation_ids.json').write_text({json.dumps({'confirmation':ids})!r})\n"
    prep+="""command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','--module','arena.span008_context_export',
    '--raw',str(RAW),'--model',str(MODEL),'--adapter',str(PARENT/'exposure/adapter'),'--manifest',str(WORK/'manifest.json'),
    '--selection',str(WORK/'confirmation_ids.json'),'--reference',str(REFERENCES/'smoke_reference.npz'),'--config',str(WORK/'config.json'),
    '--out',str(WORK/'promotion'),'--deadline',str(DEADLINE)]+([] if IS_RERUN else ['--qualify'])
"""
    nb.cells=nb.cells[:-2];sources=read_json(pub/'context_completion_bundle.json')['source_sha256'].copy()
    for name in ['context_route','span008_context_export']:
        path=REPO/f'src/arena/{name}.py';sources[str(path.relative_to(REPO))]=sha256(path)
        nb.cells.append(nbf.v4.new_code_cell(f"(MODULES/{(name+'.py')!r}).write_text({path.read_text()!r})"))
    nb.cells +=[nbf.v4.new_code_cell(prep),nbf.v4.new_code_cell(worker_cell('promotion',''))]
    for cell in nb.cells:
        if cell.cell_type=='code':compile(cell.source,'span008-promotion','exec')
    folder=root/'notebooks/promotion';folder.mkdir(parents=True,exist_ok=True);nbf.write(nb,folder/'job.ipynb')
    metadata=read_json(root/'notebooks/context_completion/kernel-metadata.json');metadata.update(id='arseniytrushin/llm-preference-span008-context-promotion',title='LLM Preference Span008 Context Promotion')
    atomic_json(folder/'kernel-metadata.json',metadata)
    actual=nbf.read(folder/'job.ipynb',4).cells[-1].source
    with tempfile.TemporaryDirectory(prefix='span008-promotion-cell-') as tmp:
        scope={'WORK':Path(tmp),'STARTED':time.time(),'DEADLINE':time.time()+10,'command':[sys.executable,'-c','print("promotion full-cell rehearsal")'],'env':None}
        exec(compile(actual,'actual-serialized-final-cell','exec'),scope);record=read_json(Path(tmp)/'session_summary.json');assert record['role']=='promotion' and record['exit_code']==0
    switch='\n'.join(line for line in nb.cells[1].source.splitlines() if line.startswith(('IS_RERUN=','DEADLINE=')))
    branch_tests=[]
    for environment,digest,expected in [({},visible,False),({},'changed-input-hash',True),({'KAGGLE_IS_COMPETITION_RERUN':'true'},visible,True)]:
        scope={'os':SimpleNamespace(environ=environment),'hashlib':SimpleNamespace(sha256=lambda data:SimpleNamespace(hexdigest=lambda:digest)),'RAW':ROOT/'raw','STARTED':0}
        exec(compile(switch,'actual-serialized-mode-switch','exec'),scope)
        assert scope['IS_RERUN']==expected and scope['DEADLINE']==(25200 if expected else 5400)-120
        branch_tests.append({'expected_hidden':expected,'actual_hidden':scope['IS_RERUN'],'deadline_seconds':scope['DEADLINE'],'passed':True})
    for path in ['scripts/build_span_008_promotion.py','scripts/evaluate_span_008_promotion.py','docs/SPAN_008_H50_PROTOCOL.md','results/span_008/candidate_selection.json']:sources[path]=sha256(REPO/path)
    atomic_json(pub/'promotion_bundle.json',{'role':'promotion','source_sha256':sources,'notebook_sha256':sha256(folder/'job.ipynb'),'metadata_sha256':sha256(folder/'kernel-metadata.json'),
        'cap_seconds':5400,'hidden_cap_seconds':25200,'internet_enabled':False,'visible_test_sha256':visible,'confirmation_ids_sha256':__import__('hashlib').sha256(json.dumps(ids).encode()).hexdigest()})
    atomic_json(pub/'promotion_cell_rehearsal.json',{'passed':True,'serialized_last_cell_sha256':__import__('hashlib').sha256(actual.encode()).hexdigest(),'result':record,'new_gpu_executions':0,
        'public_service_seconds_cap':5400,'competition_worker_seconds_cap':25200,'same_prediction_function_for_qualification_and_test':True})
    value=read_json(pub/'promotion_cell_rehearsal.json');value['serialized_mode_switch_tests']=branch_tests;atomic_json(pub/'promotion_cell_rehearsal.json',value)
    print(folder)


if __name__=='__main__':main()
