"""Reuse the verified native runtime; freeze a single concurrency ablation notebook."""
from pathlib import Path
import sys
import nbformat as nbf
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,sha256

root=ROOT/'runs/span_007';pub=REPO/'results/span_007';role='judge_stability'
assert not (root/'jobs'/role/'push_intent.json').exists()
cfg=read_json(REPO/'configs/span_007.json');ids=read_json(root/'sample_ids.json')['judge']
parent=Path(read_json(pub/'artifact_inventory.json')['input_artifacts']['native_parent']['path'])
nb=nbf.read(parent,as_version=4);nb.cells=nb.cells[:6]
old='DEADLINE=SESSION_STARTED+1680-60';assert old in nb.cells[1].source
nb.cells[1].source=nb.cells[1].source.replace(old,'DEADLINE=SESSION_STARTED+2400-120')
nb.cells[0].source='# Span007 E05 — fixed native judge, client concurrency1/16, cache-off'
sources=['data','judge','judge_runtime','span_repeatability','experiment','span007_judge']
for name in ['span_repeatability','experiment','span007_judge']:
    path=REPO/f'src/arena/{name}.py';nb.cells.append(nbf.v4.new_code_cell(f"(MODULES/'{name}.py').write_text({path.read_text()!r})"))
native_inputs=read_json(REPO/'results/span_004/artifact_inventory.json');manifest=read_json(native_inputs['portable_manifest']['path'])
import hashlib,json
canonical=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()
nb.cells.append(nbf.v4.new_code_cell(f'''import hashlib
assert hashlib.sha256(json.dumps(json.loads((BUNDLE/'manifest.json').read_text()),sort_keys=True,separators=(',',':')).encode()).hexdigest()=={canonical!r}
from arena.span007_judge import run
result=run(CONFIG,EXE,RAW,WORK/'judge_stability',{ {'sentinel':ids['sentinel'],'permutation_indices':ids['permutation_indices']}!r},{cfg!r})
(WORK/'session_summary.json').write_text(json.dumps({{'seconds':time.time()-SESSION_STARTED,'complete':True,'competition_labels_accessed':False}}))
signal.alarm(0)
'''))
for cell in nb.cells:
    if cell.cell_type=='code':compile(cell.source,'span007-native','exec')
folder=root/'notebooks'/role;folder.mkdir(parents=True,exist_ok=True);nbf.write(nb,folder/'job.ipynb')
metadata=read_json(parent.parent/'kernel-metadata.json');metadata.update(id='arseniytrushin/llm-preference-span-007-judge-stability',
    title='LLM Preference Span007 Judge Stability',enable_internet=False)
atomic_json(folder/'kernel-metadata.json',metadata)
paths=[f'src/arena/{name}.py' for name in sources]+['scripts/build_span_007_judge.py','scripts/run_experiment.py','src/arena/kaggle_control.py',
       'GOAL.md','configs/span_007.json','docs/SPAN_007_PLAN.md']
atomic_json(pub/(role+'_bundle.json'),{'role':role,'source_sha256':{p:sha256(REPO/p) for p in paths},
    'notebook_sha256':sha256(folder/'job.ipynb'),'metadata_sha256':sha256(folder/'kernel-metadata.json'),
    'cap_seconds':2400,'internet_enabled':False,'historical_parent_sha256':sha256(parent)})
print(folder)
