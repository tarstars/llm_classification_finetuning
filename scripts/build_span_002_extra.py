"""Score only the remaining fixed 1536 rows, reusing the valid 512-row current cache."""
import hashlib
import json
from pathlib import Path
import sys

import nbformat as nbf

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,json_write

root=ROOT/'runs/span_002'
assert json.loads((root/'jobs/ensemble-consistent/status.json').read_text())['status'].endswith('ERROR')
source=root/'jobs/ensemble-consistent/output/cached_current'
summary=json.loads((source/'judge_summary.json').read_text())
assert summary['rows']==512 and not summary['both_orders']
frozen=json.loads((root/'frozen_consistent_blend.json').read_text())
assert frozen['current_judge_predictions_sha256']==hashlib.sha256((source/'judge_predictions.npz').read_bytes()).hexdigest()
nb=nbf.read(root/'notebooks/ensemble-recovery/job.ipynb',as_version=4)
nb.cells[0]=nbf.v4.new_markdown_cell('''# Span 002 — remaining fixed H37 rows

The prior stage scored all 512 current-runtime control rows, then failed on a
missing import during recipe export. Those valid predictions are reused. The
recipe is now fitted/frozen locally before this job. Score only the original extra
1536 IDs under the same repaired model/prompt/packing/single-order runtime.
''')
for cell in nb.cells:
    if cell.cell_type=='code':
        cell.source=cell.source.replace('DEADLINE=SESSION_STARTED+4500','DEADLINE=SESSION_STARTED+3480')
nb.cells[-1]=nbf.v4.new_code_cell(f'''import hashlib
from arena.judge import run
from arena.data import json_write
FROZEN={frozen!r}
json_write(WORK/'current_frozen_blend.json',FROZEN)
gate={{'reused_cached_rows':512,'remaining_rows':len(SAMPLE['ids']),
    'projected_extra_seconds_with_margin':{summary['seconds']!r}/512*len(SAMPLE['ids'])*1.2+600,
    'remaining_session_seconds':DEADLINE-time.time(),
    'old_cache_pooled':False,'frozen_recipe_at_unix':FROZEN['frozen_at_unix']}}
gate['throughput_passed']=gate['projected_extra_seconds_with_margin']<gate['remaining_session_seconds']
json_write(WORK/'consistent_gate.json',gate)
print('CONSISTENT_GATE',json.dumps(gate),flush=True)
if gate['throughput_passed']:
    started=time.time()
    result=run(CONFIG,EXE,RAW,WORK/'gemma26','gemma26','dev',sample=SAMPLE,both_orders=False)
    session={{'expanded':True,'cached_rows':512,'extra_rows':result['rows'],'extra_scoring_started_at_unix':started}}
else:
    session={{'expanded':False,'reason':'Insufficient measured runtime; no extra scoring'}}
json_write(WORK/'session_summary.json',{{'session_seconds':time.time()-SESSION_STARTED,**session}})
session
''')
for cell in nb.cells:
    if cell.cell_type=='code':compile(cell.source,'span002-extra-cell','exec')
dest=root/'notebooks/ensemble-extra'
dest.mkdir(parents=True,exist_ok=True)
assert not (root/'jobs/ensemble-extra/push_intent.json').exists()
nbf.write(nb,dest/'job.ipynb')
metadata=json.loads((root/'notebooks/ensemble-recovery/kernel-metadata.json').read_text())
metadata.update(id='arseniytrushin/llm-preference-span-002-judge-extra',title='LLM Preference Span 002 Judge Extra')
json_write(dest/'kernel-metadata.json',metadata)
json_write(REPO/'results/span_002/extra_continuation_plan.json',{
    'reason':'Completed 512-row scoring retained; NameError at export due missing hashlib import; no extra rows had been requested.',
    'source_kernel':'arseniytrushin/llm-preference-span-002-judge-consistent/2','reused_cached_rows':512,
    'remaining_rows':1536,'remaining_sample_unchanged':True,
    'cached_predictions_sha256':frozen['current_judge_predictions_sha256'],
    'frozen_recipe_sha256':hashlib.sha256((root/'frozen_consistent_blend.json').read_bytes()).hexdigest(),
    'max_session_seconds':3480,'projected_seconds_with_margin':summary['seconds']/512*1536*1.2+600,
    'funding':'3480 of previously unallocated 6300 seconds; conservative role ceilings sum 40380 <= 43200.',
    'notebook_sha256':hashlib.sha256((dest/'job.ipynb').read_bytes()).hexdigest()})
print(dest)
