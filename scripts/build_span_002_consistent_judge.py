"""Re-score all fixed H37 rows uniformly after the old-cache identity gate failed."""
import hashlib
import json
from pathlib import Path
import sys

import nbformat as nbf
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,json_write

root=ROOT/'runs/span_002'
recovery=root/'jobs/ensemble-recovery'
assert json.loads((recovery/'status.json').read_text())['status'].endswith('COMPLETE')
evidence=json.loads((REPO/'results/span_002/live_progress_ensemble-recovery.json').read_text())
assert evidence['RECOVERY_GATE']['throughput_passed'] and not evidence['RECOVERY_GATE']['identity_passed']
nb=nbf.read(root/'notebooks/ensemble-recovery/job.ipynb',as_version=4)
dest=root/'notebooks/ensemble-consistent'
dest.mkdir(parents=True,exist_ok=True)
assert not (root/'jobs/ensemble-consistent/push_intent.json').exists(), 'Preserve any existing launch intent'
cached=np.load(ROOT/'runs/attempt_003/jobs/development_ctx1024/output/gemma26/judge_predictions.npz')
baseline=np.load(ROOT/'runs/attempt_002/jobs/qwen4b1024/output/joint_run/dev_predictions.npz')
lookup={int(rid):i for i,rid in enumerate(baseline['ids'])}
classifier=baseline['probabilities'][[lookup[int(rid)] for rid in cached['ids']]]
np.testing.assert_array_equal(cached['labels'],baseline['labels'][[lookup[int(rid)] for rid in cached['ids']]])
config=json.loads((REPO/'configs/span_002.json').read_text())
nb.cells[0]=nbf.v4.new_markdown_cell('''# Span 002 — H37 consistent current-runtime cohort

The original-cache numerical gate failed and remains reported. Re-score all fixed
2048 IDs under the repaired current backend: 512 first, freeze its one-weight
recipe, then score the same preselected extra 1536 only if runtime fits. No pooling
with historical judge outputs. Same model, prompt, packing and single-order view.
''')
sources={}
for name in ['span_heads','span_calibration']:
    path=REPO/f'src/arena/{name}.py'
    sources[str(path.relative_to(REPO))]=hashlib.sha256(path.read_bytes()).hexdigest()
    nb.cells.insert(len(nb.cells)-1,nbf.v4.new_code_cell(f"(MODULES/{(name+'.py')!r}).write_text({path.read_text()!r})"))
nb.cells[-1]=nbf.v4.new_code_cell(f'''from arena.judge import run
from arena.data import parse_turns,prompt_groups,json_write
from arena.span_calibration import train_blend
import numpy as np,pandas as pd,importlib.metadata,hashlib
SPAN_CONFIG={config!r}
CACHED_IDS={cached['ids'].tolist()!r}
CLASSIFIER_RAW=np.asarray({classifier.tolist()!r})
EXPECTED_LABELS=np.asarray({cached['labels'].tolist()!r})
current_summary=run(CONFIG,EXE,RAW,WORK/'cached_current','gemma26','dev',sample={{'ids':CACHED_IDS}},both_orders=False)
current=np.load(WORK/'cached_current/judge_predictions.npz')
np.testing.assert_array_equal(current['ids'],CACHED_IDS)
np.testing.assert_array_equal(current['labels'],EXPECTED_LABELS)
prompts=pd.read_csv(RAW/'train.csv',usecols=['id','prompt']).set_index('id').loc[CACHED_IDS]
prompts['prompt']=prompts['prompt'].map(parse_turns)
groups=prompt_groups(prompts).to_numpy()
fits,weight=train_blend(CLASSIFIER_RAW,current['original'],EXPECTED_LABELS,groups,SPAN_CONFIG)
frozen={{'classifier_family':'temperature','judge_family':'temperature_tie',
    'classifier_theta':fits[0].theta.tolist(),'judge_theta':fits[1].theta.tolist(),
    'judge_weight':weight,'fit_rows':512,'fit_groups':len(set(groups)),
    'frozen_at_unix':time.time(),'current_judge_predictions_sha256':hashlib.sha256((WORK/'cached_current/judge_predictions.npz').read_bytes()).hexdigest(),
    'config_sha256':hashlib.sha256(json.dumps(SPAN_CONFIG,sort_keys=True).encode()).hexdigest(),
    'software':{{name:importlib.metadata.version(name) for name in ['numpy','scipy','scikit-learn']}},
    'evaluation':'Recipe fitted/frozen on uniformly re-scored 512 rows before extra 1536 scoring; old raw cache not pooled.'}}
json_write(WORK/'current_frozen_blend.json',frozen)
gate={{'cached_rows':512,'cached_scoring_seconds':current_summary['seconds'],
    'projected_extra_seconds_with_margin':current_summary['seconds']/512*len(SAMPLE['ids'])*1.2+600,
    'remaining_session_seconds':DEADLINE-time.time(),'old_cache_pooled':False,
    'frozen_recipe_at_unix':frozen['frozen_at_unix']}}
gate['throughput_passed']=gate['projected_extra_seconds_with_margin']<gate['remaining_session_seconds']
json_write(WORK/'consistent_gate.json',gate)
print('CONSISTENT_GATE',json.dumps(gate),flush=True)
if gate['throughput_passed']:
    extra_started=time.time()
    summary=run(CONFIG,EXE,RAW,WORK/'gemma26','gemma26','dev',sample=SAMPLE,both_orders=False)
    result={{'expanded':True,'cached_rows':512,'extra_rows':summary['rows'],'extra_scoring_started_at_unix':extra_started}}
else:
    result={{'expanded':False,'cached_rows':512,'reason':'Measured runtime does not fit; no extra scoring'}}
json_write(WORK/'session_summary.json',{{'session_seconds':time.time()-SESSION_STARTED,**result}})
result
''')
for cell in nb.cells:
    if cell.cell_type=='code':compile(cell.source,'span002-consistent-cell','exec')
nbf.write(nb,dest/'job.ipynb')
metadata=json.loads((root/'notebooks/ensemble-recovery/kernel-metadata.json').read_text())
metadata.update(id='arseniytrushin/llm-preference-span-002-judge-consistent',
                title='LLM Preference Span 002 Judge Consistent')
json_write(dest/'kernel-metadata.json',metadata)
json_write(REPO/'results/span_002/consistent_cohort_plan.json',{
    'cohort_rows':2048,'recomputed_cached_rows':512,'extra_rows':1536,'old_judge_cache_pooled':False,
    'maximum_session_seconds':4500,'funded_from':'Unused H42 allocation; original total/deadline unchanged.',
    'reason':'Recovery GPU speed passed but raw identity to old cache failed; use uniform current-runtime predictions.',
    'prior_failed_identity_gate':evidence['RECOVERY_GATE'],
    'sample_ids_unchanged':True,'extra_sample_sha256':hashlib.sha256((root/'ensemble_expansion_ids.json').read_bytes()).hexdigest(),
    'notebook_sha256':hashlib.sha256((dest/'job.ipynb').read_bytes()).hexdigest(),
    'source_sha256':sources,'protocol':'docs/SPAN_002_RUNTIME.md'})
print(dest)
