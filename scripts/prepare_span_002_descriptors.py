"""Verify cached packing descriptors and freeze a matched raw-order reference."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,json_write
from arena.span_heads import order_diagnostics,normalize


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


root=ROOT/'runs/span_002'
inventory=json.loads((root/'artifact_inventory.json').read_text())
fingerprint_path=ROOT/'eda/attempt_002/feature_fingerprint.json'
feature_path=ROOT/'eda/attempt_002/features.parquet'
fingerprint=json.loads(fingerprint_path.read_text())
tokenizer=ROOT/'runs/attempt_002/jobs/qwen4b1024/output/joint_run/adapter/tokenizer.json'
parent_tokenizer=root/'baseline_resume/joint_run/checkpoints/checkpoint-800/tokenizer.json'
assert digest(parent_tokenizer)==fingerprint['tokenizer_sha256'], 'Resumed tokenizer differs from descriptor cache'
for key,path in [('raw_sha256',ROOT/'raw/train.csv'),('tokenizer_sha256',tokenizer),
                 ('preprocessing_sha256',REPO/'src/arena/joint_data.py'),
                 ('feature_program_sha256',REPO/'src/arena/eda_features.py')]:
    assert digest(path)==fingerprint[key],key
assert digest(inventory['split']['path'])==inventory['split']['sha256']
features=pd.read_parquet(feature_path,columns=['id','label','split','total_content_tokens',
    'cut_1024_cap256','prompt_cut_1024_cap256','answer_cut_1024_cap256','retained_fraction_1024_cap256'])
assert features.id.is_unique
features=features.set_index('id')
split=json.loads(Path(inventory['split']['path']).read_text())
assert set(features.index[features.split=='dev'])==set(split['dev'])
assert set(features.index[features.split=='confirmation'])==set(split['confirmation'])


def retention(ids):
    f=features.loc[ids]
    raw=f.total_content_tokens.sum()
    kept=float((f.total_content_tokens*f.retained_fraction_1024_cap256).sum())
    return {'rows':len(f),'any_field_cut_fraction':float(f.cut_1024_cap256.mean()),
        'prompt_cut_fraction':float(f.prompt_cut_1024_cap256.mean()),
        'answer_cut_fraction':float(f.answer_cut_1024_cap256.mean()),
        'mean_row_content_retained_fraction':float(f.retained_fraction_1024_cap256.mean()),
        'corpus_content_retained_fraction':kept/max(int(raw),1),
        'raw_content_tokens':int(raw),'retained_content_tokens':int(round(kept))}


baseline=np.load(inventory['baseline']['path'])
joint=np.load(inventory['joint']['path'])
for name in ['baseline','joint']:
    assert digest(inventory[name]['path'])==inventory[name]['sha256']
    data=baseline if name=='baseline' else joint
    np.testing.assert_array_equal(data['labels'],features.loc[data['ids'],'label'])
ids=joint['ids']
files=[ROOT/f'runs/span_001/jobs/joint/output/joint/features/features_rank{rank}.npz' for rank in (0,1)]
pieces=[np.load(path) for path in files]
all_ids=np.concatenate([p['ids'] for p in pieces])
assert len(all_ids)==len(set(all_ids))
lookup={int(rid):i for i,rid in enumerate(all_ids)}
idx=[lookup[int(rid)] for rid in ids]
assert not np.concatenate([p['feature_fit'] for p in pieces])[idx].any()
original=np.concatenate([p['original'] for p in pieces])[idx]
swapped=np.concatenate([p['aligned_swap'] for p in pieces])[idx]
reference=normalize((original+swapped)/2)
base_lookup={int(rid):i for i,rid in enumerate(baseline['ids'])}
drift=np.abs(reference-baseline['probabilities'][[base_lookup[int(rid)] for rid in ids]])
assert drift.mean()<.002 and drift.max()<.03
np.testing.assert_array_equal(joint['frozen_order_disagreement'],
                             (original.argmax(1)!=swapped.argmax(1)).astype(float))
destination=root/'frozen_order_reference.npz'
if destination.exists():
    old=np.load(destination)
    for key,value in [('ids',ids),('labels',joint['labels']),('original',original),('aligned_swap',swapped)]:
        np.testing.assert_array_equal(old[key],value)
else:
    np.savez_compressed(destination,ids=ids,labels=joint['labels'],original=original,aligned_swap=swapped)
cohorts={'training_partition':retention(features.index[features.split=='train']),
         'full_development':retention(baseline['ids']),'order_reference':retention(ids)}
expanded=root/'expanded_evaluated_predictions.npz'
if expanded.exists():cohorts['ensemble_combined']=retention(np.load(expanded)['ids'])
judge_paths=[root/'jobs/ensemble-consistent/output/cached_current/judge_summary.json',
             root/'jobs/ensemble-extra/output/gemma26/judge_summary.json']
judge_summaries=[json.loads(path.read_text()) for path in judge_paths]
assert all(not s['both_orders'] and s['prompt_overhead']==239 for s in judge_summaries)
judge_rows=sum(s['rows'] for s in judge_summaries)
judge_cut=sum(s['truncated_rows'] for s in judge_summaries)
result={'checked_at':datetime.now(timezone.utc).isoformat(),'confirmation_evaluated':False,
    'method':'Reuse EDA content-token accounting under unchanged original-order 1024-token packing; marker tokens excluded. Cached labels checked for row alignment only.',
    'max_length':1024,'cohorts':cohorts,'frozen_order_reference':{
        'rows':len(ids),'diagnostics':order_diagnostics(original,swapped),
        'reload_mean_probability_difference':float(drift.mean()),
        'reload_max_probability_difference':float(drift.max()),
        'note':'Coverage: original raw orientations of 1,024 development rows from span-001 frozen-weight extraction.'},
    'native_judge':{'rows':judge_rows,'truncated_rows':judge_cut,
        'truncated_fraction':judge_cut/judge_rows,
        'mean_packed_tokens':sum(s['rows']*s['packed_tokens_mean'] for s in judge_summaries)/judge_rows,
        'prompt_overhead_tokens':239,'both_orders_measured':False,
        'order_disagreement':None,
        'order_note':'The single-order source summary uses zero as an order-flip placeholder; it is not measured invariance.',
        'gpu_count':2,'physical_memory_MiB_per_gpu':15360,'peak_allocated_bytes':None,
        'memory_note':'Two-T4 execution succeeded; peak native GPU allocation was not captured. Capacity is not peak usage.'},
    'input_sha256':{'features':digest(feature_path),'feature_fingerprint':digest(fingerprint_path),
        'raw_data':fingerprint['raw_sha256'],'tokenizer':fingerprint['tokenizer_sha256'],
        'parent_checkpoint_tokenizer':digest(parent_tokenizer),
        'preprocessing':fingerprint['preprocessing_sha256'],'feature_program':fingerprint['feature_program_sha256'],
        'frozen_rank0':digest(files[0]),'frozen_rank1':digest(files[1]),
        'frozen_order_reference':digest(destination),'baseline':inventory['baseline']['sha256'],
        'current_judge_cached_summary':digest(judge_paths[0]),'current_judge_extra_summary':digest(judge_paths[1]),
        'joint':inventory['joint']['sha256'],'source':digest(Path(__file__))}}
for directory in [root,REPO/'results/span_002']:json_write(directory/'packing_diagnostics.json',result)
print(json.dumps(result,indent=2))
