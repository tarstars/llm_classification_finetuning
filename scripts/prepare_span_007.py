"""Freeze exact v7 data/cohorts and initial resources, without confirmation targets."""
from pathlib import Path
import json
import shutil
import sys

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT, prompt_groups
from arena.experiment import sha256, read_json, atomic_json, utc_now
from arena.joint_data import JointTokenizer
from arena.span006_inputs import token_digest, text_digest


def main():
    out = ROOT / 'runs/span_007'; pub = REPO / 'results/span_007'
    assert not (pub / 'artifact_inventory.json').exists(), 'Prepared inputs are immutable'
    cfg = read_json(REPO / 'configs/span_007.json')
    old = read_json(REPO / 'results/span_006/artifact_inventory.json')
    inventory = read_json(pub / 'resource_inventory_initial.json')
    assert inventory['quota']['remaining_seconds'] >= cfg['gpu_session_seconds_cap']
    assert not inventory['nonterminal_jobs'] and not inventory['quota']['paid_scaling']
    entries = dict(old['input_artifacts'])
    for entry in entries.values(): assert sha256(entry['path']) == entry['sha256']
    dev = pd.read_parquet(entries['dev.parquet']['path'])
    train = pd.read_parquet(entries['train.parquet']['path'])
    adapter = Path(entries['adapter']['path']).parent
    tokenizer = AutoTokenizer.from_pretrained(str(adapter), local_files_only=True, padding_side='right')
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    small = JointTokenizer(tokenizer, 1024).encode_frame(dev)
    large = JointTokenizer(tokenizer, 2048).encode_frame(dev)
    changed = np.array([a != b or c != d for a,b,c,d in zip(small[0],large[0],small[1],large[1])])
    # Reconstruct the original selection cohort, then take a fixed subset for monitoring.
    raw_ids = pd.read_csv(entries['raw_train']['path'], usecols=['id']).id.to_numpy()
    dev_positions = np.flatnonzero(np.isin(raw_ids, dev.id))
    original_selection = np.sort(np.random.default_rng(20261102).choice(dev_positions,1024,replace=False))
    selection = np.sort(np.random.default_rng(cfg['seed']).choice(original_selection,cfg['selection_rows'],replace=False))
    sample = read_json(ROOT / 'runs/span_006/sample_ids.json')
    judge_sample = read_json(ROOT / 'runs/span_004/sample_ids.json')
    ids = {'selection':raw_ids[selection].tolist(), 'smoke':sample['smoke'],
           'context':dev.id[changed].tolist(), 'development':dev.id.tolist(),
           'confirmation':sample['confirmation'], 'judge':judge_sample}
    atomic_json(out / 'sample_ids.json',ids)
    shutil.copyfile(ROOT / 'runs/span_006/smoke_reference.npz',out / 'smoke_reference.npz')
    for name,path in {'sample_ids':out/'sample_ids.json','smoke_reference':out/'smoke_reference.npz',
                      'confirmation_reference':ROOT/'runs/span_006/jobs/confirmation/output/confirmation/confirmation_predictions.npz',
                      'native_parent':ROOT/'runs/span_004/notebooks/repeatability_retry/job.ipynb',
                      'source_predictions':ROOT/'runs/span_006/freeze/source_predictions.npz',
                      'source_bundle':ROOT/'runs/span_006/freeze/source_bundle.joblib'}.items():
        entries[name]={'path':str(path),'sha256':sha256(path)}
    subset=lambda packed:[packed[i] for i in np.flatnonzero(changed)]
    token_hashes={f'{size}_{view}':token_digest(subset(packed)) for size,enc in [(1024,small),(2048,large)]
                  for view,packed in [('original',enc[0]),('swapped',enc[1])]}
    result={'span':7,'prepared_at':utc_now(),'started_at':cfg['started_at'],'deadline':cfg['deadline'],
            'confirmation_targets_read':False,'input_artifacts':entries,'train_rows':len(train),'dev_rows':len(dev),
            'dev_groups':int(prompt_groups(dev).nunique()),'context_changed_rows':int(changed.sum()),
            'context_text_sha256':text_digest(dev[changed]),'context_token_sha256':token_hashes,
            'old_snapshot_sha256':{str(p.relative_to(REPO)):sha256(p) for p in (REPO/'backlog/versions').glob('*') if p.is_file()},
            'user_notebook_sha256':{p:sha256(REPO/p) for p in ['notebooks/07_error_poster.ipynb','notebooks/12_backlog_v1_span.ipynb']},
            'contract_sha256':{p:sha256(REPO/p) for p in ['GOAL.md','configs/span_007.json','docs/SPAN_007_PLAN.md']}}
    plan={k:cfg[k] for k in ['span','started_at','deadline','gpu_session_seconds_cap']}
    plan.update(initial_quota_refresh_time=inventory['quota']['refresh_time'],quota_used_at_start=inventory['quota']['used_seconds'])
    for folder in [pub,out]: atomic_json(folder/'artifact_inventory.json',result); atomic_json(folder/'span_plan.json',plan)
    print(json.dumps({k:result[k] for k in ['train_rows','dev_rows','context_changed_rows','confirmation_targets_read']},indent=2))


if __name__=='__main__': main()
