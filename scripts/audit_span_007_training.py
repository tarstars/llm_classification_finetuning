"""Audit actual two-rank exposure, resumed state and saved training telemetry."""
from pathlib import Path
import json
import sys
import time
import numpy as np
import pandas as pd
from safetensors.torch import load_file
from transformers import AutoTokenizer

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,CLASSES
from arena.experiment import read_json,atomic_json,sha256,verify_sources,utc_now
from arena.joint_data import JointTokenizer
from arena.telemetry import restart_factor


def main():
    started=time.monotonic();pub=REPO/'results/span_007';run=ROOT/'runs/span_007';job=run/'jobs/continuation'
    assert read_json(job/'controller_state.json')['stage']=='complete'
    verify_sources(REPO,read_json(pub/'continuation_bundle.json')['source_sha256'])
    inv=read_json(pub/'artifact_inventory.json');out=job/'output/continuation';summary=read_json(out/'summary.json');source=read_json(out/'resume_source.json')
    assert summary['complete'] and sha256(out/'adapter/adapter_model.safetensors')==summary['adapter_sha256']
    tensors=load_file(str(out/'adapter/adapter_model.safetensors'));assert all(t.isfinite().all() for t in tensors.values())
    train=pd.read_parquet(inv['input_artifacts']['train.parquet']['path']);y=train[CLASSES].to_numpy().argmax(1)
    tokenizer=AutoTokenizer.from_pretrained(str(Path(inv['input_artifacts']['adapter']['path']).parent),local_files_only=True,padding_side='right')
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    original,swapped,_=JointTokenizer(tokenizer,1024).encode_frame(train)
    lookup={int(i):(int(label),{len(a),len(b)}) for i,label,a,b in zip(train.id,y,original,swapped)}
    ranks=[];skip_vectors=[];global_ids=set();total_tokens=total_rows=0
    for rank in [0,1]:
        events=[json.loads(line) for line in (out/f'exposure_rank{rank}.jsonl').read_text().splitlines()]
        assert len(events)==summary['new_cycles']
        assert [e['global_step'] for e in events]==list(range(1439,summary['final_step']+1))
        assert events[0]['scheduler_before']==source['parent_scheduler_position']
        assert events[0]['scale_before']==source['parent_amp_scale']
        counts={'cycles':0,'accepted_updates':0,'skipped_updates':0,'processed_tokens':0,'accepted_tokens':0,'processed_rows':0,'accepted_rows':0}
        local_ids=set();previous=source['parent_scheduler_position'];skips=[];class_counts=np.zeros(3,int)
        for event in events:
            assert event['rank']==rank and event['scheduler_before']==previous
            skip=event['skipped'];skips.append(skip);assert event['scheduler_after']-previous==int(not skip);previous=event['scheduler_after']
            assert event['scale_after']>0 and (not skip or event['scale_after']<event['scale_before'])
            ids,labels,lengths=event['ids'],event['canonical_labels'],event['token_lengths']
            assert len(ids)==len(labels)==len(lengths)>0
            for rid,label,length in zip(ids,labels,lengths):
                expected_label,expected_lengths=lookup[rid];assert label==expected_label and length in expected_lengths
            counts['cycles']+=1;counts['skipped_updates' if skip else 'accepted_updates']+=1
            counts['processed_rows']+=len(ids);counts['processed_tokens']+=sum(lengths)
            if not skip:
                counts['accepted_rows']+=len(ids);counts['accepted_tokens']+=sum(lengths);local_ids.update(ids);global_ids.update(ids)
                class_counts+=np.bincount(labels,minlength=3)
            assert counts==event['cumulative']
        reported=read_json(out/f'exposure_summary_rank{rank}.json')
        for key,value in counts.items():assert reported[key]==value
        assert reported['accepted_unique_rows']==len(local_ids);np.testing.assert_array_equal(reported['accepted_class_counts'],class_counts)
        total_tokens+=counts['accepted_tokens'];total_rows+=counts['accepted_rows'];ranks.append(reported);skip_vectors.append(skips)
    assert skip_vectors[0]==skip_vectors[1],'DDP ranks disagree on actual update acceptance'
    telemetry=[json.loads(line) for line in (out/'telemetry/events.jsonl').read_text().splitlines()]
    cfg=read_json(REPO/'configs/span_007.json')
    for event in telemetry:
        if event['event']=='cycle':
            m=event['metrics'];expected=cfg['training_lr']*restart_factor(m['scheduler_position'],source['parent_scheduler_position'],16,320)
            np.testing.assert_allclose(m['learning_rate'],expected,atol=1e-15)
    hardware=[read_json(out/f'gpu_rank{i}.json') for i in [0,1]]
    assert {r['device'] for r in hardware}=={0,1} and all('T4' in r['name'] and r['peak_allocated_bytes']>0 for r in hardware)
    for relative,digest in inv['old_snapshot_sha256'].items():assert sha256(REPO/relative)==digest
    for relative,digest in inv['user_notebook_sha256'].items():assert sha256(REPO/relative)==digest
    stop_reasons=[]
    if summary['stopped_for_budget']:stop_reasons.append('budget')
    if summary['stopped_for_instability']:stop_reasons.append('monitoring regression')
    if len(skip_vectors[0])>=8 and all(skip_vectors[0][-8:]):stop_reasons.append('eight consecutive AMP skips')
    if summary['new_cycles']==320:stop_reasons.append('fixed cycle endpoint')
    proof={'passed':True,'checked_at':utc_now(),'new_nominal_cycles':summary['new_cycles'],'new_logical_accepted_updates':ranks[0]['accepted_updates'],
           'new_logical_skipped_updates':ranks[0]['skipped_updates'],'global_accepted_rows':total_rows,'global_accepted_unique_rows':len(global_ids),
           'global_accepted_packed_tokens':total_tokens,'per_rank':ranks,'hardware':hardware,'stop_reasons':stop_reasons,
           'parent_optimizer_fingerprint_checked_before_updates':source['parent_optimizer_restored'],
           'token_lengths_reconstructed':True,'confirmation_target_columns_loaded':True,'confirmation_targets_used':False,
           'finite_adapter_tensors':len(tensors),'audit_seconds':time.monotonic()-started,'source_sha256':sha256(Path(__file__)),
           'adapter_sha256':summary['adapter_sha256'],'summary_sha256':sha256(out/'summary.json'),
           'trace_sha256':{f'exposure_rank{i}.jsonl':sha256(out/f'exposure_rank{i}.jsonl') for i in [0,1]}}
    atomic_json(pub/'training_audit.json',proof);print({k:v for k,v in proof.items() if k not in ['per_rank','hardware','trace_sha256']})


if __name__=='__main__':main()
