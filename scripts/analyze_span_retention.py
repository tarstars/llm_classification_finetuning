"""Reconstruct the fixed pilots' token retention without another GPU execution."""
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import numpy as np
from transformers import AutoTokenizer

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT, TEXT_COLUMNS, json_write
from arena.joint_data import read_competition
from arena.span_pack import SpanTokenizer


def main():
    root=ROOT/'runs/span_001'
    config=json.loads((REPO/'configs/span_001.json').read_text())
    samples=json.loads((root/'sample_ids.json').read_text())
    expected=json.loads((root/'artifact_inventory.json').read_text())['sample_ids_sha256']
    assert hashlib.sha256((root/'sample_ids.json').read_bytes()).hexdigest()==expected
    frame,_=read_competition(ROOT/'raw')
    ids=sorted(set(sum(samples.values(),[])))
    frame=frame.set_index('id').loc[ids]
    baseline=ROOT/'runs/attempt_002/jobs/qwen4b1024/output/joint_run/adapter'
    reward=root/'tokenizer_cache/models--Skywork--Skywork-Reward-V2-Qwen3-4B/snapshots'/config['reward_revision']
    report={'method':'CPU reconstruction on fixed IDs: baseline original orientation, reward common content; marker tokens excluded',
            'sample_ids_sha256':expected,'max_length':config['max_length'],'tokenizers':{}}
    private={'ids':np.array(ids)}
    for name,path in [('baseline',baseline),('reward',reward)]:
        tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        encoder=SpanTokenizer(tokenizer,config['max_length'])
        original=np.zeros((len(frame),3),dtype=int)
        retained=np.zeros_like(original)
        packed=np.zeros((len(frame),2),dtype=int)
        for index,row in enumerate(frame.itertuples()):
            fields=[tokenizer.encode(getattr(row,key),add_special_tokens=False) for key in TEXT_COLUMNS]
            selected=list(encoder.retained_parts(*fields))
            if name=='reward':
                _, rb, ra=encoder.retained_parts(fields[0],fields[2],fields[1])
                from arena.joint_data import head_tail
                selected[1]=head_tail(fields[1],min(len(selected[1]),len(ra)),encoder.truncated)
                selected[2]=head_tail(fields[2],min(len(selected[2]),len(rb)),encoder.truncated)
                sequences=encoder.reward_pair(*fields)
            else:
                sequences=[encoder.combine(*fields),encoder.combine(fields[0],fields[2],fields[1])]
            original[index]=[len(x) for x in fields]
            for k,(raw,part) in enumerate(zip(fields,selected)):
                kept=len(part)
                if len(raw)>kept and kept>len(encoder.truncated)+2:
                    kept-=len(encoder.truncated)
                retained[index,k]=kept
            packed[index]=[len(seq) for seq in sequences]
            assert max(packed[index])<=config['max_length']
        private[name+'_original_tokens']=original
        private[name+'_retained_content_tokens']=retained
        private[name+'_packed_tokens']=packed
        values={}
        for split,sample in samples.items():
            mask=np.isin(ids,sample)
            raw,keep,lengths=original[mask],retained[mask],packed[mask]
            values[split]={'rows':int(mask.sum()),'packed_length_quantiles':np.quantile(lengths,[0,.5,.9,.99,1]).tolist(),
                'any_field_clipped_fraction':float(np.mean((keep<raw).any(1))),
                'field_clipped_fraction':dict(zip(TEXT_COLUMNS,np.mean(keep<raw,axis=0).tolist())),
                'corpus_content_retained_fraction':dict(zip(TEXT_COLUMNS,(keep.sum(0)/np.maximum(raw.sum(0),1)).tolist())),
                'raw_tokens':dict(zip(TEXT_COLUMNS,raw.sum(0).tolist())),
                'retained_content_tokens':dict(zip(TEXT_COLUMNS,keep.sum(0).tolist()))}
        if name=='reward':
            files=sorted((root/'jobs/reward-tokenfix/output/reward').glob('reward_rank*.npz'))
            assert len(files)==2,'Both executed reward shards are required for packing reproduction'
            lookup={key:i for i,key in enumerate(ids)}
            reproduced=0
            for file in files:
                executed=np.load(file)
                indices=[lookup[int(key)] for key in executed['ids']]
                np.testing.assert_array_equal(packed[indices],executed['lengths'])
                reproduced+=len(indices)
            report['reward_executed_length_reproduction']={'rows':reproduced,'exact_match':True}
        report['tokenizers'][name]={'samples':values,'additional_template_clipped_rows':0,
            'tokenizer_json_sha256':hashlib.sha256((path/'tokenizer.json').read_bytes()).hexdigest(),
            'tokenizer_config_sha256':hashlib.sha256((path/'tokenizer_config.json').read_bytes()).hexdigest()}
    np.savez_compressed(root/'retention_counts.npz',**private)
    json_write(root/'retention.json',report)
    json_write(REPO/'results/span_001/retention.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
