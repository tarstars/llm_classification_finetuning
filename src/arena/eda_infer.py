"""Frozen-model order/context diagnostics on an explicit random development sample."""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer,TrainingArguments
from .data import CLASSES,TEXT_COLUMNS,parse_turns,json_write
from .joint_data import JointTokenizer,PreferenceDataset,PreferenceCollator
from .joint_infer import load_inference_model
from .joint_train import PreferenceTrainer,probabilities,metrics,temper


def run(args):
    selected=json.loads(args.selection.read_text())
    config=selected['config']
    sample=json.loads(args.sample.read_text())
    split=json.loads(args.split.read_text())
    ids=sample['ids']
    assert len(ids)==len(set(ids)) and set(ids)<=set(split['dev'])
    assert not set(ids)&set(split['confirmation'])
    assert hashlib.sha256((args.train/'joint_run/adapter/adapter_model.safetensors').read_bytes()).hexdigest()==selected['adapter_sha256']
    assert hashlib.sha256(Path(__file__).with_name('joint_data.py').read_bytes()).hexdigest()==config['preprocessing_sha256']
    frame=pd.read_csv(args.raw/'train.csv',usecols=['id',*CLASSES,*TEXT_COLUMNS]).set_index('id').loc[ids].reset_index()
    labels=frame[CLASSES].to_numpy().argmax(1)
    for name in TEXT_COLUMNS:frame[name]=frame[name].map(parse_turns)
    tokenizer=AutoTokenizer.from_pretrained(args.train/'joint_run/adapter',local_files_only=True,padding_side='right')
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    original,swapped,_=JointTokenizer(tokenizer,1024).encode_frame(frame)
    longer,longer_swap,_=JointTokenizer(tokenizer,2048).encode_frame(frame)
    changed=np.array([a!=b or c!=d for a,b,c,d in zip(original,longer,swapped,longer_swap)])
    model=load_inference_model(args.model,args.train/'joint_run/adapter',config)
    model.config.pad_token_id=tokenizer.pad_token_id
    bf16=torch.cuda.get_device_capability()[0]>=8
    training_args=TrainingArguments(output_dir=str(args.out/'temporary'),per_device_eval_batch_size=2,
        fp16=not bf16,bf16=bf16,report_to='none',remove_unused_columns=False,
        dataloader_num_workers=0,disable_tqdm=True)
    trainer=PreferenceTrainer(model=model,args=training_args,processing_class=tokenizer,
        data_collator=PreferenceCollator(tokenizer))
    def predict(a,b,y,flip=False):
        logits=trainer.predict(PreferenceDataset(a,b,y,force_swap=flip)).predictions
        if isinstance(logits,tuple):logits=logits[0]
        p=probabilities(logits)
        return p[:,[1,0,2]] if flip else p
    started=time.monotonic()
    p_original=predict(original,swapped,labels)
    p_swapped=predict(original,swapped,labels,True)
    p_average=(p_original+p_swapped)/2
    p_long_original=p_original.copy();p_long_swapped=p_swapped.copy()
    idx=np.flatnonzero(changed)
    if len(idx):
        a=[longer[i] for i in idx];b=[longer_swap[i] for i in idx]
        p_long_original[idx]=predict(a,b,labels[idx])
        p_long_swapped[idx]=predict(a,b,labels[idx],True)
    p_long_average=(p_long_original+p_long_swapped)/2
    if trainer.is_world_process_zero():
        args.out.mkdir(parents=True,exist_ok=True)
        ref=np.load(args.train/'joint_run/dev_predictions.npz')
        positions={int(v):i for i,v in enumerate(ref['ids'])}
        r=np.array([positions[int(v)] for v in ids])
        assert np.array_equal(ref['labels'][r],labels)
        drift=np.abs(ref['probabilities'][r]-p_average)
        assert drift.max()<.05,'Frozen model reload disagrees materially with saved development probabilities'
        t=selected['temperature']
        summary={'sample':{k:v for k,v in sample.items() if k!='ids'},'rows':len(ids),
            'context_changed_rows':int(changed.sum()),'temperature_frozen':t,
            'model_hash':selected['adapter_sha256'],'confirmation_used':False,
            'baseline_reload_max_absolute_probability_difference':float(drift.max()),
            'baseline_reload_mean_absolute_probability_difference':float(drift.mean()),
            'raw_original':metrics(labels,p_original),'raw_swap_average':metrics(labels,p_average),
            'calibrated_original':metrics(labels,temper(p_original,t)),
            'calibrated_swap_average':metrics(labels,temper(p_average,t)),
            'calibrated_2048':metrics(labels,temper(p_long_average,t)),
            'order_argmax_disagreement_fraction':float((p_original.argmax(1)!=p_swapped.argmax(1)).mean()),
            'order_mean_l1_probability_difference':float(np.abs(p_original-p_swapped).sum(1).mean()),
            'order_mean_original_minus_aligned_swap_probabilities':(p_original-p_swapped).mean(0).tolist(),
            'inference_seconds':time.monotonic()-started,
            'peak_gpu_bytes':torch.cuda.max_memory_allocated(),
            'caveat':'Random development pilot; fixed existing temperature; no retraining or official submission. Longer-context effect requires paired uncertainty and confirmation.'}
        json_write(args.out/'eda_model_summary.json',summary)
        np.savez_compressed(args.out/'eda_model_predictions.npz',ids=np.array(ids),labels=labels,
            original=p_original,swapped=p_swapped,average=p_average,long_original=p_long_original,
            long_swapped=p_long_swapped,long_average=p_long_average,context_changed=changed)
        print('EDA_MODEL_RESULT',json.dumps(summary),flush=True)
    trainer.accelerator.wait_for_everyone()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['raw','model','train','selection','sample','split','out']:p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
