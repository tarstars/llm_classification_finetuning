"""Frozen deployed-model probabilities for an explicit training-partition cohort."""
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


def run(a):
    selection=json.loads(a.selection.read_text());cfg=selection['config']
    sample=json.loads(a.sample.read_text());split=json.loads(a.split.read_text())
    ids=sample['ids'];assert sample['partition']=='train' and len(ids)==len(set(ids))
    assert not set(ids)&(set(split['dev'])|set(split['confirmation']))
    adapter=a.train/'joint_run/adapter'
    assert hashlib.sha256((adapter/'adapter_model.safetensors').read_bytes()).hexdigest()==selection['adapter_sha256']
    assert hashlib.sha256(Path(__file__).with_name('joint_data.py').read_bytes()).hexdigest()==cfg['preprocessing_sha256']
    frame=pd.read_csv(a.raw/'train.csv',usecols=['id',*CLASSES,*TEXT_COLUMNS]).set_index('id').loc[ids].reset_index()
    labels=frame[CLASSES].to_numpy().argmax(1)
    for name in TEXT_COLUMNS:frame[name]=frame[name].map(parse_turns)
    tokenizer=AutoTokenizer.from_pretrained(adapter,local_files_only=True,padding_side='right')
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    original,swapped,_=JointTokenizer(tokenizer,cfg['max_length']).encode_frame(frame)
    model=load_inference_model(a.model,adapter,cfg);model.config.pad_token_id=tokenizer.pad_token_id
    bf16=torch.cuda.get_device_capability()[0]>=8
    args=TrainingArguments(output_dir=str(a.out/'temporary'),per_device_eval_batch_size=4,
        bf16=bf16,fp16=not bf16,report_to='none',remove_unused_columns=False,
        dataloader_num_workers=0,disable_tqdm=True)
    trainer=PreferenceTrainer(model=model,args=args,processing_class=tokenizer,data_collator=PreferenceCollator(tokenizer))
    started=time.monotonic()
    def predict(flip):
        z=trainer.predict(PreferenceDataset(original,swapped,labels,force_swap=flip)).predictions
        if isinstance(z,tuple):z=z[0]
        p=probabilities(z);return p[:,[1,0,2]] if flip else p
    p0=predict(False);p1=predict(True);avg=(p0+p1)/2
    p=temper(avg,selection['temperature'])
    if trainer.is_world_process_zero():
        a.out.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(a.out/'poster_train_predictions.npz',ids=np.array(ids),labels=labels,
            original=p0,swapped=p1,average=avg,calibrated=p)
        summary={'partition':'train','sample':{k:v for k,v in sample.items() if k!='ids'},
            'rows':len(ids),'model_hash':selection['adapter_sha256'],'temperature':selection['temperature'],
            'metrics':metrics(labels,p),'inference_seconds':time.monotonic()-started,
            'note':'Training partition, not a held-out estimate. Only 0.61 epochs were completed; per-row exposure was not logged. Highest-loss examples are relative to this scored cohort.'}
        json_write(a.out/'poster_train_summary.json',summary)
        print('POSTER_TRAIN_RESULT',json.dumps(summary),flush=True)
    trainer.accelerator.wait_for_everyone()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['raw','model','train','selection','sample','split','out']:p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
