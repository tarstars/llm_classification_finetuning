"""Frozen adapter evaluation and internet-disabled competition prediction."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoTokenizer,TrainingArguments
from .data import CLASSES,TEXT_COLUMNS,parse_turns,json_write
from .joint_data import JointTokenizer,PreferenceCollator,read_competition,split_from_ids
from .joint_train import load_base_model,predict_swapped,metrics,temper,PreferenceTrainer


def load_inference_model(model_path,adapter_path,config):
    model,_=load_base_model(model_path,config,training=False)
    return PeftModel.from_pretrained(model,str(adapter_path),is_trainable=False).eval()


def run(raw,model_path,adapter,config,training_summary,out,mode,selection,split_path=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    assert hashlib.sha256((Path(adapter)/'adapter_model.safetensors').read_bytes()).hexdigest()==selection['adapter_sha256']
    assert training_summary['temperature']==selection['temperature']
    assert not selection['confirmation_used_for_selection']
    if config.get('preprocessing_sha256'):
        assert hashlib.sha256(Path(__file__).with_name('joint_data.py').read_bytes()).hexdigest()==config['preprocessing_sha256']
    tokenizer=AutoTokenizer.from_pretrained(str(adapter),local_files_only=True,padding_side='right')
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    if mode=='confirm':
        if split_path is None:raise ValueError('Confirmation requires explicit split IDs')
        frame,_=read_competition(raw)
        parts=split_from_ids(frame,json.loads(Path(split_path).read_text()))
        selected=frame.iloc[parts['confirmation']]
        labels=selected[CLASSES].values.argmax(1)
        counts=frame.iloc[parts['train']][CLASSES].sum().to_numpy(dtype=float)
        prior=counts/counts.sum()
    else:
        selected=pd.read_csv(Path(raw)/'test.csv')
        for name in TEXT_COLUMNS:selected[name]=selected[name].map(parse_turns)
        labels=None
    original,swapped,diagnostics=JointTokenizer(tokenizer,config['max_length']).encode_frame(selected)
    model=load_inference_model(model_path,adapter,config)
    model.config.pad_token_id=tokenizer.pad_token_id
    cuda=torch.cuda.is_available()
    bf16=cuda and torch.cuda.get_device_capability()[0]>=8
    args=TrainingArguments(output_dir=str(out/'evaluation_tmp'),per_device_eval_batch_size=config['eval_batch_size'],
        fp16=cuda and not bf16,bf16=bf16,report_to='none',remove_unused_columns=False,
        dataloader_num_workers=0,dataloader_pin_memory=cuda,disable_tqdm=True)
    trainer=PreferenceTrainer(model=model,args=args,processing_class=tokenizer,data_collator=PreferenceCollator(tokenizer))
    p=predict_swapped(trainer,original,swapped,labels)
    calibrated=temper(p,training_summary['temperature'])
    if trainer.is_world_process_zero():
        if mode=='confirm':
            result={'confirmation':metrics(labels,calibrated),'uncalibrated_confirmation':metrics(labels,p),
                'training_prior_confirmation':metrics(labels,np.tile(prior,(len(labels),1))),
                'majority_confirmation':metrics(labels,np.tile(np.eye(3)[prior.argmax()],(len(labels),1))),
                'rows':len(selected),'temperature':training_summary['temperature'],
                'model_choices_frozen_before_confirmation':True,'diagnostics':diagnostics}
            json_write(out/'confirmation.json',result)
            np.savez_compressed(out/'confirmation_predictions.npz',ids=selected.id.to_numpy(),labels=labels,probabilities=calibrated)
            print(json.dumps(result,indent=2),flush=True)
        else:
            submission=pd.DataFrame(calibrated,columns=CLASSES)
            submission.insert(0,'id',selected.id.to_numpy())
            assert submission.id.is_unique and np.isfinite(calibrated).all()
            assert np.all(calibrated>=0) and np.allclose(calibrated.sum(1),1)
            assert submission.columns.tolist()==pd.read_csv(Path(raw)/'sample_submission.csv').columns.tolist()
            submission.to_csv(out/'submission.csv',index=False)
            json_write(out/'prediction_summary.json',{'rows':len(submission),'classes':CLASSES,
                'temperature':training_summary['temperature'],'diagnostics':diagnostics,'probabilities_valid':True})
            print('Wrote submission.csv:',len(submission),'rows',flush=True)
    trainer.accelerator.wait_for_everyone()


def main():
    parser=argparse.ArgumentParser()
    for name in ['raw','model','adapter','config','summary','out','selection']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--split',type=Path)
    parser.add_argument('--mode',choices=['confirm','predict'],required=True)
    args=parser.parse_args()
    run(args.raw,args.model,args.adapter,json.loads(args.config.read_text()),json.loads(args.summary.read_text()),args.out,args.mode,json.loads(args.selection.read_text()),args.split)


if __name__=='__main__':main()
