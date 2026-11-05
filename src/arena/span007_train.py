"""One bounded H03 warm restart with complete exposure and stage telemetry."""
import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoTokenizer, TrainingArguments, TrainerCallback, set_seed

from .data import CLASSES
from .experiment import read_json, atomic_json, sha256, EventJournal
from .joint_data import JointTokenizer, PreferenceDataset, PreferenceCollator, read_competition, split_from_ids
from .joint_train import load_trainable_model, probabilities, metrics, compute_metrics
from .training_exposure import ExposureDataset, ExposureCollator, ExposureLedger, ExposureTrainer
from .telemetry import TrainingTelemetry, restart_factor, optimizer_fingerprint


class RestartControl(TrainerCallback):
    def __init__(self,cfg,deadline,baseline,source):
        self.cfg,self.deadline,self.baseline,self.source=cfg,deadline,baseline,source
        self.trainer=None;self.budget_stop=False;self.bad=0;self.plateau_stop=False

    def on_train_begin(self,args,state,control,**kwargs):
        assert state.global_step==1438
        scheduler=getattr(self.trainer.lr_scheduler,'scheduler',self.trainer.lr_scheduler)
        self.origin=int(scheduler.last_epoch)
        assert self.origin==self.source['parent_scheduler_position']
        assert optimizer_fingerprint(self.trainer.optimizer.state_dict())==self.source['parent_optimizer_fingerprint']
        scaler=self.trainer.accelerator.scaler
        if scaler is not None:assert float(scaler.get_scale())==self.source['parent_amp_scale']
        scheduler.base_lrs=[self.cfg['training_lr']]*len(scheduler.base_lrs)
        factor=lambda position:restart_factor(position,self.origin,self.cfg['training_warmup_updates'],self.cfg['training_cycles'])
        scheduler.lr_lambdas=[factor]*len(scheduler.base_lrs)
        rates=[lr*factor(self.origin) for lr in scheduler.base_lrs]
        for group,lr in zip(self.trainer.optimizer.param_groups,rates):
            group['lr']=lr;group['initial_lr']=self.cfg['training_lr']
        scheduler._last_lr=rates
        self.source.update(restarted_scheduler_origin=self.origin,first_learning_rates=rates,
                           parent_optimizer_restored=True,parent_scaler_restored=scaler is not None,
                           schedule_change='Accepted-position warm restart; not unchanged historical schedule')

    def on_step_end(self,args,state,control,**kwargs):
        new=state.global_step-1438
        stop=time.time()>=self.deadline-4200
        if torch.distributed.is_initialized():
            flag=torch.tensor(int(stop),device=torch.cuda.current_device());torch.distributed.all_reduce(flag,op=torch.distributed.ReduceOp.MAX);stop=bool(flag.item())
        if new in [128,256,320]:control.should_evaluate=True;control.should_save=True
        if stop:
            self.budget_stop=True;control.should_training_stop=True;control.should_save=True

    def on_evaluate(self,args,state,control,metrics=None,**kwargs):
        loss=(metrics or {}).get('eval_log_loss')
        if loss is not None:
            assert math.isfinite(loss)
            self.bad=self.bad+1 if loss>=self.baseline+.005 else 0
            if self.bad>=2:self.plateau_stop=True;control.should_training_stop=True;control.should_save=True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['raw','model','parent','split','base-config','config','selection','reference','out']:
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--deadline',type=float,required=True);args=parser.parse_args()
    rank=int(os.environ.get('LOCAL_RANK','0'));torch.cuda.set_device(rank)
    assert torch.cuda.device_count()==2
    out=args.out;out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists()
    journal=EventJournal(out/'stages')
    cfg=read_json(args.config);base=read_json(args.base_config);selection=read_json(args.selection)
    checkpoint=args.parent/'exposure/checkpoints/checkpoint-1438'
    expected='ca8be9cfd98309fdc8406bf740b13ed3ee413f3820ec6bcbbf7f3f4059f4b15b'
    assert sha256(checkpoint/'adapter_model.safetensors')==expected
    assert sha256(Path(__file__).with_name('joint_data.py'))==base['preprocessing_sha256']
    assert sha256(args.split)==base['split_ids_sha256']
    required=['adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt','scaler.pt',
              'rng_state_0.pth','rng_state_1.pth','trainer_state.json','training_args.bin']
    assert all((checkpoint/n).exists() for n in required)
    parent_scheduler=torch.load(checkpoint/'scheduler.pt',map_location='cpu',weights_only=True)
    parent_scaler=torch.load(checkpoint/'scaler.pt',map_location='cpu',weights_only=True)
    parent_optimizer=torch.load(checkpoint/'optimizer.pt',map_location='cpu',weights_only=True)
    source={'parent_adapter_sha256':expected,'parent_checkpoint_files':{n:sha256(checkpoint/n) for n in required},
            'parent_scheduler_position':int(parent_scheduler['last_epoch']),'parent_amp_scale':float(parent_scaler['scale']),
            'parent_optimizer_fingerprint':optimizer_fingerprint(parent_optimizer)}
    del parent_optimizer
    set_seed(base['seed'])
    tokenizer=AutoTokenizer.from_pretrained(str(checkpoint),local_files_only=True,padding_side='right')
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    if rank==0:journal.emit('preparation','started')
    frame,_=read_competition(args.raw);parts=split_from_ids(frame,read_json(args.split))
    encoder=JointTokenizer(tokenizer,1024);train=frame.iloc[parts['train']]
    original,swapped,diagnostics=encoder.encode_frame(train)
    training=ExposureDataset(original,swapped,train[CLASSES].to_numpy().argmax(1),row_ids=train.id.to_numpy(),augment=True)
    dev=frame.set_index('id',drop=False).loc[selection['selection']]
    doriginal,dswapped,_=encoder.encode_frame(dev)
    development=PreferenceDataset(doriginal,dswapped,dev[CLASSES].to_numpy().argmax(1))
    model,dtype=load_trainable_model(args.model,base)
    incompatible=set_peft_model_state_dict(model,load_file(str(checkpoint/'adapter_model.safetensors')))
    assert not incompatible.unexpected_keys
    model.config.pad_token_id=tokenizer.pad_token_id;model.config.problem_type='single_label_classification'
    # Check the actual loaded trainable model before any optimizer update.
    ref=np.load(args.reference);smoke=frame.set_index('id',drop=False).loc[ref['ids']]
    so,ss,_=encoder.encode_frame(smoke);chosen=np.arange(len(smoke))[rank::2];views=[];model.eval()
    for packed in [so,ss]:
        outputs=[]
        for start in range(0,len(chosen),base['eval_batch_size']):
            rows=chosen[start:start+base['eval_batch_size']]
            batch=tokenizer.pad([{'input_ids':packed[i],'attention_mask':[1]*len(packed[i])} for i in rows],padding=True,pad_to_multiple_of=8,return_tensors='pt')
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                outputs.extend(model(**{k:v.to(rank) for k,v in batch.items()}).logits.float().cpu().numpy())
        views.append(probabilities(np.asarray(outputs)))
    delta=np.abs((views[0]+views[1][:,[1,0,2]])/2-(ref['original'][chosen]+ref['aligned_swap'][chosen])/2)
    identity={'rank':rank,'rows':len(chosen),'mean':float(delta.mean()),'maximum':float(delta.max()),
              'passed':bool(delta.mean()<=.002 and delta.max()<=.03)}
    atomic_json(out/f'identity_rank{rank}.json',identity);assert identity['passed']
    if rank==0:journal.emit('identity','checked',passed=True)
    native=not (torch.cuda.get_device_capability()[0]>=8)
    options=TrainingArguments(output_dir=str(out/'checkpoints'),max_steps=1438+cfg['training_cycles'],
        per_device_train_batch_size=base['train_batch_size'],per_device_eval_batch_size=base['eval_batch_size'],
        gradient_accumulation_steps=base['gradient_accumulation_steps'],learning_rate=cfg['training_lr'],
        weight_decay=base['weight_decay'],warmup_steps=16,lr_scheduler_type='cosine',max_grad_norm=1.,
        gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},fp16=native,bf16=not native,
        eval_strategy='no',save_strategy='no',save_total_limit=2,load_best_model_at_end=False,
        logging_steps=5,logging_first_step=True,logging_nan_inf_filter=False,report_to='none',
        seed=base['seed'],data_seed=base['seed'],ddp_find_unused_parameters=False,ddp_timeout=900,
        dataloader_num_workers=0,dataloader_pin_memory=True,remove_unused_columns=False,optim='adamw_torch',disable_tqdm=True)
    control=RestartControl(cfg,args.deadline,selection['selection_baseline_original_loss'],source)
    ledger=ExposureLedger(out,rank)
    trainer=ExposureTrainer(model=model,args=options,train_dataset=training,eval_dataset=development,
        processing_class=tokenizer,data_collator=ExposureCollator(tokenizer),compute_metrics=compute_metrics,
        group_training_by_length=base.get('group_training_by_length',False),ledger=ledger,callbacks=[control])
    control.trainer=trainer;trainer.add_callback(TrainingTelemetry(trainer,ledger,out/'telemetry'))
    started=time.monotonic();trained=trainer.train(resume_from_checkpoint=str(checkpoint));seconds=time.monotonic()-started
    assert ledger.counts['cycles']==trainer.state.global_step-1438
    assert all(torch.isfinite(p).all() for p in model.parameters() if p.requires_grad)
    trainer.save_model(str(out/'adapter'))
    if trainer.is_world_process_zero():tokenizer.save_pretrained(out/'adapter');atomic_json(out/'resume_source.json',source)
    trainer.accelerator.wait_for_everyone()
    final=frame.iloc[parts['dev']];fo,fs,_=encoder.encode_frame(final);y=final[CLASSES].to_numpy().argmax(1)
    trainer.data_collator=PreferenceCollator(tokenizer);views=[];start=time.monotonic()
    if trainer.is_world_process_zero():journal.emit('full_development','started',rows=len(final))
    for force in [False,True]:
        values=trainer.predict(PreferenceDataset(fo,fs,y,force_swap=force)).predictions
        if isinstance(values,tuple):values=values[0]
        views.append(probabilities(values))
    scoring=time.monotonic()-start
    if trainer.is_world_process_zero():
        a,b=views[0],views[1][:,[1,0,2]];p=(a+b)/2
        np.savez_compressed(out/'dev_predictions.npz',ids=final.id.to_numpy(),labels=y,original=a,aligned_swap=b)
        summary={'complete':True,'initial_step':1438,'final_step':trainer.state.global_step,'new_cycles':ledger.counts['cycles'],
                 'training_seconds':seconds,'development_prediction_seconds':scoring,'stopped_for_budget':control.budget_stop,
                 'stopped_for_instability':control.plateau_stop,'raw_metrics':metrics(y,p),'adapter_sha256':sha256(out/'adapter/adapter_model.safetensors'),
                 'development_prediction_sha256':sha256(out/'dev_predictions.npz'),'source':source,'config':cfg,
                 'parent_config':base,'train_diagnostics':diagnostics,'dtype':str(dtype),'train_rows':len(train),
                 'selection_rows':len(dev),'train_result':trained.metrics,'confirmation_targets_read':False}
        atomic_json(out/'summary.json',summary);journal.emit('full_development','finished',rows=len(final),seconds=scoring)
        journal.emit('continuation','complete',complete=True,new_cycles=summary['new_cycles'],training_seconds=seconds)
    device=torch.cuda.current_device()
    atomic_json(out/f'gpu_rank{rank}.json',{'rank':rank,'device':device,'name':torch.cuda.get_device_name(device),
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),'peak_reserved_bytes':torch.cuda.max_memory_reserved(device)})
    trainer.accelerator.wait_for_everyone()


if __name__=='__main__':main()
