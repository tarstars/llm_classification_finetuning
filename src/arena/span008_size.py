"""Fresh supervised size pilot with accepted-update accounting, never checkpoint selection."""
from pathlib import Path
import argparse,os,time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer,TrainingArguments,TrainerCallback,set_seed
from .data import CLASSES,TEXT_COLUMNS,parse_turns
from .experiment import read_json,atomic_json,sha256,EventJournal
from .joint_data import JointTokenizer,PreferenceDataset,PreferenceCollator
from .joint_train import load_trainable_model,probabilities,metrics
from .training_exposure import ExposureDataset,ExposureCollator,ExposureLedger,ExposureTrainer
from .telemetry import TrainingTelemetry


class AcceptedStop(TrainerCallback):
    def __init__(self,ledger,cfg,deadline):self.ledger,self.cfg,self.deadline=ledger,cfg,deadline;self.budget=False
    def on_step_end(self,args,state,control,**kwargs):
        flag=int(time.time()>=self.deadline-600)
        t=torch.tensor(flag,device=torch.cuda.current_device());torch.distributed.all_reduce(t,op=torch.distributed.ReduceOp.MAX)
        self.budget=bool(t.item())
        if self.budget or self.ledger.counts['accepted_updates']>=self.cfg['size_pilot_accepted_updates']:control.should_training_stop=True


class FixedScheduleTrainer(ExposureTrainer):
    def create_scheduler(self,num_training_steps,optimizer=None):
        return super().create_scheduler(self.fixed_schedule_updates,optimizer)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['raw','model','tokenizer-reference','selection','config','base-config','out']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--size',choices=['4b','8b'],required=True);parser.add_argument('--deadline',type=float,required=True);args=parser.parse_args()
    cfg=read_json(args.config);base=read_json(args.base_config);sel=read_json(args.selection);rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);assert torch.cuda.device_count()==2
    start=time.monotonic();set_seed(cfg['seed']);out=args.out;out.mkdir(parents=True,exist_ok=True);journal=EventJournal(out/'stages')
    tokenizer=AutoTokenizer.from_pretrained(str(args.model),local_files_only=True,padding_side='right')
    reference=AutoTokenizer.from_pretrained(str(args.tokenizer_reference),local_files_only=True,padding_side='right')
    assert tokenizer.get_vocab()==reference.get_vocab(),'Model size cannot change token vocabulary in this matched pilot'
    assert tokenizer.bos_token_id==reference.bos_token_id and tokenizer.eos_token_id==reference.eos_token_id
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    # Physically load target columns only for the fixed training/dev samples.
    rows=[];ids=set(sel['train']+sel['dev'])
    for chunk in pd.read_csv(args.raw/'train.csv',usecols=['id']+TEXT_COLUMNS+CLASSES,chunksize=2048):rows.append(chunk[chunk.id.isin(ids)])
    frame=pd.concat(rows).set_index('id',drop=False)
    for key in TEXT_COLUMNS:frame[key]=frame[key].map(parse_turns)
    train=frame.loc[sel['train']];dev=frame.loc[sel['dev']];encoder=JointTokenizer(tokenizer,1024)
    original,swapped,_=encoder.encode_frame(train);do,ds,_=encoder.encode_frame(dev)
    from .span006_inputs import token_digest
    token_hashes={'train_original':token_digest(original),'train_swapped':token_digest(swapped),'dev_original':token_digest(do),'dev_swapped':token_digest(ds)}
    atomic_json(out/f'packing_rank{rank}.json',token_hashes)
    model,dtype=load_trainable_model(args.model,base);model.config.pad_token_id=tokenizer.pad_token_id;model.config.problem_type='single_label_classification'
    batch,accum=(4,4) if args.size=='4b' else (2,8);assert batch*accum*2==32
    options=TrainingArguments(output_dir=str(out/'checkpoints'),max_steps=cfg['size_pilot_nominal_cycles_cap'],
        per_device_train_batch_size=batch,per_device_eval_batch_size=4 if args.size=='4b' else 2,gradient_accumulation_steps=accum,
        learning_rate=cfg['size_pilot_learning_rate'],weight_decay=.01,warmup_steps=cfg['size_pilot_warmup_updates'],lr_scheduler_type='cosine',
        max_grad_norm=1.,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},fp16=True,bf16=False,
        eval_strategy='no',save_strategy='no',logging_steps=4,logging_first_step=True,logging_nan_inf_filter=False,report_to='none',
        seed=cfg['seed'],data_seed=cfg['seed'],ddp_find_unused_parameters=False,ddp_timeout=600,dataloader_num_workers=0,dataloader_pin_memory=True,
        remove_unused_columns=False,optim='adamw_torch',disable_tqdm=True)
    ledger=ExposureLedger(out,rank);control=AcceptedStop(ledger,cfg,args.deadline)
    training=ExposureDataset(original,swapped,train[CLASSES].to_numpy().argmax(1),row_ids=train.id.to_numpy(),augment=True)
    trainer=FixedScheduleTrainer(model=model,args=options,train_dataset=training,processing_class=tokenizer,data_collator=ExposureCollator(tokenizer),
        ledger=ledger,group_training_by_length=False)
    # ExposureCallback must record this cycle before the accepted-update stop checks it.
    trainer.add_callback(control)
    trainer.fixed_schedule_updates=cfg['size_pilot_accepted_updates']
    trainer.accelerator.scaler=torch.amp.GradScaler('cuda',init_scale=cfg['size_pilot_initial_amp_scale'])
    trainer.add_callback(TrainingTelemetry(trainer,ledger,out/'telemetry'))
    before=time.monotonic();trainer.train();training_seconds=time.monotonic()-before
    trainer.save_model(str(out/'adapter'))
    if trainer.is_world_process_zero():tokenizer.save_pretrained(out/'adapter')
    trainer.accelerator.wait_for_everyone();trainer.data_collator=PreferenceCollator(tokenizer);model.eval();views=[];before=time.monotonic()
    for force in [False,True]:
        values=trainer.predict(PreferenceDataset(do,ds,force_swap=force)).predictions
        if isinstance(values,tuple):values=values[0]
        views.append(probabilities(values))
    prediction_seconds=time.monotonic()-before;device=torch.cuda.current_device()
    atomic_json(out/f'gpu_rank{rank}.json',{'rank':rank,'device':device,'name':torch.cuda.get_device_name(device),'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),'peak_reserved_bytes':torch.cuda.max_memory_reserved(device)})
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        a,b=views[0],views[1][:,[1,0,2]];np.savez_compressed(out/'dev_predictions.npz',ids=dev.id,original=a,aligned_swap=b)
        summary={'complete':True,'size':args.size,'training_seconds':training_seconds,'prediction_seconds':prediction_seconds,'worker_seconds':time.monotonic()-start,
            'accepted_updates':ledger.counts['accepted_updates'],'nominal_cycles':ledger.counts['cycles'],'stopped_for_budget':control.budget,
            'token_hashes':token_hashes,'model_config_sha256':sha256(args.model/'config.json'),'adapter_sha256':sha256(out/'adapter/adapter_model.safetensors'),
            'source_sha256':sha256(Path(__file__)),'confirmation_targets_used':False,'fresh_classifier_head':True,'global_batch':32,'seed':cfg['seed']}
        atomic_json(out/'summary.json',summary);journal.emit(args.size,'complete',complete=True,training_seconds=training_seconds,new_cycles=ledger.counts['cycles'])


if __name__=='__main__':main()
