"""Label-free, checkpointed frozen H03 confirmation inference on two T4s."""
import argparse,hashlib,json,signal,time,importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer,TrainingArguments
from .data import TEXT_COLUMNS,CLASSES,parse_turns,json_write
from .joint_data import JointTokenizer,PreferenceDataset,PreferenceCollator
from .joint_infer import load_inference_model
from .joint_train import PreferenceTrainer,probabilities,temper
from .span006_inputs import read_inputs,text_digest,token_digest


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['raw','model','adapter','manifest','selection','reference','out']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--deadline',type=float,required=True);args=parser.parse_args();started=time.monotonic()
    def expired(*unused):raise TimeoutError('Frozen inference deadline reached')
    signal.signal(signal.SIGALRM,expired);signal.alarm(max(1,int(args.deadline-time.time())))
    manifest=json.loads(args.manifest.read_text());selection=json.loads(args.selection.read_text());config=manifest['config']
    assert manifest['class_order']==CLASSES and not manifest['confirmation_used_for_selection']
    assert digest(args.adapter/'adapter_model.safetensors')==manifest['adapter_sha256']
    assert digest(args.adapter/'tokenizer.json')==manifest['tokenizer_sha256']
    assert digest(Path(__file__).with_name('joint_data.py'))==config['preprocessing_sha256']
    torch.manual_seed(config['seed']);np.random.seed(config['seed'])
    tokenizer=AutoTokenizer.from_pretrained(str(args.adapter),local_files_only=True,padding_side='right')
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    reference=np.load(args.reference);assert list(map(int,reference['ids']))==selection['smoke']
    smoke=read_inputs(args.raw/'train.csv',selection['smoke']);confirm=read_inputs(args.raw/'train.csv',selection['confirmation'])
    assert len(confirm)==5746 and text_digest(confirm)==selection['confirmation_text_digest']
    packing=JointTokenizer(tokenizer,config['max_length']);so,ss,sd=packing.encode_frame(smoke);co,cs,cd=packing.encode_frame(confirm)
    model=load_inference_model(args.model,args.adapter,config);model.config.pad_token_id=tokenizer.pad_token_id;model.config.problem_type='single_label_classification'
    assert torch.cuda.device_count()==2
    bf16=torch.cuda.get_device_capability()[0]>=8
    options=TrainingArguments(output_dir=str(args.out/'evaluation_tmp'),per_device_eval_batch_size=config['eval_batch_size'],fp16=not bf16,bf16=bf16,
        report_to='none',remove_unused_columns=False,dataloader_num_workers=0,dataloader_pin_memory=True,disable_tqdm=True)
    trainer=PreferenceTrainer(model=model,args=options,processing_class=tokenizer,data_collator=PreferenceCollator(tokenizer))
    root=trainer.is_world_process_zero();args.out.mkdir(parents=True,exist_ok=True)
    def predict(original,swapped):
        views=[]
        for force in [False,True]:
            result=trainer.predict(PreferenceDataset(original,swapped,force_swap=force)).predictions
            if isinstance(result,tuple):result=result[0]
            views.append(probabilities(result))
        return views[0],views[1][:,[1,0,2]]
    begin=time.monotonic();a,b=predict(so,ss);average=(a+b)/2
    delta=np.abs(average-(reference['original']+reference['aligned_swap'])/2)
    smoke_summary={'rows':len(smoke),'mean_raw_average_difference':float(delta.mean()),'max_raw_average_difference':float(delta.max()),
        'passed':bool(delta.mean()<=.002 and delta.max()<=.03),'prediction_seconds':time.monotonic()-begin,
        'adapter_sha256':manifest['adapter_sha256'],'tokenizer_sha256':manifest['tokenizer_sha256']}
    if root:
        np.savez(args.out/'smoke_predictions.npz',ids=smoke.id.to_numpy(),original=a,aligned_swap=b)
        json_write(args.out/'smoke_summary.json',smoke_summary);print(json.dumps({'phase':'smoke',**smoke_summary}),flush=True)
    assert smoke_summary['passed'],'Preserve failed raw identity; do not score confirmation'
    begin=time.monotonic();all_a=[];all_b=[];chunks=[]
    for start in range(0,len(confirm),512):
        assert time.time()<args.deadline-45,'Stop before the original session cap'
        end=min(start+512,len(confirm));before=time.monotonic();a,b=predict(co[start:end],cs[start:end]);all_a.append(a);all_b.append(b)
        record={'start':start,'end':end,'seconds':time.monotonic()-before}
        if root:
            file=args.out/f'chunk_{start:05d}.npz';np.savez(file,ids=confirm.id.to_numpy()[start:end],original=a,aligned_swap=b)
            record['sha256']=digest(file);chunks.append(record);json_write(args.out/'progress.json',{'rows_done':end,'total_rows':len(confirm),'chunks':chunks})
            print(json.dumps({'phase':'confirmation',**record}),flush=True)
    a=np.vstack(all_a);b=np.vstack(all_b);p=(a+b)/2
    assert p.shape==(len(confirm),3) and np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1,atol=1e-6)
    if root:
        file=args.out/'confirmation_predictions.npz';np.savez(file,ids=confirm.id.to_numpy(),original=a,aligned_swap=b)
        summary={'complete':True,'rows':len(confirm),'prediction_seconds':time.monotonic()-begin,'program_seconds':time.monotonic()-started,
            'seconds_per_row':(time.monotonic()-begin)/len(confirm),'adapter_sha256':manifest['adapter_sha256'],'tokenizer_sha256':manifest['tokenizer_sha256'],
            'class_order':CLASSES,'confirmation_votes_read':False,'confirmation_source_names_read':False,'confirmation_text_digest':text_digest(confirm),
            'original_tokens_sha256':token_digest(co),'swapped_tokens_sha256':token_digest(cs),'token_diagnostics':cd,'chunks':chunks,
            'prediction_sha256':digest(file),'peak_gpu_allocated_bytes_rank0':torch.cuda.max_memory_allocated(),'base_config_sha256':digest(args.model/'config.json'),
            'frozen_recipe_sha256':selection['frozen_recipe_sha256'],'software':{n:importlib.metadata.version(n) for n in ['torch','transformers','peft','accelerate','bitsandbytes']}}
        json_write(args.out/'confirmation_summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='chunks'}),flush=True)
    torch.cuda.synchronize()
    device=torch.cuda.current_device()
    json_write(args.out/f'gpu_rank{trainer.args.process_index}.json',{'rank':trainer.args.process_index,'local_rank':trainer.args.local_rank,
        'device':device,'name':torch.cuda.get_device_name(device),'capability':torch.cuda.get_device_capability(device),
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),'peak_reserved_bytes':torch.cuda.max_memory_reserved(device)})
    trainer.accelerator.wait_for_everyone();signal.alarm(0)


if __name__=='__main__':main()
