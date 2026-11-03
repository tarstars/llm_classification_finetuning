"""Frozen pooled features and controlled warm-start training for H04/H09/H11."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoTokenizer, TrainingArguments, set_seed

from .data import CLASSES, TEXT_COLUMNS, json_write
from .joint_data import PreferenceDataset, PreferenceCollator, read_competition
from .joint_train import Journal, PreferenceTrainer, load_base_model, probabilities
from .span_adapters import add_span_adapter, paired_objective, ATTENTION
from .span_pack import SpanTokenizer


class PairDataset(PreferenceDataset):
    def __getitem__(self, index):
        return {'input_ids': self.original[index], 'swapped_input_ids': self.swapped[index],
                'labels': int(self.labels[index])}


class PairCollator:
    def __init__(self, tokenizer):
        self.single = PreferenceCollator(tokenizer)

    def __call__(self, rows):
        records = [{'input_ids':row['input_ids'],'attention_mask':[1]*len(row['input_ids']),
                    'labels':row['labels']} for row in rows]
        records += [{'input_ids':row['swapped_input_ids'],
                     'attention_mask':[1]*len(row['swapped_input_ids']),
                     'labels':[1,0,2][row['labels']]} for row in rows]
        result = self.single(records)
        result['paired'] = True
        return result


class SpanTrainer(PreferenceTrainer):
    def __init__(self, *args, consistency_weight=0., **kwargs):
        super().__init__(*args, **kwargs)
        self.consistency_weight = consistency_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs['labels']
        paired = inputs.get('paired', False)
        outputs = model(**{key:value for key,value in inputs.items() if key not in ('labels','paired')})
        if paired:
            loss, _ = paired_objective(outputs.logits, labels, self.consistency_weight)
        else:
            loss = torch.nn.functional.cross_entropy(outputs.logits.float(), labels)
        return (loss, outputs) if return_outputs else loss


def load_frozen(model_path, adapter, base_config, training=False):
    base, dtype = load_base_model(model_path, base_config, training=training)
    model = PeftModel.from_pretrained(base, str(adapter), adapter_name='frozen', is_trainable=False)
    model.config.use_cache = False
    return model, dtype


def select_rows(raw, samples, keys):
    frame, _ = read_competition(raw)
    ids = [key for name in keys for key in samples[name]]
    assert len(ids) == len(set(ids))
    selected = frame.loc[frame.id.isin(ids)].copy()
    assert len(selected) == len(ids)
    return selected


def collect_features(raw, model_path, adapter, samples, config, base_config, out, deadline):
    rank, world = int(os.environ.get('LOCAL_RANK','0')),int(os.environ.get('WORLD_SIZE','1'))
    torch.cuda.set_device(rank)
    selected = select_rows(raw,samples,['feature_fit','dev']).iloc[rank::world]
    tokenizer = AutoTokenizer.from_pretrained(adapter,local_files_only=True,padding_side='right')
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    encoder = SpanTokenizer(tokenizer,config['max_length'])
    views, spans = [[],[]],[[],[]]
    for row in selected.itertuples():
        p,a,b = [tokenizer.encode(getattr(row,key),add_special_tokens=False) for key in TEXT_COLUMNS]
        for index, first, second in [(0,a,b),(1,b,a)]:
            ids, sa, sb = encoder.combine_with_spans(p,first,second)
            views[index].append(ids);spans[index].append((sa,sb))
    model, _ = load_frozen(model_path,adapter,base_config)
    model.eval();model.config.pad_token_id = tokenizer.pad_token_id
    raw_model = model.get_base_model()
    hidden = model.config.hidden_size
    n = len(selected)
    pooled_a = np.full((2,n,hidden),np.nan,dtype=np.float16)
    pooled_b = np.full_like(pooled_a,np.nan)
    last = np.full_like(pooled_a,np.nan)
    logits = np.full((2,n,3),np.nan,dtype=np.float32)
    batch_size = config['eval_batch_size']
    started = time.monotonic()
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for view in range(2):
        order = np.argsort([len(ids) for ids in views[view]],kind='stable')
        for begin in range(0,n,batch_size):
            if time.time() > deadline-300:
                raise TimeoutError('Feature extraction reached the shared session ceiling')
            chosen = order[begin:begin+batch_size]
            batch = tokenizer.pad([{'input_ids':views[view][i],
                                    'attention_mask':[1]*len(views[view][i])} for i in chosen],
                                   padding=True,pad_to_multiple_of=8,return_tensors='pt')
            batch = {key:value.to(rank) for key,value in batch.items()}
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                h = raw_model.model(**batch,use_cache=False).last_hidden_state
                token_logits = raw_model.score(h)
                for offset,index in enumerate(chosen):
                    sa,sb = spans[view][index]
                    a = h[offset,sa[0]:sa[1]].mean(0) if sa[1]>sa[0] else torch.zeros(hidden,device=rank)
                    b = h[offset,sb[0]:sb[1]].mean(0) if sb[1]>sb[0] else torch.zeros(hidden,device=rank)
                    pooled_a[view,index] = a.float().cpu().numpy()
                    pooled_b[view,index] = b.float().cpu().numpy()
                    last[view,index] = h[offset,len(views[view][index])-1].float().cpu().numpy()
                    logits[view,index] = token_logits[offset,len(views[view][index])-1].float().cpu().numpy()
            if begin % 256 < batch_size:
                print('SPAN_FEATURE_PROGRESS',json.dumps({'rank':rank,'view':view,'rows_in_view':int(begin+len(chosen)),
                     'total_rows':n,'elapsed_seconds':time.monotonic()-started}),flush=True)
    original, swapped = probabilities(logits[0]),probabilities(logits[1])[:,[1,0,2]]
    values = {'ids':selected.id.to_numpy(), 'answer_a':(pooled_a[0].astype(float)+pooled_b[1])/2,
              'answer_b':(pooled_b[0].astype(float)+pooled_a[1])/2,
              'joint':(last[0].astype(float)+last[1])/2,
              'probabilities':(original+swapped)/2,'original':original,'aligned_swap':swapped,
              'feature_fit':selected.id.isin(samples['feature_fit']).to_numpy()}
    assert all(np.isfinite(value).all() for key,value in values.items() if key != 'feature_fit')
    for key in ('answer_a','answer_b','joint'): values[key] = values[key].astype(np.float16)
    np.savez_compressed(out/f'features_rank{rank}.npz',**values)
    json_write(out/f'features_rank{rank}.json',{'rank':rank,'rows':n,'complete':True,
        'seconds':time.monotonic()-started,'peak_gpu_bytes':torch.cuda.max_memory_allocated(rank)})


def run_training(raw, model_path, adapter, samples, config, base_config, out, deadline, arm):
    set_seed(config['seed'])
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    assert not (out/'summary.json').exists(), 'Never overwrite a completed arm'
    selected = select_rows(raw,samples,['training_pilot'])
    development = select_rows(raw,samples,['dev'])
    tokenizer=AutoTokenizer.from_pretrained(adapter,local_files_only=True,padding_side='right')
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    encoder=SpanTokenizer(tokenizer,config['max_length'])
    original,swapped,train_diagnostics=encoder.encode_frame(selected)
    labels=selected[CLASSES].values.argmax(1)
    paired=arm.startswith('paired')
    training=(PairDataset(original,swapped,labels) if paired else
              PreferenceDataset(original,swapped,labels,augment=True))
    dev_orig,dev_swap,dev_diagnostics=encoder.encode_frame(development)
    dev_y=development[CLASSES].values.argmax(1)
    model,dtype=load_frozen(model_path,adapter,base_config,training=True)
    model.config.pad_token_id=tokenizer.pad_token_id
    model.config.problem_type='single_label_classification'
    with torch.no_grad():
        example=tokenizer.pad([{'input_ids':dev_orig[0],'attention_mask':[1]*len(dev_orig[0])}],
                               return_tensors='pt')
        device=torch.cuda.current_device()
        example={key:value.to(device) for key,value in example.items()}
        model.eval()
        with torch.autocast('cuda',dtype=torch.float16): before=model(**example).logits.detach().clone()
    counts=add_span_adapter(model,config,broad=(arm=='random_all_linear'))
    reference=sum((module.in_features+module.out_features)*config['attention_rank']
                  for name,module in model.named_modules()
                  if name.split('.')[-1] in ATTENTION and hasattr(module,'lora_A'))
    counts['reference_attention_lora_parameters']=reference
    counts['relative_lora_count_change']=(counts['lora_parameters']-reference)/reference
    assert abs(counts['relative_lora_count_change']) < .02
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16): after=model(**example).logits
    assert torch.allclose(before,after,atol=2e-3,rtol=2e-3), 'Adding zero-initialized adapters changed baseline outputs'
    # Module initialization consumes different RNG amounts for broad coverage;
    # reset before sampling so the two random-view arms use the same row order.
    set_seed(config['seed'])
    batch=config['paired_train_batch_size' if paired else 'random_train_batch_size']
    training_cap=min(config['training_max_seconds_per_arm'],max(0,deadline-time.time()-900))
    assert training_cap >= 300, 'Insufficient remaining budget for this training/evaluation arm'
    arguments=TrainingArguments(output_dir=str(out/'checkpoints'),max_steps=config['training_steps'],
        per_device_train_batch_size=batch,per_device_eval_batch_size=config['eval_batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        learning_rate=config['learning_rate'],weight_decay=config['weight_decay'],
        warmup_steps=config['warmup_steps'],lr_scheduler_type='cosine',max_grad_norm=1.,
        gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        fp16=True,bf16=False,eval_strategy='no',save_strategy='no',report_to='none',
        logging_steps=8,logging_first_step=True,logging_nan_inf_filter=False,
        include_num_input_tokens_seen='non_padding',
        seed=config['seed'],data_seed=config['seed'],ddp_find_unused_parameters=False,
        dataloader_num_workers=0,remove_unused_columns=False,disable_tqdm=True,optim='adamw_torch')
    journal=Journal(out/'training_progress.jsonl',training_cap)
    trainer=SpanTrainer(model=model,args=arguments,train_dataset=training,
        processing_class=tokenizer,data_collator=PairCollator(tokenizer) if paired else PreferenceCollator(tokenizer),
        callbacks=[journal],group_training_by_length=True,
        consistency_weight=config['consistency_weight'] if arm=='paired_consistency' else 0.)
    started=time.monotonic()
    result=trainer.train()
    train_seconds=time.monotonic()-started
    model.save_pretrained(out/'adapter',selected_adapters=['span']) if trainer.is_world_process_zero() else None
    trainer.accelerator.wait_for_everyone()
    # Use the ordinary evaluation collator even for paired-view training.
    trainer.data_collator=PreferenceCollator(tokenizer)
    outputs=[]
    for force in (False,True):
        ds=PreferenceDataset(dev_orig,dev_swap,dev_y,force_swap=force)
        values=trainer.predict(ds).predictions
        if isinstance(values,tuple): values=values[0]
        outputs.append(probabilities(values))
    if trainer.is_world_process_zero():
        np.savez_compressed(out/'predictions.npz',ids=development.id.to_numpy(),labels=dev_y,
            original=outputs[0],aligned_swap=outputs[1][:,[1,0,2]],
            probabilities=(outputs[0]+outputs[1][:,[1,0,2]])/2)
        json_write(out/'summary.json',{'arm':arm,'complete':True,'steps':trainer.state.global_step,
            'stopped_for_budget':journal.stopped_for_budget,'training_seconds':train_seconds,
            'canonical_rows':len(selected),'development_rows':len(development),'world_size':2,
            'views_per_update':batch*config['gradient_accumulation_steps']*2*(2 if paired else 1),
            'canonical_exposures':trainer.state.global_step*batch*config['gradient_accumulation_steps']*2,
            'non_padding_tokens_seen':int(trainer.state.num_input_tokens_seen),
            'initial_logit_max_difference':float((before-after).abs().max()),'counts':counts,
            'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'train_result':result.metrics,
            'train_diagnostics':train_diagnostics,'dev_diagnostics':dev_diagnostics,'config':config})
    trainer.accelerator.wait_for_everyone()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['features','train'])
    for name in ('raw','model','adapter','samples','config','base-config','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--deadline',type=float,required=True)
    parser.add_argument('--arm')
    args=parser.parse_args()
    samples=json.loads(args.samples.read_text());config=json.loads(args.config.read_text())
    base_config=json.loads(args.base_config.read_text())
    kwargs=dict(raw=args.raw,model_path=args.model,adapter=args.adapter,samples=samples,
                config=config,base_config=base_config,out=args.out,deadline=args.deadline)
    if args.mode=='features':collect_features(**kwargs)
    else:
        assert args.arm in config['arms']
        run_training(**kwargs,arm=args.arm)


if __name__=='__main__':main()
