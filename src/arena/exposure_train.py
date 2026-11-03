"""Stateful checkpoint-800 continuation with direct exposure accounting for H03."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoTokenizer, TrainerCallback, set_seed

from .data import CLASSES, json_write
from .joint_data import JointTokenizer, PreferenceDataset, PreferenceCollator, read_competition, split_from_ids
from .joint_train import (Journal, make_trainer, load_trainable_model, probabilities, metrics, compute_metrics)
from .training_exposure import ExposureDataset, ExposureCollator, ExposureLedger, ExposureTrainer


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class PlateauStop(TrainerCallback):
    def __init__(self, baseline):
        self.baseline, self.bad_count, self.stopped = baseline, 0, False

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        value = metrics['eval_log_loss']
        self.bad_count = self.bad_count+1 if value >= self.baseline+.005 else 0
        if self.bad_count >= 2:
            self.stopped = True
            control.should_training_stop = True
            control.should_save = True


def run(args):
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(rank)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'summary.json').exists(), 'Use the existing result rather than duplicate training'
    config = json.loads(args.base_config.read_text())
    span = json.loads(args.config.read_text())
    assert digest(Path(__file__).with_name('joint_data.py')) == config['preprocessing_sha256']
    assert digest(args.split) == config['split_ids_sha256']
    checkpoint = args.parent/'joint_run/checkpoints/checkpoint-800'
    required = ['adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt',
                'scaler.pt','rng_state_0.pth','rng_state_1.pth','trainer_state.json','training_args.bin']
    assert all((checkpoint/name).exists() for name in required)
    native_state = json.loads((checkpoint/'trainer_state.json').read_text())
    assert native_state['global_step'] == 800 and native_state['max_steps'] == 1438
    adapter = args.parent/'joint_run/adapter/adapter_model.safetensors'
    initial_state = load_file(str(checkpoint/'adapter_model.safetensors'))
    selected = load_file(str(adapter))
    assert set(initial_state) == set(selected)
    assert all(torch.equal(initial_state[key], selected[key]) for key in initial_state), 'Wrong checkpoint weights'
    source = {'checkpoint_step':800,'original_max_steps':1438,
        'file_sha256':{name:digest(checkpoint/name) for name in required},
        'selected_adapter_sha256':digest(adapter),'checkpoint_weights_equal_selected':True,
        'torch_original':'2.11.0+cu128','torch_current':torch.__version__,
        'continuation_kind':'Optimizer/scheduler/scaler/RNG continuation; changed CUDA build, no bitwise-equivalence claim.'}
    if rank == 0:
        json_write(out/'resume_source.json', source)
    set_seed(config['seed'])
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True, padding_side='right')
    frame, _ = read_competition(args.raw)
    parts = split_from_ids(frame, json.loads(args.split.read_text()))
    indices = np.sort(np.random.default_rng(config['seed']).choice(parts['dev'],config['selection_dev_rows'],replace=False))
    encoder = JointTokenizer(tokenizer, config['max_length'])
    train_frame, dev_frame = frame.iloc[parts['train']], frame.iloc[indices]
    original, swapped, train_diag = encoder.encode_frame(train_frame)
    dev_original, dev_swapped, dev_diag = encoder.encode_frame(dev_frame)
    training = ExposureDataset(original, swapped, train_frame[CLASSES].values.argmax(1),
        row_ids=train_frame.id.to_numpy(), augment=True)
    development = PreferenceDataset(dev_original, dev_swapped, dev_frame[CLASSES].values.argmax(1))
    model, dtype = load_trainable_model(args.model, config)
    incompatible = set_peft_model_state_dict(model, initial_state)
    assert not incompatible.unexpected_keys
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.problem_type = 'single_label_classification'
    del initial_state, selected
    # Check the CUDA-build change on 64 fixed original selection rows before updates.
    model.eval()
    chosen = np.arange(min(64,len(indices)))[rank::int(os.environ.get('WORLD_SIZE','1'))]
    views = []
    for sequences in (dev_original, dev_swapped):
        result = []
        for begin in range(0,len(chosen),config['eval_batch_size']):
            rows = chosen[begin:begin+config['eval_batch_size']]
            batch = tokenizer.pad([{'input_ids':sequences[i],'attention_mask':[1]*len(sequences[i])} for i in rows],
                                  padding=True,pad_to_multiple_of=8,return_tensors='pt')
            batch = {key:value.to(rank) for key,value in batch.items()}
            with torch.inference_mode(), torch.autocast('cuda',dtype=torch.float16):
                result.extend(model(**batch).logits.float().cpu().numpy())
        views.append(probabilities(np.asarray(result)))
    fresh = (views[0]+views[1][:,[1,0,2]])/2
    old = np.load(args.parent/'joint_run/dev_predictions.npz')
    lookup = {int(rid):i for i,rid in enumerate(old['ids'])}
    reference = old['probabilities'][[lookup[int(rid)] for rid in dev_frame.id.iloc[chosen]]]
    difference = np.abs(fresh-reference)
    identity = {'rank':rank,'rows':len(chosen),'mean_probability_difference':float(difference.mean()),
                'max_probability_difference':float(difference.max()),'mean_limit':.002,'max_limit':.03}
    json_write(out/f'starting_identity_rank{rank}.json', identity)
    assert difference.mean() < .002 and difference.max() < .03, 'Starting inference identity guard failed'
    remaining_training = args.deadline-time.time()-4500
    assert remaining_training > 1800, 'Insufficient time for a meaningful continuation and final evaluation'
    config['max_train_seconds'] = min(config['max_train_seconds'], remaining_training)
    ledger = ExposureLedger(out,rank)
    reference_trainer = make_trainer(model,tokenizer,training,development,config,out)
    arguments = reference_trainer.args
    arguments.load_best_model_at_end = False
    arguments.save_total_limit = 3
    del reference_trainer
    journal = Journal(out/'training_progress.jsonl',config['max_train_seconds'])
    plateau = PlateauStop(native_state['best_metric'])
    trainer = ExposureTrainer(model=model,args=arguments,train_dataset=training,eval_dataset=development,
        processing_class=tokenizer,data_collator=ExposureCollator(tokenizer),compute_metrics=compute_metrics,
        group_training_by_length=config.get('group_training_by_length',False),ledger=ledger,callbacks=[journal,plateau])
    started = time.monotonic()
    result = trainer.train(resume_from_checkpoint=str(checkpoint))
    elapsed = time.monotonic()-started
    assert ledger.counts['cycles'] == trainer.state.global_step-800
    assert trainer.state.global_step <= 1438
    assert all(torch.isfinite(p).all() for p in model.parameters() if p.requires_grad)
    trainer.save_model(str(out/'adapter'))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(out/'adapter')
    trainer.accelerator.wait_for_everyone()
    final = frame.iloc[parts['dev']]
    final_original, final_swapped, final_diag = encoder.encode_frame(final)
    labels = final[CLASSES].values.argmax(1)
    trainer.data_collator = PreferenceCollator(tokenizer)
    predictions = []
    scoring_started = time.monotonic()
    for force in (False,True):
        values = trainer.predict(PreferenceDataset(final_original,final_swapped,labels,force_swap=force)).predictions
        if isinstance(values,tuple):values=values[0]
        predictions.append(probabilities(values))
    prediction_seconds = time.monotonic()-scoring_started
    if trainer.is_world_process_zero():
        p = (predictions[0]+predictions[1][:,[1,0,2]])/2
        assert np.isfinite(p).all() and np.allclose(p.sum(1),1)
        np.savez_compressed(out/'dev_predictions.npz',ids=final.id.to_numpy(),labels=labels,
            original=predictions[0],aligned_swap=predictions[1][:,[1,0,2]],probabilities=p)
        json_write(out/'summary.json',{'complete':True,'initial_step':800,'final_step':trainer.state.global_step,
            'new_nominal_cycles':ledger.counts['cycles'],'epochs_completed':float(trainer.state.epoch),
            'stopped_for_time':journal.stopped_for_budget,'stopped_for_plateau':plateau.stopped,
            'training_seconds':elapsed,'development_prediction_seconds':prediction_seconds,
            'raw_swap_averaged':metrics(labels,p),'raw_original':metrics(labels,predictions[0]),
            'train_rows':len(train_frame),'selection_rows':len(dev_frame),'final_dev_rows':len(final),
            'selection_ids_sha256':hashlib.sha256(dev_frame.id.to_numpy().tobytes()).hexdigest(),
            'adapter_sha256':digest(out/'adapter/adapter_model.safetensors'),
            'resume_source':source,'config':config,'dtype':str(dtype),'train_result':result.metrics,
            'train_diagnostics':train_diag,'selection_diagnostics':dev_diag,'final_diagnostics':final_diag,
            'peak_gpu_bytes':torch.cuda.max_memory_allocated(),
            'note':'Historical accepted exposure before step 800 is unknown; new cycles are directly measured.'})
    trainer.accelerator.wait_for_everyone()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('raw','model','parent','split','base-config','config','out'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--deadline',type=float,required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
