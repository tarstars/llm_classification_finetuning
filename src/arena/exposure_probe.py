"""Actual CUDA/Trainer forced-overflow and checkpoint-resume contract for E01."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import PreTrainedModel, PretrainedConfig, TrainingArguments, TrainerCallback, set_seed
from transformers.modeling_outputs import SequenceClassifierOutput

from .data import json_write
from .training_exposure import ExposureDataset, ExposureCollator, ExposureLedger, ExposureTrainer


class ProbeModel(PreTrainedModel):
    config_class = PretrainedConfig
    def __init__(self, config):
        super().__init__(config)
        self.embed = torch.nn.Embedding(32, 8)
        self.score = torch.nn.Linear(8, 3)
        self.post_init()

    def forward(self, input_ids, attention_mask=None, **kwargs):
        h = self.embed(input_ids)
        h = (h*attention_mask[...,None]).sum(1)/attention_mask.sum(1,keepdim=True)
        return SequenceClassifierOutput(logits=self.score(h))


def collate(rows):
    return {name:torch.tensor([row[name] for row in rows]) for name in ('input_ids','attention_mask','labels')}


class VerifyChanges(TrainerCallback):
    def __init__(self, trainer):
        self.trainer, self.events = trainer, []

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        self.before = [p.detach().clone() for p in model.parameters()]

    def on_step_end(self, args, state, control, model=None, **kwargs):
        changed = any(not torch.equal(a,b.detach()) for a,b in zip(self.before,model.parameters()))
        skipped = bool(self.trainer.accelerator.optimizer_step_was_skipped)
        assert changed == (not skipped), (state.global_step, changed, skipped)
        assert all(torch.isfinite(p).all() for p in model.parameters())
        self.events.append({'step':state.global_step,'parameters_changed':changed,'skipped':skipped})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available(), 'E01 requires actual CUDA GradScaler behavior'
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    original = [[3+(i%12),4,5,6] for i in range(48)]
    dataset = ExposureDataset(original, original, np.arange(48)%3, row_ids=np.arange(48))
    runs = []
    final = None
    for name, resume in [('continuous',None), ('resumed',args.out/'continuous/checkpoints/checkpoint-3')]:
        set_seed(20261119)
        model = ProbeModel(PretrainedConfig(num_labels=3))
        directory = args.out/name
        ledger_state = resume / f'audit_state_rank{rank}.json' if resume else None
        ledger = ExposureLedger(directory, rank, ledger_state)
        training_args = TrainingArguments(output_dir=str(directory/'checkpoints'), max_steps=5,
            per_device_train_batch_size=2, gradient_accumulation_steps=2,
            learning_rate=.01, warmup_steps=0, lr_scheduler_type='cosine', fp16=True,
            save_strategy='steps',save_steps=3,save_total_limit=3,eval_strategy='no',
            report_to='none',logging_steps=1,logging_nan_inf_filter=False,seed=20261119,
            remove_unused_columns=False,ddp_find_unused_parameters=False,disable_tqdm=True,
            dataloader_num_workers=0,optim='adamw_torch',max_grad_norm=1.)
        trainer = ExposureTrainer(model=model,args=training_args,train_dataset=dataset,
            data_collator=ExposureCollator(collator=collate),ledger=ledger)
        watch = VerifyChanges(trainer)
        trainer.add_callback(watch)
        handle = model.score.weight.register_hook(lambda gradient: torch.full_like(gradient, float('inf'))
            if trainer.state.global_step == 1 else gradient)
        trainer.train(resume_from_checkpoint=str(resume) if resume else None)
        handle.remove()
        assert ledger.counts['cycles'] == 5 and ledger.counts['accepted_updates'] == 4
        assert ledger.counts['skipped_updates'] == 1
        assert ledger.counts['accepted_tokens'] < ledger.counts['processed_tokens']
        if final is None:
            final = {k:v.detach().clone() for k,v in model.state_dict().items()}
            reference_state = ledger.state()
        else:
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, final[key], atol=1e-7, rtol=1e-6)
            assert ledger.state() == reference_state, 'Exposure counters changed after exact resume'
        runs.append({'name':name,'counts':ledger.summary(),'parameter_checks':watch.events,
                     'scaler':trainer.accelerator.scaler.state_dict(),
                     'scheduler':trainer.lr_scheduler.state_dict()})
    result = {'passed':True,'rank':rank,'torch':torch.__version__,
              'cuda_device':torch.cuda.get_device_name(rank),'runs':runs,
              'checks':['forced overflow skips actual parameter update','finite accepted parameters',
                        'scheduler counts accepted updates','checkpoint resumes counters and parameters']}
    json_write(args.out/f'probe_rank{rank}.json', result)
    print('EXPOSURE_PROBE',json.dumps(result),flush=True)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
