"""Direct accepted-update and private row/token accounting around Trainer cycles."""
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path

from transformers import TrainerCallback

from .data import json_write
from .joint_data import PreferenceDataset, PreferenceCollator
from .joint_train import PreferenceTrainer


class ExposureLedger:
    def __init__(self, directory, rank=0, resume=None):
        self.directory, self.rank = Path(directory), int(rank)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.pending = []
        self.counts = {'cycles':0, 'accepted_updates':0, 'skipped_updates':0,
                       'processed_tokens':0, 'accepted_tokens':0,
                       'processed_rows':0, 'accepted_rows':0}
        self.processed_ids, self.accepted_ids = Counter(), Counter()
        self.processed_classes, self.accepted_classes = Counter(), Counter()
        self.last_step = None
        if resume is not None:
            state = json.loads(Path(resume).read_text())
            assert state['rank'] == self.rank
            self.counts = state['counts']
            self.last_step = state['last_global_step']
            for name in ('processed_ids','accepted_ids','processed_classes','accepted_classes'):
                setattr(self, name, Counter({int(k):v for k,v in state[name].items()}))

    def add(self, ids, labels, lengths):
        assert len(ids) == len(labels) == len(lengths)
        self.pending.extend((int(i), int(y), int(n)) for i,y,n in zip(ids,labels,lengths))

    def finish(self, step, skipped, scale_before, scale_after, scheduler_before, scheduler_after):
        assert self.pending, 'An optimizer cycle must contain processed rows'
        if self.last_step is not None:
            assert step == self.last_step+1, 'Unexpected global-step discontinuity'
        assert scheduler_after-scheduler_before == int(not skipped), 'Scheduler and actual skip flag disagree'
        ids, labels, lengths = zip(*self.pending)
        self.counts['cycles'] += 1
        self.counts['skipped_updates' if skipped else 'accepted_updates'] += 1
        self.counts['processed_tokens'] += sum(lengths)
        self.counts['processed_rows'] += len(ids)
        self.processed_ids.update(ids)
        self.processed_classes.update(labels)
        if not skipped:
            self.counts['accepted_tokens'] += sum(lengths)
            self.counts['accepted_rows'] += len(ids)
            self.accepted_ids.update(ids)
            self.accepted_classes.update(labels)
        event = {'global_step':step, 'rank':self.rank, 'skipped':bool(skipped),
            'scale_before':scale_before,'scale_after':scale_after,
            'scheduler_before':scheduler_before,'scheduler_after':scheduler_after,
            'ids':list(ids),'canonical_labels':list(labels),'token_lengths':list(lengths),
            'cumulative':dict(self.counts)}
        with (self.directory / f'exposure_rank{self.rank}.jsonl').open('a') as stream:
            stream.write(json.dumps(event)+'\n')
        self.pending = []
        self.last_step = int(step)
        return event

    def state(self):
        assert not self.pending, 'Only checkpoint at an optimizer-cycle boundary'
        return {'rank':self.rank,'last_global_step':self.last_step,'counts':self.counts,
                **{name:dict(getattr(self,name)) for name in
                   ('processed_ids','accepted_ids','processed_classes','accepted_classes')}}

    def save(self, directory):
        json_write(Path(directory) / f'audit_state_rank{self.rank}.json', self.state())

    def summary(self):
        return {'rank':self.rank, **self.counts,
            'processed_unique_rows':len(self.processed_ids),'accepted_unique_rows':len(self.accepted_ids),
            'processed_class_counts':[self.processed_classes[i] for i in range(3)],
            'accepted_class_counts':[self.accepted_classes[i] for i in range(3)],
            'last_global_step':self.last_step}


class ExposureDataset(PreferenceDataset):
    def __init__(self, *args, row_ids, **kwargs):
        super().__init__(*args, **kwargs)
        self.row_ids = list(map(int, row_ids))
        assert len(self.row_ids) == len(self)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['_audit'] = {'id':self.row_ids[index], 'label':int(self.labels[index]),
                          'tokens':len(item['input_ids'])}
        return item


class ExposureCollator:
    def __init__(self, tokenizer=None, collator=None):
        self.base = collator or PreferenceCollator(tokenizer)

    def __call__(self, rows):
        batch = self.base([{k:v for k,v in row.items() if k != '_audit'} for row in rows])
        if '_audit' in rows[0]:
            batch['_audit'] = {name:[row['_audit'][key] for row in rows]
                               for name,key in [('ids','id'),('labels','label'),('lengths','tokens')]}
        return batch


class ExposureTrainer(PreferenceTrainer):
    def __init__(self, *args, ledger, **kwargs):
        self.ledger = ledger
        super().__init__(*args, **kwargs)
        self.add_callback(ExposureCallback(self, ledger))

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        metadata = inputs.pop('_audit', None)
        if model.training and metadata is not None:
            self.ledger.add(metadata['ids'], metadata['labels'], metadata['lengths'])
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)


class ExposureCallback(TrainerCallback):
    def __init__(self, trainer, ledger):
        self.trainer, self.ledger = trainer, ledger

    def scheduler_position(self):
        scheduler = self.trainer.lr_scheduler
        return int(getattr(scheduler, 'scheduler', scheduler).last_epoch)

    def scale(self):
        scaler = self.trainer.accelerator.scaler
        return float(scaler.get_scale()) if scaler is not None else None

    def on_train_begin(self, args, state, control, **kwargs):
        if self.ledger.last_step is not None:
            assert self.ledger.last_step == state.global_step, 'Exposure state does not match checkpoint'

    def on_step_begin(self, args, state, control, **kwargs):
        self.scale_before = self.scale()
        self.scheduler_before = self.scheduler_position()

    def on_step_end(self, args, state, control, **kwargs):
        self.ledger.finish(state.global_step, self.trainer.accelerator.optimizer_step_was_skipped,
            self.scale_before, self.scale(), self.scheduler_before, self.scheduler_position())

    def on_save(self, args, state, control, **kwargs):
        directory = Path(args.output_dir) / f'checkpoint-{state.global_step}'
        self.ledger.save(directory)

    def on_train_end(self, args, state, control, **kwargs):
        self.ledger.save(self.ledger.directory)
        json_write(self.ledger.directory / f'exposure_summary_rank{self.ledger.rank}.json', self.ledger.summary())
