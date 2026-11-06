"""Exercise the actual Trainer callback order and fixed accepted-update schedule on CPU."""
from pathlib import Path
from types import SimpleNamespace
import sys,tempfile,time,unittest
from unittest.mock import patch
import torch
from transformers import TrainingArguments
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from arena.span008_size import FixedScheduleTrainer,AcceptedStop
from arena.training_exposure import ExposureLedger,ExposureDataset,ExposureCollator


class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__();self.embedding=torch.nn.Embedding(12,4);self.score=torch.nn.Linear(4,3)
    def forward(self,input_ids,attention_mask=None):return SimpleNamespace(logits=self.score(self.embedding(input_ids).mean(1)))


class AcceptedUpdateTest(unittest.TestCase):
    def test_actual_callback_order_stops_on_current_accepted_cycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger=ExposureLedger(tmp);tokens=[[1+i%5,2] for i in range(8)];dataset=ExposureDataset(tokens,tokens,[i%3 for i in range(8)],row_ids=list(range(8)))
            def collate(rows):return {k:torch.tensor([r[k] for r in rows]) for k in ['input_ids','attention_mask','labels']}
            args=TrainingArguments(output_dir=tmp,use_cpu=True,max_steps=4,per_device_train_batch_size=2,gradient_accumulation_steps=2,
                warmup_steps=1,learning_rate=.001,lr_scheduler_type='cosine',eval_strategy='no',save_strategy='no',report_to='none',
                remove_unused_columns=False,disable_tqdm=True,dataloader_pin_memory=False)
            trainer=FixedScheduleTrainer(model=TinyClassifier(),args=args,train_dataset=dataset,ledger=ledger,data_collator=ExposureCollator(collator=collate))
            trainer.fixed_schedule_updates=2;trainer.add_callback(AcceptedStop(ledger,{'size_pilot_accepted_updates':2},time.time()+3600))
            with patch('torch.cuda.current_device',return_value='cpu'),patch('torch.distributed.all_reduce'):
                trainer.train()
            self.assertEqual(trainer.state.global_step,2)
            self.assertEqual(ledger.counts['accepted_updates'],2)
            self.assertEqual(ledger.counts['accepted_rows'],8)
            self.assertEqual(ledger.counts['skipped_updates'],0)
            self.assertEqual(trainer.lr_scheduler.last_epoch,2)
            self.assertEqual(trainer.optimizer.param_groups[0]['lr'],0)


if __name__=='__main__':unittest.main()
