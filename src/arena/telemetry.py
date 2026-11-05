"""Lossless finite/absent metric representation and aggregate training events."""
import math
import hashlib
import json
import time

from transformers import TrainerCallback
from .experiment import EventJournal, atomic_json


def optimizer_fingerprint(state):
    """Canonical values, independent of torch.save container serialization."""
    import torch
    digest=hashlib.sha256(json.dumps(state['param_groups'],sort_keys=True,separators=(',',':')).encode())
    for index,values in sorted(state['state'].items()):
        digest.update(str(index).encode())
        for name,value in sorted(values.items()):
            digest.update(name.encode())
            if isinstance(value,torch.Tensor):
                tensor=value.detach().cpu().contiguous()
                digest.update(str((str(tensor.dtype),list(tensor.shape))).encode())
                digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
            else:digest.update(json.dumps(value,sort_keys=True).encode())
    return digest.hexdigest()


def finite_metrics(values):
    result={}
    for name,value in values.items():
        if value is None or isinstance(value,(str,bool)):
            result[name]=value
            continue
        try:number=float(value)
        except (TypeError,ValueError):continue
        result[name]=number if math.isfinite(number) else None
        if not math.isfinite(number):result[name+'_nonfinite']=True
    return result


def restart_factor(position, origin, warmup, cycles):
    """Accepted scheduler positions, including a nonzero first warmup step."""
    index=max(0,position-origin)
    if index<warmup:return (index+1)/warmup
    return .5*(1+math.cos(math.pi*min(1.,(index-warmup)/max(1,cycles-warmup))))


class TrainingTelemetry(TrainerCallback):
    def __init__(self,trainer,ledger,directory):
        self.trainer,self.ledger=trainer,ledger
        self.journal=EventJournal(directory)
        self.started=time.monotonic()
        self.consecutive_skips=0

    def on_train_begin(self,args,state,control,**kwargs):
        self.started=time.monotonic()
        if state.is_world_process_zero:self.journal.emit('training','started',step=state.global_step,counts=self.ledger.summary())

    def on_step_end(self,args,state,control,**kwargs):
        self.consecutive_skips=self.consecutive_skips+1 if self.trainer.accelerator.optimizer_step_was_skipped else 0
        if self.consecutive_skips>=8:
            control.should_training_stop=True;control.should_save=True
        if state.is_world_process_zero and (self.ledger.counts['cycles']%5==0 or control.should_training_stop):
            import torch
            self.journal.emit('training','cycle',step=state.global_step,elapsed_seconds=time.monotonic()-self.started,
                counts=self.ledger.summary(),metrics=finite_metrics({'learning_rate':self.trainer.optimizer.param_groups[0]['lr'],
                 'amp_scale':self.trainer.accelerator.scaler.get_scale() if self.trainer.accelerator.scaler else None,
                 'scheduler_position':getattr(self.trainer.lr_scheduler,'scheduler',self.trainer.lr_scheduler).last_epoch}),
                gpu={'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()})

    def on_log(self,args,state,control,logs=None,**kwargs):
        if state.is_world_process_zero:
            self.journal.emit('training','metrics',step=state.global_step,elapsed_seconds=time.monotonic()-self.started,
                              metrics=finite_metrics(logs or {}))

    def on_evaluate(self,args,state,control,metrics=None,**kwargs):
        if state.is_world_process_zero:self.journal.emit('validation','measured',step=state.global_step,metrics=finite_metrics(metrics or {}))

    def on_save(self,args,state,control,**kwargs):
        if state.is_world_process_zero:self.journal.emit('checkpoint','saved',step=state.global_step,counts=self.ledger.summary())

    def on_train_end(self,args,state,control,**kwargs):
        if state.is_world_process_zero:self.journal.emit('training','finished',step=state.global_step,
            elapsed_seconds=time.monotonic()-self.started,counts=self.ledger.summary())
