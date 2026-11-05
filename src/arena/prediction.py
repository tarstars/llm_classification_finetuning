"""Reusable label-free frozen classifier inference; no training or target access."""
from pathlib import Path
import os
import numpy as np
import torch
from transformers import AutoTokenizer,TrainingArguments
from .joint_data import JointTokenizer,PreferenceDataset
from .joint_infer import load_inference_model
from .joint_train import PreferenceTrainer,probabilities
from .experiment import atomic_json


class InferenceCollator:
    def __init__(self,tokenizer):self.tokenizer=tokenizer
    def __call__(self,rows):
        # PreferenceDataset carries ignored labels by default; remove them before
        # prediction so an all-ignored CE does not create a meaningless NaN loss.
        return self.tokenizer.pad([{k:v for k,v in row.items() if k!='labels'} for row in rows],
            padding=True,pad_to_multiple_of=8,return_tensors='pt')


class FrozenClassifier:
    def __init__(self,model_path,adapter,config,out,batch_size=None):
        self.rank=int(os.environ.get('LOCAL_RANK','0'));torch.cuda.set_device(self.rank)
        assert torch.cuda.device_count()==2
        torch.manual_seed(config['seed']);np.random.seed(config['seed'])
        self.tokenizer=AutoTokenizer.from_pretrained(str(adapter),local_files_only=True,padding_side='right')
        if self.tokenizer.pad_token_id is None:self.tokenizer.pad_token=self.tokenizer.eos_token
        model=load_inference_model(model_path,adapter,config);model.config.pad_token_id=self.tokenizer.pad_token_id
        model.config.problem_type='single_label_classification'
        bf16=torch.cuda.get_device_capability()[0]>=8
        options=TrainingArguments(output_dir=str(Path(out)/'evaluation_tmp'),
            per_device_eval_batch_size=batch_size or config['eval_batch_size'],fp16=not bf16,bf16=bf16,
            report_to='none',remove_unused_columns=False,dataloader_num_workers=0,dataloader_pin_memory=True,disable_tqdm=True)
        self.trainer=PreferenceTrainer(model=model,args=options,processing_class=self.tokenizer,data_collator=InferenceCollator(self.tokenizer))

    @property
    def root(self):return self.trainer.is_world_process_zero()

    def encode(self,frame,max_length):return JointTokenizer(self.tokenizer,max_length).encode_frame(frame)

    def predict(self,original,swapped):
        views=[]
        for force in [False,True]:
            values=self.trainer.predict(PreferenceDataset(original,swapped,force_swap=force)).predictions
            if isinstance(values,tuple):values=values[0]
            views.append(probabilities(values))
        a,b=views[0],views[1][:,[1,0,2]]
        assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
        np.testing.assert_allclose(a.sum(1),1,atol=1e-6);np.testing.assert_allclose(b.sum(1),1,atol=1e-6)
        return a,b

    def save_hardware(self,out):
        torch.cuda.synchronize();device=torch.cuda.current_device()
        atomic_json(Path(out)/f'gpu_rank{self.rank}.json',{'rank':self.rank,'device':device,'name':torch.cuda.get_device_name(device),
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),'peak_reserved_bytes':torch.cuda.max_memory_reserved(device)})
        self.trainer.accelerator.wait_for_everyone()
