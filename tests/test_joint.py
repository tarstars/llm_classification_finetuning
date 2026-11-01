import sys
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from arena.joint_data import JointTokenizer,PreferenceDataset,PreferenceCollator


class CharacterTokenizer:
    bos_token_id=1
    eos_token_id=2
    pad_token_id=0
    def encode(self,text,add_special_tokens=False):return list(text.encode())
    def __call__(self,texts,**kwargs):return {'input_ids':[self.encode(t) for t in texts]}
    def pad(self,rows,padding,pad_to_multiple_of,return_tensors):
        length=((max(len(r['input_ids']) for r in rows)+7)//8)*8
        return {k:torch.tensor([r[k]+[0]*(length-len(r[k])) for r in rows]) for k in ['input_ids','attention_mask']}


class JointContracts(unittest.TestCase):
    def test_both_answers_survive_long_context(self):
        encoder=JointTokenizer(CharacterTokenizer(),max_length=512)
        ids=encoder.combine(list(b'p'*2000),list(b'a'*2000+b'A_END'),list(b'b'*2000+b'B_END'))
        text=bytes(ids[1:-1]).decode()
        self.assertLessEqual(len(ids),512)
        for marker in ['[USER REQUEST]','[ANSWER A]','[ANSWER B]','[DECISION]','A_END','B_END']:
            self.assertIn(marker,text)

    def test_swapping_keeps_ties_and_exchanges_winner(self):
        ds=PreferenceDataset([[4],[5],[6]],[[14],[15],[16]],np.array([0,1,2]),force_swap=True)
        self.assertEqual([ds[i]['labels'] for i in range(3)],[1,0,2])
        self.assertEqual(ds[0]['input_ids'],[14])
        augmented=PreferenceDataset([[4]],[[14]],np.array([0]),augment=True)
        with patch('arena.joint_data.random.random',return_value=0.):
            self.assertEqual(augmented[0]['labels'],1)

    def test_padding_is_masked_without_moving_labels(self):
        collator=PreferenceCollator(CharacterTokenizer())
        batch=collator([{'input_ids':[4,2],'attention_mask':[1,1],'labels':1},
                        {'input_ids':[7,8,2],'attention_mask':[1,1,1],'labels':2}])
        self.assertEqual(tuple(batch['input_ids'].shape),(2,8))
        self.assertEqual(batch['attention_mask'][0].tolist(),[1,1,0,0,0,0,0,0])
        self.assertEqual(batch['labels'].tolist(),[1,2])

    @unittest.skipUnless(importlib.util.find_spec('peft'), 'Install the train extra for trainer checks')
    def test_gradient_accumulation_matches_full_batch(self):
        from transformers import TrainingArguments
        from transformers.modeling_outputs import SequenceClassifierOutput
        from arena.joint_train import PreferenceTrainer
        class ToyClassifier(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.score=torch.nn.Linear(2,3)
            def forward(self,input_ids,labels=None,**kwargs):
                return SequenceClassifierOutput(logits=self.score(input_ids.float()))
        data=[{'input_ids':torch.tensor([i,1]),'labels':i%3} for i in range(4)]
        def collate(rows):
            return {'input_ids':torch.stack([r['input_ids'] for r in rows]),
                    'labels':torch.tensor([r['labels'] for r in rows])}
        states=[]
        with tempfile.TemporaryDirectory() as tmp:
            for batch,accumulation in [(4,1),(1,4)]:
                torch.manual_seed(11)
                model=ToyClassifier()
                args=TrainingArguments(output_dir=f'{tmp}/{batch}',use_cpu=True,
                    per_device_train_batch_size=batch,gradient_accumulation_steps=accumulation,
                    max_steps=1,learning_rate=.01,optim='sgd',max_grad_norm=0,
                    save_strategy='no',report_to='none',disable_tqdm=True,seed=11)
                trainer=PreferenceTrainer(model=model,args=args,train_dataset=data,data_collator=collate)
                inputs=collate(data)
                expected=torch.nn.functional.cross_entropy(model(**inputs).logits,inputs['labels'])
                actual=trainer.compute_loss(model,inputs,num_items_in_batch=100)
                torch.testing.assert_close(actual,expected)
                self.assertFalse(trainer.model_accepts_loss_kwargs)
                trainer.train()
                states.append({k:v.detach().clone() for k,v in model.state_dict().items()})
        for name in states[0]:
            torch.testing.assert_close(states[0][name],states[1][name],atol=1e-7,rtol=1e-6)

    @unittest.skipUnless(importlib.util.find_spec('peft'), 'Install the train extra to test adapter training')
    def test_adapter_and_classification_head_train_and_reload(self):
        from transformers import Qwen3Config,AutoModelForSequenceClassification,AutoTokenizer
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast
        from arena.joint_train import load_trainable_model,make_trainer
        from arena.joint_infer import load_inference_model
        torch.set_num_threads(2);torch.manual_seed(7)
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)/'base';base.mkdir()
            config=Qwen3Config(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=1,
                num_attention_heads=4,num_key_value_heads=2,head_dim=8,num_labels=3,pad_token_id=0)
            AutoModelForSequenceClassification.from_config(config).save_pretrained(base)
            vocab={'[PAD]':0,'[UNK]':1,'[EOS]':2,**{f'w{i}':i for i in range(3,32)}}
            tokenizer=Tokenizer(WordLevel(vocab,unk_token='[UNK]'));tokenizer.pre_tokenizer=Whitespace()
            tokenizer=PreTrainedTokenizerFast(tokenizer_object=tokenizer,pad_token='[PAD]',unk_token='[UNK]',eos_token='[EOS]')
            tokenizer.save_pretrained(base)
            cfg={'quantize':False,'lora_rank':2,'lora_alpha':4,'lora_dropout':0.,
                'target_modules':['q_proj','k_proj','v_proj','o_proj'],'epochs':1,'train_batch_size':2,
                'eval_batch_size':2,'gradient_accumulation_steps':1,'learning_rate':.001,
                'weight_decay':.01,'eval_steps':2,'seed':7,'max_train_seconds':0}
            model,_=load_trainable_model(base,cfg)
            head_name=next(n for n,p in model.named_parameters() if 'score' in n and p.requires_grad)
            before=dict(model.named_parameters())[head_name].detach().clone()
            original=[[3+i%3,8,9,2] for i in range(12)]
            swapped=[[10,3+i%3,9,2] for i in range(12)]
            data=PreferenceDataset(original,swapped,np.arange(12)%3)
            out=Path(tmp)/'run';out.mkdir()
            trainer=make_trainer(model,tokenizer,data,data,cfg,out,pilot_steps=3)
            trainer.args.load_best_model_at_end=False
            trainer.train()
            self.assertFalse(torch.equal(before,dict(model.named_parameters())[head_name]))
            self.assertTrue(any(torch.count_nonzero(p).item()>0 for n,p in model.named_parameters() if 'lora_B' in n))
            model.eval()
            inputs=PreferenceCollator(tokenizer)([data[0],data[1]])
            inputs.pop('labels')
            with torch.no_grad():expected=model(**inputs).logits
            model.save_pretrained(out/'adapter')
            restored=load_inference_model(base,out/'adapter',cfg)
            with torch.no_grad():actual=restored(**inputs).logits
            torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)


if __name__=='__main__':unittest.main()
