"""Scientific contracts for retained content, frozen initialization and paired loss."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from arena.joint_data import JointTokenizer
from arena.span_pack import SpanTokenizer
from arena.span_adapters import add_span_adapter, paired_objective
from arena.span_heads import (SymmetricHead, fit_with_inner_selection, loss_rows, paired_group_gain, normalize)
from arena.span_audit import scheduled_lr, training_audit, matched_training


class Tokens:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    def encode(self, text, add_special_tokens=False):
        return list(text.encode())
    def decode(self, ids, **kwargs):
        return bytes(ids).decode()
    def apply_chat_template(self, rows, **kwargs):
        return self.encode('U:'+rows[0]['content']+'\nA:'+rows[1]['content'])


class SpanContracts(unittest.TestCase):
    def test_invalid_probability_artifacts_are_rejected_without_prior_fill(self):
        for p in ([[0,0,0]], [[2,2,2]], [[.2,.3,.4]], [[-.1,.5,.6]], [[float('nan'),.5,.5]]):
            with self.assertRaises(AssertionError):normalize(p)
        np.testing.assert_allclose(normalize([[0.,.4,.6]]).sum(1),1.)

    def test_amp_audit_uses_lr_before_scheduler_and_counts_skips(self):
        config={'training_steps':64,'warmup_steps':4,'learning_rate':2e-5}
        records=[{'step':1,'learning_rate':0.,'grad_norm':float('nan')},
                 {'step':16,'learning_rate':scheduled_lr(12,config),'grad_norm':6.2},
                 {'step':24,'learning_rate':scheduled_lr(20,config),'grad_norm':9.3},
                 {'step':64,'learning_rate':scheduled_lr(60,config),'grad_norm':4.}]
        audit=training_audit(records,config,{'steps':64})
        self.assertEqual(audit['accepted_optimizer_steps_range'],[61,61])
        self.assertEqual(audit['skipped_optimizer_steps_range'],[3,3])
        partial=training_audit(records[:3],config,{'steps':26})
        self.assertEqual(partial['accepted_optimizer_steps_range'],[21,23])
        with self.assertRaises(ValueError):
            training_audit([{'step':1,'learning_rate':123,'grad_norm':1.}],config,{'steps':1})

    def test_training_contrast_requires_equal_exposure_and_verified_update_counts(self):
        summary=dict(steps=64,canonical_rows=2048,development_rows=1024,world_size=2,
            views_per_update=32,canonical_exposures=2048,non_padding_tokens_seen=1000,
            numerical_audit={'accepted_optimizer_steps_range':[61,61]},
            adapter_weight_audit={'all_finite':True})
        self.assertTrue(matched_training(summary,summary)['matched'])
        other=deepcopy(summary);other['numerical_audit']['accepted_optimizer_steps_range']=[60,60]
        self.assertFalse(matched_training(summary,other)['matched'])
        other=deepcopy(summary);other['numerical_audit']['accepted_optimizer_steps_range']=[60,61]
        self.assertIsNone(matched_training(summary,other)['matched'])
        other=deepcopy(summary);other['non_padding_tokens_seen']=999
        self.assertFalse(matched_training(summary,other)['matched'])

    def test_symmetric_head_gradients_and_predictions(self):
        rng = np.random.default_rng(18)
        d, s = rng.normal(size=(40,2)), np.column_stack([rng.normal(size=(40,2)),np.ones(40)])
        y = rng.integers(0,3,40)
        theta = rng.normal(size=5)*.2
        for kind in ('coupled','factorized'):
            head = SymmetricHead(kind,.1)
            _, gradient = head._objective(theta,d,s,y)
            numerical = []
            for i in range(len(theta)):
                offset = np.eye(len(theta))[i]*1e-5
                numerical.append((head._objective(theta+offset,d,s,y)[0]-
                                  head._objective(theta-offset,d,s,y)[0])/2e-5)
            np.testing.assert_allclose(gradient,numerical,atol=1e-7,rtol=1e-5)
            head.fit(d,s[:,:-1],y)
            p = head.predict_proba(d,s[:,:-1])
            swapped = head.predict_proba(-d,s[:,:-1])[:,[1,0,2]]
            np.testing.assert_allclose(p,swapped,atol=1e-12)
            self.assertTrue(np.isfinite(p).all())

    def test_head_selection_does_not_read_evaluation_labels(self):
        rng = np.random.default_rng(19)
        data = {'rewards':rng.normal(size=(180,2))}
        y = rng.integers(0,3,180)
        groups = np.arange(180)//2
        fit, evaluation = np.arange(150),np.arange(150,180)
        config = {'seed':18,'head_inner_group_folds':3,'head_pca_components':2,
                  'head_regularization_grid':[.01,.1]}
        first, details, _ = fit_with_inner_selection(data,y,groups,fit,evaluation,'reward','coupled',config)
        changed = y.copy(); changed[evaluation] = (changed[evaluation]+1)%3
        second, other_details, _ = fit_with_inner_selection(data,changed,groups,fit,evaluation,'reward','coupled',config)
        np.testing.assert_array_equal(first,second)
        self.assertEqual(details,other_details)

    def test_group_gain_direction_and_constant_effect(self):
        y = np.zeros(10,dtype=int)
        baseline = np.tile([.2,.3,.5],(10,1))
        candidate = np.tile([.4,.2,.4],(10,1))
        config = {'seed':18,'bootstrap_group_replicates':100}
        result = paired_group_gain(y,baseline,candidate,np.array([0,0,0,1,1,2,3,3,3,3]),config)
        self.assertAlmostEqual(result['gain'],np.log(2))
        np.testing.assert_allclose(result['interval_95'],[np.log(2),np.log(2)])
        self.assertEqual(result['groups'],4)

    def test_span_masks_match_frozen_baseline_in_both_orders(self):
        tokenizer = Tokens()
        old, new = JointTokenizer(tokenizer, 512), SpanTokenizer(tokenizer, 512)
        for p, a, b in [('request','first','second'),('p'*800,'a'*2000,'b'*1801),('p','', 'b'*1000)]:
            fields = [tokenizer.encode(x) for x in (p,a,b)]
            for q, first, second in (fields, [fields[0],fields[2],fields[1]]):
                ids, a_span, b_span = new.combine_with_spans(q,first,second)
                self.assertEqual(ids, old.combine(q,first,second))
                retained = new.retained_parts(q,first,second)
                self.assertEqual(ids[a_span[0]:a_span[1]], retained[1])
                self.assertEqual(ids[b_span[0]:b_span[1]], retained[2])

    def test_reward_content_is_swap_equivariant_even_at_odd_budgets(self):
        tokenizer = Tokens()
        encoder = SpanTokenizer(tokenizer, 513)
        for a, b in [('a'*2000,'b'*1801),('a','b'*3000),('', 'b')]:
            q, first, second = [tokenizer.encode(x) for x in ('q'*700,a,b)]
            original = encoder.reward_pair(q,first,second)
            swapped = encoder.reward_pair(q,second,first)
            self.assertEqual(original, swapped[::-1])

    def test_reward_template_handles_transformers_batch_encoding_return(self):
        from transformers import BatchEncoding
        class DictTokens(Tokens):
            def apply_chat_template(self, rows, **kwargs):
                return BatchEncoding({'input_ids':super().apply_chat_template(rows,**kwargs)})
        tokenizer = DictTokens()
        sequences = SpanTokenizer(tokenizer,512).reward_pair(*[tokenizer.encode(x) for x in ('q','a','b')])
        self.assertTrue(all(isinstance(x,int) for seq in sequences for x in seq))

    def test_paired_loss_alignment_symmetry_and_gradients(self):
        first = torch.tensor([[2.,0.,-1.],[-1.,0.,2.]], requires_grad=True)
        logits = torch.cat([first, first[:,[1,0,2]]])
        labels = torch.tensor([0,2,1,2])
        loss, kl = paired_objective(logits, labels, .1)
        torch.testing.assert_close(kl, torch.tensor(0.))
        torch.testing.assert_close(loss, torch.nn.functional.cross_entropy(logits,labels))
        loss.backward()
        self.assertTrue(torch.isfinite(first.grad).all())
        wrong = torch.tensor([0,2,0,2])
        with self.assertRaises(AssertionError):
            paired_objective(logits, wrong, .1)
        perturbed = torch.tensor([[3.,0.,-1.],[-2.,1.,0.],[2.,0.,-1.],[0.,-1.,2.]], requires_grad=True)
        _, penalty = paired_objective(perturbed, labels, .1)
        self.assertGreater(penalty.item(),0)

    def test_added_adapter_preserves_initial_logits_and_freezes_original_weights(self):
        from transformers import Qwen3Config, Qwen3ForSequenceClassification
        from peft import LoraConfig, TaskType, get_peft_model
        torch.set_num_threads(2)
        torch.manual_seed(18)
        config = Qwen3Config(vocab_size=40, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=8,
            num_labels=3,pad_token_id=0)
        model = get_peft_model(Qwen3ForSequenceClassification(config),
            LoraConfig(task_type=TaskType.SEQ_CLS,r=2,lora_alpha=4,lora_dropout=0.,
                       target_modules=['q_proj','k_proj','v_proj','o_proj'], modules_to_save=['score']),
            adapter_name='frozen')
        with torch.no_grad():
            for name, value in model.named_parameters():
                if '.lora_B.frozen.' in name:
                    value.normal_(std=.02)
        model.eval()
        ids = torch.tensor([[3,5,7],[4,6,0]])
        mask = ids.ne(0).long()
        expected = model(input_ids=ids,attention_mask=mask).logits.detach().clone()
        raw=model.get_base_model()
        hidden=raw.model(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state
        token_logits=raw.score(hidden)
        pooled=token_logits[torch.arange(len(ids)),mask.sum(1)-1]
        torch.testing.assert_close(pooled,expected)
        parameters = deepcopy(model.state_dict())
        plan = {'attention_rank':2,'all_linear_mlp_rank':2,'all_linear_attention_rank':1,'lora_dropout':0.}
        counts = add_span_adapter(model, plan, broad=True)
        self.assertGreater(counts['lora_parameters'], 0)
        model.eval()
        actual = model(input_ids=ids,attention_mask=mask).logits
        torch.testing.assert_close(actual,expected,atol=1e-7,rtol=1e-6)
        torch.nn.functional.cross_entropy(actual,torch.tensor([0,1])).backward()
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all()
                            for n,p in model.named_parameters() if '.span.' in n))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for n,p in model.named_parameters() if '.lora_B.span.' in n))
        for name, parameter in model.named_parameters():
            if '.frozen.' in name:
                self.assertFalse(parameter.requires_grad)
                self.assertIsNone(parameter.grad)
                torch.testing.assert_close(parameter,parameters[name])

    def test_paired_trainer_runs_and_evaluates_with_ordinary_collator(self):
        import tempfile
        from transformers import Qwen3Config,Qwen3ForSequenceClassification,TrainingArguments
        from peft import LoraConfig,TaskType,get_peft_model
        from arena.span_joint import PairDataset,PairCollator,SpanTrainer
        from arena.joint_data import PreferenceCollator,PreferenceDataset
        class PadTokens(Tokens):
            def pad(self, rows, **kwargs):
                width=max(len(x['input_ids']) for x in rows)
                return {key:torch.tensor([row[key]+[0]*(width-len(row[key])) for row in rows])
                        for key in ('input_ids','attention_mask')}
        torch.set_num_threads(2);torch.manual_seed(20)
        model=get_peft_model(Qwen3ForSequenceClassification(Qwen3Config(vocab_size=40,
            hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,
            num_key_value_heads=2,head_dim=8,num_labels=3,pad_token_id=0)),
            LoraConfig(task_type=TaskType.SEQ_CLS,r=2,lora_alpha=4,lora_dropout=0.,
                target_modules=['q_proj','k_proj','v_proj','o_proj'],modules_to_save=['score']),
            adapter_name='frozen')
        add_span_adapter(model,{'attention_rank':2,'lora_dropout':0.},broad=False)
        old={n:p.detach().clone() for n,p in model.named_parameters() if '.frozen.' in n}
        ds=PairDataset([[3,4],[5,6],[7,8],[9,10]],[[4,3],[6,5],[8,7],[10,9]],np.array([0,1,2,0]))
        tok=PadTokens()
        with tempfile.TemporaryDirectory() as directory:
            args=TrainingArguments(output_dir=directory,use_cpu=True,max_steps=2,
                per_device_train_batch_size=1,gradient_accumulation_steps=2,
                per_device_eval_batch_size=2,learning_rate=.001,save_strategy='no',
                gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
                report_to='none',remove_unused_columns=False,disable_tqdm=True,dataloader_pin_memory=False)
            trainer=SpanTrainer(model=model,args=args,train_dataset=ds,data_collator=PairCollator(tok),
                                consistency_weight=.1)
            trainer.train()
            trainer.data_collator=PreferenceCollator(tok)
            result=trainer.predict(PreferenceDataset(ds.original,ds.swapped,ds.labels))
            self.assertEqual(result.predictions.shape,(4,3))
            self.assertTrue(np.isfinite(result.predictions).all())
            model.save_pretrained(Path(directory)/'adapter',selected_adapters=['span'])
            exported=Path(directory)/'adapter/span/adapter_model.safetensors'
            self.assertTrue(exported.exists())
            from safetensors.torch import load_file
            state=load_file(exported)
            self.assertTrue(any('lora_B' in name for name in state))
            self.assertTrue(any('score' in name for name in state))
            self.assertFalse(any('.frozen.' in name for name in state))
            for name,p in model.named_parameters():
                if name in old:torch.testing.assert_close(p,old[name])


if __name__ == '__main__':
    unittest.main()
