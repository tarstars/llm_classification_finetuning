"""Add a trainable pilot adapter without merging or changing the frozen baseline."""
from copy import deepcopy

import torch
from peft import LoraConfig, TaskType
from peft.tuners.lora.layer import LoraLayer
from peft.utils.other import ModulesToSaveWrapper

ATTENTION = ['q_proj','k_proj','v_proj','o_proj']
MLP = ['gate_proj','up_proj','down_proj']


def add_span_adapter(model, config, broad=False, frozen='frozen'):
    raw = model.base_model.model
    assert isinstance(raw.score, ModulesToSaveWrapper)
    # Keep the learned head, but only the new adapter should own its trainable copy.
    raw.score = deepcopy(raw.score.modules_to_save[frozen])
    rank = config['all_linear_mlp_rank'] if broad else config['attention_rank']
    rank_pattern = {key: config['all_linear_attention_rank'] for key in ATTENTION} if broad else {}
    alpha_pattern = {key: 2*value for key, value in rank_pattern.items()}
    adapter = LoraConfig(task_type=TaskType.SEQ_CLS, r=rank, lora_alpha=2*rank,
        lora_dropout=config['lora_dropout'], bias='none', target_modules=ATTENTION + (MLP if broad else []),
        rank_pattern=rank_pattern, alpha_pattern=alpha_pattern, modules_to_save=['score'])
    model.add_adapter('span', adapter)
    model.base_model.set_adapter([frozen, 'span'])
    for module in model.modules():
        if isinstance(module, LoraLayer) and frozen in module.lora_dropout:
            module.lora_dropout[frozen] = torch.nn.Identity()
    for name, parameter in model.named_parameters():
        trainable = ('.lora_A.span.' in name or '.lora_B.span.' in name or
                     '.score.modules_to_save.span.' in name)
        parameter.requires_grad_(trainable)
    trainable = {name: p.numel() for name, p in model.named_parameters() if p.requires_grad}
    assert trainable and all('.span.' in name for name in trainable)
    assert any('.score.' in name for name in trainable)
    return {'trainable_parameters': sum(trainable.values()),
            'lora_parameters': sum(size for name, size in trainable.items() if '.lora_' in name),
            'head_parameters': sum(size for name, size in trainable.items() if '.score.' in name),
            'target_modules': sorted(adapter.target_modules), 'rank_pattern': rank_pattern}


def paired_objective(logits, labels, consistency_weight=0.):
    """Rows are [all originals, all swapped]; labels must be aligned accordingly."""
    assert len(logits) % 2 == 0
    half = len(logits) // 2
    swap = torch.tensor([1,0,2], device=labels.device)
    assert torch.equal(labels[half:], swap[labels[:half]])
    loss = torch.nn.functional.cross_entropy(logits.float(), labels)
    first = torch.log_softmax(logits[:half].float(), dim=1)
    second = torch.log_softmax(logits[half:].float()[:,[1,0,2]], dim=1)
    kl = .5*((first.exp()*(first-second)).sum(1) +
             (second.exp()*(second-first)).sum(1)).mean()
    return loss + consistency_weight*kl, kl
