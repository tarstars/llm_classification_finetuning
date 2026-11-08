"""Native Transformers4.42.3 Gemma2 head, GPTQ reference and split execution.

The mirrored checkpoint has a simple linear head and native softcapped eager
attention. It must not inherit the retained expert's MLP head or attention path.
"""
# Split-forward structure adapted from Transformers4.42.3 Gemma2Model:
# Copyright2024GoogleInc./HuggingFaceInc.;ApacheLicense2.0.
import importlib.util
import sys
import types

import torch
from torch import nn
from transformers import Gemma2Model, Gemma2PreTrainedModel

if __package__:
    from .gptq8 import native_layout_torch
else:
    from gptq8 import native_layout_torch


class MirrorClassifier(Gemma2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Gemma2Model(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(self, input_ids, attention_mask, position_ids, pool_indices):
        hidden = self.model(input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, use_cache=False, return_dict=True).last_hidden_state
        rows = torch.arange(len(hidden), device=hidden.device)
        return self.score(hidden[rows, pool_indices.to(hidden.device)]).float()

    def stage1(self, batch):
        device = self.model.embed_tokens.weight.device
        values = {k:v.to(device) for k,v in batch.items()}
        hidden = self.model.embed_tokens(values['input_ids'])
        cache_position = torch.arange(hidden.shape[1], device=device)
        mask = self.model._update_causal_mask(values['attention_mask'], hidden,
            cache_position, None, False)
        normalizer = torch.tensor(self.config.hidden_size**.5, dtype=hidden.dtype)
        hidden = hidden * normalizer
        for layer in self.model.layers[:len(self.model.layers)//2]:
            hidden = layer(hidden, attention_mask=mask, position_ids=values['position_ids'],
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=cache_position)[0]
        target = self.model.norm.weight.device
        return dict(hidden=hidden.to(target), mask=mask.to(target),
            position_ids=values['position_ids'].to(target),
            cache_position=cache_position.to(target), pool_indices=values['pool_indices'].to(target))

    def stage2(self, state):
        hidden = state['hidden']
        for layer in self.model.layers[len(self.model.layers)//2:]:
            hidden = layer(hidden, attention_mask=state['mask'], position_ids=state['position_ids'],
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=state['cache_position'])[0]
        hidden = self.model.norm(hidden)
        return self.score(hidden[torch.arange(len(hidden),device=hidden.device),state['pool_indices']]).float()


def native_quantized_class(source_path):
    """Use the pinned official GPTQ8 fallback, without an Optimum conversion."""
    spec = importlib.util.spec_from_file_location('span019_pinned_qlinear', source_path)
    module = importlib.util.module_from_spec(spec)
    # The chosen backend is the source's Torch fallback. Do not load optional
    # compiled extensions (which have their own Torch/CUDA ABI) merely to disable
    # them afterwards. Restore the import table before model construction.
    optional = ['autogptq_cuda_64', 'autogptq_cuda_256']
    previous = {name: sys.modules[name] for name in optional if name in sys.modules}
    try:
        for name in optional: sys.modules[name] = None
        spec.loader.exec_module(module)
    finally:
        for name in optional: sys.modules.pop(name, None)
        sys.modules.update(previous)

    class NativeQuantizedMirror(MirrorClassifier):
        def __init__(self, config):
            super().__init__(config)
            names = [(name, layer) for name, layer in self.named_modules()
                     if name.startswith('model.layers.') and isinstance(layer, nn.Linear)]
            for name, layer in names:
                assert layer.bias is None
                with torch.device('meta'):
                    quant = module.QuantLinear(8,128,layer.in_features,layer.out_features,False)
                # wf is a nonpersistent helper, unlike the four checkpoint buffers.
                quant.wf = torch.arange(0,32,8,dtype=torch.int32,device='cpu').unsqueeze(0)
                quant.autogptq_cuda_available = False
                parent, attribute = name.rsplit('.',1)
                setattr(self.get_submodule(parent),attribute,quant)
    return NativeQuantizedMirror


def apply_native_dense_layout(model, projection_names):
    """Keep source keys/hooks while exactly matching the native flattened GEMM."""
    def forward(layer, x):
        result = torch.matmul(x.reshape(-1,x.shape[-1]),layer.weight.t())
        return result.reshape(x.shape[:-1]+(layer.weight.shape[0],))
    for name in projection_names:
        layer = model.get_submodule(name)
        assert isinstance(layer,nn.Linear) and layer.bias is None
        layer.weight = nn.Parameter(native_layout_torch(layer.weight.detach()),requires_grad=False)
        function = types.MethodType(forward,layer)
        if hasattr(layer,'_hf_hook'):
            layer._old_forward = function
        else:
            layer.forward = function
