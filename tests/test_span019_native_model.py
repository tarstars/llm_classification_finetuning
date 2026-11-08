from pathlib import Path
import json
import os
import tempfile
import unittest

import numpy as np
import torch
import transformers
from safetensors.numpy import save_file

from arena.span019_batch import schedule, tensors


class NativeBatchTests(unittest.TestCase):
    def test_padding_masks_literal_pad_and_preserves_original_positions_and_pool(self):
        entries = [dict(input_ids=[2,0,5,1],native_position_offset=1996,pool_index=3),
                   dict(input_ids=[2,1],native_position_offset=1998,pool_index=1)]
        batch = tensors(entries,[0,1])
        self.assertEqual(batch['attention_mask'].tolist(),[[1,1,1,1],[0,0,1,1]])
        self.assertEqual(batch['position_ids'].tolist(),[[1996,1997,1998,1999],[0,0,1998,1999]])
        self.assertEqual(batch['pool_indices'].tolist(),[3,3])

    def test_padded_budget_and_complete_index_coverage(self):
        lengths=[2001,20,1500,600,900,60,2000,1500,1600,1800]
        entries=[dict(input_ids=[2]*n) for n in lengths]
        batches=schedule(entries,max_tokens=8192,max_rows=4)
        self.assertEqual(sorted(i for b in batches for i in b),list(range(len(entries))))
        for b in batches:
            self.assertLessEqual(len(b),4)
            self.assertLessEqual(len(b)*max(lengths[i] for i in b),8192)


NATIVE_PATH=Path(os.environ.get('ARENA_NATIVE_QLINEAR_PATH','/nonexistent'))


@unittest.skipUnless(transformers.__version__=='4.42.3' and NATIVE_PATH.is_file(),
                     'Run in pinned4.42.3 runtime with the audited native GPTQ source')
class CompleteNativeModelTests(unittest.TestCase):
    def test_original_quantized_dense_and_split_with_actual_saved_tiny_model(self):
        from transformers import Gemma2Config
        from arena.dense_checkpoint import expand_checkpoint
        from arena.span019_model import MirrorClassifier, apply_native_dense_layout, native_quantized_class
        torch.set_num_threads(2);torch.manual_seed(42)
        # GPTQ's source constructor packs whole32-bit words. Keep every native
        # projection input/output dimension divisible by32, as the real model is.
        config=Gemma2Config(vocab_size=64,hidden_size=64,intermediate_size=128,
            num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=16,
            query_pre_attn_scalar=16,max_position_embeddings=8192,pad_token_id=0,
            num_labels=3,use_cache=False,attn_implementation='eager')
        config._attn_implementation='eager'
        dense=MirrorClassifier(config).half().eval()
        params={k:v.detach().cpu().numpy().copy() for k,v in dense.state_dict().items()}
        rng=np.random.default_rng(42);projections=[]
        for name,layer in dense.named_modules():
            if name.startswith('model.layers.') and isinstance(layer,torch.nn.Linear):
                projections.append(name);params.pop(name+'.weight')
                params[name+'.qweight']=rng.integers(0,2**32,(layer.in_features//4,layer.out_features),dtype=np.uint32).view(np.int32)
                params[name+'.qzeros']=np.full((1,layer.out_features//4),0x7f7f7f7f,np.int32)
                params[name+'.scales']=np.full((1,layer.out_features),.001,np.float16)
                params[name+'.g_idx']=np.zeros(layer.in_features,np.int32)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);origin=root/'origin';origin.mkdir()
            save_file(params,str(origin/'model.safetensors'),metadata={'format':'pt'})
            (origin/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{k:'model.safetensors' for k in params}}))
            config.save_pretrained(origin)
            result=expand_checkpoint(origin,root/'dense')
            config.save_pretrained(root/'dense')
            Native=native_quantized_class(NATIVE_PATH)
            original,load=Native.from_pretrained(origin,config=config,torch_dtype=torch.float16,
                low_cpu_mem_usage=True,output_loading_info=True)
            self.assertFalse(any(load.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']))
            expanded,load=MirrorClassifier.from_pretrained(root/'dense',config=config,torch_dtype=torch.float16,
                low_cpu_mem_usage=True,output_loading_info=True)
            self.assertFalse(any(load.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']))
            apply_native_dense_layout(expanded,result['projection_names'])
            original.eval();expanded.eval()
            entries=[dict(input_ids=[2,0,5,1],native_position_offset=1996,pool_index=3),
                     dict(input_ids=[2,6,3,5,6,8,1],native_position_offset=1993,pool_index=6)]
            batch=tensors(entries,[0,1])
            with torch.inference_mode():
                reference=original(**batch)
                full=expanded(**batch)
                split=expanded.stage2(expanded.stage1(batch))
            self.assertTrue(torch.isfinite(reference).all())
            for value in [full,split]:
                self.assertLessEqual(float((reference.softmax(-1)-value.softmax(-1)).abs().max()),1e-5)
            self.assertEqual(result['projection_names'],sorted(projections))


if __name__=='__main__':unittest.main()
