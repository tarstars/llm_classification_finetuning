"""H74 native H60 forward implementation with fixed10240token packing.

Model loading, precision, sharding and forward operations are unchanged.
Only packing, audit hooks and existing progress/position guards differ.
Reference: tascj/kaggle-lmsys-chatbot-arena, Apache-2.0,
commit83cd93d50b9283c18711e8c63e4e1c6399c7b9ce.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd


if __package__:
    from .span027_checks import PACKING_COHORTS, conversation_budget, after_cohort
else:
    from packing_checks import PACKING_COHORTS, conversation_budget, after_cohort

CLASSES = ['winner_model_a', 'winner_model_b', 'winner_tie']
TEXT = ['prompt', 'response_a', 'response_b']


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def event(stage, **values):
    print('ARENA_EVENT ' + json.dumps({'stage': stage, **values}, allow_nan=False), flush=True)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def profile_indices(frame, uniform_rows=256, long_rows=64):
    """Label-free fixed random cohort plus a disjoint long-text stress cohort."""
    rng = np.random.default_rng(42)
    uniform = rng.choice(len(frame), min(uniform_rows, len(frame)), replace=False)
    length = frame[TEXT].fillna('').apply(lambda s: s.str.len()).sum(axis=1).to_numpy()
    available = np.ones(len(frame), dtype=bool)
    available[uniform] = False
    long = np.argsort(-length, kind='stable')
    long = long[available[long]][:long_rows]
    return uniform, long


def validate_probabilities(probs, rows):
    if probs.shape != (rows, 3) or not np.isfinite(probs).all():
        raise ValueError('Invalid probability shape or nonfinite values')
    if (probs < 0).any() or (probs > 1).any() or not np.allclose(probs.sum(1), 1, atol=1e-5):
        raise ValueError('Invalid probability simplex')


def ensemble(original, swapped_llama):
    validate_probabilities(original, len(original))
    validate_probabilities(swapped_llama, len(original))
    result = (original + swapped_llama[:, [1, 0, 2]]) / 2
    validate_probabilities(result, len(original))
    return result


def route_indices(ids, gemma, aligned_llama, lengths, work_fraction=.1, row_fraction=.2):
    """Deterministic text-dependent route; labels never enter this function."""
    ids=np.asarray(ids);lengths=np.asarray(lengths,dtype=float)
    validate_probabilities(gemma,len(ids));validate_probabilities(aligned_llama,len(ids))
    if lengths.shape!=(len(ids),2) or not np.isfinite(lengths).all() or (lengths<=0).any():
        raise ValueError('Invalid model token lengths')
    if len(np.unique(ids))!=len(ids):raise ValueError('Duplicate IDs')
    if not (0<=work_fraction<=1 and 0<=row_fraction<=1):raise ValueError('Invalid route budget')
    p=np.clip(gemma.astype(float),1e-15,1);q=np.clip(aligned_llama.astype(float),1e-15,1)
    middle=(p+q)/2
    disagreement=(np.sum(p*np.log(p/middle),1)+np.sum(q*np.log(q/middle),1))/2
    disagreement=np.maximum(disagreement,0)
    work=np.sum(lengths*np.maximum(1,lengths/2048),axis=1)
    order=np.lexsort((ids,-disagreement/work))
    selected=[];used=0.;budget=float(work.sum()*work_fraction)
    for index in order:
        if len(selected)>=int(len(ids)*row_fraction):break
        if disagreement[index]>0 and used+work[index]<=budget:
            selected.append(int(index));used+=work[index]
    return np.sort(np.asarray(selected,dtype=int)),disagreement,work


def complete_orders(base_gemma, base_llama_swapped, indices, reverse_gemma, reverse_llama):
    """Preserve base off-route; average all four aligned predictions on-route."""
    result=ensemble(base_gemma,base_llama_swapped).astype(float)
    indices=np.asarray(indices)
    if indices.ndim!=1 or not np.issubdtype(indices.dtype,np.integer):raise ValueError('Invalid indices')
    if len(set(indices.tolist()))!=len(indices) or (indices<0).any() or (indices>=len(result)).any():
        raise ValueError('Duplicate or out-of-bounds route')
    validate_probabilities(reverse_gemma,len(indices));validate_probabilities(reverse_llama,len(indices))
    result[indices]=(base_gemma[indices].astype(float)+base_llama_swapped[indices][:,[1,0,2]]+
                     reverse_gemma[:,[1,0,2]]+reverse_llama)/4
    validate_probabilities(result,len(result))
    return result


def infer_role(args):
    import torch
    import transformers
    import triton
    import xformers
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask
    from human_pref.data.processors import ProcessorPAB
    from human_pref.data.dataset import LMSYSDataset
    from human_pref.data.collators import VarlenCollator, ShardedMaxTokensCollator
    from human_pref.utils import to_device

    started = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    assert torch.cuda.device_count() == 2
    role = args.role
    is_gemma = role in ['m0','m0_reverse']
    swapped = role in ['m3','m0_reverse']
    if is_gemma:
        from human_pref.models.modeling_gemma2 import Gemma2ForSequenceClassification as Model
    else:
        from human_pref.models.modeling_llama import LlamaForSequenceClassification as Model
    model_path = Path(args.assets) / ('lmsys-checkpoints-0-0805' if is_gemma else 'lmsys-checkpoints-3-0805')
    small = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'model.safetensors.index.json']
    source = {name: digest(model_path / name) for name in small}
    index = json.loads((model_path / 'model.safetensors.index.json').read_text())
    shards = sorted(set(index['weight_map'].values()))
    source['weight_shards'] = {name: {'bytes': (model_path / name).stat().st_size,
                                     'sha256': digest(model_path / name)} for name in shards}
    save(Path(args.out) / (role + '_checkpoint.json'), source)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    processor = ProcessorPAB(tokenizer, max_length=4096, support_system_role=not is_gemma)
    layers = 42 if is_gemma else 32
    devices = {'model.embed_tokens': 'cuda:0', 'model.norm': 'cuda:1', 'score': 'cuda:1'}
    devices.update({f'model.layers.{i}': f'cuda:{int(i >= layers // 2)}' for i in range(layers)})
    event('model_loading', role=role)
    model, loading = Model.from_pretrained(model_path, torch_dtype=torch.float16,
        device_map=devices, local_files_only=True, output_loading_info=True)
    assert not any(loading.get(key) for key in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']), loading
    model.eval()
    assert {p.device.index for p in model.parameters()} == {0, 1}
    config = model.config
    dim = config.head_dim if is_gemma else config.hidden_size // config.num_attention_heads
    inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    freq0, freq1 = inv_freq.to('cuda:0'), inv_freq.to('cuda:1')
    cold = time.monotonic() - started
    summary = {'role': role, 'cold_seconds': cold, 'cohorts': {}, 'packing_tokens': 10240,
        'device_total_memory_bytes': [torch.cuda.get_device_properties(g).total_memory for g in range(2)],
        'max_position_embeddings': config.max_position_embeddings, 'versions': {
        'python': sys.version.split()[0], 'torch': torch.__version__, 'transformers': transformers.__version__,
        'triton': triton.__version__, 'xformers': xformers.__version__, 'numpy': np.__version__}, 'loading_exact': True}
    names = args.cohorts.split(',')
    assert all(name in PACKING_COHORTS for name in names)
    for name in names:
        processor.max_length = conversation_budget(role, name)
        frame = pd.read_parquet(Path(args.out) / (name + '.parquet'))
        if not len(frame):
            np.savez_compressed(Path(args.out)/(role+'_'+name+'.npz'),ids=frame.id.to_numpy(),
                                probabilities=np.empty((0,3)),lengths=np.empty(0,dtype=int))
            summary['cohorts'][name]=dict(rows=0,seconds=0,tokens=0,max_tokens=0)
            save(Path(args.out)/(role+'_summary.json'), summary)
            continue
        if swapped:
            frame[['response_a', 'response_b']] = frame[['response_b', 'response_a']].values
        # These placeholders serve the unchanged author dataset interface only.
        # Never call dataset.evaluate, and never pass original votes to the model.
        for i, col in enumerate(CLASSES):
            frame[col] = int(i == 0)
        path = Path(args.out) / (role + '_input.parquet')
        frame.to_parquet(path, index=False)
        dataset = LMSYSDataset(path, processor=processor, include_swap=False, is_parquet=True)
        # Re-forking workers after CUDA/tokenizer execution crashed in the
        # long-cohort control. Keep tokenization in this process for reliability.
        loader = DataLoader(dataset, batch_size=80, num_workers=0,
            collate_fn=ShardedMaxTokensCollator(max_tokens=10240, base_collator=VarlenCollator()))
        outs, lengths, token_hash = [], [], hashlib.sha256()
        packed_tokens = []
        previous = None
        for gpu in range(2):
            torch.cuda.reset_peak_memory_stats(gpu)
            torch.cuda.synchronize(gpu)
        begin = time.monotonic()
        last_progress = begin
        # Same two-stage pipeline and block-diagonal causal attention as author.
        for batch_number, batch in enumerate(loader):
            for micro in batch:
                assert micro['max_seq_len'] <= config.max_position_embeddings
                assert micro['input_ids'].numel() <= 10240
                packed_tokens.append(micro['input_ids'].numel())
                lengths.extend(micro['seq_lens'])
                token_hash.update(micro['input_ids'].numpy().tobytes())
                input_ids = to_device(micro['input_ids'], 'cuda:0')
                seq_info = to_device(dict(cu_seqlens=micro['cu_seqlens'],
                    position_ids=micro['position_ids'], max_seq_len=micro['max_seq_len'],
                    attn_bias=BlockDiagonalCausalMask.from_seqlens(micro['seq_lens'])), 'cuda:0')
                with torch.no_grad(), torch.cuda.amp.autocast():
                    logits = None
                    if previous is not None:
                        hidden, prev_seq = previous
                        logits = model.forward_part2(hidden, prev_seq, freq1)
                    hidden = model.forward_part1(input_ids, seq_info, freq0)
                    seq1, hidden1 = to_device([seq_info, hidden], 'cuda:1')
                    previous = hidden1, seq1
                    # Match the author: launch both GPU stages before the blocking
                    # host copy, so the two devices can overlap their work.
                    if logits is not None:
                        outs.append(logits.cpu())
                if time.monotonic()-last_progress >= 30:
                    event('inference_progress', role=role, cohort=name, batch=batch_number,
                          rows=len(lengths), seconds=time.monotonic()-begin)
                    last_progress = time.monotonic()
            event('inference_progress', role=role, cohort=name, batch=batch_number,
                  rows=len(lengths), seconds=time.monotonic()-begin)
        if previous is not None:
            with torch.no_grad(), torch.cuda.amp.autocast():
                outs.append(model.forward_part2(previous[0], previous[1], freq1).cpu())
        probs = torch.cat(outs, dim=0).softmax(-1).numpy()
        for gpu in range(2):
            torch.cuda.synchronize(gpu)
        seconds = time.monotonic() - begin
        validate_probabilities(probs, len(frame))
        np.savez_compressed(Path(args.out) / (role + '_' + name + '.npz'),
            ids=frame.id.to_numpy(), probabilities=probs, lengths=np.array(lengths))
        summary['cohorts'][name] = {'rows': len(frame), 'seconds': seconds,
            'tokens': int(sum(lengths)), 'max_tokens': max(lengths),
            'token_ids_sha256': token_hash.hexdigest(),
            'micro_batches': len(packed_tokens), 'peak_packed_tokens': max(packed_tokens),
            'peak_allocated_bytes': [torch.cuda.max_memory_allocated(g) for g in range(2)],
            'peak_reserved_bytes': [torch.cuda.max_memory_reserved(g) for g in range(2)]}
        save(Path(args.out) / (role + '_summary.json'), summary)
        path.unlink()
        event('cohort_complete', role=role, cohort=name, rows=len(frame), seconds=seconds)
        if not after_cohort(args, summary, name):
            return


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['raw','assets','out','spec']:parser.add_argument('--'+name,required=True)
    parser.add_argument('--role',choices=['m0','m3','m0_reverse'],required=True)
    parser.add_argument('--cohorts',required=True)
    infer_role(parser.parse_args())
