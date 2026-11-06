"""Frozen tascj public ensemble; author kernels/processor remain external assets.

Reference: tascj/kaggle-lmsys-chatbot-arena, Apache-2.0, commit
83cd93d50b9283c18711e8c63e4e1c6399c7b9ce; notebook tascj0/lmsys-0805.
This wrapper adds cohort selection, telemetry, strict loading and CSV checks.
It does not fit weights/calibration or use any target in inference.
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
    is_gemma = role == 'm0'
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
    summary = {'role': role, 'cold_seconds': cold, 'cohorts': {}, 'versions': {
        'python': sys.version.split()[0], 'torch': torch.__version__, 'transformers': transformers.__version__,
        'triton': triton.__version__, 'xformers': xformers.__version__}, 'loading_exact': True}
    for name in (['uniform', 'long', 'test'] if args.qualify else ['test']):
        frame = pd.read_parquet(Path(args.out) / (name + '.parquet'))
        if not is_gemma:
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
            collate_fn=ShardedMaxTokensCollator(max_tokens=8192, base_collator=VarlenCollator()))
        outs, lengths, token_hash = [], [], hashlib.sha256()
        previous = None
        for gpu in range(2):
            torch.cuda.reset_peak_memory_stats(gpu)
            torch.cuda.synchronize(gpu)
        begin = time.monotonic()
        # Same two-stage pipeline and block-diagonal causal attention as author.
        for batch_number, batch in enumerate(loader):
            for micro in batch:
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
            'peak_allocated_bytes': [torch.cuda.max_memory_allocated(g) for g in range(2)],
            'peak_reserved_bytes': [torch.cuda.max_memory_reserved(g) for g in range(2)]}
        save(Path(args.out) / (role + '_summary.json'), summary)
        path.unlink()
        event('cohort_complete', role=role, cohort=name, rows=len(frame), seconds=seconds)


def orchestrate(args):
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    raw = Path(args.raw)
    test = pd.read_csv(raw / 'test.csv', usecols=['id', *TEXT])
    assert len(test) and test.id.is_unique
    frames = {'test': test}
    if args.qualify:
        train = pd.read_csv(raw / 'train.csv', usecols=['id', *TEXT])
        uniform, long = profile_indices(train)
        frames.update(uniform=train.iloc[uniform], long=train.iloc[long])
        save(out / 'profile_selection.json', {'seed': 42, 'uses_targets': False,
            'uniform_ids': train.iloc[uniform].id.tolist(), 'long_ids': train.iloc[long].id.tolist(),
            'train_csv_sha256': digest(raw / 'train.csv'), 'local_quality_is_independent': False})
    for name, frame in frames.items():
        frame.to_parquet(out / (name + '.parquet'), index=False)
    for role in ['m0', 'm3']:
        command = [sys.executable, str(Path(__file__).resolve()), '--role', role,
            '--raw', args.raw, '--assets', args.assets, '--out', args.out]
        if args.qualify:
            command.append('--qualify')
        subprocess.run(command, check=True)
    for name, frame in frames.items():
        gemma = np.load(out / ('m0_' + name + '.npz'))
        llama = np.load(out / ('m3_' + name + '.npz'))
        assert np.array_equal(gemma['ids'], frame.id) and np.array_equal(llama['ids'], frame.id)
        probs = ensemble(gemma['probabilities'], llama['probabilities'])
        np.savez_compressed(out / ('ensemble_' + name + '.npz'), ids=frame.id.to_numpy(), probabilities=probs)
        if name == 'test':
            submission = pd.DataFrame(probs, columns=CLASSES)
            submission.insert(0, 'id', frame.id.to_numpy())
            submission.to_csv(Path(args.out).parent / 'submission.csv', index=False)
            reloaded = pd.read_csv(Path(args.out).parent / 'submission.csv')
            assert np.array_equal(reloaded.id, frame.id)
            validate_probabilities(reloaded[CLASSES].to_numpy(), len(frame))
        (out / (name + '.parquet')).unlink()
    record = {'complete': True, 'test_rows': len(test), 'class_order': CLASSES,
              'ensemble_weights': [.5, .5], 'llama_swapped': True, 'independent_local_quality': False}
    if args.qualify:
        summaries = [json.loads((out / (r + '_summary.json')).read_text()) for r in ['m0', 'm3']]
        per_row = sum(s['cohorts']['uniform']['seconds'] / s['cohorts']['uniform']['rows'] for s in summaries)
        cold = sum(s['cold_seconds'] for s in summaries)
        record.update(uniform_seconds_per_row=per_row, cold_seconds=cold, hidden_rows_assumed=25000,
                      projected_hidden_seconds=25000 * per_row * 1.2 + cold + 600)
        record['runtime_gate_passed'] = record['projected_hidden_seconds'] < 32400
    save(out / 'qualification.json', record)
    event('complete', **record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', required=True)
    parser.add_argument('--assets', default='/kaggle/input')
    parser.add_argument('--out', required=True)
    parser.add_argument('--qualify', action='store_true')
    parser.add_argument('--role', choices=['m0', 'm3'])
    args = parser.parse_args()
    (infer_role if args.role else orchestrate)(args)
