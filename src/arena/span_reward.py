"""Frozen, pinned reward-model inference for H01; no target labels enter model inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .data import CLASSES, TEXT_COLUMNS, json_write
from .joint_data import read_competition
from .span_pack import SpanTokenizer


def run(raw, model_path, samples, config, out, deadline):
    rank, world = int(os.environ.get('LOCAL_RANK', '0')), int(os.environ.get('WORLD_SIZE', '1'))
    assert torch.cuda.is_available()
    torch.cuda.set_device(rank)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    frame, _ = read_competition(raw)
    ids = samples['feature_fit'] + samples['dev']
    assert len(ids) == len(set(ids))
    selected = frame.loc[frame.id.isin(ids)].copy()
    assert len(selected) == len(ids)
    selected = selected.iloc[rank::world]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side='right')
    assert tokenizer.chat_template, 'Use the checkpoint-provided conversation template'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoder = SpanTokenizer(tokenizer, config['max_length'])
    rows, sequences, lengths = [], [], []
    for row in selected.itertuples():
        fields = [tokenizer.encode(getattr(row, name), add_special_tokens=False) for name in TEXT_COLUMNS]
        pair = encoder.reward_pair(*fields)
        rows.append(int(row.id))
        sequences.extend(pair)
        lengths.append([len(x) for x in pair])
    started = time.monotonic()
    model = AutoModelForSequenceClassification.from_pretrained(model_path,
        dtype=torch.float16, device_map={'': rank}, num_labels=1,
        attn_implementation='sdpa', local_files_only=True).eval()
    model.config.pad_token_id = tokenizer.pad_token_id
    scores = np.full(len(sequences), np.nan, dtype=np.float32)
    order = np.argsort([len(x) for x in sequences], kind='stable')
    batch_size = config['eval_batch_size']
    progress = out / f'reward_rank{rank}.npz'
    completed = 0

    def save():
        np.savez_compressed(progress, ids=np.asarray(rows), rewards=scores.reshape(-1, 2),
                            lengths=np.asarray(lengths),
                            feature_fit=np.isin(rows, samples['feature_fit']))
        summary = {'rank': rank, 'rows': len(rows), 'completed_sequences': int(np.isfinite(scores).sum()),
            'total_sequences': len(scores), 'complete': bool(np.isfinite(scores).all()),
            'seconds_since_model_load': time.monotonic()-started,
            'peak_gpu_bytes': torch.cuda.max_memory_allocated(rank),
            'dtype': 'float16', 'gpu': torch.cuda.get_device_name(rank),
            'max_packed_length': max(map(len, sequences), default=0)}
        json_write(out / f'reward_rank{rank}.json', summary)
        print('SPAN_REWARD_PROGRESS', json.dumps(summary), flush=True)

    with torch.inference_mode():
        for begin in range(0, len(order), batch_size):
            if time.time() > deadline - 60:
                save()
                raise TimeoutError('Reward pilot reached its session budget; partial scores preserved')
            chosen = order[begin:begin+batch_size]
            batch = tokenizer.pad([{'input_ids': sequences[i], 'attention_mask': [1]*len(sequences[i])}
                                   for i in chosen], padding=True, pad_to_multiple_of=8, return_tensors='pt')
            batch = {key: value.to(rank) for key, value in batch.items()}
            values = model(**batch).logits.float().squeeze(-1).cpu().numpy()
            assert values.shape == (len(chosen),) and np.isfinite(values).all()
            scores[chosen] = values
            completed += len(chosen)
            if completed % 256 < batch_size:
                save()
    save()
    assert np.isfinite(scores).all(), 'Never fill missing predictions with priors'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('raw','model','samples','config','out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    run(args.raw, args.model, json.loads(args.samples.read_text()),
        json.loads(args.config.read_text()), args.out, args.deadline)


if __name__ == '__main__':
    main()
