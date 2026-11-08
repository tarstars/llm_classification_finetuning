"""Deterministic native input tensors and bounded padded batches; no labels."""
import numpy as np
import torch


def schedule(entries, max_tokens=8192, max_rows=16):
    lengths = np.array([len(e['input_ids']) for e in entries])
    order = np.argsort(lengths,kind='stable')
    batches, batch = [], []
    for index in order:
        index = int(index)
        if lengths[index] > max_tokens:
            raise ValueError('One input exceeds padded-token budget')
        if batch and ((len(batch)+1)*lengths[index] > max_tokens or len(batch) >= max_rows):
            batches.append(batch)
            batch = []
        batch.append(index)
    if batch: batches.append(batch)
    return batches


def tensors(entries, indices):
    selected = [entries[i] for i in indices]
    width = max(len(e['input_ids']) for e in selected)
    ids = torch.zeros((len(selected),width),dtype=torch.long)
    mask, positions = torch.zeros_like(ids), torch.zeros_like(ids)
    pools = []
    for row, entry in enumerate(selected):
        size = len(entry['input_ids']);left = width-size
        ids[row,left:] = torch.tensor(entry['input_ids'],dtype=torch.long)
        mask[row,left:] = 1
        positions[row,left:] = torch.arange(size)+entry['native_position_offset']
        pools.append(left+entry['pool_index'])
    return dict(input_ids=ids,attention_mask=mask,position_ids=positions,
                pool_indices=torch.tensor(pools,dtype=torch.long))
