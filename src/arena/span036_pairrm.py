"""Literal authored joint comparator, with explicit target-free input proof."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def native_definitions(folder):
    folder = Path(folder)
    scope = dict(torch=torch, nn=nn, np=np, F=F, List=list)
    selected = [('native_ranker.py', 'CrossCompareReranker'),
        ('native_utils.py', 'get_pair_from_conv')]
    for filename, name in selected:
        tree = ast.parse((folder / filename).read_text())
        node = next(n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), filename, 'exec'), scope)
    # This native module imports only torch; no hub or training-side imports.
    exec(compile((folder / 'native_collator.py').read_text(), 'native_collator.py', 'exec'), scope)
    return scope['CrossCompareReranker'], scope['CrossCompareCollator'], scope['get_pair_from_conv']


def records(frame, get_pair):
    a, b = [], []
    for row in frame.itertuples(index=False):
        columns = []
        for value in [row.prompt, row.response_a, row.response_b]:
            turns = json.loads(value) if isinstance(value, str) else list(value)
            assert isinstance(turns, list) and turns
            assert all(x is None or isinstance(x, str) for x in turns)
            columns.append(['' if x is None else x for x in turns])
        assert len(set(map(len, columns))) == 1, 'Unaligned turns: no row dropping'
        conv_a, conv_b = [], []
        for p, x, y in zip(*columns):
            conv_a.extend([dict(role='USER', content=p), dict(role='ASSISTANT', content=x)])
            conv_b.extend([dict(role='USER', content=p), dict(role='ASSISTANT', content=y)])
        a.append(conv_a); b.append(conv_b)
    sources, left, right = get_pair(a, b)
    return [dict(source=p, candidates=[x, y]) for p, x, y in zip(sources, left, right)]


def batches(items, collator, tokenizer, reverse=False, batch_size=2):
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        if reverse:
            batch = [dict(source=x['source'], candidates=x['candidates'][::-1]) for x in batch]
        encoded = collator(batch)
        source, source_mask = encoded['source_ids'], encoded['source_attention_mask']
        candidate, masks = encoded['candidate_ids'], encoded['candidate_attention_mask']
        combined, lengths = [], []
        for i in range(len(batch)):
            pieces = [source[i][source_mask[i].bool()]]
            for j, prefix in enumerate([tokenizer.cand1_prefix_id, tokenizer.cand2_prefix_id]):
                piece = candidate[i, j][masks[i, j].bool()].clone()
                piece[piece == tokenizer.cand_prefix_id] = prefix
                pieces.append(piece)
            ids = torch.cat(pieces)
            assert all(int((ids == t).sum()) == 1 for t in [tokenizer.source_prefix_id, tokenizer.cand1_prefix_id, tokenizer.cand2_prefix_id])
            assert len(ids) <= 2054
            combined.append(ids.numpy().astype('<i8').tobytes()); lengths.append(len(ids))
        yield start, encoded, combined, lengths


def token_proof(items, collator, tokenizer, reverse=False):
    stream = hashlib.sha256(); lengths = []
    for _, _, encoded, sizes in batches(items, collator, tokenizer, reverse):
        for value in encoded:
            stream.update(len(value).to_bytes(8, 'little')); stream.update(value)
        lengths.extend(sizes)
    return dict(token_ids_sha256=stream.hexdigest(), lengths=lengths)


def construct(folder, tokenizer, ranker_cls):
    from transformers import DebertaV2Config, DebertaV2Model
    folder = Path(folder)
    config = DebertaV2Config.from_dict(json.loads((folder / 'base_config.json').read_text()))
    config.vocab_size = len(tokenizer)
    config.out_hidden_state_size = config.hidden_size
    model = DebertaV2Model(config)
    args = SimpleNamespace(**json.loads((folder / 'ranker_config.json').read_text()))
    return ranker_cls(model, args, tokenizer)
