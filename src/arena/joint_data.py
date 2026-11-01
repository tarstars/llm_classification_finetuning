"""Token-budgeted joint question/answer inputs and isolated attempt-002 splits."""
from __future__ import annotations
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset

from .data import CLASSES, TEXT_COLUMNS, parse_turns, prompt_groups

JOINT_SEED = 20261102
SWAP_LABELS = np.array([1, 0, 2])


def phase_two_split(frame):
    labels = frame[CLASSES].values.argmax(1)
    groups = prompt_groups(frame)
    folds = np.full(len(frame), -1, dtype=np.int8)
    splitter = StratifiedGroupKFold(10, shuffle=True, random_state=JOINT_SEED)
    for k, (_, idx) in enumerate(splitter.split(frame, labels, groups)):
        folds[idx] = k
    parts = {"train": np.flatnonzero(folds >= 2), "dev": np.flatnonzero(folds == 0),
             "confirmation": np.flatnonzero(folds == 1)}
    for a, b in [("train", "dev"), ("train", "confirmation"), ("dev", "confirmation")]:
        assert not set(groups.iloc[parts[a]]) & set(groups.iloc[parts[b]])
    return parts, folds


def split_from_ids(frame, manifest):
    """Use explicit IDs to keep splits identical across sklearn/runtime versions."""
    ids = frame.id.to_numpy()
    parts = {name: np.flatnonzero(np.isin(ids, manifest[name])) for name in ["dev", "confirmation"]}
    assert len(parts["dev"]) == len(manifest["dev"])
    assert len(parts["confirmation"]) == len(manifest["confirmation"])
    assert not set(manifest["dev"]) & set(manifest["confirmation"])
    excluded = np.concatenate(list(parts.values()))
    parts["train"] = np.flatnonzero(~np.isin(np.arange(len(frame)), excluded))
    groups = prompt_groups(frame)
    for a, b in [("train", "dev"), ("train", "confirmation"), ("dev", "confirmation")]:
        assert not set(groups.iloc[parts[a]]) & set(groups.iloc[parts[b]])
    return parts


def read_competition(raw):
    raw = Path(raw)
    train, test = pd.read_csv(raw / "train.csv"), pd.read_csv(raw / "test.csv")
    for frame in (train, test):
        for col in TEXT_COLUMNS:
            frame[col] = frame[col].map(parse_turns)
    return train, test


def head_tail(tokens, budget, marker):
    if len(tokens) <= budget:
        return tokens
    if budget <= len(marker) + 2:
        return tokens[:budget]
    remaining = budget - len(marker)
    first = (remaining + 1) // 2
    return tokens[:first] + marker + tokens[-(remaining-first):]


class JointTokenizer:
    """Preserve each field explicitly; truncate fields rather than dropping answer B."""
    def __init__(self, tokenizer, max_length=1024):
        self.tokenizer, self.max_length = tokenizer, max_length
        encode = lambda s: tokenizer.encode(s, add_special_tokens=False)
        self.prefix = encode("Compare both answers to the user request. Predict the human preference: answer A, answer B, or a tie.\n\n[USER REQUEST]\n")
        self.a_marker = encode("\n\n[ANSWER A]\n")
        self.b_marker = encode("\n\n[ANSWER B]\n")
        self.suffix = encode("\n\n[DECISION]\n")
        self.truncated = encode("\n[TRUNCATED]\n")
        self.bos = [] if tokenizer.bos_token_id is None else [tokenizer.bos_token_id]
        self.eos = [] if tokenizer.eos_token_id is None else [tokenizer.eos_token_id]
        self.overhead = sum(map(len, [self.prefix, self.a_marker, self.b_marker, self.suffix, self.bos, self.eos]))
        if max_length - self.overhead < 64:
            raise ValueError("Context budget is too small for all three fields")

    def combine(self, prompt, a, b):
        available = self.max_length - self.overhead
        prompt_budget = min(len(prompt), min(256, max(32, available // 4)))
        answer_budget = available - prompt_budget
        a_budget, b_budget = min(len(a), answer_budget // 2), min(len(b), answer_budget // 2)
        unused = answer_budget - a_budget - b_budget
        take = min(unused, len(a)-a_budget)
        a_budget += take; unused -= take
        b_budget += min(unused, len(b)-b_budget)
        sequence = (self.bos + self.prefix + head_tail(prompt, prompt_budget, self.truncated)
            + self.a_marker + head_tail(a, a_budget, self.truncated)
            + self.b_marker + head_tail(b, b_budget, self.truncated) + self.suffix + self.eos)
        assert len(sequence) <= self.max_length
        return sequence

    def encode_frame(self, frame):
        original, swapped = [], []
        totals = {name: 0 for name in TEXT_COLUMNS}
        maxima = {name: 0 for name in TEXT_COLUMNS}
        # Avoid retaining the entire untruncated corpus on each distributed rank.
        for start in range(0,len(frame),256):
            fields = []
            for name in TEXT_COLUMNS:
                tokens = self.tokenizer(frame[name].iloc[start:start+256].tolist(),
                    add_special_tokens=False,truncation=False,padding=False)['input_ids']
                fields.append(tokens)
                lengths = list(map(len,tokens))
                totals[name] += sum(lengths)
                maxima[name] = max(maxima[name],max(lengths,default=0))
            for p,a,b in zip(*fields):
                original.append(self.combine(p,a,b))
                swapped.append(self.combine(p,b,a))
        diagnostics = {name:{'mean_tokens':totals[name]/max(len(frame),1),
                            'max_tokens':maxima[name]} for name in TEXT_COLUMNS}
        return original, swapped, diagnostics


class PreferenceDataset(Dataset):
    def __init__(self, original, swapped, labels=None, augment=False, force_swap=False):
        self.original, self.swapped = original, swapped
        self.labels, self.augment, self.force_swap = labels, augment, force_swap

    def __len__(self):
        return len(self.original)

    def __getitem__(self, index):
        flip = self.force_swap or (self.augment and random.random() < .5)
        ids = (self.swapped if flip else self.original)[index]
        item = {"input_ids": ids, "attention_mask": [1]*len(ids)}
        if self.labels is not None:
            y = int(self.labels[index])
            item["labels"] = int(SWAP_LABELS[y]) if flip else y
        return item


class PreferenceCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, rows):
        labels = [row["labels"] for row in rows] if "labels" in rows[0] else None
        inputs = [{k:v for k,v in row.items() if k != "labels"} for row in rows]
        batch = self.tokenizer.pad(inputs, padding=True, pad_to_multiple_of=8, return_tensors="pt")
        if labels is not None:
            batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch
