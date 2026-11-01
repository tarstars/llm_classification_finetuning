from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(os.environ.get("ARENA_DATA_DIR", Path.home() / "database/llm_classification_finetuning"))
CLASSES = ["winner_model_a", "winner_model_b", "winner_tie"]
TEXT_COLUMNS = ["prompt", "response_a", "response_b"]
SEED = 20261101


def parse_turns(value):
    """Kaggle stores JSON arrays in CSV fields; preserve turn boundaries and nulls."""
    if not isinstance(value, str):
        return ""
    try:
        turns = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("Expected a JSON-encoded conversation list") from exc
    if not isinstance(turns, list):
        raise ValueError("Expected a list of conversation turns")
    return "\n\n[TURN]\n\n".join("" if turn is None else str(turn) for turn in turns)


def load_data():
    train = pd.read_csv(ROOT / "raw/train.csv")
    test = pd.read_csv(ROOT / "raw/test.csv")
    for frame in (train, test):
        for col in TEXT_COLUMNS:
            frame[col] = frame[col].map(parse_turns)
    assert train[CLASSES].isin([0, 1]).all().all()
    assert train[CLASSES].sum(axis=1).eq(1).all()
    assert train.id.is_unique and test.id.is_unique
    return train, test


def prompt_groups(frame):
    return frame.prompt.map(lambda s: hashlib.sha256(re.sub(r"\s+", " ", s).strip().casefold().encode()).hexdigest())


def split_data(frame):
    """8 train folds, 1 development fold, 1 untouched final holdout fold."""
    y = frame[CLASSES].values.argmax(1)
    groups = prompt_groups(frame)
    fold = np.full(len(frame), -1)
    cv = StratifiedGroupKFold(10, shuffle=True, random_state=SEED)
    for k, (_, idx) in enumerate(cv.split(frame, y, groups)):
        fold[idx] = k
    parts = {"train": np.flatnonzero(fold >= 2), "dev": np.flatnonzero(fold == 1), "holdout": np.flatnonzero(fold == 0)}
    for a, b in [("train", "dev"), ("train", "holdout"), ("dev", "holdout")]:
        assert not set(groups.iloc[parts[a]]) & set(groups.iloc[parts[b]])
    return y, parts, fold


def structural_features(frame):
    """Only features available in test. Never use model identities or IDs."""
    values = []
    for col in TEXT_COLUMNS:
        text = frame[col]
        n = text.str.len().to_numpy()
        words = text.str.count(r"\S+").to_numpy()
        values.append(np.column_stack([
            np.log1p(n), np.log1p(words), np.log1p(text.str.count("\n")),
            np.log1p(text.str.count("```")), np.log1p(text.str.count(r"[!?]")),
            np.log1p(text.str.count(r"(?m)^\s*(?:[-*]|\d+\.)\s")),
            text.str.contains(r"(?i)\b(?:sorry|cannot|can't|unable|as an ai)\b", regex=True).astype(float),
            text.str.count(r"[^\x00-\x7F]").to_numpy() / np.maximum(n, 1),
            (words == 0).astype(float),
        ]))
    p, a, b = values
    return np.column_stack([p, a, b, a-b, np.abs(a-b)]).astype(np.float32)


def swapped(frame):
    out = frame.copy()
    out[["response_a", "response_b"]] = frame[["response_b", "response_a"]].values
    return out


def dense_pair_features(embeddings, stats):
    p, a, b = embeddings
    return np.column_stack([p, a, b, a-b, np.abs(a-b), a*b, p*a, p*b, stats]).astype(np.float32)


def swapped_pair_features(embeddings, stats):
    p, a, b = embeddings
    d = stats.shape[1] // 5
    sp, sa, sb, delta, absolute = np.split(stats, [d, 2*d, 3*d, 4*d], axis=1)
    return dense_pair_features((p, b, a), np.column_stack([sp, sb, sa, -delta, absolute]))


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")
    temp.replace(path)
