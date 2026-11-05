"""Shared label-free input selection and exact text/token fingerprints."""
import hashlib,json
import numpy as np
import pandas as pd
from .data import TEXT_COLUMNS,parse_turns


def read_inputs(path,ids):
    frame=pd.read_csv(path,usecols=['id',*TEXT_COLUMNS]).set_index('id',drop=False).loc[ids].copy()
    for col in TEXT_COLUMNS:frame[col]=frame[col].map(parse_turns)
    assert set(frame.columns)=={'id',*TEXT_COLUMNS} and frame.id.is_unique
    return frame


def text_digest(frame):
    return hashlib.sha256(json.dumps([[int(row.id),*[getattr(row,c) for c in TEXT_COLUMNS]] for row in frame.itertuples()],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def token_digest(sequences):
    h=hashlib.sha256()
    for seq in sequences:h.update(np.asarray([len(seq),*seq],dtype='<i4').tobytes())
    return h.hexdigest()
