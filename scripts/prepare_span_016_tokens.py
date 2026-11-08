"""CPU reference tokens and retention, using the pinned native tokenizer only."""
from pathlib import Path
import json
import resource
import sys
import time

import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import transformers

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT, TEXT_COLUMNS
from arena.experiment import read_json, atomic_json, sha256, utc_now
from arena.span016_reward import conversation, encode_conversation
from arena.span016_infer import history_counts


def main():
    assert transformers.__version__ == '4.52.3'
    root, pub = ROOT / 'runs/span_016', REPO / 'results/span_016'
    assert not (pub / 'token_reference.json').exists(), 'Preserve the reference'
    begin = time.monotonic()
    selection = read_json(pub / 'profile_selection.json')
    train = pd.read_csv(ROOT / 'raw/train.csv', usecols=['id', *TEXT_COLUMNS]).set_index('id', drop=False)
    assert sha256(ROOT / 'raw/train.csv') == selection['train_csv_sha256']
    frames = dict(long=pd.read_parquet(root / 'cohort/source.parquet'),
        control=train.loc[selection['control_ids']], uniform=train.loc[selection['uniform_ids']],
        test=pd.read_csv(ROOT / 'raw/test.csv', usecols=['id', *TEXT_COLUMNS]))
    tokenizer = AutoTokenizer.from_pretrained(root / 'source_audit/8B', local_files_only=True, padding_side='right')
    folder = root / 'token_audit'
    folder.mkdir(exist_ok=True)
    records, hashes = {}, {}
    for name, frame in frames.items():
        lengths, originals, tokens, counts = [], [], [], {}
        for row in frame.to_dict('records'):
            for key, value in history_counts(row).items():
                counts[key] = counts.get(key, 0) + value
            for side in ['a', 'b']:
                ids, meta = encode_conversation(tokenizer, conversation(row, side))
                lengths.append(len(ids))
                originals.append(meta['original_tokens'])
                tokens.append(meta['token_ids_sha256'])
        lengths = np.asarray(lengths).reshape(-1, 2)
        originals = np.asarray(originals).reshape(-1, 2)
        path = folder / ('expected_' + name + '.npz')
        np.savez_compressed(path, ids=frame.id.to_numpy(), lengths=lengths,
            original_lengths=originals, token_hashes=np.asarray(tokens).reshape(-1, 2))
        hashes[path.name] = sha256(path)
        records[name] = dict(rows=len(frame), mean_pair_tokens=float(lengths.sum(1).mean()),
            pair_token_quantiles=np.quantile(lengths.sum(1), [.5, .9, .99, 1]).tolist(),
            mean_original_pair_tokens=float(originals.sum(1).mean()),
            truncated_sequences=int((originals > lengths).sum()), total_sequences=int(lengths.size),
            retained_token_fraction=float(lengths.sum() / originals.sum()), history_counts=counts,
            work_per_row=float((lengths * np.maximum(1., lengths / 2048)).sum() / len(frame)))
    ratio = records['long']['work_per_row'] / records['uniform']['work_per_row']
    result = dict(at=utc_now(), transformer_version=transformers.__version__,
        tokenizers_version=__import__('tokenizers').__version__, cohorts=records,
        source_to_target_reward_work_ratio=ratio, uses_targets=False,
        actual_reward_gpu_speed_measured=False, new_gpu_executions=0,
        elapsed_cpu_seconds=time.monotonic() - begin, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        artifact_sha256=hashes, source_sha256=sha256(Path(__file__)))
    atomic_json(pub / 'token_reference.json', result)
    print(json.dumps(dict(cohorts=records, source_to_target_reward_work_ratio=ratio,
        elapsed_cpu_seconds=result['elapsed_cpu_seconds'], new_gpu_executions=0), indent=2))


if __name__ == '__main__':
    main()
