"""Label-blinded, manually judged comparison; targets load only after freezing."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT, TEXT_COLUMNS, CLASSES, json_write

DIRECTORY = ROOT / 'runs/attempt_004_self_judge'
SEED = 20261116
JOURNAL = Path('/tmp/arena_self_judge_20261116_judgments.jsonl')
CONFIDENCE = {'low': .4, 'medium': .5, 'good': .6, 'high': .7, 'very_high': .8}


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    assert not (DIRECTORY / 'manifest.json').exists(), 'Sample already prepared'
    # These reads never access the target arrays/columns or model identities.
    with np.load(ROOT / 'runs/attempt_003/jobs/development/output/gemma26/judge_predictions.npz') as z:
        ids = z['ids'].copy()
    features = pd.read_parquet(ROOT / 'eda/attempt_002/features.parquet',
                               columns=['id', 'prompt_group']).set_index('id')
    poster = ROOT / 'eda/attempt_002/poster/site/index.html'
    excluded = set(map(int, re.findall(r'"id":\s*(\d+)', poster.read_text())))
    for path in [REPO/'results/eda_002/manual_case_notes.json',
                 REPO/'results/error_poster_002/review_notes.json']:
        excluded.update(int(r['id']) for r in json.loads(path.read_text()))
    rng = np.random.default_rng(SEED)
    eligible = [int(i) for i in rng.permutation(ids) if int(i) not in excluded]
    selected, groups = [], set()
    for rid in eligible:
        group = features.loc[rid, 'prompt_group']
        if group in groups: continue
        groups.add(group); selected.append(rid)
        if len(selected) == 300: break
    assert len(selected) == 300
    raw = pd.read_csv(ROOT / 'raw/train.csv', usecols=['id', *TEXT_COLUMNS]).set_index('id')
    cases, mapping = [], []
    for number, rid in enumerate(selected, 1):
        swap = bool(rng.integers(2))
        row = raw.loc[rid]
        case = {'case': number, 'question': json.loads(row['prompt']),
                'A': json.loads(row['response_b'] if swap else row['response_a']),
                'B': json.loads(row['response_a'] if swap else row['response_b'])}
        cases.append(case)
        mapping.append({'case': number, 'id': rid, 'swapped': swap})
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    (DIRECTORY/'inputs.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in cases))
    json_write(DIRECTORY/'mapping.json', mapping)
    manifest = {'attempt': '004_self_judge', 'rows': 300, 'seed': SEED,
        'eligible_rows_after_exclusions': len(eligible),
        'sample_source': 'Unique prompt groups in existing fixed 512-row Gemma development sample; exclude poster/manual-review IDs',
        'confidence_mapping': CONFIDENCE,
        'probability_rule': 'Chosen class gets confidence; each other class gets (1-confidence)/2',
        'judge': 'Current Codex assistant itself; no other models, subagents, web lookup or winner feedback',
        'input': 'Complete conversations, randomized answer order, hidden row IDs and model identities',
        'inputs_sha256': digest(DIRECTORY/'inputs.jsonl'),
        'mapping_sha256': digest(DIRECTORY/'mapping.json'),
        'targets_loaded': False,
        'limitations': ['Not a clean pretraining-contamination benchmark: the public dataset may have been in model training.',
                       'Full context and deliberation differ from quantized Gemma non-thinking 1024/2048-token inference.',
                       'Excluding previously inspected cases changes sample distribution; paired comparison only.',
                       'Judge identity is current assistant; exact underlying checkpoint is not established by this script.']}
    json_write(DIRECTORY/'manifest.json', manifest)
    json_write(REPO/'results/attempt_004_self_judge/protocol.json', manifest)
    print(json.dumps(manifest, indent=2))


def record(path):
    assert not (DIRECTORY/'frozen.json').exists(), 'Predictions already frozen'
    output = JOURNAL
    previous = DIRECTORY/'judgments.jsonl'
    if not output.exists() and previous.exists(): output.write_bytes(previous.read_bytes())
    current = read_lines(output) if output.exists() else []
    incoming = json.loads(path.read_text())
    ids = {r['case'] for r in current}
    for row in incoming:
        assert set(row) == {'case', 'class', 'confidence', 'reason'}
        assert 1 <= row['case'] <= 300 and row['case'] not in ids
        assert row['class'] in ['A', 'B', 'tie']
        assert row['confidence'] in CONFIDENCE and row['reason'].strip()
        ids.add(row['case'])
    with output.open('a') as stream:
        for row in incoming: stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    print(f'Saved {len(incoming)} judgments; total {len(ids)}/300. Targets remain sealed.')


def freeze():
    records = read_lines(JOURNAL)
    assert len(records) == 300 and {r['case'] for r in records} == set(range(1, 301))
    manifest = json.loads((DIRECTORY/'manifest.json').read_text())
    for name in ['inputs', 'mapping']:
        path = DIRECTORY / ('inputs.jsonl' if name == 'inputs' else 'mapping.json')
        assert digest(path) == manifest[name+'_sha256']
    (DIRECTORY/'judgments.jsonl').write_bytes(JOURNAL.read_bytes())
    frozen = {'rows': 300, 'judgments_sha256': digest(DIRECTORY/'judgments.jsonl'),
              'frozen_utc': datetime.now(timezone.utc).isoformat(),
              'labeling_source_sha256': digest(Path(__file__)),
              'inputs_sha256': manifest['inputs_sha256'], 'mapping_sha256': manifest['mapping_sha256'],
              'targets_loaded': False, 'confidence_mapping': CONFIDENCE}
    json_write(DIRECTORY/'frozen.json', frozen)
    json_write(REPO/'results/attempt_004_self_judge/frozen.json', frozen)
    print(json.dumps(frozen, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare', 'show', 'record', 'freeze'])
    p.add_argument('--start', type=int, default=1)
    p.add_argument('--end', type=int, default=10)
    p.add_argument('--file', type=Path)
    a = p.parse_args()
    if a.action == 'prepare': prepare()
    elif a.action == 'show':
        for row in read_lines(DIRECTORY/'inputs.jsonl'):
            if a.start <= row['case'] <= a.end:
                print(json.dumps(row, ensure_ascii=False))
    elif a.action == 'record': record(a.file)
    elif a.action == 'freeze': freeze()


if __name__ == '__main__': main()
