"""Freeze the cached-data H37 recipe before any extra judge predictions arrive."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT, json_write, prompt_groups
from arena.joint_data import read_competition
from arena.span_calibration import train_blend

root = ROOT / 'runs/span_002'
destination = root / 'frozen_cached_blend.json'
assert not (root/'jobs/ensemble-expansion/push_intent.json').exists(), 'Freeze before expanded inference launches'
config = json.loads((REPO/'configs/span_002.json').read_text())
inventory = json.loads((root/'artifact_inventory.json').read_text())
baseline, judge = (np.load(inventory[name]['path']) for name in ('baseline','judge'))
lookup = {int(rid):i for i,rid in enumerate(baseline['ids'])}
p = baseline['probabilities'][[lookup[int(rid)] for rid in judge['ids']]]
frame, _ = read_competition(ROOT/'raw')
groups = prompt_groups(frame.set_index('id').loc[judge['ids']]).to_numpy()
fits, weight = train_blend(p, judge['original'], judge['labels'], groups, config)
selection = {'classifier_family':'temperature','judge_family':'temperature_tie',
    'classifier_theta':fits[0].theta.tolist(),'judge_theta':fits[1].theta.tolist(),
    'judge_weight':weight,'fit_rows':len(p),'fit_groups':len(set(groups)),
    'cached_judge_sha256':inventory['judge']['sha256'],
    'protocol_sha256':hashlib.sha256((REPO/'docs/SPAN_002_PROTOCOL.md').read_bytes()).hexdigest(),
    'evaluation':'New extra rows excluding cached-fit prompt groups; recipe frozen before extra inference.'}
if destination.exists():
    assert json.loads(destination.read_text()) == selection
else:
    json_write(destination, selection)
json_write(REPO/'results/span_002/frozen_cached_blend.json', selection)
print(json.dumps(selection, indent=2))
