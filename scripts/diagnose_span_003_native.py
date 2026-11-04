"""Describe the existing 64-row native drift, without refitting or requesting inference."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT, json_write, prompt_groups
from arena.joint_data import read_competition
from arena.span_cascade import RecipeSet
from arena.span_heads import metrics, paired_group_gain

root = ROOT / 'runs/span_003'
directory = root / 'jobs/native_profile_retry/output'
old = np.load(directory / 'native_reference.npz')
current = np.load(directory / 'native/judge_predictions.npz')
np.testing.assert_array_equal(old['ids'], current['ids'])
frozen = json.loads((root / 'frozen_cached_recipes.json').read_text())
calibrator = RecipeSet.from_dict(frozen['parameters'], frozen['config']).fits[1]
previous = calibrator.predict_proba(old['original'])
observed = calibrator.predict_proba(current['original'])
frame, _ = read_competition(ROOT / 'raw')
groups = prompt_groups(frame.set_index('id').loc[current['ids']]).to_numpy()
labels = current['labels']
summary = json.loads((directory / 'native/judge_summary.json').read_text())
diagnostic = {
    'rows': len(labels), 'groups': len(set(groups)), 'new_parameters_fitted': False,
    'calibrator_source': 'Frozen 2048-row v3 fitting recipe; includes these64 known rows, not independent validation.',
    'old_raw': metrics(labels, old['original']), 'current_raw': metrics(labels, current['original']),
    'old_fixed_calibration': metrics(labels, previous), 'current_fixed_calibration': metrics(labels, observed),
    'raw_loss_effect': paired_group_gain(labels, old['original'], current['original'], groups, frozen['config']),
    'fixed_calibration_loss_effect': paired_group_gain(labels, previous, observed, groups, frozen['config']),
    'argmax_disagreement': float((old['original'].argmax(1) != current['original'].argmax(1)).mean()),
    'packing': {**{key: summary[key] for key in ['packed_tokens_mean', 'prompt_overhead', 'truncated_rows', 'both_orders']},
                'clipped_fraction': summary['truncated_rows'] / len(labels)},
    'note': 'Positive effect means old minus current loss on the same64 known rows. Diagnostic only; do not choose backend settings from labels or waive the failed numerical guard.',
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
report = json.loads((REPO / 'results/span_003/native_profile_results.json').read_text())
if 'loss_diagnostic' in report:
    assert report['loss_diagnostic'] == diagnostic
report['loss_diagnostic'] = diagnostic
for folder in [root, REPO / 'results/span_003']:
    json_write(folder / 'native_profile_results.json', report)
print(json.dumps({key: diagnostic[key] for key in ['argmax_disagreement', 'packing', 'raw_loss_effect', 'fixed_calibration_loss_effect']}, indent=2))
