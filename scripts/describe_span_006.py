"""Post-evaluation descriptive source/input shift; no fitting or candidate selection."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT, CLASSES, json_write


def read(path):
    return json.loads(Path(path).read_text())


def distribution(values):
    values = np.asarray(values, float)
    return {'mean': float(values.mean()), 'p50': float(np.median(values)),
            'p90': float(np.quantile(values, .9)), 'p99': float(np.quantile(values, .99))}


def main():
    out = ROOT / 'runs/span_006'
    pub = REPO / 'results/span_006'
    result = read(pub / 'H46_results.json')
    manifest = read(pub / 'artifact_inventory.json')
    assert (out / 'reveal_intent.json').exists(), 'Descriptive target access only after frozen evaluation'
    train = pd.read_parquet(manifest['input_artifacts']['train.parquet']['path'])
    dev = pd.read_parquet(manifest['input_artifacts']['dev.parquet']['path'])
    confirm = pd.read_parquet(manifest['input_artifacts']['confirmation_inputs']['path'])
    raw = pd.read_csv(manifest['input_artifacts']['raw_train']['path'], usecols=['id', *CLASSES, 'model_a', 'model_b']).set_index('id')
    classes = read(manifest['input_artifacts']['source_vocabulary.json']['path'])['source_classes']
    index = {name: i for i, name in enumerate(classes)}
    prior_names = np.r_[raw.loc[train.id].model_a, raw.loc[train.id].model_b]
    prior = pd.Series(prior_names).value_counts(normalize=True)
    probabilities = np.load(out / 'freeze/source_predictions.npz')
    ratings = np.load(out / 'freeze/ratings.npz')['ratings']
    cohorts = {}
    for name, frame in [('train', train), ('dev', dev), ('confirmation', confirm)]:
        label = raw.loc[frame.id]
        answers = np.r_[frame.response_a, frame.response_b]
        source_names = np.r_[label.model_a, label.model_b]
        mapped = np.array([index.get(value, index['__OTHER__']) for value in source_names])
        frequencies = pd.Series(source_names).value_counts(normalize=True)
        all_names = prior.index.union(frequencies.index)
        y = label[CLASSES].to_numpy().argmax(1)
        entry = {'rows': len(frame), 'prompt_characters': distribution(frame.prompt.map(len)),
                 'answer_characters': distribution([len(text) for text in answers]),
                 'source_input_clipped_fraction': float(np.mean([len(text) > 4096 for text in answers])),
                 'label_proportions': (np.bincount(y, minlength=3) / len(y)).tolist(),
                 'source_rare_or_unknown_fraction': float(np.mean(mapped == index['__OTHER__'])),
                 'source_name_total_variation_from_training': float(np.abs(prior.reindex(all_names, fill_value=0) - frequencies.reindex(all_names, fill_value=0)).sum() / 2)}
        if name != 'train':
            q = np.vstack([probabilities[name + '_a'], probabilities[name + '_b']])
            expected_strength = q @ ratings
            actual_strength = ratings[mapped]
            entry.update(source_accuracy=float(np.mean(q.argmax(1) == mapped)),
                         source_log_loss=float(-np.log(np.maximum(q[np.arange(len(q)), mapped], 1e-12)).mean()),
                         source_top_probability=distribution(q.max(1)),
                         source_entropy=distribution(-(q * np.log(np.maximum(q, 1e-12))).sum(1)),
                         inferred_strength_rmse=float(np.sqrt(np.mean((expected_strength - actual_strength) ** 2))),
                         inferred_strength_pearson=float(np.corrcoef(expected_strength, actual_strength)[0, 1]))
        cohorts[name] = entry
    predictions = np.load(out / 'H46/predictions.npz')
    y = predictions['labels']
    rowloss = lambda p: -np.log(np.maximum(p[np.arange(len(y)), y], 1e-12))
    gain = rowloss(predictions['calibration']) - rowloss(predictions['strength'])
    qa, qb = probabilities['confirmation_a'], probabilities['confirmation_b']
    confidence = (qa.max(1) + qb.max(1)) / 2
    bins = []
    # Descriptive fixed bins, without tuning or proposing an adopted gate.
    for low, high in [(0, .1), (.1, .2), (.2, .35), (.35, .5), (.5, 1.000001)]:
        mask = (confidence >= low) & (confidence < high)
        bins.append({'source_confidence_low': low, 'source_confidence_high': min(1., high),
                     'rows': int(mask.sum()), 'gain': float(gain[mask].mean()) if mask.any() else None})
    source = Path(__file__)
    output = {'cohorts': cohorts, 'confirmation_confidence_bins': bins,
              'descriptive_only': True, 'new_models_or_fits': 0, 'gate_unchanged': result['gate'],
              'warning': 'Inspected cohorts and descriptive slices. Bin gains are not validated gates or deployment evidence.',
              'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    json_write(pub / 'descriptive_shift.json', output)
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
