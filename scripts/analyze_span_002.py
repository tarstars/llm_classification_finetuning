"""Evaluate predeclared cached calibration and classifier/judge blend experiments."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT, CLASSES, json_write, prompt_groups
from arena.joint_data import read_competition
from arena.span_calibration import crossfit_calibration, crossfit_blend, grouped_folds
from arena.span_heads import metrics, paired_group_gain, normalize, temperature, loss_rows


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['calibration', 'blend'])
    args = parser.parse_args()
    started = time.monotonic()
    root, aggregate = ROOT / 'runs/span_002', REPO / 'results/span_002'
    inventory = json.loads((root / 'artifact_inventory.json').read_text())
    config = json.loads((REPO / 'configs/span_002.json').read_text())
    frame, _ = read_competition(ROOT / 'raw')
    frame = frame.set_index('id', drop=False)
    all_groups = prompt_groups(frame)
    for name in ('baseline', 'joint', 'judge'):
        assert digest(inventory[name]['path']) == inventory[name]['sha256']
    baseline = np.load(inventory['baseline']['path'])
    lookup = {int(rid): i for i, rid in enumerate(baseline['ids'])}
    source = {name: digest(REPO / name) for name in ['configs/span_002.json',
        'docs/SPAN_002_PROTOCOL.md', 'scripts/analyze_span_002.py', 'src/arena/span_calibration.py']}
    report = {'span': 2, 'stage': args.stage, 'observed_at': datetime.now(timezone.utc).isoformat(),
        'evaluation': 'Exploratory development; confirmation unused; intervals conditional on fitted OOF predictions.',
        'config': config, 'source_sha256': source, 'cohorts': {}}
    saved = {}

    def cohort(name, ids):
        y = frame.loc[ids, CLASSES].values.argmax(1)
        groups = all_groups.loc[ids].to_numpy()
        raw = normalize(baseline['probabilities'][[lookup[int(rid)] for rid in ids]])
        saved[name + '_ids'], saved[name + '_labels'] = ids, y
        # Preserve exact outer and per-outer inner assignments, privately.
        folds = []
        for train, val in grouped_folds(y, groups, config['outer_group_folds'], config['seed']):
            inner = grouped_folds(y[train], groups[train], config['inner_group_folds'], config['seed'])
            folds.append({'train_ids': ids[train].tolist(), 'validation_ids': ids[val].tolist(),
                'inner': [{'train_ids': ids[train[tr]].tolist(), 'validation_ids': ids[train[va]].tolist()}
                          for tr, va in inner]})
        json_write(root / (args.stage + '_' + name + '_folds.json'), folds)
        return y, groups, raw

    def contrast(y, before, after, groups):
        result = paired_group_gain(y, before, after, groups, config)
        differences = loss_rows(y, before)-loss_rows(y, after)
        result['weighted_class_contributions'] = [float(differences[y==i].sum()/len(y)) for i in range(3)]
        assert np.isclose(sum(result['weighted_class_contributions']), result['gain'])
        return result

    if args.stage == 'calibration':
        joint = np.load(inventory['joint']['path'])
        for name, ids, models in [
            ('full_baseline', baseline['ids'], {}),
            ('paired_pilot', joint['ids'], {'paired_ce': joint['paired_attention_raw'],
                                          'paired_kl': joint['paired_consistency_raw']})]:
            y, groups, raw = cohort(name, ids)
            models = {'frozen': raw, **models}
            entries, predictions = {}, {}
            for model_name, probabilities in models.items():
                output, details = crossfit_calibration(normalize(probabilities), y, groups, config)
                entries[model_name] = {'raw': metrics(y, probabilities), 'folds': details,
                    'families': {key: metrics(y, value) for key, value in output.items()},
                    'nested_gain_vs_temperature': contrast(y, output['temperature'], output['nested_selected'], groups)}
                predictions[model_name] = output
                for family, p in output.items():
                    saved[name+'_'+model_name+'_'+family] = p
                print('CALIBRATION', name, model_name, entries[model_name]['families'], flush=True)
            deployed = temperature(raw, .7220597450036835)
            saved[name+'_deployed'] = deployed
            result = {'rows': len(y), 'groups': len(set(groups)), 'models': entries,
                      'historically_calibrated_deployed': metrics(y, deployed)}
            if name == 'paired_pilot':
                before, after = predictions['frozen']['nested_selected'], predictions['paired_kl']['nested_selected']
                result['paired_kl_vs_matched_frozen'] = contrast(y, before, after, groups)
                gap = metrics(y, after)['log_loss']-metrics(y, before)['log_loss']
                tie_gap = metrics(y, after)['class_loss'][2]-metrics(y, before)['class_loss'][2]
                result['h42_gate'] = {'overall_loss_gap': gap, 'tie_class_loss_gap': tie_gap,
                    'calibration_resolves_deficit': bool(gap <= config['tie_deficit_loss_tolerance'] and
                        tie_gap <= config['tie_deficit_class_loss_tolerance']),
                    'interpretation': 'Predeclared practical stopping rule, not an equivalence test.'}
            report['cohorts'][name] = result
    else:
        judge = np.load(inventory['judge']['path'])
        ids = judge['ids']
        y, groups, raw = cohort('cached_judge', ids)
        np.testing.assert_array_equal(y, judge['labels'])
        output, details = crossfit_blend(raw, normalize(judge['original']), y, groups, config)
        for name, p in output.items():
            saved['cached_judge_'+name] = p
        report['cohorts']['cached_judge'] = {'rows': len(y), 'groups': len(set(groups)),
            'models': {name: metrics(y, p) for name,p in output.items()}, 'folds': details,
            'blend_gain_vs_classifier': contrast(y, output['classifier'], output['blend'], groups),
            'blend_gain_vs_judge': contrast(y, output['judge'], output['blend'], groups),
            'historically_calibrated_deployed': metrics(y, temperature(raw, .7220597450036835))}
    report['cpu_elapsed_seconds'] = time.monotonic()-started
    np.savez_compressed(root / (args.stage+'_predictions.npz'), **saved)
    report['private_predictions_sha256'] = digest(root / (args.stage+'_predictions.npz'))
    for directory in (root, aggregate):
        json_write(directory / (args.stage+'_results.json'), report)
    print(json.dumps({'stage': args.stage, 'seconds': report['cpu_elapsed_seconds'],
        'cohorts': {name: {key: value for key,value in row.items() if key not in ('models','folds')}
                    for name,row in report['cohorts'].items()}}, indent=2))


if __name__ == '__main__':
    main()
