"""One predeclared H60 cached information gate; original votes overlap training."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT, parse_turns, prompt_groups
from arena.experiment import read_json, atomic_json, sha256, utc_now
from arena.span010_order import CLASSES, route_indices, ensemble
from arena.span013_order import complete_gemma_order
from audit_span_010_order import loss, paired_interval


def main():
    pub = REPO / 'results/span_013'
    registry = read_json(REPO / 'backlog/hypotheses.json')
    items = {item['id']: item for item in registry['items']}
    assert registry['active_version'] == 13 and items['E16']['status'] == 'done' and items['H60']['status'] == 'running'
    assert read_json(REPO / 'results/span_012/E16_qualification.json')['engineering_gate_passed']
    assert not (pub / 'H60_cached_replay.json').exists(), 'Preserve the one fixed cached replay'
    cache = ROOT / 'runs/span_010/jobs/order_pilot/output/order_completion'
    selection = read_json(REPO / 'results/span_010/profile_selection.json')
    train = pd.read_csv(ROOT / 'raw/train.csv', usecols=['id', 'prompt', *CLASSES]).set_index('id')
    test = pd.read_csv(ROOT / 'raw/test.csv', usecols=['id'])
    records, hashes = {}, {}
    for name in ['control', 'uniform', 'long', 'test']:
        expected = np.asarray(selection[name + '_ids']) if name != 'test' else test.id.to_numpy()
        loaded = {}
        for role in ['m0', 'm3', 'm0_reverse_routed']:
            path = cache / (role + '_' + name + '.npz')
            hashes[str(path)] = sha256(path)
            loaded[role] = np.load(path)
        a, b, reverse = (loaded[role] for role in ['m0', 'm3', 'm0_reverse_routed'])
        assert np.array_equal(a['ids'], expected) and np.array_equal(b['ids'], expected)
        indices, js, work = route_indices(expected, a['probabilities'], b['probabilities'][:, [1, 0, 2]],
                                          np.c_[a['lengths'], b['lengths']])
        route_path = cache / ('routing_' + name + '.npz')
        hashes[str(route_path)] = sha256(route_path)
        old_route = np.load(route_path)
        np.testing.assert_array_equal(indices, old_route['selected_indices'])
        assert np.array_equal(reverse['ids'], expected[indices])
        base = ensemble(a['probabilities'], b['probabilities'])
        candidate = complete_gemma_order(a['probabilities'], b['probabilities'], indices, reverse['probabilities'])
        record = dict(rows=len(expected), selected_rows=len(indices), route_replayed=True,
                      selected_work_fraction=float(work[indices].sum() / work.sum()), independent=False)
        if name != 'test':
            frame = train.loc[expected]
            y = frame[CLASSES].to_numpy().argmax(1)
            groups = prompt_groups(frame.assign(prompt=frame.prompt.map(parse_turns))).to_numpy()
            delta = loss(y, base) - loss(y, candidate)
            record.update(baseline_log_loss=float(loss(y, base).mean()), candidate_log_loss=float(loss(y, candidate).mean()),
                          loss_reduction=float(delta.mean()), descriptive_paired_group95=paired_interval(delta, groups),
                          baseline_accuracy=float((base.argmax(1) == y).mean()),
                          candidate_accuracy=float((candidate.argmax(1) == y).mean()),
                          class_loss_reduction={CLASSES[k]:float(delta[y == k].mean()) for k in range(3) if np.any(y == k)})
        records[name] = record
    passed = records['uniform']['loss_reduction'] > .0005
    result = dict(at=utc_now(), information_gate_passed=passed, information_gain_threshold=.0005,
                  independent_local_quality=False, original_votes_seen_in_checkpoint_training=True,
                  rule='on route: .25 Gemma original + .25 aligned Gemma swapped + .5 aligned Llama swapped; off route: base',
                  temperature=1, work_fraction=.1, row_fraction=.2, quality_variants_tested=1,
                  coverage=records, cache_sha256=hashes,
                  source_sha256={name:sha256(REPO/name) for name in ['src/arena/span013_order.py',
                      'scripts/replay_span_013_gemma.py','scripts/audit_span_010_order.py']},
                  decision='Prepare actual three-pass qualification' if passed else 'Stop H60 without new GPU',
                  new_gpu_executions=0)
    atomic_json(pub / 'H60_cached_replay.json', result)
    print(json.dumps({k:result[k] for k in ['information_gate_passed', 'coverage', 'decision']}, indent=2))


if __name__ == '__main__':
    main()
