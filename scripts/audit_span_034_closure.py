"""Independently replay saved fold models, optimizer conditions and span closure."""
from pathlib import Path
import sys
import time

import nbformat
import numpy as np
from scipy.special import expit

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT
from arena.experiment import read_json, atomic_json, sha256, utc_now
from arena.span034_residual import predict, folds
from arena.span_heads import loss_rows, metrics, paired_group_gain


def main():
    start = time.monotonic()
    pub, run = REPO / 'results/span_034', ROOT / 'runs/span_034'
    assert not (pub / 'closure_audit.json').exists()
    result, prep, cfg = read_json(pub / 'H78_results.json'), read_json(pub / 'preparation.json'), read_json(REPO / 'configs/span_034.json')
    for name, digest in prep['artifacts'].items(): assert sha256(Path(name)) == digest, name
    for name, digest in prep['source_sha256'].items():
        assert sha256(REPO / ('docs/GOAL_V34.md' if name == 'GOAL.md' else name)) == digest, name
    assert sha256(run / 'inputs.npz') == prep['inputs_sha256']
    assert sha256(run / 'evaluation.npz') == result['evaluation_sha256']
    assert sha256(run / 'fold_records.json') == result['folds_sha256']
    original, saved = np.load(run / 'inputs.npz'), np.load(run / 'evaluation.npz')
    labels = np.load(ROOT / 'runs/span_016/cohort_matched/labels.npz')
    for name in ['ids', 'labels', 'groups']: np.testing.assert_array_equal(saved[name], labels[name])
    p, d, y, g = original['baseline'], original['reward_difference'], labels['labels'], labels['groups']
    np.testing.assert_array_equal(saved['baseline'], p)
    np.testing.assert_array_equal(saved['outer_fold'], folds(g, cfg['seed'], 5, 'outer'))
    records = read_json(run / 'fold_records.json'); assert len(records) == 5
    optimizer_checks = 0
    def check_model(model, train):
        nonlocal optimizer_checks
        train = np.asarray(train)
        assert model['fit_rows'] == len(train) and model['non_tie_rows'] == int((y[train] != 2).sum())
        np.testing.assert_allclose(model['scale'], max(np.sqrt(np.mean(d[train].astype(float) ** 2)), 1e-12), atol=1e-12)
        use = train[y[train] != 2]
        z = d[use].astype(float) / model['scale']
        s = np.log(np.maximum(p[use, 0], 1e-15)) - np.log(np.maximum(p[use, 1], 1e-15))
        a, penalty = model['alpha'], model['penalty']
        gradient = float(np.mean((expit(s + a * z) - (y[use] == 0)) * z) + penalty * a)
        assert 0 <= a <= 10 and penalty in cfg['lambda_grid']
        assert gradient >= -1e-6 if a == 0 else gradient <= 1e-6 if a == 10 else abs(gradient) < 1e-6
        optimizer_checks += 1
    for r in records:
        number, train, val = r['fold'], np.asarray(r['fit_indices']), np.asarray(r['validation_indices'])
        assert set(train) == set(np.flatnonzero(saved['outer_fold'] != number))
        assert set(val) == set(np.flatnonzero(saved['outer_fold'] == number))
        assert not set(g[train]) & set(g[val])
        inner = folds(g[train], cfg['seed'], 3, f'inner:{number}')
        scores = []
        for penalty in cfg['lambda_grid']:
            q = np.full_like(p[train], np.nan)
            traces = [t for t in r['inner_traces'] if t['penalty'] == penalty]; assert len(traces) == 3
            for t in traces:
                k, fit_ids, test_ids = t['inner_fold'], np.asarray(t['fit_indices']), np.asarray(t['validation_indices'])
                np.testing.assert_array_equal(fit_ids, train[inner != k]); np.testing.assert_array_equal(test_ids, train[inner == k])
                assert not set(g[fit_ids]) & set(g[test_ids])
                assert t['model']['penalty'] == penalty
                check_model(t['model'], fit_ids)
                q[inner == k] = predict(p[test_ids], d[test_ids], t['model']['alpha'], t['model']['scale'])
            scores.append(dict(penalty=penalty, loss=float(loss_rows(y[train], q).mean())))
        for actual, expected in zip(r['inner_scores'], scores):
            assert actual['penalty'] == expected['penalty']; np.testing.assert_allclose(actual['loss'], expected['loss'], atol=1e-14)
        assert r['model']['penalty'] == min(scores, key=lambda s: s['loss'])['penalty']
        check_model(r['model'], train)
        out = predict(p[val], d[val], r['model']['alpha'], r['model']['scale'])
        np.testing.assert_array_equal(out, saved['candidate'][val])
        np.testing.assert_array_equal(predict(p[val][:, [1, 0, 2]], -d[val], r['model']['alpha'], r['model']['scale']), out[:, [1, 0, 2]])
        counts = np.bincount(y[train], minlength=3)
        np.testing.assert_array_equal(saved['prior'][val], np.tile(counts / counts.sum(), (len(val), 1)))
        np.testing.assert_array_equal(saved['majority'][val], np.tile(np.eye(3)[counts.argmax()], (len(val), 1)))
    assert optimizer_checks == 50
    np.testing.assert_array_equal(saved['candidate'][:, 2], p[:, 2])
    gain = paired_group_gain(y, p, saved['candidate'], g, dict(cfg, seed=cfg['bootstrap_seed']))
    assert gain == result['raw_paired_gain']
    for name, key in [('baseline', 'baseline'), ('candidate', 'candidate'), ('prior', 'prior'), ('hard_majority', 'majority')]:
        assert metrics(y, saved[key]) == result[name + '_metrics']
    assert not result['useful_information'] and result['decision'] == 'stop_no_qualified_reward_residual_gain'
    assert result['raw_paired_gain']['interval_95'][1] < cfg['advancement_min_gain']
    preserve = read_json(pub / 'preservation.json'); assert len(preserve['snapshot_sha256']) == 134
    for name, digest in {**preserve['snapshot_sha256'], **preserve['user_notebook_sha256']}.items(): assert sha256(REPO / name) == digest, name
    assert (Path.home() / 'notebooks/llm_classification_finetuning').resolve() == REPO / 'notebooks'
    resources = read_json(pub / 'resource_close.json')
    assert resources['passed'] and resources['gpu_charge_seconds'] == resources['official_submissions'] == resources['public_versions'] == 0
    assert resources['leaderboard_sha256'] == sha256(pub / 'final_leaderboard.json')
    final = read_json(REPO / 'backlog/versions/v34.final.json')
    v = next(v for v in final['versions'] if v['number'] == 34); items = {i['id']: i for i in final['items']}
    assert v['focus_ids'] == ['H78'] and items['H78']['status'] == 'done'
    carried = {i for i in v['item_ids'] if items[i]['status'] not in ['done', 'rejected']}
    assert len(carried) == 36 and not any(items[i]['status'] == 'running' for i in v['item_ids'])
    nxt = read_json(REPO / 'backlog/versions/v35.start.json'); nv = next(v for v in nxt['versions'] if v['number'] == 35)
    assert nxt['active_version'] == 35 and set(nv['item_ids']) == carried and not nv.get('focus_ids')
    assert len(list((REPO / 'backlog/versions').glob('*'))) == 138
    notebook = REPO / 'notebooks/46_cheap_reward_residual.ipynb'; nb = nbformat.read(notebook, 4)
    cells = [c for c in nb.cells if c.cell_type == 'code']
    assert cells and all(c.execution_count is not None and not any(o.output_type == 'error' for o in c.outputs) for c in cells)
    for name in ['H78_results', 'resource_close']:
        assert nb.metadata['span034_' + name + '_sha256'] == sha256(pub / (name + '.json'))
    atomic_json(pub / 'closure_audit.json', dict(at=utc_now(), passed=True, sealed_version=34, active_version=35,
        optimizer_checks=optimizer_checks, nested_group_replay=True, exact_tie_and_swap=True,
        historical_snapshots_preserved=134, snapshot_files_after_transition=138,
        carried_tasks=36, new_discoveries=[], goal_achieved=False,
        report_sha256=sha256(REPO / 'docs/SPAN_034_RESULTS.md'), notebook_sha256=sha256(notebook),
        elapsed_seconds=time.monotonic() - start, source_sha256=sha256(Path(__file__))))
    print('V34 independent replay/50optimizer conditions/reporting/preservation/closure passed.')


if __name__ == '__main__':
    main()
