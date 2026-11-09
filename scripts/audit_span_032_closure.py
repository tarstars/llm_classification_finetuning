"""Verify the coverage-only H16 stop and version transition against real artifacts."""
from collections import Counter
from pathlib import Path
import sys
import time

import nbformat
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT
from arena.experiment import atomic_json, read_json, sha256, utc_now


def main():
    start = time.monotonic()
    pub = REPO / 'results/span_032'
    run = ROOT / 'runs/span_032'
    assert not (pub / 'closure_audit.json').exists()
    prep = read_json(pub / 'preparation.json')
    audit = read_json(pub / 'detector_audit.json')
    assert prep['coverage_gate_passed'] and not prep['targets_loaded'] and not prep['fitting_performed']
    assert not audit['detector_gate_passed'] and audit['clear_refusals'] + audit['ambiguous'] < audit['required_true_positives']
    assert audit['decision'] == 'stop_detector_before_fitting' and not audit['quality_metrics_measured']
    assert not audit['preference_labels_loaded'] and audit['new_gpu_executions'] == audit['official_submissions'] == 0
    for name, digest in prep['artifacts'].items(): assert sha256(Path(name)) == digest, name
    for name, digest in prep['source_sha256'].items():
        assert sha256(REPO / ('docs/GOAL_V32.md' if name == 'GOAL.md' else name)) == digest, name
    assert sha256(run / 'blind_detector_audit.parquet') == audit['input_sha256']
    assert sha256(run / 'detector_annotations.json') == audit['annotations_sha256']
    sample = pd.read_parquet(run / 'blind_detector_audit.parquet')
    assert set(sample.columns) == {'id', 'prompt', 'response_a', 'response_b', 'detected_side'}
    annotations = read_json(run / 'detector_annotations.json')
    assert annotations['training_only'] and not annotations['preference_labels_used']
    assert sample.id.tolist() == [r['id'] for r in annotations['rows']]
    counts = Counter(r['verdict'] for r in annotations['rows'])
    assert counts == Counter(clear_refusal=17, ambiguous=2, false_positive=5, no_opening_refusal_observed=8)
    failure = read_json(pub / 'preparation_assertion_failure.json')
    assert sha256(Path(failure['original_source_path'])) == failure['original_source_sha256']
    resources = read_json(pub / 'resource_close.json')
    assert resources['passed'] and resources['gpu_charge_seconds'] == resources['public_versions'] == resources['official_submissions'] == 0
    assert resources['leaderboard_sha256'] == sha256(pub / 'final_leaderboard.json')
    preserve = read_json(pub / 'preservation.json')
    assert len(preserve['snapshot_sha256']) == 126
    for name, digest in {**preserve['snapshot_sha256'], **preserve['user_notebook_sha256']}.items(): assert sha256(REPO / name) == digest, name
    assert (Path.home() / 'notebooks/llm_classification_finetuning').resolve() == REPO / 'notebooks'
    final = read_json(REPO / 'backlog/versions/v32.final.json')
    v = next(v for v in final['versions'] if v['number'] == 32)
    items = {i['id']: i for i in final['items']}
    assert v['focus_ids'] == ['H16'] and items['H16']['status'] == 'done'
    assert any(h['status'] == 'done' and 'docs/SPAN_032_RESULTS.md' in h.get('evidence', []) for h in items['H16']['history'])
    carried = {i for i in v['item_ids'] if items[i]['status'] not in ['done', 'rejected']}
    assert len(carried) == 36 and not any(items[i]['status'] == 'running' for i in v['item_ids'])
    nxt = read_json(REPO / 'backlog/versions/v33.start.json')
    nv = next(v for v in nxt['versions'] if v['number'] == 33)
    assert nxt['active_version'] == 33 and set(nv['item_ids']) == carried | {'H77'} and not nv.get('focus_ids')
    assert len(list((REPO / 'backlog/versions').glob('*'))) == 130
    notebook = REPO / 'notebooks/44_refusal_detector_audit.ipynb'
    nb = nbformat.read(notebook, 4)
    cells = [c for c in nb.cells if c.cell_type == 'code']
    assert cells and all(c.execution_count is not None and not any(o.output_type == 'error' for o in c.outputs) for c in cells)
    for name in ['preparation', 'detector_audit', 'resource_close']:
        assert nb.metadata['span032_' + name + '_sha256'] == sha256(pub / (name + '.json'))
    result = dict(at=utc_now(), passed=True, sealed_version=32, active_version=33,
        historical_snapshots_preserved=126, snapshot_files_after_transition=130,
        carried_tasks=36, new_discoveries=['H77'], goal_achieved=False,
        audit_sha256=sha256(pub / 'detector_audit.json'), notebook_sha256=sha256(notebook),
        report_sha256=sha256(REPO / 'docs/SPAN_032_RESULTS.md'),
        goal_archive_sha256=sha256(REPO / 'docs/GOAL_V32.md'),
        elapsed_seconds=time.monotonic() - start, source_sha256=sha256(Path(__file__)))
    atomic_json(pub / 'closure_audit.json', result)
    print(result)


if __name__ == '__main__':
    main()
