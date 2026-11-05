"""Verify completed source and prediction identities before opening confirmation votes."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import sys

import nbformat
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT, json_write
from arena.span006_inputs import text_digest


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    out = ROOT / 'runs/span_006'
    pub = REPO / 'results/span_006'
    assert not (out / 'reveal_intent.json').exists(), 'Keep the first reveal fence'
    inputs = read(pub / 'artifact_inventory.json')
    recipe = read(pub / 'frozen_recipe.json')
    bundle = read(pub / 'confirmation_bundle.json')
    for section in [inputs['input_artifacts'], recipe['recipe_artifacts']]:
        for name, artifact in section.items():
            assert sha(artifact['path']) == artifact['sha256'], name
    sources = dict(inputs['source_sha256'], **bundle['source_sha256'])
    for path, digest in sources.items():
        assert sha(REPO / path) == digest, path
        committed = subprocess.check_output(['git', 'show', 'HEAD:' + path], cwd=REPO)
        assert hashlib.sha256(committed).hexdigest() == digest, path
    for path, digest in inputs['old_snapshot_sha256'].items():
        assert sha(REPO / path) == digest, path
    job = out / 'jobs/confirmation'
    assert read(job / 'controller_state.json')['stage'] == 'complete'
    assert read(job / 'status.json')['status'] == 'COMPLETE'
    notebook = out / 'notebooks/confirmation/job.ipynb'
    assert sha(notebook) == bundle['notebook_sha256']
    observed = out / 'source_observation/confirmation'
    files = list(observed.glob('*.ipynb'))
    assert len(files) == 1
    cells = lambda p: [(c.cell_type, c.source) for c in nbformat.read(p, as_version=4).cells]
    assert cells(notebook) == cells(files[0])
    assert read(observed / 'kernel-metadata.json')['enable_internet'] is False
    output = job / 'output/confirmation'
    summary = read(output / 'confirmation_summary.json')
    smoke = read(output / 'smoke_summary.json')
    assert summary['complete'] and smoke['passed']
    assert not summary['confirmation_votes_read'] and not summary['confirmation_source_names_read']
    assert summary['rows'] == 5746
    assert summary['frozen_recipe_sha256'] == sha(pub / 'frozen_recipe.json')
    assert summary['adapter_sha256'] == inputs['input_artifacts']['adapter']['sha256']
    assert summary['base_config_sha256'] == '8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a'
    assert sha(output / 'confirmation_predictions.npz') == summary['prediction_sha256']
    confirm = pd.read_parquet(inputs['input_artifacts']['confirmation_inputs']['path'])
    assert set(confirm.columns) == {'id', 'prompt', 'response_a', 'response_b'}
    assert text_digest(confirm) == summary['confirmation_text_digest'] == inputs['confirmation_text_digest']
    raw = np.load(output / 'confirmation_predictions.npz')
    np.testing.assert_array_equal(raw['ids'], confirm.id)
    cursor = 0
    for chunk in summary['chunks']:
        assert chunk['start'] == cursor and chunk['end'] == min(cursor + 512, 5746)
        path = output / f'chunk_{cursor:05d}.npz'
        assert sha(path) == chunk['sha256']
        saved = np.load(path)
        np.testing.assert_array_equal(saved['ids'], raw['ids'][cursor:chunk['end']])
        for view in ['original', 'aligned_swap']:
            np.testing.assert_array_equal(saved[view], raw[view][cursor:chunk['end']])
        cursor = chunk['end']
    assert cursor == 5746 and len(summary['chunks']) == 12
    for view in ['original', 'aligned_swap']:
        p = raw[view]
        assert p.shape == (5746, 3) and np.isfinite(p).all() and (p >= 0).all()
        np.testing.assert_allclose(p.sum(1), 1, atol=1e-6)
    hardware = [read(output / f'gpu_rank{k}.json') for k in [0, 1]]
    assert {r['device'] for r in hardware} == {0, 1}
    assert all('T4' in r['name'] and r['peak_allocated_bytes'] > 0 for r in hardware)
    charge = read(job / 'quota.json')['used_seconds'] - read(job / 'push_intent.json')['quota_before']['used_seconds']
    assert 0 < charge <= 5400
    result = {'passed': True, 'checked_at': datetime.now(timezone.utc).isoformat(),
              'source_files_verified': len(sources), 'rows': 5746, 'chunks': 12,
              'remote_source_matches': True, 'internet_disabled': True,
              'actual_two_t4_inference': True, 'confirmation_votes_read': False,
              'frozen_recipe_sha256': sha(pub / 'frozen_recipe.json'),
              'prediction_sha256': summary['prediction_sha256'],
              'charged_gpu_seconds': charge, 'source_sha256': sha(Path(__file__))}
    for folder in [pub, out]:
        json_write(folder / 'pre_reveal_audit.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
