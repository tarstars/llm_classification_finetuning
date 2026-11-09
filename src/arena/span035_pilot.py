"""One private information pilot; text only, no hidden or quality-target path."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import time
import numpy as np
import pandas as pd


def run(args, execute=subprocess.run):
    start = time.monotonic(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    assert not list(out.glob('m0_*.npz')), 'Never reuse pilot predictions'
    spec = json.loads(Path(args.spec).read_text())
    assert spec['mode'] == 'pilot' and not spec['uses_targets'] and spec['packing_tokens'] == 8192
    raw = Path(args.raw) / 'train.csv'
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == spec['train_csv_sha256']
    frame = pd.read_csv(raw, usecols=['id', 'prompt', 'response_a', 'response_b']).set_index('id')
    for name, ids in spec['frame_ids'].items():
        value = frame.loc[ids].reset_index()
        value.to_parquet(out / (name + '.parquet'), index=False)
    runtime = dict(spec, control_directory=str(Path(args.spec).parent / 'controls'))
    current = out / 'runtime_spec.json'; current.write_text(json.dumps(runtime))
    names = ['warmup', 'control', 'recent_warmup', 'recent']
    execute([sys.executable, '-u', args.engine, '--raw', args.raw, '--assets', args.assets,
        '--out', str(out), '--spec', str(current), '--role', 'm0', '--cohorts', ','.join(names)], check=True)
    stopped = (out / 'engineering_stop.json').exists()
    if not stopped:
        for name in names:
            a = np.load(out / ('m0_' + name + '.npz'))
            assert a['ids'].tolist() == spec['frame_ids'][name]
            p = a['probabilities']
            assert p.shape == (len(a['ids']), 3) and np.isfinite(p).all() and (p >= 0).all()
            np.testing.assert_allclose(p.sum(1), 1, atol=1e-5)
    for path in out.glob('*.parquet'): path.unlink()
    result = dict(complete=not stopped, stopped_for_guard=stopped, mode='pilot', uses_targets=False,
        actual_text_inference=True, rows=len(spec['frame_ids']['recent']),
        seconds=time.monotonic() - start, official_submission_eligible=False)
    (out / 'pilot_complete.json').write_text(json.dumps(result, indent=2) + '\n')
    print('ARENA_EVENT ' + json.dumps(dict(stage='recent_pilot_complete', **result)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ['raw', 'assets', 'out', 'engine', 'spec']: parser.add_argument('--' + key, required=True)
    run(parser.parse_args())
