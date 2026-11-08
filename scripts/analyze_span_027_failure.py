"""Label-free description of the stopped H74 run; never promotes its outputs."""
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT
from arena.experiment import atomic_json, read_json, sha256, utc_now


def main():
    started = time.monotonic()
    pub = REPO / 'results/span_027'
    destination = pub / 'failure_description.json'
    assert not destination.exists(), 'Preserve the first diagnostic report'
    audit = read_json(pub / 'profile_audit.json')
    assert not audit['engineering_gate_passed']
    out = ROOT / 'runs/span_027/jobs/packing_profile/output/packing_profile'
    old = ROOT / 'runs/span_013/jobs/gemma_order/output/order_completion'
    results, hashes = {}, {}
    for path in sorted(out.glob('*.npz')):
        reference = old / path.name
        if not reference.exists():
            continue
        a, b = np.load(path), np.load(reference)
        np.testing.assert_array_equal(a['ids'], b['ids'])
        difference = abs(a['probabilities'].astype(float) - b['probabilities'].astype(float))
        if not difference.size:
            continue
        row_max = difference.max(axis=1)
        results[path.stem] = dict(
            rows=len(difference), max_abs=float(difference.max()),
            mean_abs=float(difference.mean()),
            row_max_quantiles=dict(zip(['min', 'p50', 'p90', 'p95', 'p99', 'max'],
                                      np.quantile(row_max, [0, .5, .9, .95, .99, 1]).tolist())),
            rows_above_original_tolerance=int((row_max > 1e-5).sum()),
            argmax_changes=int((a['probabilities'].argmax(1) != b['probabilities'].argmax(1)).sum()))
        for p in [path, reference]:
            hashes[str(p.relative_to(ROOT))] = sha256(p)
    elapsed = time.monotonic() - started
    assert elapsed < 60
    result = dict(at=utc_now(), cohorts=results, labels_read=False, quality_measured=False,
                  new_gpu_executions=0, engineering_gate_passed=False,
                  official_submission_eligible=False, elapsed_cpu_seconds=elapsed,
                  cpu_cap_seconds=60, input_sha256=hashes,
                  script_sha256=sha256(Path(__file__)),
                  original_audit_sha256=sha256(pub / 'profile_audit.json'),
                  interpretation='Probability drift is observed. Its cause and quality effect are unmeasured; unchanged argmax does not establish equivalent log loss.')
    atomic_json(destination, result)
    print({k: v for k, v in result.items() if k not in ['input_sha256']})


if __name__ == '__main__':
    main()
