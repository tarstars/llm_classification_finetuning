"""Fail-closed pilot checks; this information job can never qualify a submission."""
from pathlib import Path
import json
import numpy as np

PACKING_COHORTS = {'warmup', 'control', 'recent_warmup', 'recent'}
RUNTIME = dict(python='3.10.13', torch='2.1.2', transformers='4.42.3', triton='2.2.0',
               xformers='0.0.24+042abc8.d20240802', numpy='1.26.4')


def conversation_budget(role, name):
    assert role == 'm0' and name in PACKING_COHORTS
    return 4096


def after_cohort(args, summary, name):
    out = Path(args.out); spec = json.loads(Path(args.spec).read_text())
    assert spec['mode'] == 'pilot' and args.role == 'm0' and not spec['uses_targets']
    a = np.load(out / ('m0_' + name + '.npz')); actual = summary['cohorts'][name]
    ref = spec['token_references'][name]
    fractions = [actual['peak_allocated_bytes'][g] / summary['device_total_memory_bytes'][g] for g in range(2)]
    checks = dict(ids_match=a['ids'].tolist() == spec['frame_ids'][name],
        lengths_match=a['lengths'].tolist() == ref['lengths'],
        token_hash_match=actual['token_ids_sha256'] == ref['token_ids_sha256'],
        checkpoint_match=json.loads((out / 'm0_checkpoint.json').read_text()) == spec['checkpoint_references']['m0'],
        producer_joined=actual['producer_joined'], packing_exact=summary['packing_tokens'] == 8192,
        cpu_prefetch_exact=summary['cpu_prefetch_enabled'] and summary['queue_depth'] == 2,
        pinned_runtime=summary['versions'] == RUNTIME,
        memory_passed=all(f <= .95 for f in fractions))
    required = list(checks)
    checks['memory_fractions'] = fractions
    if name == 'control':
        gold = np.load(Path(spec['control_directory']) / 'm0.npz')
        np.testing.assert_array_equal(gold['ids'], a['ids'])
        difference = float(np.abs(a['probabilities'].astype(float) - gold['probabilities']).max())
        checks.update(canonical_max_abs=difference, canonical_passed=difference <= 1e-5)
        required.append('canonical_passed')
    checks['passed'] = all(checks[k] for k in required)
    path = out / 'm0_checks.json'; saved = json.loads(path.read_text()) if path.exists() else {}
    saved[name] = checks; path.write_text(json.dumps(saved, indent=2) + '\n')
    if not checks['passed']:
        (out / 'engineering_stop.json').write_text(json.dumps(dict(cohort=name, checks=checks, official_submission_eligible=False), indent=2))
    print('ARENA_EVENT ' + json.dumps(dict(stage='recent_input_check', cohort=name, passed=checks['passed'])), flush=True)
    return checks['passed']
