"""Freeze unchanged incumbent controls and exact CPU batch references for E18."""
from pathlib import Path
import shutil
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT
from arena.experiment import atomic_json, read_json, sha256, utc_now, verify_sources


def main():
    started = time.monotonic()
    pub = REPO / 'results/span_028'; root = ROOT / 'runs/span_028'
    assert not (pub / 'profile_preparation.json').exists()
    proof = read_json(pub / 'cpu_audit.json'); assert proof['passed']
    verify_sources(REPO, proof['source_sha256'])
    for n, h in proof['output_sha256'].items():
        assert sha256(ROOT / n) == h
    old = ROOT / 'runs/span_027'
    previous = read_json(REPO / 'results/span_027/profile_preparation.json')
    assert sha256(old / 'profile_spec.json') == previous['profile_spec_sha256']
    base = read_json(old / 'profile_spec.json')
    batches = read_json(root / 'cpu_batch_references.json')
    references = {}
    for role, cohorts in batches.items():
        references[role] = {}
        for name, values in cohorts.items():
            expected = base['token_references'][role][name]
            assert values['ids'] == expected['ids'] and values['lengths'] == expected['lengths']
            assert values['token_ids_sha256'] == expected['token_ids_sha256']
            references[role][name] = {k: values[k] for k in ['groups', 'host_tensor_sha256']}
    spec = dict(base, at=utc_now(), packing_tokens=8192, execution_mode='cpu_prefetch',
                producer_threads=1, queue_depth=2, batch_references=references,
                source_proof_sha256=sha256(pub / 'cpu_audit.json'))
    controls = root / 'controls'; controls.mkdir(exist_ok=True)
    for name, digest in previous['canonical_control_sha256'].items():
        source = old / 'controls' / name
        assert sha256(source) == digest and not (controls / name).exists()
        shutil.copyfile(source, controls / name)
    assert not (root / 'profile_spec.json').exists()
    atomic_json(root / 'profile_spec.json', spec)
    timing = read_json(REPO / 'results/span_027/runtime_reference.json')
    for n, h in timing['database_reference_sha256'].items():
        assert sha256(ROOT / n) == h
    for n, h in timing['project_reference_sha256'].items():
        assert sha256(REPO / n) == h
    timing.update(at=utc_now(), execution_mode='cpu_prefetch', packing_tokens=8192,
                  synchronous_throughput_is_comparator_not_new_mode_floor=True,
                  inherited_reference_sha256=sha256(REPO / 'results/span_027/runtime_reference.json'))
    atomic_json(pub / 'runtime_reference.json', timing)
    atomic_json(pub / 'profile_preparation.json', dict(at=utc_now(), prepared=True, uses_targets=False,
        profile_spec_sha256=sha256(root / 'profile_spec.json'),
        canonical_control_sha256=previous['canonical_control_sha256'],
        cpu_audit_sha256=sha256(pub / 'cpu_audit.json'), cpu_batch_reference_sha256=sha256(root / 'cpu_batch_references.json'),
        packing_tokens=8192, conversation_tokens=4096, queue_depth=2, producer_threads=1,
        elapsed_cpu_seconds=time.monotonic()-started, new_gpu_executions=0))
    print('Exact native prefetch references prepared; no GPU admitted.')


if __name__ == '__main__':
    main()
