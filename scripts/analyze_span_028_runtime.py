"""Describe measured prefetch timings and one two-role feasibility calculation."""
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT
from arena.experiment import atomic_json, read_json, sha256, utc_now


def main():
    started = time.monotonic(); pub = REPO / 'results/span_028'
    destination = pub / 'runtime_description.json'
    assert not destination.exists(), 'Preserve first runtime description'
    audit = read_json(pub / 'profile_audit.json')
    assert audit['all_controls_passed'] and not audit['runtime_gate_passed']
    floor = read_json(pub / 'runtime_reference.json')['overhead_floor']
    out = ROOT / 'runs/span_028/jobs/prefetch_profile/output/prefetch_profile'
    timings, hashes = {}, {}
    for role in ['m0', 'm3', 'm0_reverse']:
        path = out / (role + '_summary.json'); s = read_json(path)
        name = 'routed_uniform' if role == 'm0_reverse' else 'uniform'
        c = s['cohorts'][name]; metrics = c['cpu_prefetch_metrics']
        timings[role] = dict(rows=c['rows'], cohort_wall_seconds=c['seconds'],
            preparation_seconds=metrics['preparation_seconds'],
            producer_queue_wait_seconds=metrics['producer_queue_wait_seconds'],
            consumer_queue_wait_seconds=metrics['consumer_queue_wait_seconds'],
            forward_launch_transfer_hostcopy_wall_seconds=c['forward_launch_transfer_hostcopy_wall_seconds'])
        hashes[str(path.relative_to(ROOT))] = sha256(path)
    roles = ['m0', 'm3']; runtime = audit['runtime']
    steady = sum(timings[r]['cohort_wall_seconds'] for r in roles)*25000/512*1.2
    cold = sum(floor['cold_by_role'][r] for r in roles)
    warm = sum(floor['warm_by_role'][r] for r in roles)
    other = runtime['other_seconds']
    projection = steady+cold+warm+other+600
    result = dict(at=utc_now(), measured_uniform_timings=timings, labels_read=False,
        quality_measured=False, new_gpu_executions=0, original_three_role_projection_seconds=runtime['projection_seconds'],
        original_runtime_failure_preserved=True, cpu_overlap_not_added_to_wall_time=True,
        same_session_synchronous_comparator_measured=False,
        two_role_feasibility=dict(roles=roles, steady_seconds=steady, cold_seconds=cold,
            full_warmup_seconds=warm, other_seconds=other, margin=1.2, reserve_seconds=600,
            projection_seconds=projection, candidate_qualified=False, quality_measured=False,
            official_submission_eligible=False,
            interpretation='Uses both actually measured base roles and retained startup floors; no reverse pass or hypothetical faster timing. A new two-role candidate still needs actual qualification.'),
        elapsed_cpu_seconds=time.monotonic()-started, source_sha256=sha256(Path(__file__)),
        audit_sha256=sha256(pub/'profile_audit.json'), input_sha256=hashes)
    atomic_json(destination, result)
    print(result)


if __name__ == '__main__':
    main()
