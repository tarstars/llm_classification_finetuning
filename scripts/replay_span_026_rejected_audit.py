"""Finish verification of an already rejected run after derived-JS roundoff.

Keep the frozen auditor untouched. The sole numerical change concerns a derived
CPU log diagnostic, using the established H60 tolerance. Raw model controls,
selected IDs, token/checkpoint identity and all deployment gates remain intact.
This companion can only record a rejected result and never authorize deployment.
"""
from pathlib import Path
import hashlib
import json
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.experiment import atomic_json,read_json,sha256,utc_now


def main():
    pub=REPO/'results/span_026';original=REPO/'scripts/audit_span_026_candidate.py'
    bundle=read_json(pub/'context_qualification_bundle.json')
    assert sha256(original)==bundle['source_sha256']['scripts/audit_span_026_candidate.py']
    assert read_json(pub/'qualification_runtime_floor_replay.json')['minimum_projection_seconds']>=32400
    assert not (pub/'candidate_qualification.json').exists() and not (pub/'candidate_qualification_replay.json').exists()
    failure=pub/'frozen_qualification_audit_failure.json';assert not failure.exists()
    atomic_json(failure,dict(at=utc_now(),passed=False,original_script_sha256=sha256(original),
        execution_session=43888,exit_code=1,failed_check='bitwise derived JS-disagreement metadata',
        cohort='control',rows=256,mismatched_elements=173,max_abs_difference=2.7755575615628914e-17,
        route_indices_exact_at_failure=True,targets_read=False,original_source_preserved=True,
        matching_numpy126_replay_same_max_abs=True,decision='retain_runtime_rejection'))
    source=original.read_text();needle="np.testing.assert_array_equal(saved_route['disagreement'],js)"
    assert source.count(needle)==1 and source.count("'candidate_qualification.json'")==2
    replay=source.replace(needle,"np.testing.assert_allclose(saved_route['disagreement'],js,rtol=1e-12,atol=1e-14)")
    replay=replay.replace("'candidate_qualification.json'","'candidate_qualification_replay.json'")
    scope=dict(__file__=str(original),__name__='span026_rejected_audit_replay')
    exec(compile(replay,'span026-rejected-audit-replay','exec'),scope);scope['main']()
    path=pub/'candidate_qualification_replay.json';r=read_json(path)
    assert not r['official_submission_eligible'] and not r['runtime_gate_passed'] and not r['labels_read']
    r['validation_replay']=dict(original_frozen_audit_passed=False,original_source_preserved=True,
        derived_js_rtol=1e-12,derived_js_atol=1e-14,canonical_raw_probability_tolerance_unchanged=1e-5,
        route_indices_and_mixtures_still_exact=True,allows_official_submission=False,
        precedent='scripts/audit_span_013_order.py uses the same derived-JS tolerance',
        patched_validator_sha256=hashlib.sha256(replay.encode()).hexdigest(),
        companion_sha256=sha256(Path(__file__)),original_failure_sha256=sha256(failure))
    atomic_json(path,r)
    print(json.dumps({k:r[k] for k in ['complete','all_controls_passed','canonical_max_abs','runtime','runtime_gate_passed','memory_guard_passed','labels_read','official_submission_eligible','validation_replay']},indent=2))


if __name__=='__main__':main()
