"""Evaluate the single frozen conditional correction; no inference or fitting."""
from pathlib import Path
import json
import sys
import time
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,read_json,sha256,utc_now,verify_sources
from arena.span019_quality import compare,losses
from arena.span022_conditional import route,correct


def main():
    started=time.monotonic();pub=REPO/'results/span_022';root=ROOT/'runs/span_022'
    assert not (pub/'conditional_screen.json').exists(),'Preserve the first fixed comparison'
    registry=read_json(REPO/'backlog/hypotheses.json')
    assert registry['active_version']==22 and next(v for v in registry['versions'] if v['number']==22)['focus_ids']==['H68']
    assert next(i for i in registry['items'] if i['id']=='H68')['status']=='running'
    names=['GOAL.md','docs/SPAN_022_PLAN.md','scripts/screen_span_022.py','src/arena/span022_conditional.py',
        'tests/test_span022_conditional.py','src/arena/span019_quality.py','results/span_021/futility_screen.json',
        'results/span_021/closure_audit.json','scripts/prepare_span_022.py',
        'results/span_022/preservation.json','results/span_022/variant_contract.json']
    sources={name:sha256(REPO/name) for name in names};verify_sources(REPO,sources)
    assert read_json(REPO/'results/span_021/closure_audit.json')['passed']
    prior=read_json(REPO/'results/span_021/futility_screen.json')
    relative=next(iter(prior['output_sha256']));path=ROOT/relative
    assert sha256(path)==prior['output_sha256'][relative]
    a=np.load(path);ids=a['ids'];baseline=a['baseline'];mirror=a['mirror']
    assert len(ids)==512 and len(np.unique(ids))==512
    lengths_path=ROOT/'runs/span_020/jobs/batch_candidate/output/mirror_experiment/mirror_uniform.npz'
    assert sha256(lengths_path)==prior['model_output_sha256'][str(lengths_path.relative_to(ROOT))]
    original=np.load(lengths_path);assert np.array_equal(original['ids'],ids)
    np.testing.assert_array_equal(original['probabilities'],mirror)
    selected,score,budget=route(ids,baseline,original['lengths'])
    candidate=correct(baseline,mirror,selected)
    root.mkdir(parents=True,exist_ok=True);pub.mkdir(parents=True,exist_ok=True)
    manifest=root/'route.json';assert not manifest.exists()
    atomic_json(manifest,dict(at=utc_now(),selected_indices=selected.tolist(),selected_ids=ids[selected].tolist(),
        budget=budget,uses_labels=False,uses_mirror_predictions_for_routing=False,source_sha256=sources))
    # Route and its immutable manifest precede access to the target vector.
    y=a['labels'];groups=a['groups'];comparison=compare(y,baseline,candidate,groups)
    delta=losses(y,baseline)-losses(y,candidate)
    assert np.all(delta[y==2]==0)
    target=root/'conditional_uniform_analysis.npz';assert not target.exists()
    np.savez_compressed(target,ids=ids,labels=y,groups=groups,baseline=baseline,candidate=candidate,
        mirror=mirror,selected_indices=selected,route_scores=score,paired_loss_reduction=delta)
    passed=bool(comparison['loss_reduction']>.001);elapsed=time.monotonic()-started;assert elapsed<300
    result=dict(at=utc_now(),complete_original_screen=True,rows=len(ids),candidate_variants=1,
        conditional_mirror_weight=.25,temperature=1,budget=budget,maximum_eligible_tokens=512,
        comparison=comparison,original_information_gate_passed=passed,
        decision='inconclusive_needs_joint_later_qualification' if passed else 'stop_conditional_correction',
        full_quality_gate_passed=None if passed else False,official_submission_eligible=False,
        tie_probabilities_exact=True,offroute_predictions_exact=True,ab_mass_preserved=True,
        all_true_tie_loss_differences_zero=True,routing_uses_labels=False,routing_uses_mirror_predictions=False,
        original_training_overlap=True,previously_inspected_data=True,independent_competition_cv=False,
        hidden_score_forecast=False,later_comparison_complete=False,later_prediction_rows_available=0,
        fitted_parameters=0,new_gpu_executions=0,new_inference_executions=0,network_calls=0,
        elapsed_cpu_seconds=elapsed,source_sha256=sources,
        input_sha256={str(path.relative_to(ROOT)):sha256(path),str(lengths_path.relative_to(ROOT)):sha256(lengths_path)},
        output_sha256={str(p.relative_to(ROOT)):sha256(p) for p in [manifest,target]})
    atomic_json(pub/'conditional_screen.json',result)
    print(json.dumps({k:result[k] for k in ['budget','comparison','original_information_gate_passed',
        'decision','official_submission_eligible','elapsed_cpu_seconds']},indent=2))


if __name__=='__main__':main()
