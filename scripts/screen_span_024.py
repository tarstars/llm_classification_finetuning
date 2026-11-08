"""One frozen published rule against cached H60; no model inference."""
from collections import Counter
from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,read_json,sha256,utc_now,verify_sources
from arena.span019_quality import compare,losses
from arena.span024_structural import EMPTY,categories,override


def main():
    started=time.monotonic();pub=REPO/'results/span_024';root=ROOT/'runs/span_024'
    assert not (pub/'structural_screen.json').exists(),'Preserve first comparison'
    registry=read_json(REPO/'backlog/hypotheses.json')
    assert registry['active_version']==24
    assert next(v for v in registry['versions'] if v['number']==24)['focus_ids']==['H70']
    assert next(i for i in registry['items'] if i['id']=='H70')['status']=='running'
    names=['GOAL.md','docs/SPAN_024_PLAN.md','scripts/screen_span_024.py',
        'src/arena/span024_structural.py','tests/test_span024_structural.py',
        'src/arena/span019_quality.py','results/span_021/futility_screen.json',
        'results/span_023/source_audit.json','results/span_023/closure_audit.json']
    sources={n:sha256(REPO/n) for n in names};verify_sources(REPO,sources)
    original=read_json(REPO/'results/span_021/futility_screen.json')
    relative=next(iter(original['output_sha256']));path=ROOT/relative
    assert sha256(path)==original['output_sha256'][relative]
    csv=ROOT/'raw/train.csv'
    assert sha256(csv)==read_json(REPO/'results/span_009/profile_selection.json')['train_csv_sha256']
    df=pd.read_csv(csv,usecols=['id','response_a','response_b']);assert df.id.is_unique
    full_category=categories(df.response_a,df.response_b)
    a=np.load(path);ids=a['ids'];assert len(ids)==512
    rows=df.set_index('id').loc[ids]
    baseline=a['baseline'];candidate,category=override(baseline,rows.response_a,rows.response_b)
    unchanged=category=='unchanged'
    assert np.array_equal(candidate[unchanged],baseline[unchanged])
    root.mkdir(parents=True,exist_ok=True);pub.mkdir(parents=True,exist_ok=True)
    manifest=root/'label_free_coverage.json';assert not manifest.exists()
    coverage=dict(at=utc_now(),uses_targets=False,full_rows=len(df),
        full_category_counts=dict(Counter(full_category.tolist())),sample_rows=len(ids),
        sample_category_counts=dict(Counter(category.tolist())),
        sample_affected_ids=ids[~unchanged].tolist(),source_sha256=sources)
    atomic_json(manifest,coverage)
    # Exact rule and coverage manifest precede target-vector access.
    y=a['labels'];groups=a['groups'];comparison=compare(y,baseline,candidate,groups)
    delta=losses(y,baseline)-losses(y,candidate)
    detail={}
    for c in np.unique(category):
        mask=category==c
        detail[str(c)]=dict(rows=int(mask.sum()),loss_reduction=float(delta[mask].mean()),
            label_counts=np.bincount(y[mask],minlength=3).tolist())
    target=root/'structural_uniform_analysis.npz';assert not target.exists()
    np.savez_compressed(target,ids=ids,labels=y,groups=groups,baseline=baseline,
        candidate=candidate,category=category,paired_loss_reduction=delta)
    passed=bool(comparison['loss_reduction']>.0005);elapsed=time.monotonic()-started;assert elapsed<300
    r=dict(at=utc_now(),complete_original_screen=True,candidate_variants=1,rows=512,
        source_rule_commit='f4d2a3fbd0497e28673010787612d3dac04e7668',
        empty_strings=list(EMPTY),precedence=['empty_a','empty_b','identical'],
        label_free_coverage=coverage,comparison=comparison,category_diagnostics=detail,
        original_information_gate_passed=passed,
        decision='needs_separate_actual_qualification' if passed else 'stop_published_structural_rule',
        offroute_predictions_exact=True,fitted_parameters=0,new_inference_executions=0,
        new_gpu_executions=0,network_calls=0,official_submissions=0,official_submission_eligible=False,
        original_training_overlap=True,previously_inspected_cohort=True,
        independent_competition_cv=False,hidden_score_forecast=False,elapsed_cpu_seconds=elapsed,
        source_sha256=sources,input_sha256={str(path.relative_to(ROOT)):sha256(path),
            str(csv.relative_to(ROOT)):sha256(csv)},
        output_sha256={str(p.relative_to(ROOT)):sha256(p) for p in [manifest,target]})
    atomic_json(pub/'structural_screen.json',r)
    print(json.dumps({k:r[k] for k in ['label_free_coverage','comparison','category_diagnostics',
        'original_information_gate_passed','decision','elapsed_cpu_seconds']},indent=2))


if __name__=='__main__':main()
