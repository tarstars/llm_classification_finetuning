"""One original-data futility comparison; unavailable later data stays unknown."""
from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,parse_turns,prompt_groups
from arena.experiment import atomic_json,read_json,sha256,utc_now,verify_sources
from arena.span010_order import CLASSES
from arena.span019_quality import compare,losses,metrics


def main():
    started=time.monotonic();pub=REPO/'results/span_021';root=ROOT/'runs/span_021'
    assert not (pub/'futility_screen.json').exists(),'Preserve the first fixed comparison'
    registry=read_json(REPO/'backlog/hypotheses.json')
    assert registry['active_version']==21 and next(v for v in registry['versions'] if v['number']==21)['focus_ids']==['H67']
    assert next(i for i in registry['items'] if i['id']=='H67')['status']=='running'
    names=['GOAL.md','docs/SPAN_021_PLAN.md','scripts/screen_span_021.py','src/arena/span019_quality.py',
        'src/arena/data.py','results/span_020/closure_audit.json','results/span_020/partial_engineering_audit.json',
        'results/span_020/evaluation_selection.json','results/span_020/batch_shape_drift.json']
    sources={name:sha256(REPO/name) for name in names};verify_sources(REPO,sources)
    old=REPO/'results/span_020';audit=read_json(old/'partial_engineering_audit.json')
    assert read_json(old/'closure_audit.json')['passed'] and audit['service_status']=='ERROR'
    assert audit['original_engineering_checks_passed'] and not audit['full_recipe_complete']
    assert not audit['source1024_complete'] and not audit['actual_source_reverse_complete']
    job=ROOT/'runs/span_020/jobs/batch_candidate';retrieval=read_json(job/'retrieval.json')
    for name,digest in retrieval['artifact_sha256'].items():assert sha256(job/'output'/name)==digest
    out=job/'output/mirror_experiment'
    assert not (out/'source_inference/mirror_source.npz').exists()
    assert not (out/'parent_reverse/m0_reverse_routed_long.npz').exists()
    old_job=ROOT/'runs/span_013/jobs/gemma_order'
    baseline_path=old_job/'output/order_completion/ensemble_uniform.npz'
    assert sha256(baseline_path)==read_json(old_job/'retrieval.json')['artifact_sha256']['order_completion/ensemble_uniform.npz']
    arrays={name:np.load(out/(name+'_uniform.npz')) for name in ['m0','mirror','ensemble']}
    arrays['baseline']=np.load(baseline_path)
    selection=read_json(old/'evaluation_selection.json');ids=np.asarray(selection['uniform_ids'])
    assert len(ids)==512
    for a in arrays.values():assert np.array_equal(a['ids'],ids)
    g=arrays['m0']['probabilities'].astype(float);m=arrays['mirror']['probabilities'].astype(float)
    candidate=arrays['ensemble']['probabilities'];baseline=arrays['baseline']['probabilities']
    np.testing.assert_allclose(candidate,.5*g+.5*m,rtol=0,atol=1e-12)
    # Every input/control check precedes the first vote read.
    assert sha256(ROOT/'raw/train.csv')==selection['train_csv_sha256']
    frame=pd.read_csv(ROOT/'raw/train.csv',usecols=['id','prompt',*CLASSES]).set_index('id',drop=False).loc[ids].copy()
    assert frame[CLASSES].isin([0,1]).all().all() and frame[CLASSES].sum(1).eq(1).all()
    y=frame[CLASSES].to_numpy().argmax(1);frame['prompt']=frame.prompt.map(parse_turns)
    groups=prompt_groups(frame).to_numpy(dtype=str);comparison=compare(y,baseline,candidate,groups)
    original_pass=bool(comparison['loss_reduction']>.001)
    root.mkdir(parents=True,exist_ok=True);pub.mkdir(parents=True,exist_ok=True)
    path=root/'original_uniform_analysis.npz';assert not path.exists()
    np.savez_compressed(path,ids=ids,labels=y,groups=groups,baseline=baseline,candidate=candidate,
        gemma=g,mirror=m,paired_loss_reduction=losses(y,baseline)-losses(y,candidate))
    elapsed=time.monotonic()-started;assert elapsed<300
    result=dict(at=utc_now(),complete_original_screen=True,rows=512,unique_prompt_groups=len(set(groups)),
        primary_candidate_weights=[.5,.5],temperature=1,candidate_variants=1,fitted_parameters=0,
        comparison=comparison,descriptive_experts=dict(Gemma=metrics(y,g),Mirror=metrics(y,m)),
        original_information_gate_passed=original_pass,
        decision='inconclusive_needs_later_completion' if original_pass else 'stop_fixed_replacement',
        full_quality_gate_passed=None if original_pass else False,later_comparison_complete=False,
        later_prediction_rows_available=0,later_reverse_complete=False,official_submission_eligible=False,
        original_training_overlap=True,independent_competition_cv=False,hidden_score_forecast=False,
        new_gpu_executions=0,new_inference_executions=0,network_calls=0,elapsed_cpu_seconds=elapsed,
        source_sha256=sources,input_original_csv_sha256=selection['train_csv_sha256'],
        output_sha256={str(path.relative_to(ROOT)):sha256(path)},
        model_output_sha256={str((out/(name+'_uniform.npz')).relative_to(ROOT)):sha256(out/(name+'_uniform.npz')) for name in ['m0','mirror','ensemble']},
        baseline_sha256=sha256(baseline_path))
    atomic_json(pub/'futility_screen.json',result)
    print(json.dumps({k:result[k] for k in ['rows','comparison','descriptive_experts','original_information_gate_passed',
        'decision','later_comparison_complete','official_submission_eligible','elapsed_cpu_seconds']},indent=2))


if __name__=='__main__':main()
