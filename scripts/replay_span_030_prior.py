"""One predeclared descriptive comparison; no fitting or hidden-score forecast."""
import time
STARTED=time.monotonic()
from pathlib import Path
import sys
import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,CLASSES,parse_turns,prompt_groups
from arena.experiment import atomic_json,read_json,sha256,utc_now
from arena.span010_order import ensemble,validate_probabilities
from arena.span019_quality import compare
from arena.span030_mixture import two_expert_prior


def normalized(p):
    p=np.asarray(p,dtype=float)
    validate_probabilities(p,len(p))
    return p/p.sum(axis=1,keepdims=True)


def main():
    pub=REPO/'results/span_030'; root=ROOT/'runs/span_030'
    assert not (pub/'quality_replay.json').exists() and not (pub/'preservation.json').exists()
    old=REPO/'results/span_028'; audit=read_json(old/'profile_audit.json')
    assert audit['profile_complete'] and audit['all_controls_passed']
    assert all(d==0 for d in audit['canonical_max_abs'].values())
    assert not audit['runtime_gate_passed'],'Three-role failure is the provenance of this distinct task'
    for n,h in audit['source_sha256'].items():
        assert sha256(REPO/('docs/GOAL_V28.md' if n=='GOAL.md' else n))==h,n
    job=ROOT/'runs/span_028/jobs/prefetch_profile'
    receipt=read_json(job/'retrieval.json')
    for n,h in receipt['artifact_sha256'].items():assert sha256(job/'output'/n)==h,n
    spec=read_json(ROOT/'runs/span_028/profile_spec.json')
    previous=read_json(old/'preservation.json'); snapshots=sorted((REPO/'backlog/versions').glob('*'))
    assert len(snapshots)==118
    for n,h in previous['user_notebook_sha256'].items():assert sha256(REPO/n)==h,n
    assert (Path.home()/'notebooks/llm_classification_finetuning').resolve()==REPO/'notebooks'
    preserve=dict(at=utc_now(),snapshot_sha256={str(p.relative_to(REPO)):sha256(p) for p in snapshots},
        user_notebook_sha256=previous['user_notebook_sha256'],goal_v28_sha256=sha256(REPO/'docs/GOAL_V28.md'))
    atomic_json(pub/'preservation.json',preserve)
    sources=['GOAL.md','docs/SPAN_030_PLAN.md','scripts/replay_span_030_prior.py',
        'src/arena/span030_mixture.py','src/arena/span019_quality.py','src/arena/data.py','src/arena/span010_order.py',
        'tests/test_span030_mixture.py']
    source_hashes={n:sha256(REPO/n) for n in sources}
    csv=ROOT/'raw/train.csv';assert sha256(csv)==spec['train_csv_sha256']
    # Human targets are read only after exact accepted source/artifact provenance.
    train=pd.read_csv(csv,usecols=['id','prompt',*CLASSES]).set_index('id')
    assert train[CLASSES].isin([0,1]).all().all() and train[CLASSES].sum(axis=1).eq(1).all()
    out=job/'output/prefetch_profile'; results={}; arrays={}; input_hashes={}
    for name in ['control','uniform','long']:
        assert time.monotonic()-STARTED<285
        paths={r:out/(r+'_'+name+'.npz') for r in ['m0','m3','ensemble']}
        data={r:np.load(p) for r,p in paths.items()}; ids=data['m0']['ids']
        assert ids.tolist()==spec['token_references']['m0'][name]['ids']
        for a in data.values():np.testing.assert_array_equal(a['ids'],ids)
        frame=train.loc[ids]; y=frame[CLASSES].to_numpy().argmax(axis=1)
        groups=prompt_groups(pd.DataFrame({'prompt':frame.prompt.map(parse_turns)})).to_numpy()
        base=normalized(ensemble(data['m0']['probabilities'],data['m3']['probabilities']))
        incumbent=normalized(data['ensemble']['probabilities'])
        candidate=two_expert_prior(data['m0']['probabilities'],data['m3']['probabilities'])
        results[name]=dict(rows=len(ids),prompt_groups=len(np.unique(groups)),
            class_counts=np.bincount(y,minlength=3).tolist(),
            removal=compare(y,incumbent,base,groups),
            weight_ablation=compare(y,base,candidate,groups),
            versus_incumbent=compare(y,incumbent,candidate,groups))
        arrays.update({name+'_ids':ids,name+'_labels':y,name+'_groups':groups,
            name+'_base':base,name+'_incumbent':incumbent,name+'_candidate':candidate})
        input_hashes.update({str(p.relative_to(ROOT)):sha256(p) for p in paths.values()})
    root.mkdir(parents=True,exist_ok=True); target=root/'fixed_prior_analysis.npz';assert not target.exists()
    np.savez_compressed(target,**arrays)
    elapsed=time.monotonic()-STARTED;assert elapsed<300
    gain=results['uniform']['versus_incumbent']['loss_reduction'];passes=bool(gain>=-.005)
    result=dict(at=utc_now(),complete=True,quality_variants_tested=1,expert_weights=[.4,.6],
        sequentially_selected_on_inspected_original_cohorts=True,cumulative_twoexpert_priors_tested=[[.6,.4],[.4,.6]],
        previous_trial_sha256=sha256(REPO/'results/span_029/quality_replay.json'),
        temperature=1,reverse_pass=False,route=False,row_normalization=True,
        cohorts=results,uniform_gain_vs_incumbent=gain,fixed_gross_regression_threshold=-.005,
        gross_regression_guard_passed=passes,local_positive_gain_required=False,
        original_training_overlap=True,independent_cv=False,hidden_score_forecast=False,
        labels_read=True,new_gpu_executions=0,official_submission_eligible=False,
        decision='prepare_actual_two_role_qualification' if passes else 'stop_gross_regression',
        elapsed_cpu_seconds=elapsed,cpu_cap_seconds=300,source_sha256=source_hashes,
        cached_input_sha256=input_hashes,train_csv_sha256=sha256(csv),
        parent_audit_sha256=sha256(old/'profile_audit.json'),
        analysis_sha256=sha256(target),preservation_sha256=sha256(pub/'preservation.json'))
    atomic_json(pub/'quality_replay.json',result)
    print(dict(decision=result['decision'],uniform_gain_vs_incumbent=gain,
        weight_gain=results['uniform']['weight_ablation']['loss_reduction'],
        paired_group95=results['uniform']['versus_incumbent']['paired_group95'],
        elapsed_cpu_seconds=elapsed,independent_cv=False,hidden_score_forecast=False))


if __name__=='__main__':main()
