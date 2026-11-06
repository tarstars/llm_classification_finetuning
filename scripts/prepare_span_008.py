"""Freeze v8 input-only cohorts and references; no confirmation vote access."""
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,sha256,utc_now


def main():
    out=ROOT/'runs/span_008';pub=REPO/'results/span_008';out.mkdir(parents=True,exist_ok=True)
    assert not (pub/'artifact_inventory.json').exists()
    cfg=read_json(REPO/'configs/span_008.json');initial=read_json(pub/'resource_inventory_initial.json')
    assert not initial['nonterminal_jobs'] and initial['quota']['remaining_seconds']>=10800 and not initial['quota']['paid_scaling']
    old=read_json(REPO/'results/span_007/artifact_inventory.json');entries=dict(old['input_artifacts'])
    for value in entries.values():assert sha256(value['path'])==value['sha256']
    dev=pd.read_parquet(entries['dev.parquet']['path'],columns=['id','prompt','response_a','response_b'])
    dev.to_parquet(out/'development_inputs.parquet',index=False)
    raw=np.load(entries['classifier']['path']);np.testing.assert_array_equal(raw['ids'],dev.id)
    p=(raw['original']+raw['aligned_swap'])/2;entropy=-(p*np.log(np.maximum(p,1e-12))).sum(1)
    rng=np.random.default_rng(cfg['seed']);draw=lambda pool,n:np.sort(rng.choice(np.asarray(pool),n,replace=False))
    all_rows=np.arange(len(dev));changed_ids=read_json(ROOT/'runs/span_007/sample_ids.json')['context']
    changed=set(changed_ids);unchanged=np.flatnonzero(~dev.id.isin(changed));route=np.argsort(-entropy,kind='stable')[:int(len(dev)*.1)]
    samples={'judge_cache':dev.id.iloc[draw(all_rows,2048)].tolist(),
             'judge_profile':dev.id.iloc[draw(route,128)].tolist(),
             'classifier_profile':dev.id.iloc[draw(all_rows,128)].tolist(),
             'unchanged_profile':dev.id.iloc[draw(unchanged,128)].tolist(),
             'context_all':changed_ids,'context_overlap':changed_ids[:128],'context_missing':changed_ids[896:],
             'judge_sentinel':read_json(ROOT/'runs/span_007/sample_ids.json')['judge']['sentinel']}
    assert len(samples['context_missing'])==256
    samples['judge_swap']=np.asarray(samples['judge_cache'])[draw(np.arange(2048),64)].tolist()
    lookup={int(i):n for n,i in enumerate(dev.id)}
    for name in ['classifier_profile','unchanged_profile']:
        indices=[lookup[int(i)] for i in samples[name]]
        np.savez(out/(name+'_reference.npz'),ids=np.array(samples[name]),original=raw['original'][indices],aligned_swap=raw['aligned_swap'][indices])
    shutil.copyfile(ROOT/'runs/span_007/smoke_reference.npz',out/'smoke_reference.npz')
    native=ROOT/'runs/span_007/jobs/judge_stability/output/judge_stability/serial/initial'
    shutil.copyfile(native/'predictions.npz',out/'judge_reference.npz')
    atomic_json(out/'judge_reference_tokens.json',read_json(native/'summary.json')['token_sha256_by_id'])
    shutil.copyfile(ROOT/'runs/span_007/jobs/context/output/context/2048_0000.npz',out/'context_overlap_reference.npz')
    atomic_json(out/'sample_ids.json',samples)
    local={name:{'path':str(out/name),'sha256':sha256(out/name)} for name in
        ['development_inputs.parquet','sample_ids.json','smoke_reference.npz','classifier_profile_reference.npz',
         'unchanged_profile_reference.npz','judge_reference.npz','judge_reference_tokens.json','context_overlap_reference.npz']}
    result={'span':8,'prepared_at':utc_now(),'input_artifacts':entries,'prepared_artifacts':local,
            'context_text_sha256':old['context_text_sha256'],'context_token_sha256':old['context_token_sha256'],
            'context_parent_receipt_sha256':sha256(ROOT/'runs/span_007/jobs/context/retrieval.json'),
            'development_rows':len(dev),'confirmation_targets_read':False,'preparation_competition_target_columns_loaded':False,
            'old_snapshot_sha256':{str(p.relative_to(REPO)):sha256(p) for p in (REPO/'backlog/versions').glob('*') if p.is_file()},
            'user_notebook_sha256':{p:sha256(REPO/p) for p in ['notebooks/07_error_poster.ipynb','notebooks/12_backlog_v1_span.ipynb']},
            'contract_sha256':{p:sha256(REPO/p) for p in ['GOAL.md','configs/span_008.json','docs/SPAN_008_PLAN.md']},
            'source_sha256':sha256(Path(__file__))}
    plan={k:cfg[k] for k in ['span','started_at','deadline','gpu_session_seconds_cap']}
    plan.update(initial_quota_refresh_time=initial['quota']['refresh_time'],quota_used_at_start=initial['quota']['used_seconds'])
    for folder in [out,pub]:atomic_json(folder/'artifact_inventory.json',result);atomic_json(folder/'span_plan.json',plan)
    print({'rows':len(dev),'sample_sizes':{k:len(v) for k,v in samples.items()},'confirmation_targets_read':False})


if __name__=='__main__':main()
