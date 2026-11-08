"""Bounded cached population/retention audit; no fitting or new inference."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
STARTED=time.monotonic()

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,parse_turns,prompt_groups
from arena.experiment import read_json,atomic_json,sha256,utc_now,verify_sources
from arena.span010_order import CLASSES,TEXT,ensemble
from arena.span011_calibration import temperature_scale
from arena.span017_transport import population,disagreement,score_implied_term


def main():
    begin=STARTED;pub=REPO/'results/span_017';root=ROOT/'runs/span_017'
    plan=read_json(pub/'span_plan.json');registry=read_json(REPO/'backlog/hypotheses.json')
    assert registry['active_version']==17
    assert next(v for v in registry['versions'] if v['number']==17)['focus_ids']==['E17','H62']
    assert next(i for i in registry['items'] if i['id']=='E17')['status']=='running'
    assert not (pub/'E17_transport.json').exists(),'Preserve the fixed diagnostic output'
    assert not plan['E17_fitting_allowed']
    temperature=read_json(REPO/'results/span_011/frozen_temperature.json')['temperature']
    assert temperature==plan['E17_existing_temperature']==1.3161518671742531
    sources={n:sha256(REPO/n) for n in ['GOAL.md','docs/SPAN_017_PLAN.md',
        'results/span_017/span_plan.json','scripts/analyze_span_017_transport.py',
        'src/arena/span017_transport.py','src/arena/span011_calibration.py','src/arena/span010_order.py']}
    verify_sources(REPO,sources)
    torch.set_num_threads(2)
    research=ROOT/'runs/span_009/research'
    checkpoint=research/'lmsys-checkpoints-0-0805'
    processor_path=research/'lmsys-modules-0805/data/processors.py'
    assert sha256(processor_path)==read_json(REPO/'results/span_009/full_pab_length_audit.json')['processor_sha256']
    original=ROOT/'runs/span_009/jobs/public_ensemble_resume/output/public_ensemble'
    assert sha256(checkpoint/'tokenizer.json')==read_json(original/'m0_checkpoint.json')['tokenizer.json']
    spec=importlib.util.spec_from_file_location('span017_author_processor',processor_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True)
    capped=module.ProcessorPAB(tokenizer,4096,False)
    uncapped=module.ProcessorPAB(tokenizer,1_000_000,False)
    def encode(processor,row):
        return processor.build_input({**{c:row[c] for c in TEXT},
            'winner_model_a':1,'winner_model_b':0,'winner_tie':0})['input_ids'].numpy()
    raw_hash=sha256(ROOT/'raw/train.csv')
    assert raw_hash==read_json(REPO/'results/span_010/profile_selection.json')['train_csv_sha256']
    train=pd.read_csv(ROOT/'raw/train.csv',usecols=['id',*TEXT,*CLASSES]).set_index('id',drop=False)
    hashes={'raw/train.csv':raw_hash};cohorts={};groups_by_name={};output_hashes={}
    destination=root/'transport';destination.mkdir(exist_ok=True)
    def digest(path):
        value=sha256(path);hashes[str(path.relative_to(ROOT))]=value;return value
    def add(name,job,suffix,kind,frame_path=None,label_path=None):
        assert time.monotonic()-begin<plan['cpu_seconds_caps']['E17']
        retrieval=read_json(job/'retrieval.json')['artifact_sha256']
        dirname='calibration' if 'calibration_' in job.name else 'reward_experiment' if job.name=='reward_pilot' else 'order_completion'
        folder=job/'output'/dirname
        a,b=[np.load(folder/(role+'_'+suffix+'.npz')) for role in ['m0','m3']]
        for role in ['m0','m3']:
            file=folder/(role+'_'+suffix+'.npz')
            assert digest(file)==retrieval[dirname+'/'+file.name]
        ids=a['ids'];np.testing.assert_array_equal(ids,b['ids'])
        if frame_path is None:
            frame=train.loc[ids].reset_index(drop=True)
            y=frame[CLASSES].to_numpy().argmax(1)
        else:
            digest(frame_path);digest(label_path)
            frame=pd.read_parquet(frame_path);targets=np.load(label_path)
            np.testing.assert_array_equal(ids,targets['ids'])
            y=targets['labels']
        np.testing.assert_array_equal(ids,frame.id)
        p=ensemble(a['probabilities'],b['probabilities'])
        if dirname=='calibration':
            stored=folder/('ensemble_'+suffix+'.npz');assert digest(stored)==retrieval[dirname+'/'+stored.name]
            np.testing.assert_array_equal(p,np.load(stored)['base_probabilities'])
        scaled=temperature_scale(p,temperature)
        actual,full,turns,nulls=[],[],[],[];fingerprint=hashlib.sha256()
        for row in frame.to_dict('records'):
            assert time.monotonic()-begin<plan['cpu_seconds_caps']['E17']
            prepared=encode(capped,row);all_tokens=encode(uncapped,row)
            fingerprint.update(prepared.tobytes());actual.append(len(prepared));full.append(len(all_tokens))
            arrays=[json.loads(row[c]) for c in TEXT]
            assert all(isinstance(values,list) for values in arrays)
            turns.append(max(map(len,arrays)));nulls.append(any(x is None for values in arrays for x in values))
        actual,full=np.asarray(actual),np.asarray(full)
        if kind=='later_votes':
            assert all(count==1 for count in turns),'Source single-turn filters must reproduce'
        np.testing.assert_array_equal(actual,a['lengths'])
        summary=read_json(folder/'m0_summary.json')['cohorts'][suffix]
        assert fingerprint.hexdigest()==summary['token_ids_sha256']
        assert np.all(actual<=full)
        group=prompt_groups(frame.assign(prompt=frame.prompt.map(parse_turns))).to_numpy()
        groups_by_name[name]=set(group.tolist())
        path=destination/(name+'.npz')
        np.savez_compressed(path,ids=ids,gemma_lengths=actual,gemma_full_lengths=full,
            base_confidence=p.max(1),base_entropy=-(np.clip(p,1e-15,1)*np.log(np.clip(p,1e-15,1))).sum(1))
        output_hashes[str(path.relative_to(ROOT))]=sha256(path)
        cohorts[name]=dict(rows=len(ids),exposure=kind,
            original_labels_seen_in_backbone_training=kind=='original_training_overlap',
            calibration_fit_population=name=='later_learning',independent_competition_cv=False,
            base=population(p,y,temperature),frozen_temperature=population(scaled,y,temperature),
            expert_disagreement=disagreement(a['probabilities'],b['probabilities'][:,[1,0,2]]),
            lengths=dict(mean_Gemma_prepared=float(actual.mean()),mean_Llama_prepared=float(b['lengths'].mean()),
                mean_Gemma_full=float(full.mean()),Gemma_retained_fraction=float(actual.sum()/full.sum()),
                truncated_rows=int((actual<full).sum()),multiturn_rows=int((np.asarray(turns)>1).sum()),
                null_message_rows=int(np.asarray(nulls).sum()),
                prepared_quantiles=np.quantile(actual,[0,.25,.5,.75,.9,1]).tolist()),
            CPU_GPU_prepared_tokens_identical=True)
        print(json.dumps(dict(stage='population_complete',population=name,rows=len(ids),seconds=time.monotonic()-begin)),flush=True)
    v11=ROOT/'runs/span_011'
    add('original_control',v11/'jobs/calibration_learning','control','original_training_overlap')
    add('original_uniform',ROOT/'runs/span_010/jobs/order_pilot','uniform','original_training_overlap')
    for stage in ['learning','confirmation']:
        label=v11/'cohort'/(stage+'_labels.npz')
        assert sha256(label)==read_json(REPO/'results/span_011/cohort_manifest.json')['label_sha256'][stage]
        add('later_'+stage,v11/('jobs/calibration_'+stage),stage,'later_votes',
            v11/'cohort'/(stage+'.parquet'),label)
    v16=ROOT/'runs/span_016';label=v16/'cohort_matched/labels.npz'
    assert sha256(label)==read_json(REPO/'results/span_016/matched_cohort_manifest.json')['artifact_sha256']['labels.npz']
    add('later_v16_matched',v16/'jobs/reward_pilot','long','later_votes',v16/'cohort_matched/source.parquet',label)
    official=read_json(REPO/'results/span_011/official_result.json')
    assert float(official['public_score'])==.84797
    implied=score_implied_term(.83053,.84797,temperature)
    overlap={a+' vs '+b:len(groups_by_name[a]&groups_by_name[b]) for i,a in enumerate(cohorts) for b in list(cohorts)[i+1:]}
    elapsed=time.monotonic()-begin;assert elapsed<plan['cpu_seconds_caps']['E17']
    result=dict(at=utc_now(),complete=True,cohorts=cohorts,frozen_temperature=temperature,
        historical_official_baseline_loss=.83053,historical_official_scaled_loss=.84797,
        official_score_implied_normalization_term=implied,normalized_prompt_overlaps=overlap,
        source_sha256=sources,input_sha256=hashes,output_sha256=output_hashes,
        processor_sha256=sha256(processor_path),tokenizer_sha256=sha256(checkpoint/'tokenizer.json'),
        elapsed_cpu_seconds=elapsed,new_parameter_fits=0,new_gpu_executions=0,new_submissions=0,
        independent_competition_cv=False,hidden_target_rows_accessed=0,
        decision='Treat later votes as qualified robustness evidence, not a target calibration surrogate. Original controls remain training-overlapping; no unique cause or new calibration is established.')
    atomic_json(pub/'E17_transport.json',result)
    print(json.dumps(dict(complete=True,elapsed_cpu_seconds=elapsed,official_implied_term=implied,new_parameter_fits=0)))


if __name__=='__main__':
    main()
