"""Freeze identical-pair IDs and prove author-tokenizer identity, without labels."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import time

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,read_json,sha256,utc_now


def main():
    started=time.monotonic();pub=REPO/'results/span_025';root=ROOT/'runs/span_025'
    pub.mkdir(parents=True,exist_ok=True);root.mkdir(parents=True,exist_ok=True)
    assert not (pub/'cohort_audit.json').exists(),'Preserve first token/coverage audit'
    r=read_json(REPO/'backlog/hypotheses.json');assert r['active_version']==25
    assert next(v for v in r['versions'] if v['number']==25)['focus_ids']==['H71']
    csv=ROOT/'raw/train.csv'
    assert sha256(csv)==read_json(REPO/'results/span_009/profile_selection.json')['train_csv_sha256']
    df=pd.read_csv(csv,usecols=['id','prompt','response_a','response_b']);assert df.id.is_unique
    selected=df.loc[df.response_a==df.response_b].sort_values('id').reset_index(drop=True)
    assert len(selected)==275
    control_ids=read_json(REPO/'results/span_009/profile_selection.json')['uniform_ids']
    control=df.set_index('id').loc[control_ids].reset_index()
    manifest=root/'cohort_manifest.json'
    selection=dict(uses_targets=False,full_original_rows=len(df),
        identical_ids=selected.id.tolist(),control_ids=control_ids,warmup_ids=control_ids[:80],
        train_csv_sha256=sha256(csv),rule='exact raw response_a == response_b',
        fixed_probabilities=[.06,.06,.88])
    if manifest.exists():
        saved=read_json(manifest)
        assert all(saved[k]==v for k,v in selection.items()),'Never replace a frozen selection'
    else:atomic_json(manifest,dict(at=utc_now(),**selection))
    module=ROOT/'runs/span_009/research/lmsys-modules-0805/data/processors.py'
    spec=importlib.util.spec_from_file_location('author_pab_span025',module)
    author=importlib.util.module_from_spec(spec);spec.loader.exec_module(author)
    reference=ROOT/'runs/span_013/jobs/gemma_order/output/order_completion'
    roles={};output={}
    for role in ['m0','m3']:
        asset=ROOT/'runs/span_009/research'/('lmsys-checkpoints-0-0805' if role=='m0' else 'lmsys-checkpoints-3-0805')
        expected=read_json(reference/(role+'_checkpoint.json'))
        for name in ['config.json','tokenizer.json','tokenizer_config.json','model.safetensors.index.json']:
            assert sha256(asset/name)==expected[name]
        tokenizer=AutoTokenizer.from_pretrained(asset,local_files_only=True)
        processor=author.ProcessorPAB(tokenizer,4096,role=='m3')
        def encode(record,swap=False):
            row=dict(record)
            if swap:row['response_a'],row['response_b']=row['response_b'],row['response_a']
            row.update(winner_model_a=1,winner_model_b=0,winner_tie=0)
            return processor.build_input(row)['input_ids'].numpy()
        control_hash=hashlib.sha256()
        for row in control.to_dict('records'):control_hash.update(encode(row,role=='m3').tobytes())
        original=read_json(reference/(role+'_summary.json'))['cohorts']['control']
        if control_hash.hexdigest()!=original['token_ids_sha256']:
            import transformers,tokenizers
            failure=pub/'local_control_token_failure.json'
            if not failure.exists():
                atomic_json(failure,dict(at=utc_now(),role=role,stage='canonical_control_token_identity',
                    expected=original['token_ids_sha256'],observed=control_hash.hexdigest(),
                    transformers=transformers.__version__,tokenizers=tokenizers.__version__,
                    tokenizer_file_matches_H60=True,uses_targets=False,new_gpu_executions=0,
                    decision='stop_before_gpu_and_check_exact_native_tokenizer_runtime'))
        assert control_hash.hexdigest()==original['token_ids_sha256'],role+' canonical control token mismatch'
        fingerprint=hashlib.sha256();lengths=[]
        for row in selected.to_dict('records'):
            tokens=encode(row);swapped=encode(row,True)
            assert np.array_equal(tokens,swapped),'Identical strings produced changed swapped tokens'
            fingerprint.update(tokens.tobytes());lengths.append(len(tokens))
        output[role+'_lengths']=np.asarray(lengths,dtype=np.int32)
        roles[role]=dict(control_token_ids_sha256=control_hash.hexdigest(),control_identity=True,
            identical_token_ids_sha256=fingerprint.hexdigest(),all_swapped_tokens_identical=True,
            rows=len(selected),mean_tokens=float(np.mean(lengths)),max_tokens=int(max(lengths)),
            tokenizer_sha256=expected['tokenizer.json'],checkpoint_record_sha256=sha256(reference/(role+'_checkpoint.json')))
    target=root/'identical_lengths.npz';assert not target.exists()
    np.savez_compressed(target,ids=selected.id.to_numpy(),**output)
    assert time.monotonic()-started<300
    users=read_json(REPO/'results/span_024/preservation.json')['user_notebook_sha256']
    for name,digest in users.items():assert sha256(REPO/name)==digest
    snapshots=list((REPO/'backlog/versions').glob('*'));assert len(snapshots)==98
    atomic_json(pub/'preservation.json',dict(at=utc_now(),snapshot_sha256={str(p.relative_to(REPO)):sha256(p) for p in snapshots},user_notebook_sha256=users))
    result=dict(at=utc_now(),preparation_passed=True,gpu_admitted=False,uses_targets=False,
        rows=len(selected),full_original_rows=len(df),prevalence=len(selected)/len(df),roles=roles,
        elapsed_cpu_seconds=time.monotonic()-started,processor_sha256=sha256(module),
        input_sha256={'raw/train.csv':sha256(csv)},
        output_sha256={str(p.relative_to(ROOT)):sha256(p) for p in [manifest,target]},
        fixed_probabilities=[.06,.06,.88],new_gpu_executions=0,new_inference_executions=0,
        candidate_quality_measured=False,official_submission_eligible=False)
    atomic_json(pub/'cohort_audit.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
