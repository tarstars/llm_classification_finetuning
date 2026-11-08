"""Exact native 4k/6k input coverage before labels or any GPU admission."""
from collections import defaultdict
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import time

import numpy as np
import pandas as pd
import transformers
import tokenizers
from transformers import AutoTokenizer

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,read_json,sha256,utc_now,verify_sources
from arena.span026_context import body_and_spans,retained_characters


def main():
    started=time.monotonic();pub=REPO/'results/span_026';root=ROOT/'runs/span_026'
    pub.mkdir(parents=True,exist_ok=True);root.mkdir(parents=True,exist_ok=True)
    assert not (pub/'coverage_audit.json').exists(),'Preserve the first source/coverage observation'
    assert transformers.__version__=='4.42.3' and tokenizers.__version__=='0.19.1'
    registry=read_json(REPO/'backlog/hypotheses.json');assert registry['active_version']==26
    assert next(v for v in registry['versions'] if v['number']==26)['focus_ids']==['H72']
    sources={n:sha256(REPO/n) for n in ['GOAL.md','docs/SPAN_026_PLAN.md',
        'scripts/audit_span_026_context.py','src/arena/span026_context.py','tests/test_span026_context.py']}
    verify_sources(REPO,sources)
    native=ROOT/'runs/span_009/research/lmsys-modules-0805/data/processors.py'
    assert sha256(native)==read_json(REPO/'results/span_025/cohort_audit.json')['processor_sha256']
    assets=ROOT/'runs/span_009/research/lmsys-checkpoints-0-0805'
    reference=ROOT/'runs/span_013/jobs/gemma_order/output/order_completion'
    checkpoint=read_json(reference/'m0_checkpoint.json')
    for name in ['config.json','tokenizer.json','tokenizer_config.json','model.safetensors.index.json']:
        assert sha256(assets/name)==checkpoint[name]
    model_config=read_json(assets/'config.json');position_limit=model_config['max_position_embeddings']
    spec=importlib.util.spec_from_file_location('native_pab_span026',native)
    author=importlib.util.module_from_spec(spec);spec.loader.exec_module(author)
    tokenizer=AutoTokenizer.from_pretrained(assets,local_files_only=True)
    assert tokenizer.truncation_side=='right','The declared prefix-coverage mechanism does not apply'
    old=author.ProcessorPAB(tokenizer,4096,False);new=author.ProcessorPAB(tokenizer,6144,False)
    csv=ROOT/'raw/train.csv';csv_hash=sha256(csv)
    assert csv_hash==read_json(REPO/'results/span_009/profile_selection.json')['train_csv_sha256']
    df=pd.read_csv(csv,usecols=['id','prompt','response_a','response_b']);assert df.id.is_unique
    previous=ROOT/'runs/span_009/analysis/full_pab_lengths.npz'
    assert sha256(previous)==read_json(REPO/'results/span_009/full_pab_length_audit.json')['artifact_sha256']
    cached=np.load(previous);assert np.array_equal(cached['ids'],df.id)
    control_ids=read_json(REPO/'results/span_013/profile_selection.json')['control_ids']
    def prepared(record):return dict(record,winner_model_a=1,winner_model_b=0,winner_tie=0)
    control_hash=hashlib.sha256()
    for row in df.set_index('id').loc[control_ids].reset_index().to_dict('records'):
        control_hash.update(old.build_input(prepared(row))['input_ids'].numpy().tobytes())
    expected=read_json(reference/'m0_summary.json')['cohorts']['control']['token_ids_sha256']
    assert control_hash.hexdigest()==expected
    snapshots=list((REPO/'backlog/versions').glob('*'));assert len(snapshots)==102
    users=read_json(REPO/'results/span_025/preservation.json')['user_notebook_sha256']
    for n,h in users.items():assert sha256(REPO/n)==h
    atomic_json(pub/'preservation.json',dict(at=utc_now(),snapshot_sha256={str(p.relative_to(REPO)):sha256(p) for p in snapshots},user_notebook_sha256=users))
    n=len(df);old_lengths=np.zeros(n,dtype=np.int32);new_lengths=old_lengths.copy()
    changed=np.zeros(n,dtype=bool);old_hash=[];new_hash=[];details=[]
    totals=defaultdict(int);old_stream=hashlib.sha256();new_stream=hashlib.sha256()
    for i,row in enumerate(df.to_dict('records')):
        a=old.build_input(prepared(row));b=new.build_input(prepared(row))
        x=a['input_ids'].numpy();z=b['input_ids'].numpy()
        assert len(x)==cached['lengths'][i],'Original cached length mismatch'
        assert len(z)<=position_limit,'New input exceeds model position limit'
        old_lengths[i]=len(x);new_lengths[i]=len(z);changed[i]=not np.array_equal(x,z)
        old_hash.append(hashlib.sha256(x.tobytes()).hexdigest());new_hash.append(hashlib.sha256(z.tobytes()).hexdigest())
        old_stream.update(x.tobytes());new_stream.update(z.tobytes())
        if changed[i]:
            body,spans=body_and_spans(row)
            tx=tokenizer(body,add_special_tokens=False,max_length=6144,truncation=True,return_offsets_mapping=True)
            expected_conversation='<|The Start of Conversation between a User and two Assistants|>'+tokenizer.decode(tx.input_ids[:4096])+'<|The End of Conversation between a User and two Assistants|>\n'
            assert old.build_conversation(*[json.loads(row[k]) for k in ['prompt','response_a','response_b']])==expected_conversation
            cuts=[max((end for _,end in tx.offset_mapping[:limit]),default=0) for limit in [4096,6144]]
            before,total=retained_characters(spans,cuts[0]);after,_=retained_characters(spans,cuts[1])
            for key in total:
                assert before[key]<=after[key]<=total[key]
                totals['total_'+key]+=total[key];totals['old_'+key]+=before[key];totals['new_'+key]+=after[key]
            totals['old_B_absent']+=int(before['response_b']==0 and total['response_b']>0)
            totals['new_B_absent']+=int(after['response_b']==0 and total['response_b']>0)
            totals['newly_visible_B']+=int(before['response_b']==0 and after['response_b']>0)
            details.append(dict(id=int(row['id']),old_chars=before,new_chars=after,total_chars=total))
        assert time.monotonic()-started<1200,'Original CPU elapsed cap exceeded'
        if (i+1)%5000==0:
            progress=dict(at=utc_now(),rows=i+1,total_rows=n,changed_rows=int(changed[:i+1].sum()),
                elapsed_seconds=time.monotonic()-started,uses_targets=False)
            atomic_json(pub/'progress.json',progress);print(json.dumps(progress),flush=True)
    target=root/'context_coverage.npz'
    np.savez_compressed(target,ids=df.id.to_numpy(),old_lengths=old_lengths,new_lengths=new_lengths,
        changed=changed,old_input_sha256=np.asarray(old_hash),new_input_sha256=np.asarray(new_hash))
    detail_path=root/'changed_field_coverage.json';atomic_json(detail_path,dict(uses_targets=False,rows=details))
    old_work=old_lengths.astype(float)*np.maximum(1,old_lengths/2048)
    new_work=new_lengths.astype(float)*np.maximum(1,new_lengths/2048)
    extra=float(new_work[changed].sum());ratio=extra/float(old_work.sum());count=int(changed.sum())
    eligible=bool(count>=64 and count/n<=.01 and ratio<=.15)
    changed_positions=np.flatnonzero(changed)
    selected=np.sort(np.random.default_rng(44).choice(changed_positions,min(384,count),replace=False))
    selected_ids=sorted(df.id.iloc[selected].tolist())
    manifest=root/'candidate_cohort_manifest.json'
    atomic_json(manifest,dict(at=utc_now(),uses_targets=False,seed=44,selected_ids=selected_ids,
        full_changed_rows=count,full_original_rows=n,conversation_tokens=6144,parent_tokens=4096,
        train_csv_sha256=csv_hash,coverage_sha256=sha256(target),cpu_feasibility_passed=eligible))
    near=old_lengths>=4300
    result=dict(at=utc_now(),complete=True,uses_targets=False,rows=n,changed_rows=count,
        changed_fraction=count/n,near_cap_proxy_rows=int(near.sum()),
        changed_rows_missed_by_near_cap_proxy=int((changed&~near).sum()),
        exact_original_lengths_reproduced=True,canonical_control_tokens_exact=True,
        old_input_stream_sha256=old_stream.hexdigest(),new_input_stream_sha256=new_stream.hexdigest(),
        parent_conversation_tokens=4096,candidate_conversation_tokens=6144,model_position_limit=position_limit,
        original_mean_tokens=float(old_lengths.mean()),candidate_changed_mean_tokens=float(new_lengths[changed].mean()) if count else None,
        candidate_max_tokens=int(new_lengths.max()),original_work_proxy=float(old_work.sum()),
        extra_full_pass_work_proxy=extra,extra_work_fraction_of_original_gemma=ratio,
        counts_full_extra_pass_not_only_increment=True,changed_character_coverage=dict(totals),
        field_coverage_measures_semantic_usefulness=False,cpu_feasibility_passed=eligible,
        decision='prepare_fixed_engineering_pilot' if eligible else 'stop_all_truncated_6k_variant',
        gpu_admitted=False,new_gpu_executions=0,new_inference_executions=0,quality_measured=False,
        elapsed_cpu_seconds=time.monotonic()-started,runtime=dict(transformers=transformers.__version__,tokenizers=tokenizers.__version__),
        source_sha256=sources,processor_sha256=sha256(native),tokenizer_sha256=sha256(assets/'tokenizer.json'),
        input_sha256={'raw/train.csv':csv_hash,str(previous.relative_to(ROOT)):sha256(previous)},
        output_sha256={str(p.relative_to(ROOT)):sha256(p) for p in [target,detail_path,manifest]})
    assert result['elapsed_cpu_seconds']<1200
    atomic_json(pub/'coverage_audit.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
