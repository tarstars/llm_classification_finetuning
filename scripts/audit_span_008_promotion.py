"""Reconstruct qualification masks/chunks and independently replay the frozen contrast."""
from pathlib import Path
import json,resource,signal,sys,time
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,CLASSES,TEXT_COLUMNS,parse_turns,prompt_groups
from arena.experiment import read_json,atomic_json,sha256,utc_now,verify_sources
from arena.joint_data import JointTokenizer
from arena.context_route import changed_mask
from arena.prediction_chunks import ChunkStore,object_digest
from arena.span_heads import temperature,paired_group_gain


def main():
    start=time.monotonic();root=ROOT/'runs/span_008';pub=REPO/'results/span_008';job=root/'jobs/promotion';out=job/'output/promotion';cfg=read_json(REPO/'configs/span_008.json')
    assert not (pub/'promotion_audit.json').exists()
    def expired(*unused):raise TimeoutError('Original audit CPU cap reached')
    signal.signal(signal.SIGALRM,expired);signal.alarm(cfg['cpu_stage_seconds_caps']['analysis'])
    result=read_json(pub/'H50_results.json');assert result['confirmation_complete'];bundle=read_json(pub/'promotion_bundle.json');verify_sources(REPO,bundle['source_sha256'])
    inv=read_json(pub/'artifact_inventory.json');adapter=Path(inv['input_artifacts']['adapter']['path']).parent
    tok=AutoTokenizer.from_pretrained(str(adapter),local_files_only=True,padding_side='right');encoder_short=JointTokenizer(tok,1024);encoder_long=JointTokenizer(tok,2048)
    frames={'confirmation':pd.read_parquet(inv['input_artifacts']['confirmation_inputs']['path']),'test':pd.read_csv(ROOT/'raw/test.csv',usecols=['id']+TEXT_COLUMNS)}
    for col in TEXT_COLUMNS:frames['test'][col]=frames['test'][col].map(parse_turns)
    checks={}
    for name,frame in frames.items():
        saved=np.load(out/name/'predictions.npz');np.testing.assert_array_equal(saved['ids'],frame.id)
        so,ss,_=encoder_short.encode_frame(frame);lo,ls,_=encoder_long.encode_frame(frame);mask=changed_mask(so,ss,lo,ls);np.testing.assert_array_equal(saved['changed'],mask)
        chunks=0
        for branch,indices,length,batch in [('short',np.flatnonzero(~mask),1024,4),('long',np.flatnonzero(mask),2048,8)]:
            path=out/name/branch;m=read_json(path/'manifest.json');contract=m['contract'];ids=frame.id.iloc[indices].tolist()
            assert contract['ids_sha256']==object_digest(ids) and contract['length']==length and contract['batch_per_gpu']==batch
            assert contract['adapter_sha256']==sha256(adapter/'adapter_model.safetensors') and contract['tokenizer_sha256']==sha256(adapter/'tokenizer.json')
            assert contract['packing_sha256']==sha256(REPO/'src/arena/joint_data.py') and contract['route_sha256']==sha256(REPO/'src/arena/context_route.py')
            assert contract['source_sha256']==bundle['source_sha256']['src/arena/span008_context_export.py']
            starts=sorted(map(int,m['chunks']));chunks+=len(starts)
            if not len(indices):assert not starts;continue
            joined=ChunkStore(path,contract).assemble(starts,ids)
            for view in ['original','aligned_swap']:np.testing.assert_array_equal(joined[view],saved[view][indices])
            for rank in [0,1]:
                memory=[read_json(out/name/f'{branch}_{begin}_gpu_rank{rank}.json') for begin in starts]
                assert all(v['device']==v['rank']==rank and 'T4' in v['name'] and v['peak_allocated_bytes']>0 for v in memory)
                assert [v['peak_allocated_bytes'] for v in memory]==sorted(v['peak_allocated_bytes'] for v in memory)
        p=temperature((saved['original']+saved['aligned_swap'])/2,.9441963863130951);np.testing.assert_allclose(p,saved['probabilities'],atol=1e-6)
        checks[name]={'rows':len(frame),'changed_rows':int(mask.sum()),'verified_chunks':chunks,'route_reconstructed_from_tokens':True,'arrays_equal_measured_chunks':True,'both_rank_memory_verified':True}
    frame=frames['confirmation'];raw=np.load(out/'confirmation/predictions.npz');base=np.load(inv['input_artifacts']['confirmation_reference']['path']);np.testing.assert_array_equal(base['ids'],frame.id)
    labels=pd.read_csv(inv['input_artifacts']['raw_train']['path'],usecols=['id']+CLASSES).set_index('id').loc[frame.id];y=labels[CLASSES].to_numpy().argmax(1);groups=prompt_groups(frame).to_numpy(dtype=str)
    p=temperature((base['original']+base['aligned_swap'])/2,.9441963863130951);q=temperature((raw['original']+raw['aligned_swap'])/2,.9441963863130951)
    contrast=paired_group_gain(y,p,q,groups,dict(cfg,seed=cfg['bootstrap_seed']));assert contrast==result['contrast']
    expected=contrast['gain']>=cfg['confirmation_practical_gain'] and contrast['interval_95'][0]>0
    assert result['quality_gate_passed']==expected and not result['eligible_for_official']
    assert not (root/'official/submission_intent.json').exists(),'The failed quality gate must prevent an official mutation'
    audit={'passed':True,'checked_at':utc_now(),'checks':checks,'contrast_replayed':contrast,'quality_gate_passed':bool(expected),'no_official_intent':True,
        'fit_operations':0,'new_gpu_executions':0,'seconds':time.monotonic()-start,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        'source_sha256':sha256(Path(__file__)),'confirmation_reveal_intent_sha256':sha256(root/'confirmation_reveal_intent.json')}
    atomic_json(pub/'promotion_audit.json',audit);print(audit)


if __name__=='__main__':main()
