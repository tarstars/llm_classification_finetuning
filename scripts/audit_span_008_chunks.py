"""Independently reconcile compatible chunks, coverage, overlap and device telemetry."""
from pathlib import Path
import sys,time
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,sha256,utc_now
from arena.prediction_chunks import ChunkStore,object_digest


def main():
    start=time.monotonic();root=ROOT/'runs/span_008';pub=REPO/'results/span_008';cfg=read_json(REPO/'configs/span_008.json');inv=read_json(pub/'artifact_inventory.json');selection=read_json(root/'sample_ids.json')
    assert not (pub/'chunk_audit.json').exists();checks={}
    for role in ['serial_cascade','context_completion']:
        job=root/'jobs'/role;out=job/'output'/role;state=read_json(job/'controller_state.json');assert state['stage'] in ['complete','failed']
        receipt=read_json(job/'retrieval.json')
        for file,digest in receipt['artifact_sha256'].items():assert sha256(job/'output'/file)==digest
        if not (out/'chunks/manifest.json').exists():
            checks[role]={'chunk_store_created':False,'full_coverage':False,'service_status':state['status']};continue
        m=read_json(out/'chunks/manifest.json');contract=m['contract'];bundle=read_json(pub/(role+'_bundle.json'))
        if role=='context_completion':
            assert contract['adapter_sha256']==sha256(inv['input_artifacts']['adapter']['path'])
            assert contract['tokenizer_sha256']==sha256(Path(inv['input_artifacts']['adapter']['path']).parent/'tokenizer.json')
            assert contract['packing_sha256']==sha256(REPO/'src/arena/joint_data.py')
            assert contract['source_sha256']==bundle['source_sha256']['src/arena/span008_classifier.py']
            assert contract['batch_per_gpu']==8 and contract['length']==2048 and contract['parent_receipt_sha256']==inv['context_parent_receipt_sha256']
            ids=selection['context_missing'];assert contract['ids_sha256']==object_digest(ids)
        else:
            assert contract['model_sha256']=='4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5'
            assert contract['client_slots']==1 and contract['cache_prompt'] is False
            assert contract['source_sha256']==bundle['source_sha256']['src/arena/span008_serial.py'];ids=selection['judge_cache'];assert contract['ids']==ids
        store=ChunkStore(out/'chunks',contract);covered=[];starts=sorted(map(int,m['chunks']))
        for begin in starts:
            assert begin==len(covered);n=m['chunks'][str(begin)]['rows'];raw=store.read(begin,ids[begin:begin+n]);covered.extend(raw['ids'].tolist())
        full=covered==ids;record={'chunk_store_created':True,'chunks':len(starts),'verified_rows':len(covered),'expected_rows':len(ids),'full_coverage':full,'contract_sha256':m['contract_sha256'],'service_status':state['status']}
        if full:
            assembled=store.assemble(starts,ids);target='completed_long.npz' if role=='context_completion' else 'judge_cache.npz';saved=np.load(out/target)
            for key in ['ids','original']+(['aligned_swap'] if role=='context_completion' else []):np.testing.assert_array_equal(assembled[key],saved[key])
            record['aggregate_arrays_equal_chunks']=True
        if role=='context_completion' and full:
            ref=np.load(root/'context_overlap_reference.npz');overlap=np.load(out/'overlap.npz');np.testing.assert_array_equal(ref['ids'],overlap['ids'])
            delta=np.abs(np.r_[ref['original']-overlap['original'],ref['aligned_swap']-overlap['aligned_swap']]);limits=cfg['raw_identity_limits']
            assert delta.mean()<=limits['mean'] and delta.max()<=limits['maximum']
            record['overlap']={'rows':len(ref['ids']),'mean_probability_difference':float(delta.mean()),'max_probability_difference':float(delta.max()),'passed':True}
            hardware=[]
            for rank in [0,1]:
                stages=[read_json(out/f'{phase}_gpu_rank{rank}.json') for phase in ['smoke','overlap','long_0','long_128','unchanged_profile']]
                assert all(h['rank']==rank and h['device']==rank and 'T4' in h['name'] and h['peak_allocated_bytes']>0 for h in stages)
                assert [h['peak_allocated_bytes'] for h in stages]==sorted(h['peak_allocated_bytes'] for h in stages)
                hardware.append(read_json(out/f'gpu_rank{rank}.json'))
            record['per_rank_final_peak_memory']=hardware;record['scope']='Whole worker per rank, including model loading; not isolated allocation per chunk'
            old=read_json(REPO/'results/span_007/context_partial_audit.json');oldout=ROOT/'runs/span_007/jobs/context/output/context';oldids=[]
            for file,digest in sorted(old['coverage']['2048']['prediction_files_sha256'].items()):assert sha256(oldout/file)==digest;oldids.extend(np.load(oldout/file)['ids'].tolist())
            np.testing.assert_array_equal(oldids+covered,selection['context_all']);record['stitched_long_rows']=len(oldids)+len(covered)
        checks[role]=record
    result={'passed':True,'checked_at':utc_now(),'checks':checks,'competition_targets_read':False,'new_gpu_executions':0,'seconds':time.monotonic()-start,
        'interruption_test':'Uncommitted temporary ignored; resume cache hit performs zero fake inference calls; changed source contract, corruption, reordered IDs, incomplete coverage and overwrites are rejected',
        'recovery_scope':'Aligned atomic chunks and compatible partial reuse. No automatic retry of a failed GPU version.',
        'source_sha256':sha256(Path(__file__)),'helper_sha256':sha256(REPO/'src/arena/prediction_chunks.py')}
    atomic_json(pub/'chunk_audit.json',result);print(result)


if __name__=='__main__':main()
