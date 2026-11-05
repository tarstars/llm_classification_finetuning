"""Audit saved partial H48 coverage without loading votes or estimating full quality."""
from pathlib import Path
import sys
import numpy as np

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,sha256,utc_now


def main():
    pub=REPO/'results/span_007';run=ROOT/'runs/span_007';job=run/'jobs/context';out=job/'output/context'
    assert not (pub/'context_partial_audit.json').exists()
    assert read_json(job/'controller_state.json')['stage']=='failed' and not (out/'summary.json').exists()
    assert read_json(pub/'jobs/context/source_verification.json')['passed']
    receipt=read_json(job/'retrieval.json')
    for file,digest in receipt['artifact_sha256'].items():assert sha256(job/'output'/file)==digest
    ids=read_json(run/'sample_ids.json')['context'];assert len(ids)==1152
    identity=read_json(out/'identity.json');assert identity['passed'] and identity['fixed_eval_batch_size_per_gpu']==8
    events=[read_json(out/'stages/latest_event.json')]
    import json
    events=[json.loads(line) for line in (out/'stages/events.jsonl').read_text().splitlines()]
    coverage={}
    for size in [1024,2048]:
        first=[];second=[];covered=[];files=sorted(out.glob(str(size)+'_*.npz'));expected_start=0
        for file in files:
            start=int(file.stem.split('_')[1]);assert start==expected_start
            values=np.load(file);end=start+len(values['ids']);np.testing.assert_array_equal(values['ids'],ids[start:end])
            for view in ['original','aligned_swap']:
                p=values[view];assert p.shape==(end-start,3) and np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
                np.testing.assert_allclose(p.sum(1),1,atol=1e-6)
            first.append(values['original']);second.append(values['aligned_swap']);covered.extend(values['ids']);expected_start=end
        assert len(covered)==len(set(covered))
        records=[e for e in events if e['stage']=='context_'+str(size) and e['event']=='chunk_completed']
        assert len(records)==len(files) and [r['rows'] for r in records]==list(range(128,len(covered)+1,128))
        aggregate=out/f'predictions_{size}.npz'
        if aggregate.exists():
            data=np.load(aggregate);np.testing.assert_array_equal(data['ids'],covered)
            for view,parts in [('original',first),('aligned_swap',second)]:np.testing.assert_array_equal(data[view],np.vstack(parts))
        seconds=sum(r['seconds'] for r in records)
        coverage[str(size)]={'rows':len(covered),'chunks':len(files),'completed_chunk_seconds':seconds,
                             'seconds_per_completed_row':seconds/len(covered),'aggregate_complete':len(covered)==len(ids) and aggregate.exists(),
                             'prediction_files_sha256':{file.name:sha256(file) for file in files}}
    paired=min(v['rows'] for v in coverage.values());assert paired<len(ids)
    result={'passed':True,'scope':'Integrity and coverage of saved chunks only; full quality and deployment are unqualified.',
            'checked_at':utc_now(),'expected_changed_rows':len(ids),'paired_complete_rows':paired,'missing_long_rows':len(ids)-paired,
            'coverage':coverage,'identity':identity,'competition_targets_read':False,'full_quality_estimated':False,
            'complete_worker_summary_available':False,'measured_final_peak_memory_available':False,
            'charged_gpu_seconds':read_json(job/'controller_state.json')['charged_seconds'],
            'source_sha256':sha256(Path(__file__)),'retrieval_sha256':sha256(job/'retrieval.json')}
    atomic_json(pub/'context_partial_audit.json',result);print({k:v for k,v in result.items() if k!='coverage'});print(coverage)


if __name__=='__main__':main()
