"""Post-submission label-free CPU length audit; require exact GPU token identity."""
from pathlib import Path
import hashlib
import importlib.util
import json
import resource
import sys
import time

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,sha256,utc_now


def main():
    start=time.monotonic();run=ROOT/'runs/span_009';pub=REPO/'results/span_009'
    model=run/'research/lmsys-checkpoints-0-0805'
    module=run/'research/lmsys-modules-0805/data/processors.py'
    prior=run/'jobs/public_ensemble_resume/output/public_ensemble'
    expected=json.loads((prior/'m0_checkpoint.json').read_text())
    assert sha256(model/'tokenizer.json')==expected['tokenizer.json']
    spec=importlib.util.spec_from_file_location('original_pab_processor',module)
    author=importlib.util.module_from_spec(spec);spec.loader.exec_module(author)
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    processor=author.ProcessorPAB(tokenizer,4096,False)
    frame=pd.read_csv(ROOT/'raw/train.csv',usecols=['id','prompt','response_a','response_b'])
    ids=json.loads((pub/'profile_selection.json').read_text())['uniform_ids']
    def encode(row):
        row.update(winner_model_a=1,winner_model_b=0,winner_tie=0)
        return processor.build_input(row)['input_ids'].numpy()
    fingerprint=hashlib.sha256()
    for row in frame.set_index('id').loc[ids].reset_index().to_dict('records'):
        fingerprint.update(encode(row).tobytes())
    reference=json.loads((prior/'m0_summary.json').read_text())['cohorts']['uniform']
    identity=fingerprint.hexdigest()==reference['token_ids_sha256']
    base=dict(at=utc_now(),post_submission_diagnostic=True,model_or_submission_changed=False,
        uses_targets=False,new_gpu_executions=0,uniform_token_ids_identical=identity,
        processor_sha256=sha256(module),tokenizer_sha256=expected['tokenizer.json'],
        script_sha256=sha256(Path(__file__)),train_sha256=sha256(ROOT/'raw/train.csv'))
    if not identity:
        atomic_json(pub/'full_pab_length_audit.json',dict(base,complete=False,
            reason='Local tokenizer/backend differs from accepted GPU tokens; do not extrapolate.'))
        print('Stopped: exact token fingerprint differs; no full-corpus claim.')
        return
    lengths=[]
    for n,row in enumerate(frame.to_dict('records')):
        lengths.append(len(encode(row)))
        if n%5000==0:
            print(json.dumps({'rows':n,'seconds':time.monotonic()-start}),flush=True)
        assert time.monotonic()-start<570,'Bounded CPU audit exceeded its original allowance'
    lengths=np.asarray(lengths,dtype=np.int32)
    target=run/'analysis';target.mkdir(exist_ok=True)
    artifact=target/'full_pab_lengths.npz'
    np.savez_compressed(artifact,ids=frame.id.to_numpy(),lengths=lengths)
    uniform=np.load(prior/'m0_uniform.npz')['lengths']
    result=dict(base,complete=True,rows=len(lengths),elapsed_seconds=time.monotonic()-start,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        full_mean_tokens=float(lengths.mean()),uniform_mean_tokens=float(uniform.mean()),
        full_to_uniform_mean_ratio=float(lengths.mean()/uniform.mean()),
        quantiles={str(q):float(np.quantile(lengths,q)) for q in [0,.25,.5,.75,.9,.95,.99,1]},
        near_context_cap_rows=int((lengths>=4300).sum()),
        near_cap_definition='Final packed input>=4300tokens; a proxy, not exact truncation detection.',
        artifact_sha256=sha256(artifact),original_runtime_gate_unchanged=True)
    atomic_json(pub/'full_pab_length_audit.json',result)
    print(json.dumps({k:result[k] for k in ['rows','elapsed_seconds','full_mean_tokens',
        'uniform_mean_tokens','full_to_uniform_mean_ratio','near_context_cap_rows','peak_rss_bytes']}))


if __name__=='__main__':main()
