"""Reference native reward tokens for the replacement, preserving the first audit."""
from pathlib import Path
import sys
import json
import time

import numpy as np
import pandas as pd
import transformers
from transformers import AutoTokenizer

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,read_json,sha256,utc_now
from arena.span016_reward import conversation,encode_conversation


def main():
    assert transformers.__version__=='4.52.3'
    begin=time.monotonic();root=ROOT/'runs/span_016';pub=REPO/'results/span_016'
    assert not (pub/'matched_token_reference.json').exists()
    cohort=read_json(pub/'matched_cohort_manifest.json');path=root/'cohort_matched/source.parquet'
    assert sha256(path)==cohort['artifact_sha256']['source.parquet']
    frame=pd.read_parquet(path);tokenizer=AutoTokenizer.from_pretrained(root/'source_audit/8B',local_files_only=True)
    lengths,originals,hashes=[],[],[]
    for row in frame.to_dict('records'):
        for side in ['a','b']:
            ids,meta=encode_conversation(tokenizer,conversation(row,side))
            lengths.append(len(ids));originals.append(meta['original_tokens']);hashes.append(meta['token_ids_sha256'])
    lengths=np.asarray(lengths).reshape(-1,2);originals=np.asarray(originals).reshape(-1,2)
    out=root/'token_audit_matched';out.mkdir()
    target=out/'expected_long.npz'
    np.savez_compressed(target,ids=frame.id.to_numpy(),lengths=lengths,original_lengths=originals,
        token_hashes=np.asarray(hashes).reshape(-1,2))
    baseline=read_json(pub/'token_reference.json')['cohorts']['uniform']
    result=dict(at=utc_now(),rows=len(frame),mean_pair_tokens=float(lengths.sum(1).mean()),
        mean_original_pair_tokens=float(originals.sum(1).mean()),truncated_sequences=int((originals>lengths).sum()),
        retained_token_fraction=float(lengths.sum()/originals.sum()),
        source_to_target_reward_work_ratio=float(lengths.sum(1).mean()/baseline['mean_pair_tokens']),
        transformer_version=transformers.__version__,uses_targets=False,actual_reward_gpu_speed_measured=False,
        artifact_sha256={'expected_long.npz':sha256(target)},elapsed_cpu_seconds=time.monotonic()-begin,
        initial_reference_preserved=True,source_sha256=sha256(Path(__file__)),new_gpu_executions=0)
    atomic_json(pub/'matched_token_reference.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
