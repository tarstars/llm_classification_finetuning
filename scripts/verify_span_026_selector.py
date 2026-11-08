"""Prove the deployable text-only selector matches all audited changed inputs."""
from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd
import transformers,tokenizers
from transformers import AutoTokenizer

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,read_json,sha256,utc_now
from arena.span026_runtime import prefix_mask


def main():
    started=time.monotonic();pub=REPO/'results/span_026';root=ROOT/'runs/span_026'
    assert not (pub/'selector_audit.json').exists(),'Preserve first selector proof'
    assert transformers.__version__=='4.42.3' and tokenizers.__version__=='0.19.1'
    original=read_json(pub/'coverage_audit.json');path=root/'context_coverage.npz'
    assert sha256(path)==original['output_sha256']['runs/span_026/context_coverage.npz']
    expected=np.load(path)
    csv=ROOT/'raw/train.csv';assert sha256(csv)==original['input_sha256']['raw/train.csv']
    df=pd.read_csv(csv,usecols=['id','prompt','response_a','response_b'])
    assert np.array_equal(df.id,expected['ids'])
    assets=ROOT/'runs/span_009/research/lmsys-checkpoints-0-0805'
    assert sha256(assets/'tokenizer.json')==original['tokenizer_sha256']
    tokenizer=AutoTokenizer.from_pretrained(assets,local_files_only=True)
    assert tokenizer.truncation_side=='right'
    actual=prefix_mask(df.to_dict('records'),tokenizer)
    elapsed=time.monotonic()-started
    assert original['elapsed_cpu_seconds']+elapsed<1200,'Aggregate CPU source cap exceeded'
    identical=np.array_equal(actual,expected['changed'])
    result=dict(at=utc_now(),rows=len(df),uses_targets=False,selector_matches_all_native_changed_inputs=identical,
        actual_selected_rows=int(actual.sum()),native_changed_rows=int(expected['changed'].sum()),
        false_positive_rows=int((actual&~expected['changed']).sum()),false_negative_rows=int((~actual&expected['changed']).sum()),
        elapsed_cpu_seconds=elapsed,aggregate_cpu_source_seconds=original['elapsed_cpu_seconds']+elapsed,
        cpu_cap_seconds=1200,new_gpu_executions=0,new_inference_executions=0,gpu_admitted=False,
        source_sha256={n:sha256(REPO/n) for n in ['scripts/verify_span_026_selector.py','src/arena/span026_runtime.py']},
        decision='prepare_actual_engineering_pilot' if identical else 'stop_selector_mismatch')
    atomic_json(pub/'selector_audit.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
