"""Seal native pilot streams/control files; preserve the completed corpus proof."""
from pathlib import Path
import hashlib
import importlib.util
import json
import shutil
import sys
import time
import numpy as np
import pandas as pd
import transformers, tokenizers
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT, TEXT_COLUMNS
from arena.experiment import read_json, atomic_json, sha256, utc_now
from arena.span035_recent import processor_class


def main():
    start = time.monotonic(); pub, run = REPO / 'results/span_035', ROOT / 'runs/span_035'
    assert not (pub / 'pilot_preparation.json').exists()
    assert transformers.__version__ == '4.42.3' and tokenizers.__version__ == '0.19.1'
    prep = read_json(pub / 'preparation.json'); assert prep['cpu_feasibility_passed']
    revision = read_json(pub / 'binding_revision.json')
    for name, digest in revision['binding_sources'].items(): assert sha256(REPO / name) == digest
    for name, digest in prep['source_sha256'].items():
        path = Path(revision['initial_proposal'][name]['path']) if name in revision['initial_proposal'] else REPO / name
        assert sha256(path) == digest, name
    raw = ROOT / 'raw/train.csv'; assert sha256(raw) == prep['raw_sha256']
    frame = pd.read_csv(raw, usecols=['id', *TEXT_COLUMNS]).set_index('id')
    module = ROOT / 'runs/span_009/research/lmsys-modules-0805/data/processors.py'
    assert sha256(module) == prep['processor_sha256']
    loader = importlib.util.spec_from_file_location('native_recent_pilot_ref', module)
    native = importlib.util.module_from_spec(loader); loader.loader.exec_module(native)
    asset = ROOT / 'runs/span_009/research/lmsys-checkpoints-0-0805'
    tokenizer = AutoTokenizer.from_pretrained(asset, local_files_only=True)
    original = native.ProcessorPAB(tokenizer, 4096, False)
    recent = processor_class(native.ProcessorPAB)(tokenizer, 4096, False)
    selection = read_json(REPO / 'results/span_013/profile_selection.json')
    ids = dict(warmup=selection['control_ids'][:80], control=selection['control_ids'],
        recent_warmup=prep['changed_ids'][:80], recent=prep['changed_ids'])
    references = {}
    for name, values in ids.items():
        stream = hashlib.sha256(); lengths = []
        processor = recent if name.startswith('recent') else original
        for row in frame.loc[values].reset_index().to_dict('records'):
            data = dict(row, winner_model_a=1, winner_model_b=0, winner_tie=0)
            tokens = processor.build_input(data)['input_ids'].numpy()
            stream.update(tokens.tobytes()); lengths.append(len(tokens))
        references[name] = dict(lengths=lengths, token_ids_sha256=stream.hexdigest())
    parent = ROOT / 'runs/span_013/jobs/gemma_order/output/order_completion'
    assert references['control']['token_ids_sha256'] == read_json(parent / 'm0_summary.json')['cohorts']['control']['token_ids_sha256']
    controls = run / 'controls'; controls.mkdir()
    shutil.copyfile(parent / 'm0_control.npz', controls / 'm0.npz')
    spec = dict(mode='pilot', uses_targets=False, packing_tokens=8192, train_csv_sha256=prep['raw_sha256'],
        frame_ids=ids, token_references=references, checkpoint_references=dict(m0=read_json(parent / 'm0_checkpoint.json')))
    atomic_json(run / 'pilot_spec.json', spec)
    elapsed = time.monotonic() - start
    assert prep['elapsed_seconds'] + elapsed < 1200
    result = dict(at=utc_now(), prepared=True, uses_targets=False, rows=len(ids['recent']),
        aggregate_cpu_preparation_seconds=prep['elapsed_seconds'] + elapsed, stage_seconds=elapsed,
        spec_sha256=sha256(run / 'pilot_spec.json'), control_sha256=sha256(controls / 'm0.npz'), new_gpu_executions=0)
    atomic_json(pub / 'pilot_preparation.json', result); print(result)


if __name__ == '__main__': main()
