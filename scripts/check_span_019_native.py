"""Check declared mirror inputs and GPTQ fallback math without model inference."""
from pathlib import Path
import ast
import importlib.util
import json
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT
from arena.experiment import atomic_json, read_json, sha256, utc_now, verify_sources
from arena.gptq8 import dense_weight, native_layout_torch
from arena.span019_input import encode, flatten_turns, native_pool_index, render


def main():
    pub, root = REPO / 'results/span_019', ROOT / 'runs/span_019/source_audit'
    assert not (pub / 'native_cpu_controls.json').exists(), 'Preserve first CPU controls'
    observation = read_json(pub / 'source_observation.json')
    assert observation['structural_source_checks_passed']
    for name, digest in observation['source_files_sha256'].items():
        assert sha256(root / name) == digest
    source_names = ['scripts/check_span_019_native.py', 'src/arena/gptq8.py',
                    'src/arena/span019_input.py', 'docs/SPAN_019_ASSET_PROTOCOL.md']
    sources = {name: sha256(REPO / name) for name in source_names}
    verify_sources(REPO, sources)
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    text = (root / 'README.md').read_text()
    start = text.index('def create_rounds(')
    end = text.index('    return prompt', start) + len('    return prompt')
    function = ast.parse(text[start:end])
    assert not any(isinstance(n, (ast.Call, ast.Import, ast.ImportFrom, ast.Attribute)) for n in ast.walk(function))
    scope = {}
    exec(compile(function, 'pinned-card-pure-formatter', 'exec'), scope)
    fixtures = [('', '', ''), ('q', 'a', 'b'), ('first\nsecond', 'null\n', 'x'),
                ('quote """', '<pad> inline', '<eos> inline'),
                ('long '*6000, 'a', 'b'), ('q', 'long '*6000, 'b'),
                ('q', 'a', 'long '*6000), ('q', '<pad> '+'long '*6000, 'b')]
    inputs = []
    for query, answer_a, answer_b in fixtures:
        row = dict(zip(['prompt','response_a','response_b'], [json.dumps([v]) for v in [query,answer_a,answer_b]]))
        actual = encode(row, tokenizer)
        native_text = scope['create_rounds'](query, answer_a, answer_b)
        assert native_text == render(query, answer_a, answer_b)
        ids = tokenizer(native_text, truncation=True, max_length=2000)['input_ids']
        ids.append(tokenizer.eos_token_id)
        padded = tokenizer.pad({'input_ids':[ids]}, padding='max_length', max_length=2000)['input_ids'][0]
        assert ids == actual['input_ids']
        assert native_pool_index(padded, tokenizer.pad_token_id) == actual['pool_index'] + actual['native_position_offset']
        inputs.append(dict(tokens=len(ids), padded_tokens=len(padded), offset=actual['native_position_offset'],
                           pool_index=actual['pool_index'], identical=True))
    path = root / 'native/qlinear_cuda_v0.7.1.py'
    spec = importlib.util.spec_from_file_location('pinned_gptq_native', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(2)
    arithmetic = []
    for seed in [42, 43, 44]:
        rng = np.random.default_rng(seed)
        qweight = rng.integers(0, 2**32, (32,128), dtype=np.uint32).view(np.int32)
        qzeros = rng.integers(0, 2**32, (4,32), dtype=np.uint32).view(np.int32)
        scales = rng.uniform(.001, .05, (4,128)).astype(np.float16)
        g_idx = rng.integers(0,4,128,dtype=np.int32)
        native = module.QuantLinear(8,32,128,128,False)
        for name, value in [('qweight',qweight),('qzeros',qzeros),('scales',scales),('g_idx',g_idx)]:
            setattr(native, name, torch.from_numpy(value.copy()))
        x = torch.from_numpy(rng.normal(size=(128,128)).astype(np.float16))
        with torch.no_grad():
            reference = native(x)
            weight = native_layout_torch(torch.from_numpy(dense_weight(qweight,qzeros,scales,g_idx)))
            converted = torch.nn.functional.linear(x, weight)
        difference = float((reference-converted).abs().max())
        assert difference <= 1e-5
        arithmetic.append(dict(seed=seed, max_abs=difference, weight_stride=list(weight.stride()), passed=True))
    # Freeze complete IDs from existing selections. No human labels are read.
    selection = read_json(REPO / 'results/span_016/profile_selection.json')
    later_path = ROOT / 'runs/span_016/cohort_matched/source.parquet'
    later_manifest = read_json(REPO / 'results/span_016/matched_cohort_manifest.json')
    assert sha256(later_path) == later_manifest['artifact_sha256']['source.parquet']
    later = pd.read_parquet(later_path, columns=['id'])
    assert len(later) == 1024 and later.id.is_unique
    atomic_json(pub / 'evaluation_selection.json', dict(at=utc_now(), uses_targets_for_selection=False,
        control_ids=selection['control_ids'], uniform_ids=selection['uniform_ids'],
        long_ids=selection['long_ids'], later_ids=later.id.tolist(),
        train_csv_sha256=selection['train_csv_sha256'], later_source_sha256=sha256(later_path),
        later_manifest_sha256=sha256(REPO/'results/span_016/matched_cohort_manifest.json'),
        original_votes_seen_in_checkpoint_training=True, later_cohort_previously_inspected=True))
    result = dict(at=utc_now(), passed=True, source_sha256=sources, synthetic_input_checks=inputs,
        synthetic_native_GPTQ_arithmetic=arithmetic, real_model_logits_compared=False,
        config_pad_token_id=observation['pad_token_id'], effective_native_pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id, padding_side=tokenizer.padding_side,
        maximum_input_tokens=2001, full_weights_downloaded=False, new_gpu_executions=0,
        human_votes_read=False, quality_measured=False,
        selection_sha256=sha256(pub/'evaluation_selection.json'), gpu_admitted=False)
    atomic_json(pub / 'native_cpu_controls.json', result)
    print(json.dumps({k:result[k] for k in ['passed','synthetic_native_GPTQ_arithmetic',
        'effective_native_pad_token_id','maximum_input_tokens','real_model_logits_compared','gpu_admitted']}))


if __name__ == '__main__':
    main()
