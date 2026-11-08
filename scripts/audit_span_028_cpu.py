"""One bounded, label-free proof of original native batches with CPU prefetch."""
import time
STARTED = time.monotonic()
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

import numpy as np
import pandas as pd
import torch
import transformers
import tokenizers
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.cpu_prefetch import CpuPrefetch, require_host
from arena.data import ROOT
from arena.experiment import atomic_json, read_json, sha256, utc_now


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


class TaggedDataset:
    """Audit-only row tags do not enter native packing or tensor construction."""
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        samples = self.dataset[index]
        for sample in samples:
            sample['row_id'] = int(self.dataset.df.iloc[index].id)
        return samples


def consume(iterator, deadline):
    tensor_hash = hashlib.sha256()
    tokens = hashlib.sha256()
    ids, lengths, groups = [], [], []
    preparation = 0.
    iterator = iter(iterator)
    while True:
        assert time.monotonic() < deadline, 'Preserve the CPU cap'
        begin = time.monotonic()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        finally:
            preparation += time.monotonic() - begin
        require_host(batch)
        batch_groups = []
        for micro in batch:
            lens = [int(n) for n in micro['seq_lens']]
            assert sum(lens) <= 8192
            assert micro['cu_seqlens'].tolist() == np.r_[0, np.cumsum(lens)].tolist()
            assert torch.equal(micro['position_ids'], torch.cat([torch.arange(n) for n in lens]).unsqueeze(0))
            ids.extend(micro['row_id']); lengths.extend(lens); batch_groups.append(lens)
            tokens.update(micro['input_ids'].numpy().tobytes())
            for key in ['input_ids', 'position_ids', 'cu_seqlens', 'label']:
                value = micro[key].contiguous()
                tensor_hash.update(json.dumps([key, str(value.dtype), list(value.shape)]).encode())
                tensor_hash.update(value.numpy().tobytes())
        groups.append(batch_groups)
    return dict(ids=ids, lengths=lengths, groups=groups, token_ids_sha256=tokens.hexdigest(),
                host_tensor_sha256=tensor_hash.hexdigest(), consumer_next_seconds=preparation)


def main():
    pub = REPO / 'results/span_028'; root = ROOT / 'runs/span_028'
    assert not (pub / 'cpu_audit.json').exists() and not (pub / 'preservation.json').exists()
    assert transformers.__version__ == '4.42.3' and tokenizers.__version__ == '0.19.1'
    torch.set_num_threads(2)
    deadline = STARTED + 285
    source_files = ['GOAL.md', 'docs/SPAN_028_PLAN.md', 'scripts/audit_span_028_cpu.py',
                    'src/arena/cpu_prefetch.py', 'tests/test_cpu_prefetch.py']
    snapshots = sorted((REPO / 'backlog/versions').glob('*'))
    assert len(snapshots) == 110
    previous = read_json(REPO / 'results/span_027/preservation.json')
    for n, h in previous['user_notebook_sha256'].items():
        assert sha256(REPO / n) == h
    assert (Path.home() / 'notebooks/llm_classification_finetuning').resolve() == REPO / 'notebooks'
    preservation = dict(at=utc_now(), snapshot_sha256={str(p.relative_to(REPO)): sha256(p) for p in snapshots},
                        user_notebook_sha256=previous['user_notebook_sha256'],
                        goal_v27_sha256=sha256(REPO / 'docs/GOAL_V27.md'))
    atomic_json(pub / 'preservation.json', preservation)
    native = ROOT / 'runs/span_009/research/lmsys-modules-0805/data'
    native_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in [native / (n + '.py') for n in ['collators', 'dataset', 'processors']]}
    old = read_json(REPO / 'results/span_027/packing_cpu_audit.json')
    assert sha256(native / 'collators.py') == old['native_collator_sha256']
    assert sha256(native / 'processors.py') == old['native_processor_sha256']
    collators = module(native / 'collators.py', 'native_prefetch_collators')
    processors = module(native / 'processors.py', 'native_prefetch_processors')
    datasets = module(native / 'dataset.py', 'native_prefetch_dataset')
    spec_path = ROOT / 'runs/span_027/profile_spec.json'
    assert sha256(spec_path) == read_json(REPO / 'results/span_027/profile_preparation.json')['profile_spec_sha256']
    spec = read_json(spec_path)
    fields = ['id', 'prompt', 'response_a', 'response_b']
    frames = {name: pd.read_csv(ROOT / ('raw/' + name + '.csv'), usecols=fields).set_index('id') for name in ['train', 'test']}
    assert sha256(ROOT / 'raw/train.csv') == spec['train_csv_sha256']
    root.mkdir(parents=True, exist_ok=True)
    references, results = {}, {}
    for role in ['m0', 'm3', 'm0_reverse']:
        gemma = role != 'm3'
        assets = ROOT / ('runs/span_009/research/lmsys-checkpoints-' + ('0' if gemma else '3') + '-0805')
        for name in ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'model.safetensors.index.json']:
            assert sha256(assets / name) == spec['checkpoint_references'][role][name]
        tokenizer = transformers.AutoTokenizer.from_pretrained(assets, local_files_only=True)
        processor = processors.ProcessorPAB(tokenizer, 4096, not gemma)
        references[role], results[role] = {}, {}
        for name, expected in spec['token_references'][role].items():
            assert time.monotonic() < deadline
            frame = frames['test' if name.endswith('test') else 'train'].loc[expected['ids']].reset_index()
            if role != 'm0':
                frame[['response_a', 'response_b']] = frame[['response_b', 'response_a']].values
            for i, col in enumerate(['winner_model_a', 'winner_model_b', 'winner_tie']):
                frame[col] = int(i == 0)
            path = root / 'cpu_input.parquet'
            frame.to_parquet(path, index=False)
            def source():
                dataset = TaggedDataset(datasets.LMSYSDataset(path, processor=processor, is_parquet=True))
                return DataLoader(dataset, batch_size=80, num_workers=0,
                                  collate_fn=collators.ShardedMaxTokensCollator(8192, collators.VarlenCollator()))
            baseline = consume(source(), deadline)
            with CpuPrefetch(source()) as prefetched:
                observed = consume(prefetched, deadline)
            assert not prefetched.thread.is_alive()
            for key in ['ids', 'lengths', 'groups', 'token_ids_sha256', 'host_tensor_sha256']:
                assert observed[key] == baseline[key], (role, name, key)
            assert observed['ids'] == expected['ids'] and observed['lengths'] == expected['lengths']
            assert observed['token_ids_sha256'] == expected['token_ids_sha256']
            references[role][name] = baseline
            results[role][name] = dict(rows=len(observed['ids']), micro_batches=sum(map(len, observed['groups'])),
                exact_host_batches=True, exact_row_order=True, exact_positions=True,
                token_ids_sha256=observed['token_ids_sha256'], host_tensor_sha256=observed['host_tensor_sha256'],
                local_synchronous_preparation_seconds=baseline['consumer_next_seconds'],
                local_prefetch_metrics=dict(prefetched.metrics))
            path.unlink()
    target = root / 'cpu_batch_references.json'
    assert not target.exists()
    atomic_json(target, references)
    elapsed = time.monotonic() - STARTED
    assert elapsed < 300
    result = dict(at=utc_now(), passed=True, labels_read=False, quality_measured=False,
                  gpu_probability_identity_measured=False, gpu_speedup_measured=False, new_gpu_executions=0,
                  packing_tokens=8192, queue_depth=2, producer_threads=1, cpu_cap_seconds=300,
                  elapsed_cpu_seconds=elapsed, actual_native_cohorts=results,
                  source_sha256={n: sha256(REPO / n) for n in source_files},
                  native_source_sha256=native_hashes, token_reference_sha256=sha256(spec_path),
                  output_sha256={str(target.relative_to(ROOT)): sha256(target)},
                  preservation_sha256=sha256(pub / 'preservation.json'))
    atomic_json(pub / 'cpu_audit.json', result)
    print(dict(passed=True, elapsed_cpu_seconds=elapsed, native_cohorts=sum(map(len, results.values())), new_gpu_executions=0))


if __name__ == '__main__':
    main()
