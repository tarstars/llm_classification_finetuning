"""Prove recent-turn inputs against all possible changed rows, without targets."""
from collections import Counter
from pathlib import Path
import hashlib
import importlib.util
import json
import resource
import signal
import sys
import time

import numpy as np
import pandas as pd
import transformers
import tokenizers
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT, TEXT_COLUMNS
from arena.experiment import read_json, atomic_json, sha256, utc_now
from arena.span035_recent import processor_class, select_recent


def main():
    start = time.monotonic()
    pub, run = REPO / 'results/span_035', ROOT / 'runs/span_035'
    assert not (pub / 'preparation.json').exists(), 'Preserve the first real native proof'
    cfg = read_json(REPO / 'configs/span_035.json')
    def expired(*unused): raise TimeoutError('V35 CPU preparation cap')
    signal.signal(signal.SIGALRM, expired); signal.alarm(cfg['cpu_preparation_seconds_cap'])
    assert transformers.__version__ == '4.42.3' and tokenizers.__version__ == '0.19.1'
    run.mkdir(parents=True, exist_ok=True)
    prior = read_json(REPO / 'results/span_026/coverage_audit.json')
    raw = ROOT / 'raw/train.csv'
    assert sha256(raw) == prior['input_sha256']['raw/train.csv']
    coverage_path = ROOT / 'runs/span_026/context_coverage.npz'
    assert sha256(coverage_path) == prior['output_sha256'][str(coverage_path.relative_to(ROOT))]
    coverage = np.load(coverage_path)
    old_hashes = dict(zip(coverage['ids'].tolist(), coverage['old_input_sha256'].tolist()))
    possible = set(coverage['ids'][coverage['changed']].tolist()); assert len(possible) == 357
    native_path = ROOT / 'runs/span_009/research/lmsys-modules-0805/data/processors.py'
    assert sha256(native_path) == prior['processor_sha256']
    loader = importlib.util.spec_from_file_location('native_recent_reference', native_path)
    native = importlib.util.module_from_spec(loader); loader.loader.exec_module(native)
    asset = ROOT / 'runs/span_009/research/lmsys-checkpoints-0-0805'
    parent = ROOT / 'runs/span_013/jobs/gemma_order/output/order_completion'
    checkpoint = read_json(parent / 'm0_checkpoint.json')
    for name in ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'model.safetensors.index.json']:
        assert sha256(asset / name) == checkpoint[name]
    tokenizer = AutoTokenizer.from_pretrained(asset, local_files_only=True)
    old = native.ProcessorPAB(tokenizer, 4096, False)
    new = processor_class(native.ProcessorPAB)(tokenizer, 4096, False)
    frame = pd.read_csv(raw, usecols=['id', *TEXT_COLUMNS]); assert len(frame) == 57477 and frame.id.is_unique
    selection = read_json(REPO / 'results/span_013/profile_selection.json')
    controls = {n: selection[n + '_ids'] for n in ['control', 'uniform', 'long']}
    inspect_ids = sorted(possible | set(sum(controls.values(), [])))
    indexed = frame.set_index('id', drop=False)
    records, reasons, tokens_old, tokens_new = [], Counter(), {}, {}
    for index, rid in enumerate(inspect_ids):
        row = indexed.loc[rid].to_dict()
        dummy = dict(row, winner_model_a=1, winner_model_b=0, winner_tie=0)
        a, b = old.build_input(dummy), new.build_input(dummy)
        x, z = a['input_ids'].numpy(), b['input_ids'].numpy()
        assert hashlib.sha256(x.tobytes()).hexdigest() == old_hashes[rid]
        info = new.last_selection_info.copy(); reasons[info['reason']] += 1
        tokens_old[rid], tokens_new[rid] = x, z
        if info['changed']:
            assert rid in possible and len(z) <= len(x) + cfg['maximum_input_length_increment']
            columns = [json.loads(row[k]) for k in TEXT_COLUMNS]
            first = info['first_kept_turn']
            retained = {k: json.dumps(v[first:], ensure_ascii=False) for k, v in zip(TEXT_COLUMNS, columns)}
            proof = old.build_input(dict(retained, winner_model_a=1, winner_model_b=0, winner_tie=0))
            np.testing.assert_array_equal(z, proof['input_ids'].numpy())
            assert b['input_text'] == proof['input_text']
            _, reverse_info = select_recent(columns[0], columns[2], columns[1], tokenizer, 4096)
            assert reverse_info['changed'] and reverse_info['first_kept_turn'] == first
            assert len(z) <= read_json(asset / 'config.json')['max_position_embeddings']
        else:
            np.testing.assert_array_equal(x, z)
            assert a['input_text'] == b['input_text']
        records.append(dict(id=int(rid), old_length=len(x), new_length=len(z), **info,
            old_sha256=hashlib.sha256(x.tobytes()).hexdigest(), new_sha256=hashlib.sha256(z.tobytes()).hexdigest()))
        if (index + 1) % 100 == 0:
            progress = dict(at=utc_now(), rows=index + 1, total=len(inspect_ids), elapsed_seconds=time.monotonic() - start, targets_loaded=False)
            atomic_json(pub / 'preparation_progress.json', progress); print(progress, flush=True)
    for name, ids in controls.items():
        stream = hashlib.sha256()
        for rid in ids: stream.update(tokens_old[rid].tobytes())
        expected = read_json(parent / 'm0_summary.json')['cohorts'][name]['token_ids_sha256']
        assert stream.hexdigest() == expected
    changed = [r for r in records if r['changed']]
    assert all(r['id'] in possible for r in changed)
    proof_path = run / 'native_input_proof.json'
    atomic_json(proof_path, dict(targets_loaded=False, rows=records))
    changed_ids = [r['id'] for r in changed]
    # All possible changes are covered; outside the old exact clipped set the
    # new function takes its byte-equivalent original-prefix branch.
    np.savez_compressed(run / 'recent_inputs.npz', ids=np.asarray(changed_ids),
        old_lengths=np.asarray([r['old_length'] for r in changed]), new_lengths=np.asarray([r['new_length'] for r in changed]))
    examples = []
    if changed:
        train_source = read_json(REPO / 'results/span_031/artifact_inventory.json')['input_artifacts']['train.parquet']
        train_ids = set(pd.read_parquet(train_source['path'], columns=['id']).id)
        available = [r for r in changed if r['id'] in train_ids]
        chosen = np.random.default_rng(cfg['seed']).choice(len(available), min(4, len(available)), replace=False)
        for j in chosen:
            record = available[j]; row = indexed.loc[record['id']]
            parsed = [json.loads(row[k]) for k in TEXT_COLUMNS]
            examples.append(dict(**record, final_prompt=parsed[0][-1], final_a=parsed[1][-1], final_b=parsed[2][-1],
                first_omitted_prompt=parsed[0][0], retained_start_prompt=parsed[0][record['first_kept_turn']]))
    atomic_json(run / 'training_retention_examples.json', dict(training_only=True, targets_loaded=False, rows=examples))
    snapshots = sorted((REPO / 'backlog/versions').glob('*')); assert len(snapshots) == 138
    preserve = read_json(REPO / 'results/span_034/preservation.json')
    for name, digest in preserve['user_notebook_sha256'].items(): assert sha256(REPO / name) == digest
    atomic_json(pub / 'preservation.json', dict(at=utc_now(), snapshot_sha256={str(p.relative_to(REPO)): sha256(p) for p in snapshots}, user_notebook_sha256=preserve['user_notebook_sha256']))
    resource_prior = read_json(REPO / 'results/span_034/resource_close.json')['quota']
    plan = dict(started_at=cfg['started_at'], deadline=cfg['deadline'], quota_used_at_start=resource_prior['used_seconds'],
        initial_quota_refresh_time=resource_prior['refresh_time'], last_verified_quota=resource_prior,
        gpu_session_seconds_cap=9000, public_version_limit=2, official_candidate_limit=1, hidden_cap_seconds=32400, parent_span=34)
    atomic_json(pub / 'span_plan.json', plan)
    passed = len(changed) >= cfg['minimum_changed_rows'] and len(changed) / len(frame) <= cfg['maximum_changed_fraction']
    sources = ['GOAL.md', 'docs/SPAN_035_PLAN.md', 'configs/span_035.json', 'src/arena/span035_recent.py',
        'scripts/prepare_span_035_recent.py', 'tests/test_span035_recent.py']
    result = dict(at=utc_now(), task='H27', prepared=True, targets_loaded=False, quality_measured=False,
        possible_changed_rows=357, inspected_native_rows=len(records), changed_rows=len(changed), changed_ids=changed_ids,
        full_original_rows=len(frame), changed_fraction=len(changed) / len(frame), reason_counts=dict(reasons),
        input_bounds_passed=True, unchanged_branches_exact=True, original_native_control_tokens_exact=True,
        all_retained_turns_native_exact=True, selected_turn_indices_swap_consistent=True,
        new_max_input_tokens=max((r['new_length'] for r in changed), default=0),
        old_changed_mean_input_tokens=float(np.mean([r['old_length'] for r in changed])) if changed else None,
        new_changed_mean_input_tokens=float(np.mean([r['new_length'] for r in changed])) if changed else None,
        cpu_feasibility_passed=bool(passed), gpu_admitted=False,
        raw_sha256=sha256(raw), processor_sha256=sha256(native_path), tokenizer_sha256=sha256(asset / 'tokenizer.json'),
        old_coverage_sha256=sha256(coverage_path), proof_sha256=sha256(proof_path),
        examples_sha256=sha256(run / 'training_retention_examples.json'), inputs_sha256=sha256(run / 'recent_inputs.npz'),
        source_sha256={n: sha256(REPO / n) for n in sources}, elapsed_seconds=time.monotonic() - start,
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        runtime=dict(transformers=transformers.__version__, tokenizers=tokenizers.__version__))
    assert result['peak_rss_gib'] < cfg['rss_guard_gib']
    atomic_json(pub / 'preparation.json', result)
    print({k: v for k, v in result.items() if k not in ['changed_ids', 'source_sha256']}, flush=True)


if __name__ == '__main__': main()
