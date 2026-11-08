"""Select fresh post-release groups into fixed pre-existing PAB length quotas."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sqlite3
import sys
import time

os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT, TEXT_COLUMNS
from arena.experiment import read_json, atomic_json, exclusive_lock, sha256, utc_now
from arena.span011_data import LABELS, write_inference_inputs


def main():
    torch.set_num_threads(2)
    root, pub = ROOT / 'runs/span_016', REPO / 'results/span_016'
    out = root / 'cohort_matched'
    out.mkdir(exist_ok=True)
    with exclusive_lock(out / 'prepare.lock'):
        assert not (pub / 'matched_cohort_manifest.json').exists(), 'Preserve selected cohort'
        assert not list((root / 'jobs').glob('*/push_intent.json')), 'Refine before any GPU'
        begin = time.monotonic()
        cutoff = datetime.fromisoformat(read_json(pub / 'cohort_manifest.json')['post_publication_cutoff'])
        old = ROOT / 'runs/span_011/cohort'
        excluded = set(read_json(root / 'cohort/selection.json')['groups'])
        for name in ['learning', 'confirmation']:
            excluded.update(pd.read_parquet(old / (name + '_mapping.parquet'), columns=['group_id']).group_id)
        meta = read_json(REPO / 'results/span_011/source_metadata_audit.json')
        bins, quotas = np.asarray(meta['target_length_bins']), np.asarray(meta['learning_bin_quotas'])
        assert quotas.tolist() == [244, 248, 219, 207, 55, 33, 11, 7] and quotas.sum() == 1024
        run9 = ROOT / 'runs/span_009'
        model = run9 / 'research/lmsys-checkpoints-0-0805'
        module = run9 / 'research/lmsys-modules-0805/data/processors.py'
        expected = read_json(run9 / 'jobs/public_ensemble_resume/output/public_ensemble/m0_checkpoint.json')
        assert sha256(model / 'tokenizer.json') == expected['tokenizer.json']
        assert sha256(module) == read_json(REPO / 'results/span_009/full_pab_length_audit.json')['processor_sha256']
        spec = importlib.util.spec_from_file_location('span016_original_processor', module)
        author = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(author)
        processor = author.ProcessorPAB(AutoTokenizer.from_pretrained(model, local_files_only=True), 4096, False)
        def encode(row):
            return processor.build_input({**{c: row[c] for c in TEXT_COLUMNS},
                'winner_model_a': 1, 'winner_model_b': 0, 'winner_tie': 0})['input_ids'].numpy()
        ids = read_json(pub / 'profile_selection.json')['control_ids']
        control = pd.read_csv(ROOT / 'raw/train.csv', usecols=['id', *TEXT_COLUMNS]).set_index('id').loc[ids]
        fingerprint = hashlib.sha256()
        for row in control.to_dict('records'):
            fingerprint.update(encode(row).tobytes())
        assert fingerprint.hexdigest() == read_json(run9 / 'jobs/public_ensemble_resume/output/public_ensemble/m0_summary.json')['cohorts']['uniform']['token_ids_sha256']
        connection = sqlite3.connect('file:' + str(old / 'rows.sqlite') + '?mode=ro', uri=True)
        candidates = []
        for payload, group in connection.execute('SELECT r.payload,p.group_id FROM representatives p JOIN rows r ON r.uid=p.uid'):
            row = json.loads(payload)
            if group in excluded or datetime.fromisoformat(row['timestamp']) <= cutoff:
                continue
            row['group_id'] = group
            priority = hashlib.sha256(('42:span016:lengthmatched:' + group).encode()).hexdigest()
            candidates.append((priority, row))
        candidates.sort(key=lambda r: (r[0], r[1]['source_uuid']))
        filled, selected, tokenized = np.zeros(len(quotas), dtype=int), [], 0
        for _, row in candidates:
            length = len(encode(row))
            index = int(np.searchsorted(bins, length, side='right') - 1)
            assert 0 <= index < len(quotas)
            tokenized += 1
            if filled[index] < quotas[index]:
                row.update(prepared_tokens=length, length_bin=index)
                selected.append(row)
                filled[index] += 1
            assert time.monotonic() - begin < 1800, 'Bounded CPU preparation'
            if np.array_equal(filled, quotas):
                break
        assert np.array_equal(filled, quotas), 'Insufficient support; stop without changing quotas'
        selected.sort(key=lambda r: r['id'])
        selection = dict(at=utc_now(), uses_targets=False, seed=42,
            ids=[r['id'] for r in selected], groups=[r['group_id'] for r in selected],
            cutoff=cutoff.isoformat(), bins=bins.tolist(), quotas=quotas.tolist(),
            excluded_groups=len(excluded), available_groups=len(candidates))
        atomic_json(out / 'selection.json', selection)
        write_inference_inputs(out / 'source.parquet', selected)
        mapping = pd.DataFrame([{k: r[k] for k in ['id', 'source_uuid', 'group_id', 'timestamp', 'prepared_tokens', 'length_bin']} for r in selected])
        mapping.to_parquet(out / 'mapping.parquet', index=False)
        y = [LABELS[connection.execute('SELECT winner FROM rows WHERE uid=?', (r['source_uuid'],)).fetchone()[0]] for r in selected]
        connection.close()
        np.savez_compressed(out / 'labels.npz', ids=np.asarray(selection['ids'], dtype=np.int64),
            labels=np.asarray(y), groups=np.asarray(selection['groups']))
        result = dict(at=utc_now(), rows=1024, uses_targets_for_selection=False,
            targets_read_after_durable_selection=True, original_cohort_preserved=True,
            original_cohort_manifest_sha256=sha256(pub / 'cohort_manifest.json'),
            post_publication_cutoff=cutoff.isoformat(), timestamp_range=[mapping.timestamp.min(), mapping.timestamp.max()],
            excluded_groups=len(excluded), available_groups=len(candidates), groups_tokenized=tokenized,
            bins=bins.tolist(), selected_bin_counts=filled.tolist(), cpu_gpu_token_identity=True,
            processor_sha256=sha256(module), tokenizer_sha256=expected['tokenizer.json'],
            artifact_sha256={n: sha256(out / n) for n in ['selection.json', 'source.parquet', 'labels.npz', 'mapping.parquet']},
            out_of_time_source=True, independent_competition_cv=False, elapsed_cpu_seconds=time.monotonic()-begin,
            source_sha256=sha256(Path(__file__)), new_gpu_executions=0)
        atomic_json(pub / 'matched_cohort_manifest.json', result)
        print(json.dumps({k: result[k] for k in ['rows', 'selected_bin_counts', 'available_groups', 'groups_tokenized', 'elapsed_cpu_seconds']}, indent=2))


if __name__ == '__main__':
    main()
