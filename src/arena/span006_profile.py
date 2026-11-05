"""Conditional Internet-disabled source/classifier/residual inference qualification."""
import argparse
import hashlib
import json
import resource
import signal
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, TrainingArguments

from .data import CLASSES, TEXT_COLUMNS, json_write, parse_turns
from .joint_data import JointTokenizer, PreferenceDataset, PreferenceCollator
from .joint_infer import load_inference_model
from .joint_train import PreferenceTrainer, probabilities
from .span006_inputs import read_inputs, token_digest
from .source_portable import PortableSource, compact_features, residual_predict


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['raw', 'model', 'adapter', 'manifest', 'reference', 'assets', 'contract', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    started = time.monotonic()

    def expired(*unused):
        raise TimeoutError('Original offline profile deadline reached')

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(max(1, int(args.deadline - time.time())))
    manifest = json.loads(args.manifest.read_text())
    contract = json.loads(args.contract.read_text())
    asset_manifest = json.loads((args.assets / 'asset_manifest.json').read_text())
    for name, digest in asset_manifest['artifact_sha256'].items():
        assert sha(args.assets / name) == digest, name
    assert sha(args.adapter / 'adapter_model.safetensors') == manifest['adapter_sha256']
    assert sha(args.adapter / 'tokenizer.json') == manifest['tokenizer_sha256']
    config = manifest['config']
    assert sha(Path(__file__).with_name('joint_data.py')) == config['preprocessing_sha256']
    assert manifest['class_order'] == CLASSES
    reference = np.load(args.reference)
    smoke = read_inputs(args.raw / 'train.csv', reference['ids'])
    assert len(smoke) == contract['smoke_rows'] == 64
    visible = pd.read_csv(args.raw / 'test.csv', usecols=['id', *TEXT_COLUMNS])
    for name in TEXT_COLUMNS:
        visible[name] = visible[name].map(parse_turns)
    # Qualification runs against the exposed three-row example, not hidden test data.
    assert len(visible) == 3
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    tokenizer = AutoTokenizer.from_pretrained(str(args.adapter), local_files_only=True, padding_side='right')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    packed = [JointTokenizer(tokenizer, config['max_length']).encode_frame(f) for f in [smoke, visible]]
    model = load_inference_model(args.model, args.adapter, config)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.problem_type = 'single_label_classification'
    assert torch.cuda.device_count() == 2
    bf16 = torch.cuda.get_device_capability()[0] >= 8
    options = TrainingArguments(output_dir=str(args.out / 'evaluation_tmp'),
        per_device_eval_batch_size=config['eval_batch_size'], fp16=not bf16, bf16=bf16,
        report_to='none', remove_unused_columns=False, dataloader_num_workers=0,
        dataloader_pin_memory=True, disable_tqdm=True)
    trainer = PreferenceTrainer(model=model, args=options, processing_class=tokenizer,
                                data_collator=PreferenceCollator(tokenizer))
    root = trainer.is_world_process_zero()
    args.out.mkdir(parents=True, exist_ok=True)
    views = []
    gpu_seconds = []
    for original, swapped, diagnostics in packed:
        begin = time.monotonic()
        pair = []
        for force in [False, True]:
            values = trainer.predict(PreferenceDataset(original, swapped, force_swap=force)).predictions
            if isinstance(values, tuple):
                values = values[0]
            pair.append(probabilities(values))
        views.append((pair[0], pair[1][:, [1, 0, 2]]))
        gpu_seconds.append(time.monotonic() - begin)
    if root:
        begin = time.monotonic()
        source = PortableSource(args.assets)
        ratings = np.load(args.assets / 'ratings.npz')['ratings']
        head = np.load(args.assets / 'strength_head.npz')
        source_setup = time.monotonic() - begin
        cpu_seconds = []
        results = []
        for frame, pair in zip([smoke, visible], views):
            begin = time.monotonic()
            qa = source.predict(frame.response_a)
            qb = source.predict(frame.response_b)
            average = (pair[0] + pair[1]) / 2
            logits = np.log(np.clip(average, 1e-12, 1)) / manifest['temperature']
            p = np.exp(logits - np.logaddexp.reduce(logits, axis=1, keepdims=True))
            d, s = compact_features(qa, qb, ratings)
            joint = residual_predict(p, d, s, head)
            swapped = residual_predict(p[:, [1, 0, 2]], -d, s, head)[:, [1, 0, 2]]
            np.testing.assert_allclose(joint, swapped, atol=1e-10)
            assert np.isfinite(joint).all() and (joint >= 0).all()
            np.testing.assert_allclose(joint.sum(1), 1, atol=1e-6)
            cpu_seconds.append(time.monotonic() - begin)
            results.append((qa, qb, average, joint))
        qa, qb, average, joint = results[0]
        checks = {}
        for name, actual, expected in [('raw', average, (reference['original'] + reference['aligned_swap']) / 2),
                                       ('source', np.vstack([qa, qb]), np.vstack([reference['qa'], reference['qb']])),
                                       ('joint', joint, reference['joint'])]:
            delta = np.abs(actual - expected)
            checks[name] = {'mean': float(delta.mean()), 'maximum': float(delta.max()),
                            'passed': bool(delta.mean() <= contract[name + '_mean_tolerance'] and delta.max() <= contract[name + '_max_tolerance'])}
        summary = {'complete': True, 'passed': all(c['passed'] for c in checks.values()), 'identity': checks,
                   'rows': 64, 'visible_rows': 3, 'gpu_prediction_seconds': gpu_seconds,
                   'cpu_prediction_seconds': cpu_seconds, 'source_setup_seconds': source_setup,
                   'classifier_seconds_per_row': gpu_seconds[0] / 64,
                   'source_meta_seconds_per_row': cpu_seconds[0] / 64,
                   'program_seconds': time.monotonic() - started,
                   'peak_rss_gib_rank0': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
                   'smoke_original_tokens_sha256': token_digest(packed[0][0]),
                   'smoke_swapped_tokens_sha256': token_digest(packed[0][1]),
                   'confirmation_votes_read': False, 'model_fit_operations': 0,
                   'asset_manifest_sha256': sha(args.assets / 'asset_manifest.json'),
                   'contract_sha256': sha(args.contract), 'adapter_sha256': manifest['adapter_sha256']}
        np.savez(args.out / 'smoke_predictions.npz', ids=smoke.id.to_numpy(), original=views[0][0],
                 aligned_swap=views[0][1], qa=qa, qb=qb, joint=joint)
        submission = pd.DataFrame(results[1][3], columns=CLASSES)
        submission.insert(0, 'id', visible.id.to_numpy())
        assert submission.id.is_unique and submission.id.tolist() == visible.id.tolist()
        assert submission.columns.tolist() == pd.read_csv(args.raw / 'sample_submission.csv').columns.tolist()
        submission.to_csv(args.out / 'submission.csv', index=False)
        summary['csv_valid'] = True
        json_write(args.out / 'profile_summary.json', summary)
        print(json.dumps(summary), flush=True)
    torch.cuda.synchronize()
    device = torch.cuda.current_device()
    json_write(args.out / f'gpu_rank{trainer.args.process_index}.json',
               {'rank': trainer.args.process_index, 'device': device, 'name': torch.cuda.get_device_name(device),
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)})
    trainer.accelerator.wait_for_everyone()
    signal.alarm(0)


if __name__ == '__main__':
    main()
