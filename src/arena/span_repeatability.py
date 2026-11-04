"""Label-independent replay/order/restart diagnostic for the pinned native judge."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time

import numpy as np
import pandas as pd

from . import judge as judge_module
from . import judge_runtime
from .data import TEXT_COLUMNS


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate(p):
    p = np.asarray(p, dtype=float)
    assert p.ndim == 2 and p.shape[1] == 3 and np.isfinite(p).all() and (p >= 0).all()
    np.testing.assert_allclose(p.sum(1), 1, atol=1e-8)
    return p


def difference(reference_ids, reference, observed_ids, observed, limits):
    assert len(reference_ids) == len(set(map(int, reference_ids)))
    assert len(observed_ids) == len(set(map(int, observed_ids)))
    assert set(map(int, reference_ids)) == set(map(int, observed_ids))
    lookup = {int(rid): index for index, rid in enumerate(observed_ids)}
    aligned = validate(observed)[[lookup[int(rid)] for rid in reference_ids]]
    reference = validate(reference)
    delta = np.abs(aligned - reference)
    return {'rows': len(aligned), 'mean_probability_difference': float(delta.mean()),
            'max_probability_difference': float(delta.max()),
            'argmax_disagreement': float((aligned.argmax(1) != reference.argmax(1)).mean()),
            'limits': limits, 'passed': bool(delta.mean() <= limits['mean'] and delta.max() <= limits['maximum'])}


def request_payload(payload, cache_prompt):
    assert isinstance(cache_prompt, bool)
    return {**payload, 'cache_prompt': cache_prompt}


def run(config, executable, raw, out, sample, plan):
    """All model/packing settings are unchanged; only cache_prompt differs by case."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'diagnostic_summary.json').exists(), 'Never overwrite a diagnostic execution'
    model = judge_runtime.model_file(config['models']['gemma26'])
    model_identity = {'source': config['models']['gemma26'], 'filename': model.name,
                      'bytes': model.stat().st_size, 'sha256': digest(model)}
    judge_runtime.write_json(out / 'model_identity.json', model_identity)
    frame = pd.read_csv(Path(raw) / 'train.csv', usecols=['id'] + TEXT_COLUMNS).set_index('id', drop=False)
    ids = np.asarray(sample['sentinel'], dtype=np.int64)
    permutation = np.asarray(sample['permutation_indices'])
    assert len(ids) == len(set(ids)) == 64 and sorted(permutation.tolist()) == list(range(64))
    scope, tls, guard = {}, threading.local(), threading.Lock()
    original_post = judge_module.post
    memory, stop = [], threading.Event()
    def memory_worker():
        while not stop.is_set():
            try:
                data = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                               text=True, timeout=5)
                values = [int(value.strip()) for value in data.splitlines()]
                assert len(values) == 2
                memory.append(values)
            except (subprocess.SubprocessError, ValueError, AssertionError):
                pass
            stop.wait(.5)
    def captured_post(url, path, payload):
        if path != '/completion':
            return original_post(url, path, payload)
        actual = request_payload(payload, scope['cache_prompt'])
        response = original_post(url, path, actual)
        tokens = actual['prompt']
        assert isinstance(tokens, list) and all(isinstance(token, int) for token in tokens)
        entry = {'id': getattr(tls, 'row_id', None), 'pass': scope['pass'],
                 'cache_prompt': actual['cache_prompt'], 'prepared_tokens': tokens,
                 'prepared_tokens_sha256': hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                 'request_sha256': hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest(),
                 'response': response}
        with guard, (scope['directory'] / 'requests.jsonl').open('a') as stream:
            stream.write(json.dumps(entry) + '\n')
        return response
    class TracedJudge(judge_module.PromptJudge):
        def row(self, row, both_orders):
            tls.row_id = int(row['id'])
            return super().row(row, both_orders)
    def stop_server(process, log):
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        log.close()
    def one_session(pass_names, restart):
        server_out = out / ('server_restart' if restart else 'server_initial')
        server_out.mkdir(parents=True, exist_ok=False)
        load_start = time.monotonic()
        process, log, url = judge_runtime.start_server(executable, model, server_out, slots=config['slots'],
            context=config['max_length'], batch_size=config['batch_size'], ubatch_size=config['ubatch_size'],
            http_threads=config['http_threads'])
        loaded = time.monotonic() - load_start
        record = {'load_seconds': loaded, 'server_command': process.args, 'cases': {}}
        try:
            for case in plan['diagnostic_configurations']:
                memory_start = len(memory)
                scope.update(cache_prompt=case['cache_prompt'], directory=server_out, **{'pass': 'preflight'})
                judge = TracedJudge(url, config)
                preflight = []
                for fields, expected in [(['Compute 2+2.', '4', '5'], 0), (['Compute 2+2.', '5', '4'], 1),
                                         (['Say hello.', 'Hello.', 'Hello.'], 2)]:
                    probability, info = judge.score(fields, [judge.tokenize(text) for text in fields])
                    preflight.append({'expected': expected, 'probabilities': probability.tolist(),
                                      'prediction': int(probability.argmax()), 'tokens': info['tokens']})
                    if expected != 2:
                        assert probability.argmax() == expected, 'Arithmetic/class mapping failed'
                case_record = {'label_tokens': judge.token_ids, 'bos_prefix': judge.bos_prefix,
                               'prompt_overhead': judge.overhead, 'preflight': preflight, 'passes': {}}
                for name in pass_names:
                    folder = out / case['name'] / name
                    folder.mkdir(parents=True, exist_ok=False)
                    scope.update(directory=folder, **{'pass': name})
                    ordered = ids[permutation] if name == 'permuted' else ids
                    results, seconds = judge_module.score_frame(frame.loc[ordered].reset_index(drop=True), judge, folder, False)
                    p = validate([row['original'] for row in results])
                    np.savez_compressed(folder / 'predictions.npz', ids=ordered, original=p)
                    traces = [json.loads(line) for line in (folder / 'requests.jsonl').read_text().splitlines()]
                    assert len(traces) == 64 and set(row['id'] for row in traces) == set(map(int, ids))
                    token_hashes = {str(row['id']): row['prepared_tokens_sha256'] for row in traces}
                    case_record['passes'][name] = {'rows': 64, 'seconds': seconds, 'seconds_per_row': seconds / 64,
                        'prediction_sha256': digest(folder / 'predictions.npz'),
                        'requests_sha256': digest(folder / 'requests.jsonl'), 'token_sha256_by_id': token_hashes,
                        'mean_tokens': float(np.mean([row['original_info']['tokens'] for row in results])),
                        'truncated_rows': sum(row['original_info']['truncated'] for row in results)}
                    judge_runtime.write_json(folder / 'summary.json', case_record['passes'][name])
                    print('REPLAY_PASS', json.dumps({'configuration': case['name'], 'pass': name, 'rows': 64,
                          'seconds': seconds, 'mean_tokens': case_record['passes'][name]['mean_tokens']}), flush=True)
                assert memory[memory_start:], 'Missing actual device memory samples'
                case_record.update(sampled_device_used_peak_mib=np.max(np.asarray(memory[memory_start:]),axis=0).tolist(),
                                   memory_samples=len(memory)-memory_start)
                record['cases'][case['name']] = case_record
                judge_runtime.write_json(server_out / 'session.json', record)
        finally:
            stop_server(process, log)
        judge_runtime.write_json(server_out / 'session.json', record)
        return record
    judge_module.post = captured_post
    worker = threading.Thread(target=memory_worker, daemon=True)
    worker.start()
    configurations = {}
    started = time.monotonic()
    try:
        first = one_session(['initial', 'repeat', 'permuted'], False)
        second = one_session(['restart'], True)
        for case in plan['diagnostic_configurations']:
            initial_case, restart_case = first['cases'][case['name']], second['cases'][case['name']]
            passes = {**initial_case['passes'], **restart_case['passes']}
            baseline = np.load(out / case['name'] / 'initial/predictions.npz')
            comparisons = {}
            for name in ['repeat', 'permuted', 'restart']:
                observed = np.load(out / case['name'] / name / 'predictions.npz')
                comparison = difference(baseline['ids'], baseline['original'], observed['ids'], observed['original'],
                                        plan['repeatability_limits'])
                comparison['prepared_tokens_identical'] = passes[name]['token_sha256_by_id'] == passes['initial']['token_sha256_by_id']
                comparisons[name] = comparison
            peak = np.maximum(initial_case['sampled_device_used_peak_mib'], restart_case['sampled_device_used_peak_mib'])
            assert len(peak) == 2 and min(peak) > 1024, 'Actual allocation on both native GPUs required'
            result = {'cache_prompt': case['cache_prompt'], 'passes': passes, 'comparisons': comparisons,
                'repeatability_passed': all(value['passed'] and value['prepared_tokens_identical'] for value in comparisons.values()),
                'native_both_devices_verified': True, 'sampled_device_used_peak_mib': peak.tolist(),
                'memory_samples': initial_case['memory_samples'] + restart_case['memory_samples'],
                'server_load_seconds': [first['load_seconds'], second['load_seconds']],
                'label_tokens': initial_case['label_tokens'], 'bos_prefix': initial_case['bos_prefix'],
                'prompt_overhead': initial_case['prompt_overhead']}
            assert initial_case['label_tokens'] == restart_case['label_tokens'] and initial_case['bos_prefix'] == restart_case['bos_prefix']
            configurations[case['name']] = result
            judge_runtime.write_json(out / case['name'] / 'result.json', result)
            print('REPLAY_GATE', json.dumps({'configuration': case['name'], 'passed': result['repeatability_passed'],
                                            'comparisons': comparisons}), flush=True)
    finally:
        stop.set()
        worker.join(timeout=6)
        judge_module.post = original_post
        judge_runtime.write_json(out / 'device_memory_samples.json', memory)
    summary = {'checked_at': datetime.now(timezone.utc).isoformat(), 'rows': 64,
               'configurations': configurations, 'model_identity': model_identity,
               'scoring_and_server_seconds': time.monotonic()-started,
               'competition_labels_accessed': False, 'model_parameters_fitted': False,
               'memory_note': 'Sampled total device-used MiB including runtime/driver, not allocator peak.',
               'source_sha256': digest(__file__)}
    judge_runtime.write_json(out / 'diagnostic_summary.json', summary)
    return summary
