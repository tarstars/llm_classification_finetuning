"""Pure-prompt preference classification using a local llama.cpp server."""
from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .data import CLASSES, TEXT_COLUMNS, parse_turns
from .judge_runtime import model_file, post, start_server, write_json

LABELS = ['A', 'B', 'C']
EMPTY_THOUGHT = '<|channel>thought\n<channel|>'


def head_tail(tokens, budget, marker):
    if len(tokens) <= budget:
        return tokens
    if budget <= len(marker) + 2:
        return tokens[:budget]
    remaining = budget - len(marker)
    left = (remaining + 1) // 2
    return tokens[:left] + marker + tokens[-(remaining - left):]


def field_budgets(lengths, available, prompt_cap):
    p, a, b = lengths
    pb = min(p, prompt_cap, max(0, available // 4))
    rest = available - pb
    ab, bb = min(a, rest // 2), min(b, rest // 2)
    unused = rest - ab - bb
    take = min(unused, a - ab)
    ab += take
    unused -= take
    bb += min(unused, b - bb)
    return [pb, ab, bb]


def label_probabilities(response, token_ids):
    rows = response.get('completion_probabilities', response.get('probs'))
    if not rows or len(rows) != 1:
        raise ValueError('Expected one scored decision token: ' + repr(response)[:1500])
    choices = rows[0].get('top_probs', rows[0].get('top_logprobs'))
    if not choices:
        raise ValueError('Missing label probabilities: ' + repr(rows[0]))
    by_id = {}
    for choice in choices:
        value = choice.get('prob')
        if value is None:
            value = float(np.exp(choice['logprob']))
        by_id[int(choice['id'])] = float(value)
    if not set(token_ids).issubset(by_id):
        raise ValueError('The server did not return all three label scores: ' + repr(choices))
    p = np.array([by_id[t] for t in token_ids], dtype=np.float64)
    if not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1, atol=1e-4):
        raise ValueError('Label scores are not a complete probability distribution: ' + repr(p))
    return p / p.sum()


class PromptJudge:
    def __init__(self, url, config):
        self.url, self.config = url, config
        self.marker = self.tokenize('\n[CONTENT OMITTED]\n')
        self.token_ids = []
        for label in LABELS:
            ids = self.tokenize(label)
            assert len(ids) == 1, (label, ids)
            self.token_ids.append(ids[0])
        assert len(set(self.token_ids)) == 3
        self.overhead = len(self.tokenize(self.chat_prompt(['', '', ''])))

    def tokenize(self, text):
        return post(self.url, '/tokenize', {'content': text, 'add_special': False, 'parse_special': True})['tokens']

    def detokenize(self, ids):
        return post(self.url, '/detokenize', {'tokens': ids})['content']

    def chat_prompt(self, fields):
        prompt, a, b = fields
        content = ('[USER CONVERSATION]\n' + prompt + '\n[END USER CONVERSATION]\n\n'
                   '[CANDIDATE A]\n' + a + '\n[END CANDIDATE A]\n\n'
                   '[CANDIDATE B]\n' + b + '\n[END CANDIDATE B]\n\n'
                   'Which candidate would the human prefer? A, B, or C (tie):')
        result = post(self.url, '/apply-template', {'messages': [
            {'role': 'system', 'content': self.config['system_prompt']},
            {'role': 'user', 'content': content}], 'add_generation_prompt': True,
            'chat_template_kwargs': {'enable_thinking': False}})['prompt']
        # Non-thinking Gemma 31B/26B still expects an empty thought block before
        # final-answer tokens. Some server versions include it in the template.
        if not result.endswith(EMPTY_THOUGHT):
            if result.endswith('<|channel>thought\n'):
                result += '<channel|>'
            else:
                assert result.endswith('<|turn>model\n'), repr(result[-150:])
                result += EMPTY_THOUGHT
        return result

    def pack(self, fields, tokens):
        available = self.config['max_length'] - self.overhead - 17
        for _ in range(4):
            budgets = field_budgets(list(map(len, tokens)), available, self.config['prompt_cap'])
            parts = [self.detokenize(head_tail(ids, budget, self.marker)) if len(ids) > budget else text
                     for text, ids, budget in zip(fields, tokens, budgets)]
            prompt = self.chat_prompt(parts)
            ids = self.tokenize(prompt)
            if len(ids) < self.config['max_length']:
                return prompt, {'tokens': len(ids), 'field_tokens': list(map(len, tokens)),
                                'budgets': budgets, 'truncated': any(len(x) > y for x, y in zip(tokens, budgets))}
            available -= len(ids) - self.config['max_length'] + 17
        raise ValueError('Cannot pack all fields within context')

    def score(self, fields, tokens):
        prompt, info = self.pack(fields, tokens)
        response = post(self.url, '/completion', {'prompt': prompt, 'n_predict': 1,
            'temperature': 1.0, 'samplers': ['temperature'], 'grammar': 'root ::= "A" | "B" | "C"',
            'n_probs': 3, 'post_sampling_probs': True, 'return_tokens': True,
            'cache_prompt': True, 'seed': self.config['seed']})
        assert not response.get('truncated', False), 'Server silently truncated a prepared input'
        info['timings'] = response.get('timings', {})
        return label_probabilities(response, self.token_ids), info

    def row(self, row, both_orders):
        fields = [parse_turns(row[name]) for name in TEXT_COLUMNS]
        tokens = [self.tokenize(text) for text in fields]
        p0, i0 = self.score(fields, tokens)
        if both_orders:
            p1, i1 = self.score([fields[0], fields[2], fields[1]], [tokens[0], tokens[2], tokens[1]])
            p1 = p1[[1, 0, 2]]
        else:
            p1, i1 = p0.copy(), None
        return {'id': int(row['id']), 'original': p0.tolist(), 'swapped': p1.tolist(),
                'average': ((p0 + p1) / 2).tolist(), 'original_info': i0, 'swapped_info': i1}


def metrics(labels, p):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    y = np.asarray(labels, dtype=int)
    loss = -np.log(np.clip(p[np.arange(len(y)), y], np.finfo(np.float64).eps, 1))
    return {'rows': len(y), 'log_loss': float(loss.mean()), 'accuracy': float(accuracy_score(y, p.argmax(1))),
            'macro_f1': float(f1_score(y, p.argmax(1), labels=[0, 1, 2], average='macro', zero_division=0)),
            'confusion_matrix': confusion_matrix(y, p.argmax(1), labels=[0, 1, 2]).tolist()}


def score_frame(frame, judge, out, both_orders=True):
    out = Path(out)
    checkpoint = out / 'scored_rows.jsonl'
    prior = {}
    if checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            result = json.loads(line)
            prior[result['id']] = result
    rows = frame.to_dict('records')
    remaining = [r for r in rows if int(r['id']) not in prior]
    started = time.monotonic()
    with checkpoint.open('a') as log, ThreadPoolExecutor(max_workers=judge.config['slots']) as pool:
        # map preserves row order; requests execute concurrently against server slots.
        for result in pool.map(lambda row: judge.row(row, both_orders), remaining):
            prior[result['id']] = result
            log.write(json.dumps(result) + '\n')
            log.flush()
            n = len(prior)
            if n <= 4 or n % 16 == 0 or n == len(rows):
                progress = {'rows': n, 'total': len(rows), 'elapsed_seconds': time.monotonic() - started}
                write_json(out / 'progress.json', progress)
                print('JUDGE_PROGRESS', json.dumps(progress), flush=True)
    results = [prior[int(r['id'])] for r in rows]
    return results, time.monotonic() - started


def run(config, executable, raw, out, model_key, mode, sample=None, selection=None, both_orders=True):
    out, raw = Path(out), Path(raw)
    out.mkdir(parents=True, exist_ok=True)
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config_path = out / 'judge_config.json'
    if config_path.exists(): assert json.loads(config_path.read_text()) == config
    write_json(config_path, config)
    if mode in ['profile', 'dev']:
        frame = pd.read_csv(raw / 'train.csv').set_index('id').loc[sample['ids']].reset_index()
        if mode == 'profile': frame = frame.iloc[:config['profile_rows']]
        labels = frame[CLASSES].to_numpy().argmax(1)
    elif mode == 'predict':
        frame = pd.read_csv(raw / 'test.csv')
        labels = None
        assert selection['config_sha256'] == config_hash and selection['model_key'] == model_key
        assert both_orders == selection['both_orders']
    else:
        raise ValueError(mode)
    source = config['models'][model_key]
    identity = {'config_sha256': config_hash, 'model_key': model_key, 'mode': mode,
                'both_orders': both_orders, 'ids_sha256': hashlib.sha256(frame.id.to_numpy().tobytes()).hexdigest()}
    identity_path = out / 'run_identity.json'
    if identity_path.exists(): assert json.loads(identity_path.read_text()) == identity
    write_json(identity_path, identity)
    model = model_file(source)
    process, log, url = start_server(executable, model, out, slots=config['slots'], context=config['max_length'])
    try:
        judge = PromptJudge(url, config)
        results, seconds = score_frame(frame, judge, out, both_orders)
        original = np.array([r['original'] for r in results])
        swapped = np.array([r['swapped'] for r in results])
        average = np.array([r['average'] for r in results])
        summary = {'mode': mode, 'model_key': model_key, 'model_source': source, 'rows': len(frame),
                   'both_orders': both_orders, 'seconds': seconds, 'config_sha256': config_hash,
                   'label_tokens': judge.token_ids, 'prompt_overhead': judge.overhead,
                   'truncated_rows': sum(r['original_info']['truncated'] for r in results),
                   'packed_tokens_mean': float(np.mean([r['original_info']['tokens'] for r in results])),
                   'projected_25000_seconds': seconds / max(len(frame), 1) * config['profile_test_rows_assumption'],
                   'weights_trained': False, 'blended_with_classifier': False}
        summary['judge_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        summary['runtime_source_sha256'] = hashlib.sha256(Path(__file__).with_name('judge_runtime.py').read_bytes()).hexdigest()
        if labels is not None:
            summary.update(original=metrics(labels, original), swap_average=metrics(labels, average),
                order_flip_fraction=float((original.argmax(1) != swapped.argmax(1)).mean()))
            np.savez_compressed(out / 'judge_predictions.npz', ids=frame.id.to_numpy(), labels=labels,
                                original=original, swapped=swapped, average=average)
        else:
            p = calibrate(average, selection['temperature'], selection['tie_bias'])
            assert np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(1), 1)
            submission = pd.DataFrame(p, columns=CLASSES)
            submission.insert(0, 'id', frame.id.to_numpy())
            assert submission.id.is_unique
            assert submission.columns.tolist() == pd.read_csv(raw / 'sample_submission.csv').columns.tolist()
            submission.to_csv(out / 'submission.csv', index=False)
            summary.update(temperature=selection['temperature'], tie_bias=selection['tie_bias'], probabilities_valid=True)
        write_json(out / 'judge_summary.json', summary)
        print('JUDGE_RESULT', json.dumps(summary), flush=True)
    finally:
        process.terminate()
        try: process.wait(timeout=30)
        except Exception:
            process.kill(); process.wait()
        log.close()
    return summary


def calibrate(p, temperature=1., tie_bias=0.):
    z = np.log(np.clip(p, 1e-15, 1)) / temperature
    z[:, 2] += tie_bias
    z -= z.max(1, keepdims=True)
    q = np.exp(z)
    return q / q.sum(1, keepdims=True)
