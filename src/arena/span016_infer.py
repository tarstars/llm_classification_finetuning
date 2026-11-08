"""Pinned native Qwen3 scalar inference; labels never enter this process."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

if __package__:
    from .span016_reward import conversation, encode_conversation
else:
    from reward_mapping import conversation, encode_conversation


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def history_counts(row):
    p, a, b = [json.loads(row[k]) for k in ['prompt', 'response_a', 'response_b']]
    if not all(isinstance(x, list) for x in [p, a, b]):
        raise ValueError('Expected conversation arrays')
    return dict(multiturn_rows=int(max(map(len, [p, a, b])) > 1),
        mismatched_turn_rows=int(len({len(p), len(a), len(b)}) > 1),
        null_messages_in_rendered_pairs=2 * sum(x is None for x in p) + sum(x is None for x in a + b),
        missing_user_turns_in_pairs=sum(max(0, len(x) - len(p)) for x in [a, b]),
        missing_assistant_turns_in_pairs=sum(max(0, len(p) - len(x)) for x in [a, b]))


def main():
    parser = argparse.ArgumentParser()
    for name in ['model', 'out', 'cohorts', 'size']:
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    assert transformers.__version__ == '4.52.3'
    assert torch.cuda.device_count() == 2
    torch.set_num_threads(2)
    torch.manual_seed(42)
    started = time.monotonic()
    out, source = Path(args.out), Path(args.model)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, padding_side='right')
    assert tokenizer.pad_token_id == 151654 and tokenizer.eos_token_id == 151645
    if args.size == '8B':
        devices = {'model.embed_tokens': 0, 'model.rotary_emb': 0, 'model.norm': 1, 'score': 1}
        devices.update({f'model.layers.{i}': int(i >= 18) for i in range(36)})
    else:
        assert args.size == '0.6B'
        devices = {'': 0}
    model, loading = AutoModelForSequenceClassification.from_pretrained(source,
        torch_dtype=torch.float16, device_map=devices, attn_implementation='sdpa',
        output_loading_info=True, local_files_only=True)
    assert not any(loading.get(k) for k in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']), loading
    assert model.config.num_labels == 1 and model.config.pad_token_id == tokenizer.pad_token_id
    assert tuple(model.score.weight.shape) == (1, model.config.hidden_size)
    assert torch.isfinite(model.score.weight).all() and model.score.weight.abs().sum() > 0
    model.eval()
    used = sorted({p.device.index for p in model.parameters()})
    assert used == ([0, 1] if args.size == '8B' else [0])
    summary = dict(size=args.size, transformer_version=transformers.__version__,
        torch_version=torch.__version__, devices=used, loading_exact=True,
        scalar_head_restored=True, cold_seconds=time.monotonic() - started,
        dtype='float16', max_length=2048, batch_sequences=2, cohorts={})
    native_checked = False
    for name in args.cohorts.split(','):
        frame = pd.read_parquet(out / (name + '.parquet'))
        assert set(frame.columns) == {'id', 'prompt', 'response_a', 'response_b'}
        begin = time.monotonic()
        sequences, metadata, history = [], [], {}
        for row in frame.to_dict('records'):
            for key, value in history_counts(row).items():
                history[key] = history.get(key, 0) + value
            for side in ['a', 'b']:
                tokens, details = encode_conversation(tokenizer, conversation(row, side))
                sequences.append(tokens)
                metadata.append(details)
        scores = np.full(len(sequences), np.nan, dtype=np.float32)
        order = np.argsort([len(s) for s in sequences], kind='stable')
        for gpu in used:
            torch.cuda.reset_peak_memory_stats(gpu)
        with torch.inference_mode():
            for start in range(0, len(order), 2):
                indices = order[start:start + 2]
                batch = tokenizer.pad([dict(input_ids=sequences[i], attention_mask=[1] * len(sequences[i]))
                    for i in indices], padding=True, return_tensors='pt')
                batch = {k: v.to('cuda:0') for k, v in batch.items()}
                values = model(**batch, use_cache=False).logits.float().reshape(-1)
                assert values.shape == (len(indices),) and torch.isfinite(values).all()
                if not native_checked:
                    # Official native forward and the explicit token-head/pooling
                    # reference must agree on identical tensors, not a different batch.
                    hidden = model.model(**batch, use_cache=False).last_hidden_state
                    logits = model.score(hidden)
                    positions = torch.arange(batch['input_ids'].shape[1], device='cuda:0')
                    last = ((batch['input_ids'] != tokenizer.pad_token_id) * positions).max(-1).values
                    reference = logits[torch.arange(len(indices), device=logits.device), last.to(logits.device)].float().reshape(-1)
                    reference = reference.to(values.device)
                    difference = float((values - reference).abs().max())
                    assert difference < 1e-5, 'Native scalar pooling mismatch'
                    repeated = model(**batch, use_cache=False).logits.float().reshape(-1)
                    repeat_difference = float((values - repeated).abs().max())
                    assert repeat_difference < 1e-5, 'Repeated identical-tensor mismatch'
                    summary['native_controls'] = dict(passed=True, native_max_abs=difference,
                        repeated_max_abs=repeat_difference, identical_input_tensors=True)
                    native_checked = True
                scores[indices] = values.cpu().numpy()
                if start % 256 == 0:
                    print('ARENA_EVENT ' + json.dumps(dict(stage='reward_progress', size=args.size,
                        cohort=name, completed_sequences=min(start + 2, len(order)), total_sequences=len(order),
                        seconds=time.monotonic() - begin)), flush=True)
        for gpu in used:
            torch.cuda.synchronize(gpu)
        assert np.isfinite(scores).all()
        lengths = np.asarray([len(s) for s in sequences], dtype=np.int64).reshape(-1, 2)
        original = np.asarray([m['original_tokens'] for m in metadata], dtype=np.int64).reshape(-1, 2)
        hashes = np.asarray([m['token_ids_sha256'] for m in metadata]).reshape(-1, 2)
        np.savez_compressed(out / ('reward_' + args.size + '_' + name + '.npz'),
            ids=frame.id.to_numpy(), rewards=scores.reshape(-1, 2), lengths=lengths,
            original_lengths=original, token_hashes=hashes)
        summary['cohorts'][name] = dict(rows=len(frame), seconds=time.monotonic() - begin,
            retained_tokens=int(lengths.sum()), original_tokens=int(original.sum()),
            truncated_sequences=int((original > lengths).sum()),
            history_counts=history,
            input_token_hash=hashlib.sha256(hashes.tobytes()).hexdigest(),
            peak_allocated_bytes=[torch.cuda.max_memory_allocated(g) for g in used])
        save(out / ('reward_' + args.size + '_summary.json'), summary)
    assert native_checked


if __name__ == '__main__':
    main()
