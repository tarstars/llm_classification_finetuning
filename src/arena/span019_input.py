"""The pinned mirror's formatter and pooling semantics, without winner labels."""
import json


def flatten_turns(value):
    turns = json.loads(value)
    if not isinstance(turns, list):
        raise ValueError('Expected JSON conversation list')
    return '\n'.join('' if turn is None else str(turn) for turn in turns)


def render(query, answer_a, answer_b):
    return f'''User question:
    """{query}"""
    Answer A:
    """{answer_a}"""
    Answer B:
    """{answer_b}"""
    '''


def native_pool_index(input_ids, pad_id):
    """Match the card's argmax(first pad)-1, including left-pad and no-pad cases."""
    if not input_ids:
        raise ValueError('Empty input')
    first = next((i for i, token in enumerate(input_ids) if token == pad_id), 0)
    return (first - 1) % len(input_ids)


def encode(row, tokenizer):
    if tokenizer.pad_token_id != 0 or tokenizer.eos_token_id != 1 or tokenizer.padding_side != 'left':
        raise ValueError('Pinned native tokenizer contract changed')
    text = render(*(flatten_turns(row[k]) for k in ['prompt', 'response_a', 'response_b']))
    ids = tokenizer(text, truncation=True, max_length=2000)['input_ids']
    ids.append(tokenizer.eos_token_id)
    # Native pad(max_length=2000) does not truncate the appended 2001st token.
    offset = max(0, 2000 - len(ids))
    padded = [tokenizer.pad_token_id] * offset + ids
    pool = native_pool_index(padded, tokenizer.pad_token_id) - offset
    if not 0 <= pool < len(ids):
        raise ValueError('Native classifier pooled outside the actual sequence')
    return dict(input_ids=ids, native_position_offset=offset, pool_index=pool)
