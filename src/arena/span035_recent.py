"""Complete recent-turn extraction; the native budget and serialization remain fixed."""
import json

HEAD = '<|The Start of Conversation between a User and two Assistants|>'
TAIL = '<|The End of Conversation between a User and two Assistants|>\n'


def turn_text(prompt, a, b):
    p, a, b = ['null' if x is None else str(x) for x in [prompt, a, b]]
    return f'\n### User:\n{p}\n\n### Assistant A:\n{a}\n\n### Assistant B:\n{b}\n'


def select_recent(prompts, responses_a, responses_b, tokenizer, limit=4096):
    columns = [prompts, responses_a, responses_b]
    turns = list(zip(*columns))
    parts = [turn_text(*t) for t in turns]
    def encode(text):
        return tokenizer(text, add_special_tokens=False, max_length=limit + 1, truncation=True).input_ids
    whole = encode(''.join(parts))
    original = HEAD + tokenizer.decode(whole[:limit]) + TAIL
    info = dict(changed=False, total_turns=len(turns), first_kept_turn=0, kept_turns=len(turns), reason='not_clipped')
    if len(whole) <= limit:
        return original, info
    info['reason'] = 'single_or_unaligned'
    if len(turns) < 2 or len({len(c) for c in columns}) != 1:
        return original, info
    swapped_parts = [turn_text(p, b, a) for p, a, b in turns]
    if len(encode(''.join(swapped_parts))) <= limit:
        info['reason'] = 'orientation_threshold'
        return original, info
    best, first = None, len(turns) - 1
    for start in range(len(turns) - 1, -1, -1):
        normal = encode(''.join(parts[start:]))
        reverse = encode(''.join(swapped_parts[start:]))
        if max(len(normal), len(reverse)) > limit:
            break
        best, first = normal, start
    if best is None:
        info['reason'] = 'latest_turn_too_long'
        return original, info
    info.update(changed=True, first_kept_turn=first, kept_turns=len(turns) - first,
                reason='complete_recent_suffix', body_tokens=len(best))
    return HEAD + tokenizer.decode(best) + TAIL, info


def processor_class(native_class):
    class RecentProcessor(native_class):
        def build_conversation(self, prompts, responses_a, responses_b):
            conversation, info = select_recent(prompts, responses_a, responses_b, self.tokenizer, self.max_length)
            self.last_selection_info = info
            return conversation
    return RecentProcessor


def describe_record(record, tokenizer, limit=4096):
    return select_recent(*[json.loads(record[k]) for k in ['prompt', 'response_a', 'response_b']], tokenizer, limit)
