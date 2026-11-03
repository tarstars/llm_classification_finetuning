"""Span-specific retained-content helpers; frozen baseline preprocessing stays unchanged."""
from .joint_data import JointTokenizer, head_tail


class SpanTokenizer(JointTokenizer):
    def retained_parts(self, prompt, a, b):
        available = self.max_length - self.overhead
        question = min(len(prompt), min(256, max(32, available // 4)))
        remaining = available - question
        a_budget, b_budget = min(len(a), remaining // 2), min(len(b), remaining // 2)
        unused = remaining - a_budget - b_budget
        take = min(unused, len(a) - a_budget)
        a_budget += take
        unused -= take
        b_budget += min(unused, len(b) - b_budget)
        return (head_tail(prompt, question, self.truncated),
                head_tail(a, a_budget, self.truncated), head_tail(b, b_budget, self.truncated))

    def combine_with_spans(self, prompt, a, b):
        p, a, b = self.retained_parts(prompt, a, b)
        first = len(self.bos) + len(self.prefix) + len(p) + len(self.a_marker)
        second = first + len(a) + len(self.b_marker)
        ids = self.bos + self.prefix + p + self.a_marker + a + self.b_marker + b + self.suffix + self.eos
        assert len(ids) <= self.max_length
        return ids, (first, first + len(a)), (second, second + len(b))

    def reward_pair(self, prompt, a, b):
        """Use common retained content from both orientations (at most one token shorter)."""
        p, first, second = self.retained_parts(prompt, a, b)
        _, swapped_b, swapped_a = self.retained_parts(prompt, b, a)
        first = head_tail(a, min(len(first), len(swapped_a)), self.truncated)
        second = head_tail(b, min(len(second), len(swapped_b)), self.truncated)
        decode = lambda ids: self.tokenizer.decode(ids, skip_special_tokens=False)
        q, answer_a, answer_b = map(decode, (p, first, second))
        sequences = []
        for answer in (answer_a, answer_b):
            encoded = self.tokenizer.apply_chat_template(
                [{'role': 'user', 'content': q}, {'role': 'assistant', 'content': answer}],
                tokenize=True, add_generation_prompt=False, return_dict=False)
            # Transformers versions differ in the default/result container.
            if hasattr(encoded, 'keys'):
                encoded = encoded['input_ids']
            if hasattr(encoded, 'tolist'):
                encoded = encoded.tolist()
            assert isinstance(encoded, list) and all(isinstance(x, int) for x in encoded), 'Expected flat token IDs'
            sequences.append(encoded)
        assert all(len(ids) <= self.max_length for ids in sequences), 'Reward template exceeds the fixed budget'
        return sequences
