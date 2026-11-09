from types import SimpleNamespace
import unittest

from arena.span035_recent import select_recent, turn_text, HEAD, TAIL


class CharacterTokenizer:
    def __call__(self, text, **kw):
        return SimpleNamespace(input_ids=[ord(c) for c in text][:kw['max_length']])
    def decode(self, ids): return ''.join(chr(i) for i in ids)


class RecentTests(unittest.TestCase):
    def setUp(self): self.tokenizer = CharacterTokenizer()
    def test_short_and_single_long_are_native(self):
        for p, limit in [(['hello'], 200), (['x' * 500], 100)]:
            body = turn_text(p[0], 'A', 'B')
            actual, info = select_recent(p, ['A'], ['B'], self.tokenizer, limit)
            self.assertEqual(actual, HEAD + body[:limit] + TAIL)
            self.assertFalse(info['changed'])
    def test_complete_recent_turns_and_swap(self):
        p, a, b = ['old' * 60, 'middle', 'latest'], ['oldA', 'middleA', 'lastA'], ['oldB', 'middleB', 'lastB']
        limit = len(turn_text(p[1], a[1], b[1]) + turn_text(p[2], a[2], b[2]))
        actual, info = select_recent(p, a, b, self.tokenizer, limit)
        self.assertTrue(info['changed']); self.assertEqual(info['first_kept_turn'], 1)
        self.assertEqual(actual, HEAD + turn_text(p[1], a[1], b[1]) + turn_text(p[2], a[2], b[2]) + TAIL)
        reverse, ri = select_recent(p, b, a, self.tokenizer, limit)
        self.assertEqual(ri['first_kept_turn'], info['first_kept_turn'])
        self.assertIn('lastB\n\n### Assistant B:\nlastA', reverse)
    def test_oversized_latest_and_unaligned_keep_original(self):
        for p, a, b in [(['old', 'latest' * 100], ['A', 'A'], ['B', 'B']),
                        (['long' * 100, 'extra'], ['A'], ['B'])]:
            body = ''.join(turn_text(*v) for v in zip(p, a, b))
            actual, info = select_recent(p, a, b, self.tokenizer, 100)
            self.assertFalse(info['changed']); self.assertEqual(actual, HEAD + body[:100] + TAIL)
    def test_null_content_is_not_rewritten(self):
        p, a, b = ['old' * 80, 'last'], ['old', None], ['old', 'B']
        actual, info = select_recent(p, a, b, self.tokenizer, 100)
        self.assertTrue(info['changed']); self.assertIn('### Assistant A:\nnull', actual)


if __name__ == '__main__': unittest.main()
