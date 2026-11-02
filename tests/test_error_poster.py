import json
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from arena.error_poster import marked_segments, safe_json, top_union
from arena.joint_data import head_tail


class ErrorPoster(unittest.TestCase):
    def test_untrusted_text_cannot_end_embedded_json_script(self):
        payload = {'text': '</script><script>alert("x")</script>&\u2028\u2029😺'}
        encoded = safe_json(payload)
        self.assertNotIn('<', encoded)
        self.assertNotIn('&', encoded)
        self.assertNotIn('\u2028', encoded)
        self.assertEqual(json.loads(encoded), payload)
        with self.assertRaises(ValueError):
            safe_json({'loss': float('nan')})

    def test_highlighting_matches_actual_head_tail_selection(self):
        text = 'abcdefghijklmno'
        offsets = [(i, i + 1) for i in range(len(text))]
        for budget in range(len(text) + 2):
            with self.subTest(budget=budget):
                segments = marked_segments(text, offsets, budget, 3)
                self.assertEqual(''.join(s['text'] for s in segments), text)
                retained = ''.join(s['text'] for s in segments if s['kept'])
                tokens = head_tail(list(text), budget, ['!'] * 3)
                self.assertEqual(retained, ''.join(t for t in tokens if t != '!'))
        self.assertEqual(marked_segments('', [], 0, 3), [{'text': '', 'kept': True}])

    def test_category_selection_ranks_separately_and_deduplicates(self):
        frame = pd.DataFrame([
            {'cohort': 'train', 'id': 9, 'loss': 5., 'tag_code': False},
            {'cohort': 'train', 'id': 8, 'loss': 5., 'tag_code': False},
            {'cohort': 'train', 'id': 7, 'loss': 3., 'tag_code': True},
            {'cohort': 'test', 'id': 6, 'loss': 2., 'tag_code': True},
            {'cohort': 'test', 'id': 5, 'loss': 1., 'tag_code': True},
        ])
        selected = top_union(frame, [('all', '', ''), ('code', '', '')], 1)
        self.assertEqual(selected.id.tolist(), [8, 7, 6])
        self.assertFalse(selected.index.duplicated().any())


if __name__ == '__main__':
    unittest.main()
