import unittest

from arena.span026_context import body_and_spans,retained_characters


class NativePrefixTests(unittest.TestCase):
    def test_exact_native_body_and_content_spans(self):
        r=dict(prompt='["Q", null]',response_a='["AAA", "x"]',response_b='["BBBB", "y"]')
        text,spans=body_and_spans(r)
        self.assertEqual(text,'\n### User:\nQ\n\n### Assistant A:\nAAA\n\n### Assistant B:\nBBBB\n'
            '\n### User:\nnull\n\n### Assistant A:\nx\n\n### Assistant B:\ny\n')
        for span in spans:
            self.assertEqual(len(text[span['start']:span['end']]),span['end']-span['start'])
        a,t=retained_characters(spans,text.index('BBBB')+2)
        self.assertEqual(a,dict(prompt=1,response_a=3,response_b=2))
        self.assertEqual(t,dict(prompt=5,response_a=4,response_b=5))

    def test_native_zip_boundary_and_prefix_monotonicity(self):
        text,spans=body_and_spans(dict(prompt='["a","b"]',response_a='["c"]',response_b='["d"]'))
        self.assertNotIn('b',text)
        before={k:0 for k in ['prompt','response_a','response_b']}
        for cutoff in range(len(text)+1):
            now,total=retained_characters(spans,cutoff)
            self.assertTrue(all(before[k]<=now[k]<=total[k] for k in now));before=now
        self.assertEqual(before,total)
        with self.assertRaises(ValueError):retained_characters(spans,-1)


if __name__=='__main__':unittest.main()
