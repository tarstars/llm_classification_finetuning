import unittest
import numpy as np

from arena.span026_runtime import prefix_mask,projection


class ContextRuntimeTests(unittest.TestCase):
    def test_selector_only_uses_raw_formatted_text(self):
        records=[dict(prompt='["p"]',response_a='["a"]',response_b='["b"]',winner_model_a=1),
            dict(prompt='["'+('x'*4200)+'"]',response_a='["a"]',response_b='["b"]',winner_model_a=0)]
        def tokenizer(texts,**kwargs):
            self.assertEqual(kwargs,dict(add_special_tokens=False,max_length=4097,truncation=True,padding=False))
            self.assertTrue(all(isinstance(t,str) and 'winner_model_a' not in t for t in texts))
            return dict(input_ids=[list(range(min(len(t),4097))) for t in texts])
        np.testing.assert_array_equal(prefix_mask(records,tokenizer,batch_size=1),[False,True])

    def test_full_pass_warmup_selector_and_worst_parent_are_charged(self):
        r=projection(30912,274,276,270,128,64,.006,200,100,extra_parent_startup_seconds=20)
        expected=30912+(276-274)*25000/512*1.2+20+128/64*.006*25000*1.2+200+100
        self.assertAlmostEqual(r['projection_seconds'],expected)
        self.assertFalse(r['divides_already_parallel_wall_time_by_gpu_count'])
        self.assertFalse(r['official_submission_eligible'])
        slower=projection(30912,274,276,290,128,64,.006,200,100)
        self.assertGreater(slower['projection_seconds'],r['projection_seconds'])
        self.assertFalse(slower['engineering_screen_passed'])

    def test_invalid_runtime_inputs_fail(self):
        with self.assertRaises(ValueError):projection(1,1,1,1,1,0,.01,1,1)
        with self.assertRaises(ValueError):projection(1,1,1,1,1,1,1.1,1,1)


if __name__=='__main__':unittest.main()
