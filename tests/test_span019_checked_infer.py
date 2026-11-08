import unittest
import numpy as np
from arena.span019_checked_infer import repeated_logit_guard


class RepeatedLogitTests(unittest.TestCase):
    def test_rejects_a_common_logit_shift_even_if_softmax_would_match(self):
        reports=[]
        def forward(models,entries,cohort):
            logits=np.array([[1.,2.,3.]],dtype=np.float32)
            if cohort=='parallel_repeat':logits+=1
            return logits,{'rows':1}
        checked=repeated_logit_guard(forward,reports.append)
        checked([],[],'parallel_reference')
        with self.assertRaisesRegex(ValueError,'raw logits differ'):
            checked([],[],'parallel_repeat')
        self.assertFalse(reports[0]['passed'])
        self.assertEqual(reports[0]['raw_logit_max_abs'],1.)

    def test_passes_through_original_logits_and_profile(self):
        logits=np.array([[1.,2.,3.]],dtype=np.float32);profile={'rows':1};reports=[]
        checked=repeated_logit_guard(lambda *a:(logits,profile),reports.append)
        checked([],[],'parallel_reference')
        actual,actual_profile=checked([],[],'parallel_repeat')
        self.assertIs(actual,logits);self.assertIs(actual_profile,profile)
        self.assertTrue(reports[0]['passed'])


if __name__=='__main__':unittest.main()
