import unittest
from arena.kaggle_control import sanitize_event


class ProgressEvents(unittest.TestCase):
    def test_worker_identity_survives_and_text_secrets_do_not(self):
        record=dict(stage='inference_progress',role='m0_reverse',cohort='uniform',
                    batch=7,rows=640,seconds=12.3,prompt='private conversation',
                    response_a='private answer',api_key='secret',cookie='secret')
        self.assertEqual(sanitize_event(record),dict(stage='inference_progress',
            role='m0_reverse',cohort='uniform',batch=7,rows=640,seconds=12.3))

    def test_identifiers_are_bounded_and_batch_is_a_nonnegative_integer(self):
        for value in ['text with spaces','a'*65,{'prompt':'private'},None]:
            self.assertNotIn('role',sanitize_event(dict(role=value)))
        for value in [-1,True,1.5,'3',None]:
            self.assertNotIn('batch',sanitize_event(dict(batch=value)))
        self.assertEqual(sanitize_event(dict(role='m3',cohort='routed.test',batch=0)),
                         dict(role='m3',cohort='routed.test',batch=0))


if __name__=='__main__':unittest.main()
