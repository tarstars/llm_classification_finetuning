import copy
import unittest
from arena.official_outcome import assess


class OutcomeTests(unittest.TestCase):
    def fixture(self,rank=19,score=.8289):
        official=dict(ref=58143104,status='SubmissionStatus.COMPLETE',public_score=score)
        receipt=dict(ref=58143104)
        board=dict(complete=True,pagination=dict(complete=True,pages=4),team_id=17244411,
            observed_at='2026-11-09T08:00:01+00:00',rank=rank,teams=310,
            score=min(.82987,score),target_achieved=rank<=19)
        return official,receipt,'2026-11-09T08:00:00+00:00',board

    def test_rank19_completes_but_rank20_does_not(self):
        self.assertTrue(assess(*self.fixture(19))['goal_achieved'])
        self.assertFalse(assess(*self.fixture(20))['goal_achieved'])

    def test_pending_receipt_never_completes_even_with_good_board(self):
        args=list(self.fixture(1));args[0]['status']='SubmissionStatus.PENDING';args[0]['public_score']=''
        self.assertFalse(assess(*args)['goal_achieved'])

    def test_official_display_rounding_is_allowed(self):
        args=list(self.fixture(19,.828895));args[3]['score']=.82890
        self.assertTrue(assess(*args)['goal_achieved'])

    def test_stale_incomplete_or_wrong_team_board_is_rejected(self):
        for change in ['stale','incomplete','wrong_team','wrong_score','wrong_flag']:
            with self.subTest(change=change):
                args=list(copy.deepcopy(self.fixture()))
                if change=='stale':args[3]['observed_at']='2026-11-09T07:59:59+00:00'
                if change=='incomplete':args[3]['pagination']['complete']=False
                if change=='wrong_team':args[3]['team_id']=1
                if change=='wrong_score':args[3]['score']=.80
                if change=='wrong_flag':args[3]['target_achieved']=False
                with self.assertRaises(ValueError):assess(*args)

    def test_worse_candidate_keeps_known_incumbent_without_claiming_gain(self):
        result=assess(*self.fixture(23,.84))
        self.assertFalse(result['goal_achieved']);self.assertFalse(result['candidate_improved_best'])
        self.assertEqual(result['leaderboard_best_score'],.82987)

    def test_wrong_receipt_or_nonfinite_complete_score_is_rejected(self):
        args=list(self.fixture());args[0]['ref']=123
        with self.assertRaises(ValueError):assess(*args)
        args=list(self.fixture());args[0]['public_score']=float('nan')
        with self.assertRaises(ValueError):assess(*args)


if __name__=='__main__':unittest.main()
