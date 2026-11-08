import unittest
from arena.leaderboard import collect_board


class LeaderboardTests(unittest.TestCase):
    def test_replayed_first_page_retries_same_cursor_without_inventing_teams(self):
        first=[dict(teamId=1,score='.8'),dict(teamId=2,score='.9')]
        second=[dict(teamId=3,score='1.0')]
        requests=[]
        def fetch(cursor):
            requests.append(cursor)
            if len(requests) in [1,2]:return first,'after_first'
            return second,None
        rows,proof=collect_board(fetch)
        self.assertEqual(requests,[None,'after_first','after_first'])
        self.assertEqual([r['teamId'] for r in rows],[1,2,3])
        self.assertTrue(proof['complete']);self.assertEqual(len(proof['rejected_pages']),1)

    def test_persistent_replays_never_report_complete(self):
        def fetch(cursor):return [dict(teamId=1,score='.8')],'cursor'
        with self.assertRaises(RuntimeError):collect_board(fetch,retries=2)


if __name__=='__main__':unittest.main()
