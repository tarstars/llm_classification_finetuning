from contextlib import nullcontext
from types import SimpleNamespace
import unittest

from scripts.submit_span_010_order import leaderboard


class LeaderboardGoal(unittest.TestCase):
    def fake_api(self,rank):
        rows=[dict(teamId=1000+i,score=str(.8+i*.0001),submissionDate='date') for i in range(235)]
        rows[rank-1]['teamId']=17244411
        state=dict(cursor=None,calls=0)
        def paging(request,size,cursor):state['cursor']=cursor
        def get(request):
            state['calls']+=1;start=int(state['cursor'] or 0)
            page=rows[start:start+100]
            return SimpleNamespace(submissions=[SimpleNamespace(to_dict=lambda r=r:r) for r in page],
                next_page_token=str(start+100) if start+100<len(rows) else None)
        api=SimpleNamespace(_set_paging=paging,build_kaggle_client=lambda:nullcontext(
            SimpleNamespace(competitions=SimpleNamespace(competition_api_client=SimpleNamespace(get_leaderboard=get)))))
        return api,state

    def test_above_twentieth_requires_rank_nineteen_or_better(self):
        for rank,achieved in [(19,True),(20,False),(219,False)]:
            api,state=self.fake_api(rank);result=leaderboard(api)
            self.assertEqual(result['rank'],rank)
            self.assertEqual(result['target_achieved'],achieved)
            self.assertEqual(result['teams'],235)
            self.assertEqual(state['calls'],3)

    def test_duplicate_team_is_not_a_completion_proof(self):
        api,_=self.fake_api(19)
        with api.build_kaggle_client() as client:pass
        original=client.competitions.competition_api_client.get_leaderboard
        def duplicate(request):
            result=original(request)
            if result.next_page_token=='100':
                result.submissions[0]=SimpleNamespace(to_dict=lambda:dict(teamId=17244411,score='.8'))
            return result
        client.competitions.competition_api_client.get_leaderboard=duplicate
        api.build_kaggle_client=lambda:nullcontext(client)
        with self.assertRaises(AssertionError):leaderboard(api)


if __name__=='__main__':unittest.main()
