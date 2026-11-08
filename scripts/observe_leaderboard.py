"""Observe the full official board; never merge replayed pages into a rank."""
from pathlib import Path
import argparse
import json
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.experiment import atomic_json,utc_now,sha256
from arena.leaderboard import collect_board
from arena.span004_support import api_client,read_call


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='results/span_019/latest_leaderboard.json')
    args=parser.parse_args();api=api_client()
    from kaggle.api.kaggle_api_extended import ApiGetLeaderboardRequest
    def fetch(cursor):
        request=ApiGetLeaderboardRequest();request.competition_name='llm-classification-finetuning'
        api._set_paging(request,100,cursor)
        with api.build_kaggle_client() as client:
            r=read_call(lambda:client.competitions.competition_api_client.get_leaderboard(request))
        return [x.to_dict() for x in r.submissions],r.next_page_token
    rows,pagination=collect_board(fetch)
    own=[(i+1,r) for i,r in enumerate(rows) if int(r['teamId'])==17244411]
    assert len(own)==1 and len(rows)>=20
    rank,row=own[0]
    result=dict(observed_at=utc_now(),complete=True,team_id=17244411,rank=rank,teams=len(rows),
        score=float(row['score']),rank19_score=float(rows[18]['score']),rank20_score=float(rows[19]['score']),
        leader_score=float(rows[0]['score']),target_achieved=rank<=19,
        latest_team_submission_date=row.get('submissionDate'),pagination=pagination,
        source_sha256={str(Path(__file__).relative_to(REPO)):sha256(Path(__file__)),
            'src/arena/leaderboard.py':sha256(REPO/'src/arena/leaderboard.py')})
    atomic_json(REPO/args.output,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
