"""Interpret a freshly observed official receipt and complete current leaderboard."""
from datetime import datetime
import math

TERMINAL={'COMPLETE','ERROR','FAILED','INVALID','CANCELED','CANCELLED'}


def instant(value):
    parsed=datetime.fromisoformat(value.replace('Z','+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('Observation timestamp must have a timezone')
    return parsed


def assess(official,receipt,official_observed_at,board=None,incumbent=.82987):
    if str(official['ref'])!=str(receipt['ref']):
        raise ValueError('Observed result belongs to another receipt')
    status=str(official['status']).split('.')[-1]
    observed=instant(official_observed_at)
    result=dict(official_receipt=str(receipt['ref']),official_status=status,
        official_terminal=status in TERMINAL,goal_achieved=False)
    if status!='COMPLETE':
        return result
    score=float(official['public_score'])
    if not math.isfinite(score) or score<0:
        raise ValueError('Completed receipt has no finite nonnegative score')
    if board is None or not board.get('complete') or not board.get('pagination',{}).get('complete'):
        raise ValueError('A complete paginated current board is required')
    if instant(board['observed_at'])<observed:
        raise ValueError('Board predates the confirmed receipt observation')
    if board['team_id']!=17244411:
        raise ValueError('Board describes another team')
    rank=board['rank'];teams=board['teams']
    if isinstance(rank,bool) or not isinstance(rank,int) or not 1<=rank<=teams or teams<20:
        raise ValueError('Invalid board rank or coverage')
    expected=min(float(incumbent),score)
    if not math.isfinite(float(board['score'])) or not math.isclose(float(board['score']),expected,rel_tol=0,abs_tol=.0000051):
        raise ValueError('Board best score does not match known legitimate completed candidates')
    achieved=rank<=19
    if board.get('target_achieved')!=achieved:
        raise ValueError('Recorded target flag contradicts actual rank')
    result.update(candidate_score=score,incumbent_score=float(incumbent),
        gain_vs_incumbent=float(incumbent)-score,candidate_improved_best=score<float(incumbent),
        rank=rank,teams=teams,leaderboard_best_score=float(board['score']),goal_achieved=achieved,
        board_observed_at=board['observed_at'])
    return result
