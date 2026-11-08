"""Read a coherent sorted board despite a server replaying an earlier page."""
import math


def collect_board(fetch, retries=4, max_pages=100):
    rows,ids,cursors,rejected=[],set(),set(),[]
    cursor=None
    for page in range(max_pages):
        for attempt in range(retries):
            values,next_cursor=fetch(cursor)
            scores=[float(v['score']) for v in values]
            keys=[int(v['teamId']) for v in values]
            reason=None
            if any(not math.isfinite(s) or s<0 for s in scores):reason='invalid_scores'
            elif scores!=sorted(scores):reason='unsorted_page'
            elif len(set(keys))!=len(keys):reason='duplicate_within_page'
            elif ids.intersection(keys):reason='replayed_or_overlapping_page'
            elif rows and scores and scores[0]<float(rows[-1]['score']):reason='backward_page'
            elif not values and next_cursor:reason='empty_nonterminal_page'
            if reason:
                rejected.append(dict(page=page+1,attempt=attempt+1,reason=reason,rows=len(values)))
                # Crucially keep the REQUESTEDcursor. A replayed first page's
                # returned cursor would otherwise restart the pagination chain.
                continue
            break
        else:
            raise RuntimeError('Could not read a coherent leaderboard page; completion unproven')
        rows.extend(values);ids.update(keys)
        if not next_cursor:
            return rows,dict(complete=True,pages=page+1,rejected_pages=rejected)
        if next_cursor in cursors:raise RuntimeError('Repeated accepted leaderboard cursor')
        cursors.add(next_cursor);cursor=next_cursor
    raise RuntimeError('Leaderboard page limit reached; completion unproven')
