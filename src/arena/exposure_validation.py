"""Reconcile actual cycle records with checkpoint counters and training-row truth."""
from collections import Counter
import math


def validate_exposure(events, state, first_step, final_step, training_labels, max_length):
    assert [event['global_step'] for event in events]==list(range(first_step+1,final_step+1)), 'Missing or repeated optimizer cycles'
    counts={'cycles':0,'accepted_updates':0,'skipped_updates':0,'processed_tokens':0,
            'accepted_tokens':0,'processed_rows':0,'accepted_rows':0}
    row_counts={kind:Counter() for kind in ('processed','accepted')}
    classes={kind:Counter() for kind in ('processed','accepted')}
    previous=None
    for event in events:
        assert event['rank']==state['rank']
        ids,labels,lengths=event['ids'],event['canonical_labels'],event['token_lengths']
        assert len(ids)==len(labels)==len(lengths)>0
        assert all(rid in training_labels and training_labels[rid]==y for rid,y in zip(ids,labels)), 'Non-training row or wrong canonical class'
        assert all(0<int(length)<=max_length for length in lengths)
        skipped=event['skipped']
        assert isinstance(skipped,bool)
        assert event['scheduler_after']-event['scheduler_before']==int(not skipped)
        for key in ('scale_before','scale_after'):
            assert event[key] is not None and math.isfinite(event[key]) and event[key]>0
        if skipped:
            assert event['scale_after']<=event['scale_before']
        if previous:
            assert event['scheduler_before']==previous['scheduler_after']
            assert event['scale_before']==previous['scale_after']
        counts['cycles']+=1
        counts['skipped_updates' if skipped else 'accepted_updates']+=1
        for kind in ['processed']+([] if skipped else ['accepted']):
            counts[kind+'_rows']+=len(ids)
            counts[kind+'_tokens']+=sum(lengths)
            row_counts[kind].update(ids)
            classes[kind].update(labels)
        assert event['cumulative']==counts, 'Per-cycle cumulative counters disagree with actual rows'
        previous=event
    assert state['counts']==counts and state['last_global_step']==final_step
    for kind in ('processed','accepted'):
        assert Counter({int(k):v for k,v in state[kind+'_ids'].items()})==row_counts[kind]
        assert Counter({int(k):v for k,v in state[kind+'_classes'].items()})==classes[kind]
    return {'verified':True,'rank':state['rank'],'cycles':len(events),
            'accepted_updates':counts['accepted_updates'],'skipped_updates':counts['skipped_updates'],
            'first_scheduler_position':events[0]['scheduler_before'] if events else None,
            'last_scheduler_position':events[-1]['scheduler_after'] if events else None,
            'skip_vector':[event['skipped'] for event in events]}
