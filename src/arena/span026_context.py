"""Explain the native conversation prefix using exact raw content spans."""
import json


def body_and_spans(record):
    columns=[json.loads(record[k]) for k in ['prompt','response_a','response_b']]
    parts=[];spans=[];position=0
    for turn,values in enumerate(zip(*columns)):
        for field,prefix,value in zip(['prompt','response_a','response_b'],
                ['\n### User:\n','\n\n### Assistant A:\n','\n\n### Assistant B:\n'],values):
            text='null' if value is None else str(value)
            parts.extend([prefix,text]);position+=len(prefix)
            spans.append(dict(field=field,turn=turn,start=position,end=position+len(text)))
            position+=len(text)
        parts.append('\n');position+=1
    return ''.join(parts),spans


def retained_characters(spans,cutoff):
    if cutoff<0:raise ValueError('Negative prefix cutoff')
    retained={k:0 for k in ['prompt','response_a','response_b']}
    total=retained.copy()
    for span in spans:
        length=span['end']-span['start'];assert length>=0
        total[span['field']]+=length
        retained[span['field']]+=max(0,min(span['end'],cutoff)-span['start'])
    return retained,total
