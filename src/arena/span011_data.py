"""Label-blind parsing, grouping and quota selection for later Arena votes."""
from datetime import datetime,timezone
import hashlib
import json
import re
import unicodedata

LABELS={'model_a':0,'model_b':1,'tie':2,'both_bad':2}
CUTOFF=datetime(2024,8,6,tzinfo=timezone.utc)


def digest(value):return hashlib.sha256(value.encode('utf-8')).hexdigest()


def normalize(text):
    return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',text)).strip().casefold()


def pair_key(prompts,answers_a,answers_b):
    p=[normalize('null' if x is None else str(x)) for x in prompts]
    a=[normalize('null' if x is None else str(x)) for x in answers_a]
    b=[normalize('null' if x is None else str(x)) for x in answers_b]
    return digest(json.dumps([p,*sorted([a,b])],ensure_ascii=False,separators=(',',':')))


def text_message(message,role):
    if not isinstance(message,dict) or message.get('role')!=role:return None
    parts=message.get('content')
    if not isinstance(parts,list) or not parts:return None
    text=[]
    for part in parts:
        if not isinstance(part,dict) or part.get('type')!='text' or part.get('image'):
            return None
        value=part.get('text')
        if not isinstance(value,str):return None
        text.append(value)
    result='\n'.join(text)
    return result if result.strip() else None


def parse_pair(row):
    """Return eligibility/input metadata; deliberately never access winner/model IDs."""
    if row.get('language')!='en':return None,'language'
    if row.get('evaluation_order')!=1:return None,'evaluation_order'
    stamp=row.get('timestamp')
    if not isinstance(stamp,datetime):return None,'timestamp'
    if stamp.tzinfo is None:stamp=stamp.replace(tzinfo=timezone.utc)
    stamp=stamp.astimezone(timezone.utc)
    if stamp<CUTOFF:return None,'before_cutoff'
    aa,bb=row.get('conversation_a'),row.get('conversation_b')
    if not isinstance(aa,list) or not isinstance(bb,list) or len(aa)!=2 or len(bb)!=2:
        return None,'turn_count'
    user_a,user_b=text_message(aa[0],'user'),text_message(bb[0],'user')
    a,b=text_message(aa[1],'assistant'),text_message(bb[1],'assistant')
    if any(x is None for x in [user_a,user_b,a,b]):return None,'nontext_or_empty'
    if user_a!=user_b:return None,'mismatched_user'
    uid,session=row.get('id'),row.get('evaluation_session_id')
    if not isinstance(uid,str) or not uid or not isinstance(session,str) or not session:
        return None,'missing_identity'
    return dict(id=int(digest('numeric:'+uid)[:16],16)&((1<<63)-1),source_uuid=uid,
        session=session,prompt_group=digest(normalize(user_a)),pair_hash=pair_key([user_a],[a],[b]),
        row_priority=digest('42:representative:'+uid),timestamp=stamp.isoformat(),
        prompt=json.dumps([user_a],ensure_ascii=False),
        response_a=json.dumps([a],ensure_ascii=False),response_b=json.dumps([b],ensure_ascii=False)),None


class Groups:
    def __init__(self):self.parent={}
    def find(self,key):
        self.parent.setdefault(key,key)
        root=key
        while self.parent[root]!=root:root=self.parent[root]
        while self.parent[key]!=key:
            following=self.parent[key];self.parent[key]=root;key=following
        return root
    def join(self,left,right):
        a,b=self.find(left),self.find(right)
        if a!=b:self.parent[max(a,b)]=min(a,b)


def split_selected(selected,learning_quotas):
    result=[]
    for index,amount in enumerate(learning_quotas):
        records=[r for r in selected if r['length_bin']==index]
        records.sort(key=lambda r:(digest('43:split:'+r['group_id']),r['source_uuid']))
        if len(records)<amount:raise ValueError('Insufficient bin support')
        for rank,row in enumerate(records):
            result.append(dict(row,split='learning' if rank<amount else 'confirmation'))
    return result


def write_inference_inputs(path,records):
    """Plain Parquet v1 types work in the pinned older GPU image; targets excluded."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    columns=['id','prompt','response_a','response_b']
    schema=pa.schema([('id',pa.int64()),*[(k,pa.string()) for k in columns[1:]]])
    table=pa.Table.from_pylist([{k:r[k] for k in columns} for r in records],schema=schema)
    pq.write_table(table,path,version='1.0',compression='snappy')
