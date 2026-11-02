"""Reproducible, local dataset descriptors for the frozen preference experiment."""
from __future__ import annotations
import hashlib,importlib.metadata,json,os,re,time
from pathlib import Path
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from .data import ROOT,CLASSES,TEXT_COLUMNS,parse_turns,prompt_groups,json_write
from .joint_data import JointTokenizer

SEED=20261109
EDA_ROOT=ROOT/'eda/attempt_002'
REPO=Path(__file__).resolve().parents[2]
RESULTS=REPO/'results/eda_002'
TASK_PATTERNS={
    'code':r'```|\b(?:python|javascript|typescript|java|c\+\+|sql|debug|programming|algorithm|function|code|html|css|rust|pandas)\b',
    'math':r'\\(?:frac|sqrt|sum|int|begin)|\b(?:calculate|equation|algebra|integral|derivative|theorem|probability|mathematics|geometry)\b',
    'creative':r'\b(?:poem|poetry|story|fiction|lyrics|creative|character|screenplay)\b',
    'transformation':r'\b(?:translate|translation|summarize|summary|rewrite|rephrase|proofread|extract|classify)\b',
    'explanation':r'\b(?:explain|describe|why|what is|what are|how does|difference between)\b',
}
REFUSAL=r"\b(?:as an ai|i (?:cannot|can't|am unable to) (?:help|assist|provide|fulfil|fulfill)|i'?m sorry|i apologize|unable to (?:assist|help))\b"


def digest_text(s):return hashlib.sha256(s.encode()).hexdigest()


def canonical_comparison(prompt,a,b,label):
    reverse=a>b
    key=digest_text(json.dumps([prompt,min(a,b),max(a,b)],ensure_ascii=False))
    return key,([1,0,2][int(label)] if reverse else int(label))


def budgets(pn,an,bn,context,overhead,prompt_cap=256,marker_size=5):
    available=context-overhead
    pb=np.minimum(pn,min(prompt_cap,max(32,available//4)))
    answer=available-pb
    ab=np.minimum(an,answer//2);bb=np.minimum(bn,answer//2)
    unused=answer-ab-bb
    take=np.minimum(unused,an-ab);ab+=take;unused-=take
    bb+=np.minimum(unused,bn-bb)
    cuts=np.column_stack([pn>pb,an>ab,bn>bb])
    kept=np.column_stack([pb,ab,bb])-cuts*marker_size
    return cuts,kept


def load_frame():
    raw=pd.read_csv(ROOT/'raw/train.csv')
    parsed=raw.copy()
    turns={}
    for name in TEXT_COLUMNS:
        turns[name]=raw[name].map(lambda x:json.loads(x) if isinstance(x,str) else [])
        assert turns[name].map(lambda x:isinstance(x,list)).all()
        parsed[name]=raw[name].map(parse_turns)
    return raw,parsed,turns


def prepare(force=False):
    EDA_ROOT.mkdir(parents=True,exist_ok=True);RESULTS.mkdir(parents=True,exist_ok=True)
    selection=json.loads((ROOT/'runs/attempt_002/model_selection.json').read_text())
    tokenizer_dir=ROOT/'runs/attempt_002/jobs/qwen4b1024/output/joint_run/adapter'
    fingerprint={'raw_sha256':hashlib.sha256((ROOT/'raw/train.csv').read_bytes()).hexdigest(),
        'tokenizer_sha256':hashlib.sha256((tokenizer_dir/'tokenizer.json').read_bytes()).hexdigest(),
        'preprocessing_sha256':hashlib.sha256(Path(__file__).with_name('joint_data.py').read_bytes()).hexdigest(),
        'feature_program_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'lingua_version':importlib.metadata.version('lingua-language-detector')}
    assert fingerprint['preprocessing_sha256']==selection['config']['preprocessing_sha256']
    if not force and (EDA_ROOT/'features.parquet').exists() and (EDA_ROOT/'feature_fingerprint.json').exists():
        if json.loads((EDA_ROOT/'feature_fingerprint.json').read_text())==fingerprint:
            print('Using verified EDA feature cache',flush=True);return
    raw,frame,turns=load_frame()
    manifest=json.loads((ROOT/'runs/attempt_002/split_ids.json').read_text())
    split=np.full(len(raw),'train',dtype=object)
    for name in ['dev','confirmation']:split[np.isin(raw.id,manifest[name])]=name
    y=raw[CLASSES].to_numpy().argmax(1)
    assert raw[CLASSES].sum(1).eq(1).all() and raw[CLASSES].isin([0,1]).all().all()
    f=pd.DataFrame({'id':raw.id,'label':y,'split':split,'prompt_group':prompt_groups(frame),
        'model_a':raw.model_a,'model_b':raw.model_b})
    for name in TEXT_COLUMNS:
        text=frame[name]
        f[name+'_chars']=text.str.len()
        f[name+'_words']=text.str.count(r'\S+')
        f[name+'_turns']=turns[name].map(len)
        f[name+'_null_turns']=turns[name].map(lambda a:sum(v is None for v in a))
        f[name+'_non_string_turns']=turns[name].map(lambda a:sum(v is not None and not isinstance(v,str) for v in a))
        f[name+'_empty']=text.str.strip().eq('')
        f[name+'_nonascii_fraction']=text.str.count(r'[^\x00-\x7f]')/text.str.len().clip(lower=1)
        f[name+'_code']=text.str.contains(TASK_PATTERNS['code'],flags=re.I,regex=True)
        f[name+'_refusal']=text.str.contains(REFUSAL,flags=re.I,regex=True)
        f[name+'_list_count']=text.str.count(r'(?m)^\s*(?:[-*]|\d+\.)\s')
        f[name+'_heading_count']=text.str.count(r'(?m)^\s*#{1,6}\s')
    f['turn_mismatch']=(f.prompt_turns!=f.response_a_turns)|(f.prompt_turns!=f.response_b_turns)
    f['multi_turn']=f.prompt_turns>1
    f['answer_equal_exact']=frame.response_a.eq(frame.response_b)
    normalize=lambda s:re.sub(r'\s+',' ',s).strip()
    f['answer_equal_whitespace']=frame.response_a.map(normalize).eq(frame.response_b.map(normalize))
    f['one_answer_empty']=f.response_a_empty^f.response_b_empty
    f['both_answers_empty']=f.response_a_empty&f.response_b_empty
    f['one_refusal']=f.response_a_refusal^f.response_b_refusal
    f['both_refusals']=f.response_a_refusal&f.response_b_refusal
    f['one_code_answer']=f.response_a_code^f.response_b_code
    f['any_code_answer']=f.response_a_code|f.response_b_code
    f['wrapper_marker_in_content']=frame[TEXT_COLUMNS].apply(lambda c:c.str.contains(r'\[(?:USER REQUEST|ANSWER A|ANSWER B|DECISION|TRUNCATED)\]',regex=True)).any(axis=1)
    for task,pattern in TASK_PATTERNS.items():f['task_'+task]=frame.prompt.str.contains(pattern,flags=re.I,regex=True)
    f['task_tag_count']=f[['task_'+t for t in TASK_PATTERNS]].sum(1)
    f['answer_length_ratio']=(f[['response_a_chars','response_b_chars']].max(1)+1)/(f[['response_a_chars','response_b_chars']].min(1)+1)
    f['longer_side']=np.where(f.response_a_chars>=f.response_b_chars,0,1)
    f['refusal_side']=np.where(f.response_a_refusal,0,1)
    f['list_difference']=f.response_a_list_count-f.response_b_list_count
    overlap=[];comparison=[];canonical=[];canonical_labels=[]
    for p,a,b,label in zip(frame.prompt,frame.response_a,frame.response_b,y):
        sa=set(re.findall(r'\w+',a.lower()));sb=set(re.findall(r'\w+',b.lower()))
        overlap.append(len(sa&sb)/max(len(sa|sb),1))
        comparison.append(digest_text(json.dumps([p,a,b],ensure_ascii=False)))
        key,cy=canonical_comparison(p,a,b,label);canonical.append(key);canonical_labels.append(cy)
    f['answer_word_jaccard']=overlap
    f['comparison_hash']=comparison;f['canonical_comparison_hash']=canonical;f['canonical_label']=canonical_labels
    print('Basic features ready; tokenizing full corpus in bounded batches',flush=True)
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_dir,local_files_only=True)
    encoder=JointTokenizer(tokenizer,1024)
    for name in TEXT_COLUMNS:
        lens=[]
        for start in range(0,len(frame),256):
            ids=tokenizer(frame[name].iloc[start:start+256].tolist(),add_special_tokens=False,truncation=False,padding=False)['input_ids']
            lens.extend(map(len,ids))
        f[name+'_tokens']=lens
        print('Tokenized',name,flush=True)
    pn,an,bn=[f[k+'_tokens'].to_numpy() for k in TEXT_COLUMNS]
    f['total_content_tokens']=pn+an+bn
    for context,cap in [(1024,256),(2048,256),(2048,512),(4096,256),(4096,1024)]:
        tag=f'{context}_cap{cap}'
        cut,kept=budgets(pn,an,bn,context,encoder.overhead,cap,len(encoder.truncated))
        f['cut_'+tag]=cut.any(1);f['prompt_cut_'+tag]=cut[:,0]
        f['answer_cut_'+tag]=cut[:,1:].any(1)
        f['retained_fraction_'+tag]=kept.sum(1)/np.maximum(pn+an+bn,1)
    # Language IDs are estimates on prose-rich prompts; short/code-heavy text stays unknown.
    from lingua import LanguageDetectorBuilder
    detector=LanguageDetectorBuilder.from_all_spoken_languages().with_minimum_relative_distance(.15).build()
    prose=frame.prompt.map(lambda s:re.sub(r'```.*?```',' ',s,flags=re.S)[:3000])
    eligible=prose.map(lambda s:sum(c.isalpha() for c in s)>=50)
    language=np.full(len(f),'UNKNOWN_SHORT_OR_CODE',dtype=object)
    indices=np.flatnonzero(eligible)
    for start in range(0,len(indices),512):
        idx=indices[start:start+512]
        detected=detector.detect_languages_in_parallel_of(prose.iloc[idx].tolist())
        language[idx]=[v.name if v is not None else 'UNKNOWN_LOW_CONFIDENCE' for v in detected]
    f['prompt_language_estimated']=language
    f['language_group']=np.where(language=='ENGLISH','ENGLISH',np.where(pd.Series(language).str.startswith('UNKNOWN'),'UNKNOWN','OTHER_IDENTIFIED'))
    f.to_parquet(EDA_ROOT/'features.parquet',index=False)
    json_write(EDA_ROOT/'feature_fingerprint.json',fingerprint)
    schema={'rows':len(raw),'columns':raw.columns.tolist(),'dtypes':raw.dtypes.astype(str).to_dict(),
        'missing_cells':raw.isna().sum().to_dict(),'duplicate_ids':int(raw.id.duplicated().sum()),
        'invalid_target_rows':int((raw[CLASSES].sum(1)!=1).sum()),
        'visible_test_rows':len(pd.read_csv(ROOT/'raw/test.csv')),
        'test_columns':pd.read_csv(ROOT/'raw/test.csv',nrows=0).columns.tolist(),
        'raw_turn_element_types':{k:turns[k].map(lambda a:','.join(sorted({type(x).__name__ for x in a}))).value_counts().to_dict() for k in TEXT_COLUMNS},
        'feature_notes':{'language':'Lingua high-accuracy mode, minimum relative distance 0.15, 50+ alphabetic prompt characters after fenced-code removal; estimates, not verified labels.',
            'tasks':'Overlapping English regex tags; code/math tags also inspect explicit syntax. Not comprehensive semantic labels.',
            'duplicate':'Exact parsed-text triples; canonical comparison sorts answer texts and remaps A/B labels. No automatic deduplication.',
            'model_names':'Diagnostic metadata only, unavailable in test and excluded from deployed features.'},
        'task_patterns':TASK_PATTERNS,'refusal_pattern':REFUSAL,'fingerprint':fingerprint}
    json_write(RESULTS/'schema.json',schema)
    print('Saved EDA features:',f.shape,flush=True)
