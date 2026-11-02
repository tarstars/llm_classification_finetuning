"""Select high-loss cases and build a standalone, local full-text error poster."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from .data import ROOT,TEXT_COLUMNS,parse_turns,json_write
from .joint_data import JointTokenizer
from .eda_analysis import temper,losses,scores

REPO=Path(__file__).resolve().parents[2]
POSTER_ROOT=ROOT/'eda/attempt_002/poster'
CATEGORIES=[
 ('all','All examples','Highest log loss within each scored cohort.'),
 ('confident_wrong','Confident wrong','Incorrect argmax with at least 80% predicted probability.'),
 ('true_tie','Human chose tie','Cases where the human label is tie.'),
 ('predicted_tie','Model chose tie','The model predicts tie but the human chose A or B.'),
 ('similar_answers','Similar answers','Answer embedding cosine above 0.9; an exploratory similarity cue.'),
 ('truncated','Any content omitted','At least one field loses content under the 1,024-token budget.'),
 ('prompt_cut','Question omitted','The question itself exceeds its allocated budget.'),
 ('multi_turn','Multiple turns','A conversation with more than one question/answer turn.'),
 ('code','Code-like wording','Lexical/syntax cue, not verified task annotation; false positives are possible.'),
 ('math','Math-like wording','English mathematical keywords or explicit mathematical syntax.'),
 ('creative','Creative writing','English creative-writing keywords; not a comprehensive topic label.'),
 ('transformation','Text transformation','Translation, summary, rewrite or extraction keywords.'),
 ('refusal','Refusal-like answer','At least one answer contains a refusal-like phrase.'),
 ('length_gap','Large length difference','One answer is more than four times as long as the other.'),
 ('nonascii','Non-ASCII question','More than 10% of question characters are non-ASCII; not a language label.'),
 ('identical','Identical answers','The two parsed answer texts are identical.'),
 ('order_sensitive','Order-sensitive','The raw winner changes after answer-order alignment; available for the training sample and development pilot.'),
 ('context_hurts','More context hurt','The paired 2,048-token diagnostic increases loss; available only on pilot rows.'),
 ('context_helps','More context helped','The paired diagnostic reduces loss; examples are still ranked by baseline loss.'),
 ('reviewed','Reviewed examples','Cases with existing analyst notes; selected reviews do not establish prevalence.'),
]


def safe_json(data):
    return json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('&','\\u0026').replace('<','\\u003c').replace('>','\\u003e').replace('\u2028','\\u2028').replace('\u2029','\\u2029')


def top_union(frame,categories,per_category):
    selected=set()
    for _,part in frame.groupby('cohort',sort=False):
        for category,_,_ in categories:
            mask=np.ones(len(part),bool) if category=='all' else part['tag_'+category].to_numpy(dtype=bool)
            selected.update(part.loc[mask].sort_values(['loss','id'],ascending=[False,True]).head(per_category).index)
    return frame.loc[sorted(selected)].copy()


def marked_segments(text,offsets,budget,marker_size):
    n=len(offsets)
    if n<=budget:return [{'text':text,'kept':True}]
    if budget<=marker_size+2:
        end=offsets[budget-1][1] if budget else 0
        return [{'text':text[:end],'kept':True},{'text':text[end:],'kept':False}]
    left=(budget-marker_size+1)//2;right=budget-marker_size-left
    end=offsets[left-1][1] if left else 0
    start=offsets[-right][0] if right else len(text)
    start=max(start,end)
    return [{'text':text[:end],'kept':True},{'text':text[end:start],'kept':False},{'text':text[start:],'kept':True}]


def build(per_category=12,allow_missing_train=False):
    if per_category<1:raise ValueError('per_category must be positive')
    selection=json.loads((ROOT/'runs/attempt_002/model_selection.json').read_text())
    features=pd.read_parquet(ROOT/'eda/attempt_002/all_enriched_features.parquet').set_index('id')
    cohorts=[];cohort_meta=[]
    def add(name,title,ids,y,p,scope,source):
        assert len(ids)==len(set(ids)) and np.isfinite(p).all() and np.allclose(p.sum(1),1)
        f=features.loc[ids].reset_index().copy()
        expected={'train':'train','dev':'dev','test':'confirmation'}[name]
        assert f.split.eq(expected).all() and np.array_equal(f.label,y)
        f['cohort']=name;f['loss']=losses(y,p);f['pred']=p.argmax(1);f['confidence']=p.max(1)
        for i in range(3):f['p'+str(i)]=p[:,i]
        ranks=pd.Series(np.arange(1,len(f)+1),index=f.sort_values(['loss','id'],ascending=[False,True]).index)
        f['cohort_rank']=ranks
        cohorts.append(f)
        cohort_meta.append({'key':name,'title':title,'rows':len(ids),'scope':scope,'source':source,
            'metrics':scores(y,p),'total_loss':float(f.loss.sum()),'model_hash':selection['adapter_sha256']})
    run=ROOT/'runs/attempt_002/jobs'
    train_path=run/'poster_train2048/output/poster_train_predictions.npz'
    if train_path.exists():
        train=np.load(train_path)
        train_summary=json.loads(train_path.with_name('poster_train_summary.json').read_text())
        sample=json.loads((POSTER_ROOT/'train_sample.json').read_text())
        assert train_summary['model_hash']==selection['adapter_sha256']
        assert train_summary['temperature']==selection['temperature']
        assert train_summary['rows']==len(train['ids'])==2048
        assert np.array_equal(train['ids'],sample['ids'])
        np.testing.assert_allclose(train['calibrated'],temper(train['average'],selection['temperature']),atol=1e-6)
        add('train','Training sample',train['ids'],train['labels'],train['calibrated'],
            'Fixed uniform sample of 2,048 out of 45,985 training-partition rows, seed 20261111. Highest losses are within this sample. Only 0.61 epochs were completed; per-row exposure was not logged.',
            'Frozen model; both answer orders; saved calibrated probabilities, temperature applied once.')
    elif not allow_missing_train:raise FileNotFoundError('Training cohort inference is pending')
    dev=np.load(run/'qwen4b1024/output/joint_run/dev_predictions.npz')
    add('dev','Development',dev['ids'],dev['labels'],temper(dev['probabilities'],selection['temperature']),
        'All 5,746 development rows. Used for checkpoint selection/calibration and subsequent EDA.',
        'Saved raw swap-average probabilities, then frozen temperature applied once.')
    test=np.load(run/'confirmation/output/confirmation_predictions.npz')
    add('test','Held-out test (local)',test['ids'],test['labels'],test['probabilities'],
        'All 5,746 local confirmation rows. This is not Kaggle hidden test data. Inspection now makes these examples available for future qualitative decisions.',
        'Saved confirmation probabilities already calibrated; no second temperature application.')
    frame=pd.concat(cohorts,ignore_index=True)
    reviewed=json.loads((REPO/'results/eda_002/manual_case_notes.json').read_text())
    reviewed+=json.loads((REPO/'results/error_poster_002/review_notes.json').read_text())
    review_map={int(r['id']):r for r in reviewed}
    diagnostic=np.load(run/'eda_diagnostic512/output/eda_model_predictions.npz')
    diagnostic_map={}
    if train_path.exists():
        for i,rid in enumerate(train['ids']):
            diagnostic_map[int(rid)]={'original':train['original'][i].tolist(),
                'aligned_swapped':train['swapped'][i].tolist(),
                'baseline_calibrated':train['calibrated'][i].tolist(),
                'order_flip':bool(train['original'][i].argmax()!=train['swapped'][i].argmax())}
    for i,rid in enumerate(diagnostic['ids']):
        base=temper(diagnostic['average'][i:i+1],selection['temperature'])[0]
        longer=temper(diagnostic['long_average'][i:i+1],selection['temperature'])[0]
        y=int(diagnostic['labels'][i])
        delta=float(-np.log(base[y])+np.log(longer[y]))
        diagnostic_map[int(rid)]={'original':diagnostic['original'][i].tolist(),
            'aligned_swapped':diagnostic['swapped'][i].tolist(),'baseline_calibrated':base.tolist(),
            'long_calibrated':longer.tolist(),'context_gain':delta,
            'context_changed':bool(diagnostic['context_changed'][i]),
            'order_flip':bool(diagnostic['original'][i].argmax()!=diagnostic['swapped'][i].argmax())}
    for cohort in cohort_meta:
        ids=frame.loc[frame.cohort.eq(cohort['key']),'id']
        cohort['order_diagnostic_rows']=sum(int(rid) in diagnostic_map for rid in ids)
        cohort['context_diagnostic_rows']=sum('long_calibrated' in diagnostic_map.get(int(rid),{}) for rid in ids)
    frame['tag_all']=True
    frame['tag_confident_wrong']=(frame.pred!=frame.label)&(frame.confidence>=.8)
    frame['tag_true_tie']=frame.label.eq(2);frame['tag_predicted_tie']=frame.pred.eq(2)&frame.label.ne(2)
    frame['tag_similar_answers']=frame.answer_embedding_cosine>.9
    frame['tag_truncated']=frame.cut_1024_cap256;frame['tag_prompt_cut']=frame.prompt_cut_1024_cap256
    for tag,col in [('multi_turn','multi_turn'),('code','task_code'),('math','task_math'),('creative','task_creative'),('transformation','task_transformation')]:frame['tag_'+tag]=frame[col]
    frame['tag_refusal']=frame.one_refusal|frame.both_refusals
    frame['tag_length_gap']=frame.answer_length_ratio>4
    frame['tag_nonascii']=frame.prompt_nonascii_fraction>.1
    frame['tag_identical']=frame.answer_equal_exact
    frame['tag_reviewed']=frame.id.isin(review_map)
    frame['tag_order_sensitive']=frame.id.map(lambda x:diagnostic_map.get(int(x),{}).get('order_flip',False))
    frame['tag_context_hurts']=frame.id.map(lambda x:diagnostic_map.get(int(x),{}).get('context_gain',0)<-.01)
    frame['tag_context_helps']=frame.id.map(lambda x:diagnostic_map.get(int(x),{}).get('context_gain',0)>.01)
    selected=top_union(frame,CATEGORIES,per_category)
    raw=pd.read_csv(ROOT/'raw/train.csv',usecols=['id',*TEXT_COLUMNS]).set_index('id')
    tokenizer_dir=run/'qwen4b1024/output/joint_run/adapter'
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_dir,local_files_only=True)
    encoder=JointTokenizer(tokenizer,selection['config']['max_length'])
    cases=[]
    for _,row in selected.iterrows():
        rid=int(row.id)
        texts={k:parse_turns(raw.loc[rid,k]) for k in TEXT_COLUMNS}
        turns={k:json.loads(raw.loc[rid,k]) for k in TEXT_COLUMNS}
        encoded={k:tokenizer(texts[k],add_special_tokens=False,return_offsets_mapping=True) for k in TEXT_COLUMNS}
        p,a,b=[encoded[k]['input_ids'] for k in TEXT_COLUMNS]
        available=encoder.max_length-encoder.overhead
        pb=min(len(p),min(256,max(32,available//4)));remaining=available-pb
        ab=min(len(a),remaining//2);bb=min(len(b),remaining//2)
        unused=remaining-ab-bb;take=min(unused,len(a)-ab);ab+=take;unused-=take;bb+=min(unused,len(b)-bb)
        fields={}
        for k,budget in zip(TEXT_COLUMNS,[pb,ab,bb]):
            parts=marked_segments(texts[k],encoded[k]['offset_mapping'],budget,len(encoder.truncated))
            assert ''.join(x['text'] for x in parts)==texts[k]
            fields[k]={'text':texts[k],'turns':['' if v is None else str(v) for v in turns[k]],
                'tokens':len(encoded[k]['input_ids']),'budget':budget,'segments':parts}
        original=encoder.combine(p,a,b);swapped=encoder.combine(p,b,a)
        assert len(original)<=1024 and len(swapped)<=1024
        cases.append({'key':f'{row.cohort}-{rid}','id':rid,'cohort':row.cohort,'label':int(row.label),
            'prediction':int(row.pred),'probabilities':[float(row['p'+str(i)]) for i in range(3)],
            'loss':float(row.loss),'cohort_rank':int(row.cohort_rank),'confidence':float(row.confidence),
            'tags':[k for k,_,_ in CATEGORIES if bool(row['tag_'+k])],
            'fields':fields,'packed_original':tokenizer.decode(original,skip_special_tokens=False),
            'packed_swapped':tokenizer.decode(swapped,skip_special_tokens=False),
            'token_counts':{'original':len(original),'swapped':len(swapped)},
            'review':review_map.get(rid),'diagnostic':diagnostic_map.get(rid),
            'loss_share_percent':float(row.loss/next(c['total_loss'] for c in cohort_meta if c['key']==row.cohort)*100)})
    categories=[]
    for key,title,description in CATEGORIES:
        categories.append({'key':key,'title':title,'description':description,
            'cohort_counts':{c['key']:int(((frame.cohort==c['key'])&frame['tag_'+key]).sum()) for c in cohort_meta},
            'cohort_metrics':{c['key']:{'rows':len(part),'log_loss':float(part.loss.mean()),
                'loss_share_percent':float(100*part.loss.sum()/c['total_loss']),
                'accuracy':float(part.pred.eq(part.label).mean())}
                for c in cohort_meta if len(part:=frame.loc[(frame.cohort==c['key'])&frame['tag_'+key]])}})
    data={'model':selection['config']['model_id'],'model_hash':selection['adapter_sha256'],
        'temperature':selection['temperature'],'official_score':.97025,'max_length':1024,
        'cohorts':cohort_meta,'categories':categories,'cases':cases,'top_per_category':per_category,
        'preview':not train_path.exists(),'hidden_test_note':'Kaggle hidden examples/labels are unavailable. Local held-out test means the confirmation fold.',
        'metric_note':'Loss is -ln(probability assigned to the observed human label), after swap averaging and the frozen calibration. Human labels are observed votes, not a claim of objective truth.',
        'gallery_scope':'Union of the top examples per category and cohort. Search and filters operate on this selected gallery; cohort summaries use every scored row.'}
    output=POSTER_ROOT/'site';output.mkdir(parents=True,exist_ok=True)
    template=(REPO/'web/error_poster.html').read_text()
    payload=safe_json(data)
    html=template.replace('/*__POSTER_DATA__*/',payload)
    assert html.count('/*__POSTER_DATA__*/')==0
    (output/'index.html').write_text(html)
    summary={k:v for k,v in data.items() if k!='cases'}
    summary['gallery_cases']=len(cases);summary['html_bytes']=len(html.encode())
    summary['html_sha256']=hashlib.sha256(html.encode()).hexdigest()
    summary['path']=str(output/'index.html')
    json_write(POSTER_ROOT/'summary.json',summary)
    json_write(REPO/'results/error_poster_002/summary.json',summary)
    print(json.dumps({'path':str(output/'index.html'),'cases':len(cases),'bytes':len(html.encode()),'cohorts':[c['key'] for c in cohort_meta]},indent=2))
    return output/'index.html'
