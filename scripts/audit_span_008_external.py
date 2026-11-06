"""Bounded local source, target-semantics and text-overlap audit; no model training."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,heapq,json,re,resource,signal,sys,time,os
import numpy as np
import pandas as pd
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,TEXT_COLUMNS,parse_turns
from arena.experiment import read_json,atomic_json,write_intent,sha256,utc_now,EventJournal


def canonical(t):return re.sub(r'\s+',' ',str(t)).strip().casefold()
def exact(t):return hashlib.sha256(canonical(t).encode()).hexdigest()
def shingles(t):
    words=re.findall(r'\w+',canonical(t))
    return {' '.join(words[i:i+5]) for i in range(len(words)-4)}
def fingerprints(s):return heapq.nsmallest(8,{int.from_bytes(hashlib.blake2b(v.encode(),digest_size=8).digest(),'little') for v in s})
def assistant(messages):return '\n\n[TURN]\n\n'.join(v['content'] or '' for v in messages if v['role']=='assistant')
def profile(texts):
    a=np.array([len(t) for t in texts]);return {'rows':len(a),'characters_quantiles':np.quantile(a,[0,.25,.5,.75,.95,.99,1]).tolist(),
        'code_fraction':float(np.mean(['```' in t for t in texts])),'non_ascii_fraction':float(np.mean([any(ord(c)>127 for c in t) for t in texts])),
        'refusal_proxy_fraction':float(np.mean([bool(re.search(r'(?i)\b(sorry|cannot|unable|as an ai)\b',t)) for t in texts]))}


def main():
    start=time.monotonic();root=ROOT/'runs/span_008/external';cfg=read_json(REPO/'configs/span_008.json');out=root/'audit';out.mkdir(exist_ok=True)
    if (out/'started.json').exists():
        # One CPU-only memory-engineering recovery; retain original clock and evidence.
        prior=read_json(out/'started.json');assert not (out/'recovery_started.json').exists()
        from datetime import datetime,timezone
        elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(prior['at'])).total_seconds()
        atomic_json(out/'recovery_started.json',{'at':utc_now(),'pid':os.getpid(),'original_start':prior,'reason':'Stop before retaining all original answer shingle strings; keep retrieval index only and verify candidates on demand','source_sha256':sha256(Path(__file__))})
    else:
        elapsed=0.;write_intent(out/'started.json',{'at':utc_now(),'pid':os.getpid(),'source_sha256':sha256(Path(__file__))})
    def expired(*unused):raise TimeoutError('Original external-audit CPU cap reached')
    signal.signal(signal.SIGALRM,expired);signal.alarm(max(1,int(cfg['cpu_stage_seconds_caps']['data_audit']-elapsed)));journal=EventJournal(out)
    sources=read_json(root/'source_access.json');source=next(s for s in sources['sources'] if s['downloaded'])
    path=Path(source['files'][0]['file']);assert sha256(path)==source['files'][0]['sha256']
    ext=pd.read_parquet(path,columns=['prompt','chosen','rejected','score_chosen','score_rejected','source'])
    ext['a']=ext.chosen.map(assistant);ext['b']=ext.rejected.map(assistant)
    split=read_json(ROOT/'runs/attempt_002/split_ids.json');dev=set(split['dev']);confirm=set(split['confirmation'])
    raw=pd.read_csv(ROOT/'raw/train.csv',usecols=['id']+TEXT_COLUMNS)
    for key in TEXT_COLUMNS:raw[key]=raw[key].map(parse_turns)
    parts=np.array(['dev' if int(i) in dev else 'confirmation' if int(i) in confirm else 'train' for i in raw.id])
    prompt_sets={part:{exact(t) for t in raw.prompt[parts==part] if canonical(t)} for part in ['train','dev','confirmation']}
    answer_sets={part:{exact(t) for col in ['response_a','response_b'] for t in raw.loc[parts==part,col] if canonical(t)} for part in prompt_sets}
    overlap={};remove=np.zeros(len(ext),bool);extph=[exact(t) for t in ext.prompt];extah=[[exact(t) for t in ext[c]] for c in ['a','b']]
    for part in prompt_sets:
        pm=np.array([h in prompt_sets[part] for h in extph]);am=np.array([a in answer_sets[part] or b in answer_sets[part] for a,b in zip(*extah)])
        overlap[part]={'exact_prompt_rows':int(pm.sum()),'exact_any_answer_rows':int(am.sum())};remove|=pm|am
    # Explicitly approximate retrieval: eight minimum hashes of five-word shingles.
    # Every original prompt and answer is indexed; query all external prompts/answers.
    # Exact Jaccard verification prevents false positives, but recall is not guaranteed.
    near={};candidate_pairs_total=0
    for kind in ['prompt','answer']:
        texts=raw.prompt.tolist() if kind=='prompt' else raw.response_a.tolist()+raw.response_b.tolist()
        membership=parts if kind=='prompt' else np.r_[parts,parts]
        index=defaultdict(list)
        for i,t in enumerate(texts):
            s=shingles(t)
            for f in fingerprints(s):index[f].append(i)
            if i%1000==0:assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2<cfg['rss_guard_gib']
        queries=ext.prompt.tolist() if kind=='prompt' else ext.a.tolist()+ext.b.tolist();counts=Counter();hits=[]
        for i,t in enumerate(queries):
            s=shingles(t);candidates=set(j for f in fingerprints(s) for j in index.get(f,()))
            candidate_pairs_total+=len(candidates)
            matched=set()
            for j in candidates:
                z=shingles(texts[j])
                if not s or min(len(s),len(z))/max(len(s),len(z))<.8:continue
                score=len(s&z)/len(s|z)
                if score>=.8:matched.add(str(membership[j]));hits.append({'external_row':i%len(ext),'original_row':j%len(raw),'jaccard':score,'kind':kind})
            for part in matched:counts[part]+=1
            if matched:remove[i%len(ext)]=True
            if i%5000==0:
                assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2<cfg['rss_guard_gib'];journal.emit('external_'+kind,'progress',rows=i,seconds=time.monotonic()-start)
        near[kind]={'matched_external_texts_by_partition':dict(counts),'verified_pairs':len(hits),'query_texts':len(queries),'original_texts':len(texts)}
        atomic_json(out/(kind+'_near_pairs.json'),hits)
        del index
    gap=ext.score_chosen.to_numpy()-ext.score_rejected.to_numpy();equal=gap==0;reversed_=gap<0
    same=np.array([canonical(a)==canonical(b) for a,b in zip(ext.a,ext.b)]);blank=np.array([not canonical(a) or not canonical(b) for a,b in zip(ext.a,ext.b)])
    safe=~(remove|equal|reversed_|same|blank);np.savez_compressed(out/'safe_rows.npz',indices=np.flatnonzero(safe))
    result={'task':'H17','audit_complete':True,'training_performed':False,'sources':sources,'external_rows':len(ext),
        'target_semantics':{'annotation':'Synthetic GPT-4 ranked responses; chosen/rejected is not a human three-class vote','equal_score_rows':int(equal.sum()),
            'reversed_score_rows':int(reversed_.sum()),'identical_answer_rows':int(same.sum()),'empty_answer_rows':int(blank.sum()),'gap_quantiles':np.quantile(gap,[0,.25,.5,.75,.95,1]).tolist(),
            'human_tie_labels_available':False,'equal_scores_are_not_valid_human_tie_labels':True},
        'source_counts':{str(k):int(v) for k,v in ext.source.value_counts().items()},'exact_overlap':overlap,'near_overlap':near,
        'near_method':{'retrieval':'Eight minimum BLAKE2b hashes of unique five-word shingles; any hash candidate, exact Jaccard verification >=0.8',
            'scope':'All prompts and all answers against all competition partitions, using text only','candidate_pairs_verified':candidate_pairs_total,
            'recall_guaranteed':False,'short_texts_under_five_words':'Exact matching only','semantic_overlap_tested':False},
        'purged_external_rows':int(remove.sum()),'provisionally_usable_rows':int(safe.sum()),'safe_rows_sha256':sha256(out/'safe_rows.npz'),
        'profiles':{'competition_prompt':profile(raw.prompt),'competition_answer':profile(raw.response_a.tolist()+raw.response_b.tolist()),
                    'external_prompt':profile(ext.prompt),'external_answer':profile(ext.a.tolist()+ext.b.tolist())},
        'proposal':'Future separate auxiliary decisive-preference pretraining, <=20%external batches, randomize A/B, then human-vote-only finetuning; never relabel score ties as human ties. Purge exact and detected near overlaps from all held cohorts; approximate matching leaves residual contamination risk.',
        'decision':'Preparation complete; mixed-data training deferred to a separately budgeted span and source-license review. No gain claimed.',
        'competition_targets_loaded':False,'seconds':time.monotonic()-start+elapsed,'recovery_elapsed_before_seconds':elapsed,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        'source_sha256':sha256(Path(__file__))}
    atomic_json(REPO/'results/span_008/H17_audit.json',result);print({k:result[k] for k in ['external_rows','target_semantics','exact_overlap','near_overlap','provisionally_usable_rows','seconds','peak_rss_gib']},flush=True)


if __name__=='__main__':main()
