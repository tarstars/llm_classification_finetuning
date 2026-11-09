"""Fixed, target-blind covariate and model-name support comparisons."""
from collections import Counter
from datetime import datetime
from pathlib import Path
import json
import re
import resource
import signal
import sqlite3
import sys
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial.distance import jensenshannon

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,sha256,utc_now
from arena.span038_arithmetic import MATH,RISK


def main():
    started=time.monotonic();pub,run=REPO/'results/span_039',ROOT/'runs/span_039'
    assert not (pub/'H81_results.json').exists();run.mkdir(parents=True,exist_ok=True)
    def expired(*unused):raise TimeoutError('Frozen CPU preparation limit')
    signal.signal(signal.SIGALRM,expired);signal.alarm(1200)
    raw=ROOT/'raw/train.csv';old=read_json(REPO/'results/span_038/coverage.json')
    assert sha256(raw)==old['raw_sha256']
    sql=ROOT/'runs/span_011/cohort/rows.sqlite';assert sha256(sql)==old['source_sql_sha256']
    provenance={'raw/train.csv':sha256(raw),'rows.sqlite':sha256(sql)}
    def claims(name):
        path=ROOT/'runs/span_038'/(name+'_coverage.jsonl');assert sha256(path)==old['artifacts'][path.name]
        provenance['span038/'+path.name]=sha256(path)
        return {r['id']:(bool(r['claims_a'] or r['claims_b']),r['feature']!=0) for r in map(json.loads,path.open())}
    original_claims,later_claims=claims('original'),claims('later_public')
    downloads=read_json(REPO/'results/span_011/source_download.json');meta=[]
    for item in downloads['files']:
        path=ROOT/'runs/span_011/source_data'/item['name'];assert sha256(path)==item['sha256']
        provenance['public/'+item['name']]=item['sha256']
        meta.append(pd.read_parquet(path,columns=['id','model_a','model_b','timestamp','evaluation_order','language']))
    meta=pd.concat(meta,ignore_index=True);meta=meta.loc[meta.evaluation_order==1]
    assert meta.id.is_unique;meta=meta.set_index('id')
    def parse(v):return json.loads(v) if isinstance(v,str) else list(v)
    def features(row,mapping):
        p,a,b=[parse(row[k]) for k in ['prompt','response_a','response_b']]
        strings=lambda xs:['' if x is None else str(x) for x in xs]
        p,a,b=map(strings,[p,a,b]);prompt='\n'.join(p);aa='\n'.join(a);bb='\n'.join(b);text='\n'.join([prompt,aa,bb])
        checked,nonzero=mapping[int(row['id'])]
        return dict(id=int(row['id']),turns=len(p),log_total_chars=float(np.log1p(len(text))),
            log_prompt_chars=float(np.log1p(len(prompt))),log_answer_chars=float(np.log1p(len(aa)+len(bb))),
            answer_imbalance=abs(len(aa)-len(bb))/max(1,len(aa)+len(bb)),
            nonascii_share=sum(ord(c)>127 for c in text)/max(1,len(text)),
            code_fence='```' in text,markdown_heading=bool(re.search(r'^#{1,6}\s',text,re.M)),
            math_delimiter=bool(re.search(r'\$|\\\[|\\\(',text)),math_request=bool(MATH.search(prompt)),
            risk_context=bool(RISK.search(text)),checked_claim=checked,nonzero_arithmetic=nonzero,
            total_chars=len(text),model_a=str(row['model_a']),model_b=str(row['model_b']))
    original=pd.read_csv(raw,usecols=['id','model_a','model_b','prompt','response_a','response_b'])
    orig=pd.DataFrame([features(r,original_claims) for r in original.to_dict('records')]);del original
    later=[];cutoff=datetime.fromisoformat('2025-07-06T18:16:33+00:00')
    connection=sqlite3.connect('file:'+str(sql)+'?mode=ro',uri=True)
    for payload,group in connection.execute('SELECT r.payload,p.group_id FROM representatives p JOIN rows r ON r.uid=p.uid'):
        row=json.loads(payload)
        if datetime.fromisoformat(row['timestamp'])<=cutoff:continue
        info=meta.loc[row['source_uuid']];assert info['language']=='en'
        f=features(dict(row,model_a=info.model_a,model_b=info.model_b),later_claims)
        f.update(group=group,source_uuid=row['source_uuid'],timestamp=row['timestamp']);later.append(f)
    connection.close();later=pd.DataFrame(later);assert len(orig)==57477 and len(later)==11215
    matched_manifest=read_json(REPO/'results/span_016/matched_cohort_manifest.json')
    mapping_path=ROOT/'runs/span_016/cohort_matched/mapping.parquet'
    assert sha256(mapping_path)==matched_manifest['artifact_sha256']['mapping.parquet']
    mapping=pd.read_parquet(mapping_path,columns=['id','prepared_tokens','length_bin'])
    matched=later.set_index('id',drop=False).loc[mapping.id].reset_index(drop=True);assert len(matched)==1024
    selection=read_json(REPO/'results/span_013/profile_selection.json')['uniform_ids'];assert len(selection)==512
    uniform=orig.set_index('id',drop=False).loc[selection].reset_index(drop=True)
    single=orig.loc[orig.turns==1].copy();assert (later.turns==1).all()
    views=dict(original_all=orig,original_single_turn=single,later_all=later,later_matched1024=matched,original_uniform512=uniform)
    numeric=['turns','log_total_chars','log_prompt_chars','log_answer_chars','answer_imbalance','nonascii_share',
        'code_fence','markdown_heading','math_delimiter','math_request','risk_context','checked_claim','nonzero_arithmetic']
    def summary(frame,w=None):
        w=np.ones(len(frame))/len(frame) if w is None else np.asarray(w)/np.sum(w)
        return dict(rows=len(frame),means={k:float(np.sum(frame[k].to_numpy(float)*w)) for k in numeric},
            stds={k:float(np.sqrt(np.sum((frame[k].to_numpy(float)-np.sum(frame[k].to_numpy(float)*w))**2*w))) for k in numeric})
    def smd(a,b):
        out={}
        for k in numeric:
            d=np.sqrt((a['stds'][k]**2+b['stds'][k]**2)/2);diff=b['means'][k]-a['means'][k]
            out[k]=float(diff/d) if d else (0. if not diff else None)
        return out
    def distribution(frame,w=None,prefix=False):
        w=np.ones(len(frame))/len(frame) if w is None else np.asarray(w)/np.sum(w)
        c=Counter()
        for col in ['model_a','model_b']:
            for name,weight in zip(frame[col],w):
                key=re.match(r'[a-z]+',name.lower()).group(0) if prefix and re.match(r'[a-z]+',name.lower()) else name
                c[key]+=float(weight)/2
        return c
    def compare_models(a,b,w=None,prefix=False):
        pa,pb=distribution(a,prefix=prefix),distribution(b,w,prefix);keys=sorted(set(pa)|set(pb))
        p=np.array([pa[k] for k in keys]);q=np.array([pb[k] for k in keys]);names=set(pa)
        support=np.asarray([(x in names and y in names) for x,y in zip(b.model_a,b.model_b)])
        return dict(exact_or_prefix='literal_name_prefix' if prefix else 'exact_model_id',
            original_unique=len(pa),source_unique=len(pb),shared=len(set(pa)&set(pb)),
            overlap_mass=float(np.minimum(p,q).sum()),total_variation=float(abs(p-q).sum()/2),
            js_distance_base2=float(jensenshannon(p,q,base=2)),
            both_exact_ids_supported=float(np.average(support,weights=w)) if not prefix else None,
            original_top10=pa.most_common(10),source_top10=pb.most_common(10))
    summaries={k:summary(v) for k,v in views.items()}
    bins=np.unique(np.quantile(single.total_chars,np.linspace(0,1,11)));bins[0]=-np.inf;bins[-1]=np.inf
    target_bin=np.digitize(single.total_chars,bins[1:-1]);source_bin=np.digitize(later.total_chars,bins[1:-1]);n=len(bins)-1
    target_share=np.bincount(target_bin,minlength=n)/len(single);source_share=np.bincount(source_bin,minlength=n)/len(later)
    factors=np.divide(target_share,source_share,out=np.zeros(n),where=source_share>0)
    weights=np.minimum(factors[source_bin],20);assert weights.sum()>0
    standardized=summary(later,weights);differences=smd(summaries['original_single_turn'],standardized)
    model_comparisons={k:compare_models(orig,views[k]) for k in ['later_all','later_matched1024']}
    model_comparisons['later_length_standardized']=compare_models(single,later,weights)
    material=model_comparisons['later_length_standardized']['overlap_mass']<.5 and sum(v is not None and abs(v)>.25 for k,v in differences.items() if k!='turns')>=2
    native=read_json(REPO/'results/span_011/source_metadata_audit.json')
    target=np.asarray(native['target_bin_counts'],float);target/=target.sum()
    observed=np.bincount(mapping.length_bin,minlength=len(target))/len(mapping)
    for k,v in views.items():v.to_parquet(run/(k+'.parquet'),index=False)
    np.savez_compressed(run/'length_standardization.npz',ids=later.id,weights=weights,bins=bins)
    snapshots=sorted((REPO/'backlog/versions').glob('*'));assert len(snapshots)==154
    users=read_json(REPO/'results/span_038/preservation.json')['user_notebook_sha256']
    for k,h in users.items():assert sha256(REPO/k)==h
    atomic_json(pub/'preservation.json',dict(at=utc_now(),snapshot_sha256={str(p.relative_to(REPO)):sha256(p) for p in snapshots},user_notebook_sha256=users))
    result=dict(at=utc_now(),task='H81',complete=True,quality_measured=False,preference_targets_loaded=False,
        identities_diagnostic_only=True,new_gpu_executions=0,new_official_submissions=0,official_submission_eligible=False,
        view_summaries=summaries,standardized_mean_differences={k:smd(summaries['original_all'],summaries[k]) for k in ['later_all','later_matched1024']},
        length_standardization=dict(source_rows=len(later),target_rows=len(single),weight_cap=20,ess=float(weights.sum()**2/np.sum(weights**2)),
            missing_target_bin_mass=float(target_share[source_share==0].sum()),summary=standardized,smd_vs_original_single=differences),
        model_comparisons=model_comparisons,literal_prefix_comparison=compare_models(orig,later,prefix=True),
        matched_native_length_total_variation=float(abs(target-observed).sum()/2),material_mismatch=bool(material),
        decision='record_material_support_mismatch' if material else 'no_predeclared_material_mismatch',
        limitations=['Exact IDs may include renaming; prefix groups are naming summaries, not aliases','Finite selected populations, no hidden-test sampling or quality transfer claim','Original teacher training overlap is not removed by covariate matching','Parser claims were not blind-certified'],
        input_sha256=provenance,artifacts={p.name:sha256(p) for p in run.iterdir() if p.is_file()},
        method_sha256={k:sha256(REPO/k) for k in ['GOAL.md','docs/SPAN_039_PLAN.md','src/arena/span038_arithmetic.py',str(Path(__file__).relative_to(REPO))]},
        elapsed_seconds=time.monotonic()-started,peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    assert result['peak_rss_gib']<6
    atomic_json(pub/'H81_results.json',result)
    print(json.dumps({k:result[k] for k in ['decision','model_comparisons','matched_native_length_total_variation','length_standardization','elapsed_seconds','peak_rss_gib']},indent=2))


if __name__=='__main__':main()
