"""Exploratory dataset and frozen-model analyses; all model diagnostics use dev only."""
from __future__ import annotations
import json,math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize,minimize_scalar
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,balanced_accuracy_score,brier_score_loss,confusion_matrix,f1_score,log_loss,roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from .data import ROOT,CLASSES,TEXT_COLUMNS,json_write
from .eda_features import EDA_ROOT,RESULTS,SEED,load_frame


def temper(p,t):
    z=np.log(np.clip(p,1e-12,1))/t;z-=z.max(1,keepdims=True)
    q=np.exp(z);return q/q.sum(1,keepdims=True)


def losses(y,p):return -np.log(np.clip(p[np.arange(len(y)),y],np.finfo(float).eps,1))


def scores(y,p):
    return {'rows':len(y),'log_loss':float(losses(y,p).mean()),
        'accuracy':float(accuracy_score(y,p.argmax(1))),
        'balanced_accuracy':float(balanced_accuracy_score(y,p.argmax(1))),
        'macro_f1':float(f1_score(y,p.argmax(1),labels=[0,1,2],average='macro',zero_division=0)),
        'brier':float(np.mean(np.sum((p-np.eye(3)[y])**2,axis=1)))}


class GroupBootstrap:
    def __init__(self,groups,repeats=600,seed=SEED):
        self.codes,unique=pd.factorize(np.asarray(groups),sort=True)
        self.n_groups=len(unique)
        self.weights=np.random.default_rng(seed).multinomial(self.n_groups,np.full(self.n_groups,1/self.n_groups),size=repeats).astype(np.float32)
    def means(self,values,mask=None):
        if mask is None:mask=np.ones(len(self.codes),bool)
        numer=np.bincount(self.codes,weights=np.asarray(values)*mask,minlength=self.n_groups)
        denom=np.bincount(self.codes,weights=np.asarray(mask,dtype=float),minlength=self.n_groups)
        d=self.weights@denom
        return np.divide(self.weights@numer,d,out=np.full(len(d),np.nan),where=d>0)
    def interval(self,values,mask=None):
        x=self.means(values,mask);x=x[np.isfinite(x)]
        return np.quantile(x,[.025,.975]).tolist() if len(x) else [None,None]


def reliability(y,p,bins=10):
    c=p.max(1);correct=p.argmax(1)==y
    edges=np.linspace(0,1,bins+1);rows=[]
    for i in range(bins):
        mask=(c>=edges[i])&((c<=edges[i+1]) if i==bins-1 else (c<edges[i+1]))
        if mask.any():rows.append({'lower':edges[i],'upper':edges[i+1],'rows':int(mask.sum()),
            'confidence':float(c[mask].mean()),'accuracy':float(correct[mask].mean())})
    return {'ece':float(sum(r['rows']*abs(r['confidence']-r['accuracy']) for r in rows)/len(y)),
        'mean_confidence':float(c.mean()),'accuracy':float(correct.mean()),'bins':rows}


def fit_temperature(y,p):
    return float(minimize_scalar(lambda t:losses(y,temper(p,t)).mean(),bounds=(.5,3.),method='bounded').x)


def tie_calibrate(p,t,bias):
    z=np.log(np.clip(p,1e-12,1))/t;z[:,2]+=bias;z-=z.max(1,keepdims=True)
    q=np.exp(z);return q/q.sum(1,keepdims=True)


def fit_tie_calibration(y,p):
    fit=minimize(lambda x:losses(y,tie_calibrate(p,x[0],x[1])).mean(),x0=[.75,0.],
        bounds=[(.5,3.),(-1.,1.)],method='L-BFGS-B')
    assert fit.success
    return fit.x


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):return [clean(v) for v in value]
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def save(name,data):json_write(RESULTS/name,clean(data))


def enrich_clusters(f,frame):
    path=ROOT/'embeddings/qwen3_256.npz'
    if not path.exists():return f,[]
    vectors=np.load(path)
    prompt=vectors['prompt'][:len(f)]
    a=vectors['response_a'][:len(f)];b=vectors['response_b'][:len(f)]
    assert prompt.shape==(len(f),256)
    f['answer_embedding_cosine']=np.sum(a*b,axis=1)
    f['relevance_cosine_a']=np.sum(prompt*a,axis=1)
    f['relevance_cosine_b']=np.sum(prompt*b,axis=1)
    train=f.split.eq('train').to_numpy()
    cluster=MiniBatchKMeans(n_clusters=12,random_state=SEED,n_init=3,batch_size=1024,max_iter=100)
    cluster.fit(prompt[train]);f['prompt_cluster']=cluster.predict(prompt)
    # Names use frequent training-only words; clustering uses pretrained embeddings.
    tfidf=TfidfVectorizer(max_features=10000,min_df=20,stop_words='english',ngram_range=(1,2))
    x=tfidf.fit_transform(frame.loc[train,'prompt'].str.slice(0,6000))
    terms=np.array(tfidf.get_feature_names_out());labels=f.loc[train,'prompt_cluster'].to_numpy()
    rows=[]
    for k in range(12):
        mask=labels==k
        weights=np.asarray(x[mask].mean(0)).ravel()
        rows.append({'cluster':k,'train_rows':int(mask.sum()),'keywords':terms[np.argsort(weights)[-10:][::-1]].tolist()})
    import joblib
    joblib.dump(cluster,EDA_ROOT/'prompt_cluster.joblib')
    return f,rows


def dataset_analysis(f,clusters):
    describe_cols=[k+'_tokens' for k in TEXT_COLUMNS]+['total_content_tokens','prompt_turns','answer_word_jaccard','answer_length_ratio']
    overview={'rows':len(f),'split_counts':f.split.value_counts().to_dict(),
        'global_class_counts':f.label.map(dict(enumerate(CLASSES))).value_counts().to_dict(),
        'class_proportions_by_split':pd.crosstab(f.split,f.label,normalize='index').to_dict(orient='index'),
        'numeric_quantiles':f[describe_cols].quantile([0,.01,.05,.25,.5,.75,.9,.95,.99,1]).to_dict(),
        'unique_prompt_groups':int(f.prompt_group.nunique()),
        'prompt_group_size_distribution':f.groupby('prompt_group').size().value_counts().sort_index().to_dict(),
        'prompt_groups_crossing_splits':int((f.groupby('prompt_group').split.nunique()>1).sum()),
        'structural_counts':{k:int(f[k].sum()) for k in ['multi_turn','turn_mismatch','answer_equal_exact','answer_equal_whitespace',
            'one_answer_empty','both_answers_empty','one_refusal','both_refusals','any_code_answer','wrapper_marker_in_content']},
        'null_turn_counts':{k:int(f[k+'_null_turns'].sum()) for k in TEXT_COLUMNS},
        'nonstring_turn_counts':{k:int(f[k+'_non_string_turns'].sum()) for k in TEXT_COLUMNS},
        'language_counts_estimated':f.prompt_language_estimated.value_counts().to_dict(),
        'language_group_by_split':pd.crosstab(f.split,f.language_group,normalize='index').to_dict(orient='index'),
        'prompt_clusters':clusters,'target_association_partition':'train and development; excludes confirmation'}
    assert overview['prompt_groups_crossing_splits']==0
    discover=f[f.split.ne('confirmation')]
    duplicates={}
    for name,label in [('comparison_hash','label'),('canonical_comparison_hash','canonical_label')]:
        g=discover.groupby(name)[label].agg(['size','nunique'])
        repeated=g[g['size']>1];conflict=repeated[repeated['nunique']>1]
        duplicates[name]={'distinct':len(g),'repeated_groups':len(repeated),'rows_in_repeated_groups':int(repeated['size'].sum()),
            'conflicting_groups':len(conflict),'rows_in_conflicting_groups':int(conflict['size'].sum()),
            'note':'Repeated identical inputs with observed different labels; not an estimate of global irreducible noise.'}
    overview['comparison_duplicates_train_dev']=duplicates
    save('dataset_summary.json',overview)
    table=[]
    for col in ['multi_turn','turn_mismatch','answer_equal_exact','one_answer_empty','both_answers_empty','one_refusal','both_refusals',
                'task_code','task_math','task_creative','task_transformation','task_explanation','language_group']:
        for value,part in discover.groupby(col,observed=True):
            counts=np.bincount(part.label,minlength=3)
            table.append({'feature':col,'value':str(value),'rows':len(part),
                **{c:float(counts[i]/len(part)) for i,c in enumerate(CLASSES)}})
    save('dataset_target_associations.json',table)
    ret=[]
    for split,part in f.groupby('split'):
        for tag in ['1024_cap256','2048_cap256','2048_cap512','4096_cap256','4096_cap1024']:
            ret.append({'split':split,'packing':tag,'any_cut':float(part['cut_'+tag].mean()),
                'prompt_cut':float(part['prompt_cut_'+tag].mean()),'answer_cut':float(part['answer_cut_'+tag].mean()),
                'mean_content_retained':float(part['retained_fraction_'+tag].mean())})
    save('retention.json',ret)
    # Identity metadata is not available in test; it is used only to describe coverage.
    identities=pd.concat([discover[['model_a','label']].rename(columns={'model_a':'model'}),
        discover[['model_b','label']].rename(columns={'model_b':'model'})]).model.value_counts()
    save('model_identity_counts.json',identities.to_dict())


def analyze():
    RESULTS.mkdir(parents=True,exist_ok=True)
    f=pd.read_parquet(EDA_ROOT/'features.parquet')
    raw,frame,turns=load_frame()
    assert np.array_equal(f.id,raw.id)
    print('Clustering existing prompt embeddings (training-only fit)',flush=True)
    f,clusters=enrich_clusters(f,frame)
    dataset_analysis(f,clusters)
    run=ROOT/'runs/attempt_002'
    selection=json.loads((run/'model_selection.json').read_text())
    ref=np.load(run/'jobs/qwen4b1024/output/joint_run/dev_predictions.npz')
    d=f.set_index('id').loc[ref['ids']].reset_index()
    y=d.label.to_numpy();rawp=ref['probabilities'];p=temper(rawp,selection['temperature'])
    assert np.array_equal(y,ref['labels']) and d.split.eq('dev').all()
    assert np.isclose(scores(y,p)['log_loss'],selection['calibrated_development']['log_loss'])
    loss=losses(y,p);rawloss=losses(y,rawp)
    train=f.split.eq('train');prior=np.bincount(f.loc[train,'label'],minlength=3)/train.sum()
    bootstrap=GroupBootstrap(d.prompt_group)
    metrics={'frozen_calibrated':scores(y,p),'raw_swap_average':scores(y,rawp),
        'prior':scores(y,np.tile(prior,(len(y),1))),
        'majority':scores(y,np.tile(np.eye(3)[prior.argmax()],(len(y),1))),
        'uniform':scores(y,np.full_like(p,1/3)),
        'log_loss_95_group_bootstrap':bootstrap.interval(loss),
        'accuracy_95_group_bootstrap':bootstrap.interval((p.argmax(1)==y).astype(float)),
        'groups':bootstrap.n_groups,'bootstrap_repeats':len(bootstrap.weights),
        'calibration_gain':float((rawloss-loss).mean()),'calibration_gain_95_group_bootstrap':bootstrap.interval(rawloss-loss),
        'prior_gain':float((losses(y,np.tile(prior,(len(y),1)))-loss).mean()),
        'confusion_counts':confusion_matrix(y,p.argmax(1),labels=[0,1,2]).tolist(),
        'reliability_raw':reliability(y,rawp),'reliability_calibrated':reliability(y,p),
        'temperature':selection['temperature'],'partition':'development','confirmation_used':False,
        'uncertainty_note':'Exploratory prompt-group bootstrap; intervals do not correct multiple slice comparisons or selection on development.'}
    per_class=[]
    for i,name in enumerate(CLASSES):
        mask=y==i;pred=p.argmax(1)==i
        per_class.append({'class':name,'rows':int(mask.sum()),'prevalence':float(mask.mean()),
            'mean_predicted_probability':float(p[:,i].mean()),'prediction_fraction':float(pred.mean()),
            'recall':float(pred[mask].mean()),'precision':float(mask[pred].mean()) if pred.any() else None,
            'log_loss':float(loss[mask].mean()),'loss_share':float(loss[mask].sum()/loss.sum()),
            'one_vs_rest_auc':float(roc_auc_score(mask,p[:,i]))})
    metrics['per_class']=per_class
    tail=[]
    for fraction in [.01,.05,.1,.2]:
        k=max(1,int(np.ceil(len(loss)*fraction)));idx=np.argsort(loss)[-k:]
        tail.append({'top_fraction':fraction,'rows':k,'loss_share':float(loss[idx].sum()/loss.sum()),
            'min_loss':float(loss[idx].min()),'tie_fraction':float((y[idx]==2).mean()),
            'truncated_fraction':float(d.iloc[idx].cut_1024_cap256.mean())})
    metrics['loss_tail']=tail
    confident_wrong=(p.argmax(1)!=y)&(p.max(1)>=.8)
    metrics['confident_wrong']={'rows':int(confident_wrong.sum()),'fraction':float(confident_wrong.mean()),
        'loss_share':float(loss[confident_wrong].sum()/loss.sum())}
    save('model_summary.json',metrics)
    # Per-row diagnostics remain outside GitHub.
    for i in range(3):d['p_'+str(i)]=p[:,i]
    d['loss']=loss;d['prediction']=p.argmax(1);d['confidence']=p.max(1)
    d.to_parquet(EDA_ROOT/'development_model_features.parquet',index=False)
    f.to_parquet(EDA_ROOT/'all_enriched_features.parquet',index=False)
    print('Computing exploratory slices and uncertainty',flush=True)
    slices=[]
    def add(feature,value,mask):
        mask=np.asarray(mask);n=int(mask.sum())
        if not n:return
        lo,hi=bootstrap.interval(loss,mask) if n>=30 else (None,None)
        other=~mask
        delta_boot=bootstrap.means(loss,mask)-bootstrap.means(loss,other) if other.any() and n>=100 else None
        slices.append({'feature':feature,'value':str(value),'rows':n,'fraction':n/len(y),
            'log_loss':float(loss[mask].mean()),'log_loss_ci_low':lo,'log_loss_ci_high':hi,
            'accuracy':float((p.argmax(1)[mask]==y[mask]).mean()),'tie_rate':float((y[mask]==2).mean()),
            'truncation_rate':float(d.loc[mask,'cut_1024_cap256'].mean()),
            'loss_delta_vs_rest':float(loss[mask].mean()-loss[other].mean()) if other.any() else None,
            'delta_ci_low':float(np.nanquantile(delta_boot,.025)) if delta_boot is not None else None,
            'delta_ci_high':float(np.nanquantile(delta_boot,.975)) if delta_boot is not None else None})
    cols=['multi_turn','turn_mismatch','answer_equal_exact','answer_equal_whitespace','one_answer_empty','both_answers_empty',
        'one_refusal','both_refusals','any_code_answer','one_code_answer','language_group','prompt_language_estimated',
        'cut_1024_cap256','prompt_cut_1024_cap256','answer_cut_1024_cap256','prompt_cluster',
        'task_code','task_math','task_creative','task_transformation','task_explanation']
    for col in cols:
        if col in d:
            for value in d[col].unique():add(col,value,d[col].eq(value))
    for col,bins in [('total_content_tokens',[0,256,512,768,1024,1536,2048,4096,np.inf]),
                     ('prompt_tokens',[0,32,128,243,256,512,1024,np.inf]),
                     ('answer_length_ratio',[1,1.25,2,4,10,np.inf]),
                     ('answer_word_jaccard',[0,.1,.3,.5,.7,.9,1.00001]),
                     ('prompt_nonascii_fraction',[-.0001,0,.01,.1,.5,1.00001]),
                     ('answer_embedding_cosine',[-1,.5,.75,.9,.98,1.00001])]:
        if col not in d:continue
        binned=pd.cut(d[col],bins,include_lowest=True)
        for value in binned.cat.categories:add(col,str(value),binned==value)
    for language in ['ENGLISH','OTHER_IDENTIFIED','UNKNOWN']:
        for cut in [False,True]:add('language_x_truncation',f'{language}; cut={cut}',d.language_group.eq(language)&d.cut_1024_cap256.eq(cut))
    for column in ['multi_turn','task_code']:
        for value in [False,True]:
            for cut in [False,True]:add(column+'_x_truncation',f'{value}; cut={cut}',d[column].eq(value)&d.cut_1024_cap256.eq(cut))
    save('model_slices.json',slices)
    style=[]
    for column,bins in [('answer_length_ratio',[1,1.25,2,4,10,np.inf]),('answer_word_jaccard',[0,.1,.3,.5,.7,.9,1.00001])]:
        cats=pd.cut(d[column],bins,include_lowest=True)
        for value in cats.cat.categories:
            mask=(cats==value).to_numpy();n=mask.sum()
            if not n:continue
            side=d.longer_side.to_numpy()
            observed=(y==side).astype(float);predicted=p[np.arange(len(p)),side]
            gap=predicted-observed
            style.append({'feature':column,'bin':str(value),'rows':int(n),'human_longer_win_rate':float(observed[mask].mean()),
                'model_mean_longer_win_probability':float(predicted[mask].mean()),'prediction_minus_observed':float(gap[mask].mean()),
                'gap_95_group_bootstrap':bootstrap.interval(gap,mask),'human_tie_rate':float((y[mask]==2).mean()),
                'model_mean_tie_probability':float(p[mask,2].mean())})
    mask=d.one_refusal.to_numpy()
    side=d.refusal_side.to_numpy();gap=p[np.arange(len(p)),side]-(y==side)
    style.append({'feature':'one_refusal','bin':'refusal-like phrases','rows':int(mask.sum()),
        'human_refusal_side_win_rate':float((y[mask]==side[mask]).mean()) if mask.any() else None,
        'model_mean_refusal_side_probability':float(p[np.arange(len(p)),side][mask].mean()) if mask.any() else None,
        'prediction_minus_observed':float(gap[mask].mean()) if mask.any() else None,
        'gap_95_group_bootstrap':bootstrap.interval(gap,mask) if mask.any() else [None,None]})
    save('style_calibration.json',style)
    # Calibration and blending are cross-fitted postprocessors on fixed model predictions.
    # They are not out-of-fold evaluations of the already selected language model.
    gb_all=np.load(run/'baselines/structural__gbdt_predictions.npy')
    positions=pd.Series(np.arange(len(f)),index=f.id)
    gb=gb_all[positions.loc[d.id].to_numpy()]
    assert np.isclose(scores(y,gb)['log_loss'],json.loads((run/'baselines.json').read_text())['structural_gbdt_development']['log_loss'])
    cv=StratifiedGroupKFold(5,shuffle=True,random_state=SEED)
    oof_single=np.zeros_like(p);oof_blend=np.zeros_like(p);oof_tie=np.zeros_like(p);folds=[]
    for k,(fit,test) in enumerate(cv.split(rawp,y,d.prompt_group)):
        t=fit_temperature(y[fit],rawp[fit]);oof_single[test]=temper(rawp[test],t)
        tt,tb=fit_tie_calibration(y[fit],rawp[fit]);oof_tie[test]=tie_calibrate(rawp[test],tt,tb)
        candidates=[]
        for w in np.linspace(0,1,11):
            mixed=w*rawp[fit]+(1-w)*gb[fit]
            mt=fit_temperature(y[fit],mixed)
            candidates.append((losses(y[fit],temper(mixed,mt)).mean(),w,mt))
        _,w,mt=min(candidates)
        oof_blend[test]=temper(w*rawp[test]+(1-w)*gb[test],mt)
        folds.append({'fold':k,'calibration_temperature':t,'lora_weight':w,'blend_temperature':mt,
            'tie_temperature':float(tt),'tie_logit_bias':float(tb)})
    gain=losses(y,oof_single)-losses(y,oof_blend)
    post={'folds':folds,'calibration_oof':scores(y,oof_single),'structural_blend_oof':scores(y,oof_blend),
        'blend_gain':float(gain.mean()),'blend_gain_95_group_bootstrap':bootstrap.interval(gain),
        'tie_calibration_oof':scores(y,oof_tie),
        'tie_calibration_gain':float((losses(y,oof_single)-losses(y,oof_tie)).mean()),
        'tie_calibration_gain_95_group_bootstrap':bootstrap.interval(losses(y,oof_single)-losses(y,oof_tie)),
        'limitation':'Cross-fitting postprocessors only; base checkpoint was already selected on development. This is exploratory, not fresh confirmation.',
        'structural_base':scores(y,gb)}
    save('postprocessing_diagnostics.json',post)
    # Diagnostic identity-only probabilities: training pair frequencies, unavailable at test.
    def pair_key(a,b):return '|'.join(sorted([a,b]))
    t=f.loc[train].copy();t['pair']=[pair_key(a,b) for a,b in zip(t.model_a,t.model_b)]
    reversed_=t.model_a>t.model_b
    t['canonical_pair_label']=np.where(reversed_,np.array([1,0,2])[t.label],t.label)
    counts=pd.crosstab(t.pair,t.canonical_pair_label).reindex(columns=[0,1,2],fill_value=0)
    prior_canon=np.bincount(t.canonical_pair_label,minlength=3)/len(t)
    dp=[pair_key(a,b) for a,b in zip(d.model_a,d.model_b)]
    probs=[];train_counts=[]
    for key,a,b in zip(dp,d.model_a,d.model_b):
        cnt=counts.loc[key].to_numpy() if key in counts.index else np.zeros(3)
        q=(cnt+20*prior_canon)/(cnt.sum()+20)
        probs.append(q[[1,0,2]] if a>b else q);train_counts.append(int(cnt.sum()))
    save('identity_diagnostic.json',{'diagnostic_only_not_deployable':True,'metrics':scores(y,np.array(probs)),
        'training_model_count':len(set(t.model_a)|set(t.model_b)),'training_pair_count':len(counts),
        'development_rows_unseen_pair':int((np.array(train_counts)==0).sum()),
        'note':'Model identity exists only in the labeled CSV; it is excluded from actual models. Frequencies mix task difficulty and model strength.'})
    # Private HTML gallery contains raw competition text; never copied into the repo.
    import html
    picks=[]
    for name,mask in [('worst_loss',np.ones(len(d),bool)),('true_tie',y==2),('truncated',d.cut_1024_cap256.to_numpy()),
                      ('code',d.task_code.to_numpy()),('non_english',d.language_group.eq('OTHER_IDENTIFIED').to_numpy()),
                      ('confident_wrong',confident_wrong)]:
        idx=np.flatnonzero(mask);idx=idx[np.argsort(loss[idx])[::-1]][:8]
        picks.extend((name,int(i)) for i in idx)
    inventory=[];blocks=['<meta charset="utf-8"><title>Private development error gallery</title><style>body{max-width:1100px;margin:2em auto;font:16px sans-serif}pre{white-space:pre-wrap}</style><h1>Private development error gallery</h1>']
    text_rows=frame.set_index('id')
    for category,i in picks:
        row=d.iloc[i];item={'category':category,'id':int(row.id),'label':CLASSES[y[i]],'prediction':CLASSES[p[i].argmax()],
            'loss':float(loss[i]),'probabilities':p[i].tolist(),'truncated':bool(row.cut_1024_cap256)}
        inventory.append(item)
        blocks.append(f'<details><summary>{html.escape(str(item))}</summary>')
        for col in TEXT_COLUMNS:blocks.append(f'<h3>{col}</h3><pre>{html.escape(text_rows.loc[row.id,col])}</pre>')
        blocks.append('</details>')
    (EDA_ROOT/'private_error_gallery.html').write_text('\n'.join(blocks))
    json_write(EDA_ROOT/'case_inventory.json',inventory)
    analyze_inference_diagnostic(f)
    print('EDA analysis complete',flush=True)


def analyze_inference_diagnostic(f):
    source=ROOT/'runs/attempt_002/jobs/eda_diagnostic512/output'
    if not (source/'eda_model_predictions.npz').exists():return
    summary=json.loads((source/'eda_model_summary.json').read_text())
    data=np.load(source/'eda_model_predictions.npz')
    d=f.set_index('id').loc[data['ids']].reset_index()
    y=data['labels'];assert d.split.eq('dev').all() and np.array_equal(y,d.label)
    b=GroupBootstrap(d.prompt_group,repeats=2000,seed=SEED+1)
    t=summary['temperature_frozen']
    po=data['original'];ps=data['swapped'];pa=data['average'];pl=data['long_average']
    orig=losses(y,po);avg=losses(y,pa)
    co=losses(y,temper(po,t));ca=losses(y,temper(pa,t));cl=losses(y,temper(pl,t))
    changed=data['context_changed'];flips=(po.argmax(1)!=ps.argmax(1)).astype(float)
    second_advantage=((ps[:,0]-po[:,0])+(po[:,1]-ps[:,1]))/2
    summary.update({'groups':b.n_groups,'bootstrap_repeats':len(b.weights),
        'order_flip_fraction_95_group_bootstrap':b.interval(flips),
        'raw_swap_averaging_gain':float((orig-avg).mean()),'raw_swap_averaging_gain_95_group_bootstrap':b.interval(orig-avg),
        'calibrated_swap_averaging_gain':float((co-ca).mean()),
        'calibrated_swap_averaging_gain_95_group_bootstrap':b.interval(co-ca),
        'second_position_mean_probability_advantage':float(second_advantage.mean()),
        'second_position_advantage_95_group_bootstrap':b.interval(second_advantage),
        'original_vs_aligned_swapped_argmax_confusion':confusion_matrix(po.argmax(1),ps.argmax(1),labels=[0,1,2]).tolist(),
        'context_2048_gain':float((ca-cl).mean()),'context_2048_gain_95_group_bootstrap':b.interval(ca-cl),
        'context_changed_only_gain':float((ca[changed]-cl[changed]).mean()),
        'context_changed_only_gain_95_group_bootstrap':b.interval(ca-cl,changed),
        'context_2048_accuracy_change':float(((pl.argmax(1)==y).astype(float)-(pa.argmax(1)==y)).mean()),
        'context_accuracy_change_95_group_bootstrap':b.interval((pl.argmax(1)==y).astype(float)-(pa.argmax(1)==y)),
        'paired_context_rows_improved':int((cl<ca-1e-8).sum()),'paired_context_rows_worsened':int((cl>ca+1e-8).sum()),
        'important':'Subset scores are not full-development or Kaggle estimates. Existing deployed model already uses swap averaging. Fixed-temperature 2048 result is a diagnostic, not a new submission.'})
    save('inference_diagnostic.json',summary)
