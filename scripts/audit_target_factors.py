"""Target associations, metadata diagnostics and leakage controls on fixed splits."""
import hashlib
import json
import os
import sys
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from scipy.stats import chi2_contingency

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT, CLASSES, TEXT_COLUMNS, parse_turns, json_write
from arena.judge import metrics

directory=ROOT/'runs/attempt_005_target_audit';directory.mkdir(parents=True,exist_ok=True)
dest=REPO/'results/attempt_005_target_audit';dest.mkdir(parents=True,exist_ok=True)
f=pd.read_parquet(ROOT/'eda/attempt_002/features.parquet')
y=f.label.to_numpy()
train=np.flatnonzero(f.split.eq('train'))
parts={name:np.flatnonzero(f.split.eq(name)) for name in ['dev','confirmation']}
for idx in parts.values():assert not set(f.prompt_group.iloc[train])&set(f.prompt_group.iloc[idx])
prior=np.bincount(y[train],minlength=3)/len(train)
f['ordered_model_pair']=f.model_a+' | '+f.model_b
f['log_answer_word_ratio']=np.log1p(f.response_a_words)-np.log1p(f.response_b_words)
f['response_word_difference']=f.response_a_words-f.response_b_words
f['refusal_difference']=f.response_a_refusal.astype(int)-f.response_b_refusal.astype(int)
f['empty_difference']=f.response_a_empty.astype(int)-f.response_b_empty.astype(int)
f['row_position']=np.arange(len(f))
for modulus in [2,4,8,10,16,32,100,256]:
    f['id_mod_'+str(modulus)]=f.id%modulus
    f['row_mod_'+str(modulus)]=f.row_position%modulus


def scores(p):
    return {name:metrics(y[idx],p[idx]) for name,idx in parts.items()}


def factor(name):
    values=f[name]
    if name not in ['model_a','model_b','ordered_model_pair'] and not name.startswith(('id_mod_','row_mod_')) and values.nunique()>20:
        edges=np.unique(np.quantile(values.iloc[train].dropna(),np.linspace(0,1,11)))
        values=pd.Series(np.searchsorted(edges[1:-1],values,side='right'),index=f.index).astype(str)
    else:values=values.astype(str)
    counts=pd.crosstab(values.iloc[train],y[train]).reindex(columns=[0,1,2],fill_value=0)
    q=(counts+30*prior).div(counts.sum(axis=1)+30,axis=0)
    p=q.reindex(values).fillna(dict(enumerate(prior))).to_numpy()
    contingency=pd.crosstab(values,y).reindex(columns=[0,1,2],fill_value=0)
    chi=chi2_contingency(contingency,correction=False)[0] if len(contingency)>1 else 0.
    return {'factor':name,'categories':len(counts),'mutual_information_nats_training':float(mutual_info_score(values.iloc[train],y[train])),
        'cramers_v_all_rows':float(np.sqrt(chi/(len(f)*min(max(len(contingency)-1,1),2)))),
        'metrics':scores(p),'interpretation':'Unavailable in test' if name in ['model_a','model_b','ordered_model_pair'] else 'Test-computable descriptor'},p


excluded={'id','label','canonical_label','row_position'}
numeric=[c for c in f.select_dtypes(include=['number','bool']).columns if c not in excluded and not c.startswith(('id_mod_','row_mod_'))]
assert not set(numeric)&{'label','canonical_label','id','row_position','model_a','model_b','split'}
names=['model_a','model_b','ordered_model_pair','id','row_position',*numeric,
       *[c for c in f if c.startswith(('id_mod_','row_mod_'))]]
factors=[]
for name in dict.fromkeys(names):
    row,_=factor(name);factors.append(row)
factors.sort(key=lambda r:r['metrics']['dev']['log_loss'])
print('FACTORS',json.dumps([{ 'factor':r['factor'],'dev_loss':r['metrics']['dev']['log_loss'],
    'confirmation_loss':r['metrics']['confirmation']['log_loss']} for r in factors[:12]]),flush=True)

models=[]
predictions={}
def record(name,p,features,note):
    results={'model':name,'features':features,'note':note,'metrics':scores(p)}
    models.append(results);predictions[name]=p
    print('MODEL',json.dumps({'model':name,'metrics':results['metrics']}),flush=True)
record('training_prior',np.tile(prior,(len(f),1)),[], 'Train-partition frequencies only')

def gbdt(name,columns,labels=y,note=''):
    x=f[columns].astype(float).replace([np.inf,-np.inf],np.nan)
    model=LGBMClassifier(n_estimators=250,num_leaves=15,max_depth=-1,learning_rate=.05,
        min_child_samples=150,reg_lambda=5,n_jobs=4,verbosity=-1,random_state=20261117,
        deterministic=True,force_col_wise=True)
    model.fit(x.iloc[train],labels[train])
    p=model.predict_proba(x)
    record(name,p,columns,note)
    return model

id_columns=['id','row_position',*[c for c in f if c.startswith(('id_mod_','row_mod_'))]]
gbdt('id_and_position_only',id_columns,note='Negative-control metadata; no text or model identity')
lengths=['prompt_chars','response_a_chars','response_b_chars','response_word_difference','log_answer_word_ratio','answer_length_ratio']
gbdt('length_only',lengths,note='Legitimate but shallow answer-length information; no identities')
shape_model=gbdt('input_structure',numeric,note='Explicit allowlist excludes all label-derived and identity fields')
importance=pd.DataFrame({'feature':numeric,'gain':shape_model.booster_.feature_importance(importance_type='gain')}).sort_values('gain',ascending=False)

identity=make_pipeline(OneHotEncoder(handle_unknown='ignore'),LogisticRegression(max_iter=500,C=1.))
identity.fit(f[['model_a','model_b']].iloc[train],y[train])
identity_p=identity.predict_proba(f[['model_a','model_b']])
record('model_identity_only',identity_p,['model_a','model_b'],'DIAGNOSTIC ONLY: names absent from test; no answer text or target-derived inputs')

# Canonical pair frequencies preserve strength differences regardless of A/B order.
reverse=f.model_a>f.model_b
pair=pd.Series([' | '.join(sorted([a,b])) for a,b in zip(f.model_a,f.model_b)])
cy=np.where(reverse,np.array([1,0,2])[y],y)
counts=pd.crosstab(pair.iloc[train],cy[train]).reindex(columns=[0,1,2],fill_value=0)
canonical_prior=np.bincount(cy[train],minlength=3)/len(train)
q=(counts+30*canonical_prior).div(counts.sum(axis=1)+30,axis=0)
p=q.reindex(pair).fillna(dict(enumerate(canonical_prior))).to_numpy()
p[reverse]=p[reverse][:,[1,0,2]]
record('model_pair_frequency',p,['model_a','model_b'],'DIAGNOSTIC ONLY: train-only smoothed pair counts; names absent from test')

# Identity + structure uses categorical names, never their observed target rates.
categorical=f[numeric+['model_a','model_b']].copy()
for c in ['model_a','model_b']:categorical[c]=categorical[c].astype('category')
combined=LGBMClassifier(n_estimators=250,num_leaves=15,learning_rate=.05,min_child_samples=150,
    reg_lambda=5,n_jobs=4,verbosity=-1,random_state=20261117,deterministic=True,force_col_wise=True)
combined.fit(categorical.iloc[train],y[train],categorical_feature=['model_a','model_b'])
record('identity_plus_structure',combined.predict_proba(categorical),numeric+['model_a','model_b'],
    'DIAGNOSTIC ONLY: unavailable model names plus legitimate shape descriptors')

shuffled=y.copy();shuffled[train]=np.random.default_rng(20261117).permutation(y[train])
gbdt('structure_shuffled_training_targets',numeric,labels=shuffled,note='Negative control; evaluated against actual labels')

# Real target-derived EDA metadata: deliberate positive control, never a predictor.
raw=pd.read_csv(ROOT/'raw/train.csv',usecols=['id',*TEXT_COLUMNS]).set_index('id').loc[f.id]
a=raw.response_a.map(parse_turns);b=raw.response_b.map(parse_turns)
text_reverse=(a>b).to_numpy()
recovered=np.where(text_reverse,np.array([1,0,2])[f.canonical_label],f.canonical_label)
assert np.array_equal(recovered,y)
positive_control={'field':'canonical_label','purpose':'Canonicalized human vote for duplicate-consistency EDA',
    'reconstruction':'Swap A/B in canonical_label when parsed response_a > parsed response_b lexicographically',
    'reconstructed_accuracy':float((recovered==y).mean()),'deployed_input':False,
    'warning':'This is actual target-derived metadata. Feeding it to a model would leak the answer; it is excluded from every predictive experiment above.'}

duplicates={}
for name in ['prompt_group','comparison_hash','canonical_comparison_hash']:
    grouped=f.groupby(name).agg(rows=('id','size'),splits=('split','nunique'))
    duplicates[name]={'duplicate_groups':int(grouped.rows.gt(1).sum()),
        'rows_in_duplicate_groups':int(grouped.loc[grouped.rows.gt(1),'rows'].sum()),
        'cross_partition_groups':int(grouped.splits.gt(1).sum())}

associations=[]
for name in ['answer_equal_exact','answer_equal_whitespace','both_answers_empty','one_answer_empty',
             'both_refusals','one_refusal','refusal_difference','longer_side']:
    for value,part in f.groupby(name):
        associations.append({'factor':name,'value':str(value),'rows':len(part),
            **dict(zip(CLASSES,np.bincount(part.label,minlength=3)/len(part)))})

pair_table=counts.copy()
pair_table['rows']=counts.sum(axis=1)
pair_table['max_class_rate']=counts.max(axis=1)/counts.sum(axis=1)
strong_pairs=[]
for name,row in pair_table.loc[pair_table.rows.ge(100)].sort_values('max_class_rate',ascending=False).head(12).iterrows():
    strong_pairs.append({'pair':name,'train_rows':int(row.rows),'largest_class_fraction':float(row.max_class_rate),
                         'class_counts':row[[0,1,2]].astype(int).tolist()})

summary={'attempt':'005_target_audit','rows':len(f),'class_names':CLASSES,
    'splits':f.split.value_counts().to_dict(),'models':models,'univariate_factors':factors,
    'structure_feature_importance':importance.to_dict(orient='records'),
    'conditional_associations':associations,'strong_model_pairs':strong_pairs,
    'duplicate_audit':duplicates,'target_derived_positive_control':positive_control,
    'protocol':{'train_partition_only_for_fitting':True,'prompt_overlap_between_partitions':False,
        'categorical_smoothing_pseudocount':30,'numeric_factor_bins':'Train-only decile edges',
        'gbdt':'250 trees, 15 leaves, min child 150, learning rate 0.05, lambda 5, no validation tuning',
        'confirmation':'Previously inspected confirmation; exploratory replication, not an untouched evaluation',
        'test_unavailable_fields':['model_a','model_b'],
        'public_visible_test_rows':len(pd.read_csv(ROOT/'raw/test.csv')),
        'scope':'No new model training on GPU, external APIs or competition submission'}}
json_write(directory/'results.json',summary);json_write(dest/'results.json',summary)
np.savez_compressed(directory/'predictions.npz',ids=f.id.to_numpy(),labels=y,**predictions)
print('Saved target-factor audit',flush=True)
