"""Development-only error/retention audit for the frozen attempt-002 model."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from transformers import AutoTokenizer

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,CLASSES,TEXT_COLUMNS,parse_turns,json_write
from arena.joint_data import JointTokenizer

run=ROOT/'runs/attempt_002'
selection=json.loads((run/'model_selection.json').read_text())
preprocessing_sha=hashlib.sha256((REPO/'src/arena/joint_data.py').read_bytes()).hexdigest()
assert preprocessing_sha==selection['config']['preprocessing_sha256']
saved=np.load(run/'jobs'/selection['job']/'output/joint_run/dev_predictions.npz')
manifest=json.loads((run/'split_ids.json').read_text())
assert set(saved['ids'])==set(manifest['dev'])
frame=pd.read_csv(ROOT/'raw/train.csv',usecols=['id',*CLASSES,*TEXT_COLUMNS])
frame=frame.set_index('id').loc[saved['ids']].reset_index()
labels=frame[CLASSES].to_numpy().argmax(1)
assert np.array_equal(labels,saved['labels'])
for name in TEXT_COLUMNS:frame[name]=frame[name].map(parse_turns)
p=saved['probabilities']
logits=np.log(np.clip(p,1e-12,1))/selection['temperature']
logits-=logits.max(1,keepdims=True)
p=np.exp(logits);p/=p.sum(1,keepdims=True)
loss=-np.log(np.clip(p[np.arange(len(p)),labels],np.finfo(float).eps,1))
assert np.isclose(loss.mean(),selection['calibrated_development']['log_loss'])
tokenizer=AutoTokenizer.from_pretrained(run/'jobs'/selection['job']/'output/joint_run/adapter',local_files_only=True)
lengths={}
for name in TEXT_COLUMNS:
    counts=[]
    for start in range(0,len(frame),256):
        tokens=tokenizer(frame[name].iloc[start:start+256].tolist(),add_special_tokens=False,
            truncation=False,padding=False)['input_ids']
        counts.extend(map(len,tokens))
    lengths[name]=np.asarray(counts)
pn,an,bn=[lengths[k] for k in TEXT_COLUMNS]


def retention(context):
    encoder=JointTokenizer(tokenizer,context)
    available=context-encoder.overhead
    prompt_budget=np.minimum(pn,min(256,max(32,available//4)))
    remaining=available-prompt_budget
    a_budget=np.minimum(an,remaining//2)
    b_budget=np.minimum(bn,remaining//2)
    unused=remaining-a_budget-b_budget
    take=np.minimum(unused,an-a_budget)
    a_budget+=take;unused-=take
    b_budget+=np.minimum(unused,bn-b_budget)
    prompt_cut=pn>prompt_budget
    answer_cut=(an>a_budget)|(bn>b_budget)
    return prompt_cut,answer_cut


def metrics(mask):
    return {'rows':int(mask.sum()),'fraction':float(mask.mean()),
        'log_loss':float(loss[mask].mean()) if mask.any() else None,
        'accuracy':float((p.argmax(1)[mask]==labels[mask]).mean()) if mask.any() else None}


prompt_cut,answer_cut=retention(selection['config']['max_length'])
cut=prompt_cut|answer_cut
prospective={}
for context in [1024,2048,4096]:
    a,b=retention(context)
    prospective[str(context)]={'any_field_truncated_fraction':float((a|b).mean()),
        'answer_truncated_fraction':float(b.mean()),'prompt_truncated_fraction':float(a.mean())}
confusion=confusion_matrix(labels,p.argmax(1),labels=[0,1,2])
result={'model_job':selection['job'],'partition':'development','confirmation_used':False,
    'preprocessing_sha256':preprocessing_sha,
    'tokenizer_sha256':hashlib.sha256((run/'jobs'/selection['job']/'output/joint_run/adapter/tokenizer.json').read_bytes()).hexdigest(),
    'rows':len(frame),'temperature':selection['temperature'],
    'overall':metrics(np.ones(len(frame),dtype=bool)),
    'by_truncation':{'no_field_truncated':metrics(~cut),'any_field_truncated':metrics(cut),
        'answer_truncated':metrics(answer_cut),'prompt_truncated':metrics(prompt_cut)},
    'prospective_retention_only':prospective,
    'by_true_class':{name:{**metrics(labels==i),
        'prediction_fraction':float((p.argmax(1)==i).mean())} for i,name in enumerate(CLASSES)},
    'confusion_matrix_true_rows_predicted_columns':confusion.tolist(),
    'caveat':'Length groups differ in difficulty; this is observational, not evidence that longer context improves quality. Larger-context quality has not been evaluated.'}
out=run/'diagnostics';out.mkdir(exist_ok=True)
pd.DataFrame({'id':frame.id,'label':labels,'loss':loss,'prompt_tokens':pn,'answer_a_tokens':an,
    'answer_b_tokens':bn,'prompt_cut':prompt_cut,'answer_cut':answer_cut}).to_parquet(out/'development_errors.parquet',index=False)
json_write(out/'development_error_audit.json',result)
json_write(REPO/'results/attempt_002/development_error_audit.json',result)
print(json.dumps(result,indent=2))
