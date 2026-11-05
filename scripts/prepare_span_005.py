"""Freeze v5 input identities and exact folds; no predictor is fitted."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys,importlib.metadata
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,CLASSES,TEXT_COLUMNS,prompt_groups,json_write
from arena.joint_data import read_competition,split_from_ids
from arena.vectorizers import clip_text
from arena.source_models import answer_hash


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    out=ROOT/'runs/span_005';pub=REPO/'results/span_005';out.mkdir(parents=True,exist_ok=True);pub.mkdir(exist_ok=True)
    assert not (out/'inputs.npz').exists(),'Preparation is immutable; inspect existing inputs instead of overwriting'
    cfg=json.loads((REPO/'configs/span_005.json').read_text())
    previous=json.loads((REPO/'results/span_004/artifact_inventory.json').read_text())
    files={k:Path(previous[k]['path']) for k in ['classifier','adapter','split','raw_train','candidate']}
    for k,p in files.items():assert digest(p)==previous[k]['sha256'],k
    files.update(embeddings=ROOT/'embeddings/qwen3_256.npz',embedding_metadata=ROOT/'embeddings/qwen3_256.json',raw_test=ROOT/'raw/test.csv')
    frame,test=read_competition(ROOT/'raw');split=json.loads(files['split'].read_text());parts=split_from_ids(frame,split)
    train=frame.iloc[parts['train']].copy();dev=frame.iloc[parts['dev']].copy();dev=dev.set_index('id').loc[split['dev']].reset_index()
    assert len(train)==45985 and len(dev)==5746
    y=dev[CLASSES].values.argmax(1);yt=train[CLASSES].values.argmax(1);groups=prompt_groups(dev).to_numpy();tg=prompt_groups(train).to_numpy()
    raw=np.load(files['classifier']);lookup={int(i):j for j,i in enumerate(raw['ids'])};ix=[lookup[int(i)] for i in dev.id]
    np.testing.assert_array_equal(raw['labels'][ix],y)
    p=(raw['original'][ix]+raw['aligned_swap'][ix])/2
    assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-6)
    folds=np.full(len(dev),-1,int)
    for k,(fit,val) in enumerate(StratifiedGroupKFold(cfg['outer_group_folds'],shuffle=True,random_state=cfg['seed']).split(p,y,groups)):
        assert not set(groups[fit])&set(groups[val]);folds[val]=k
    ug=np.unique(tg);order=np.random.default_rng(cfg['seed']).permutation(len(ug));assignment=np.empty(len(ug),int);assignment[order]=np.arange(len(ug))%cfg['source_group_folds']
    source_folds=assignment[np.searchsorted(ug,tg)]
    metadata=json.loads(files['embedding_metadata'].read_text());both=pd.concat([frame,test],ignore_index=True)
    texts=[clip_text(t) for col in TEXT_COLUMNS for t in both[col]];unique=list(dict.fromkeys(texts))
    provenance=hashlib.sha256((metadata['model']+str(metadata['dimensions'])+json.dumps(unique,ensure_ascii=False)).encode()).hexdigest()
    assert provenance==metadata['sha256'] and len(unique)==metadata['unique_texts']
    emb=np.load(files['embeddings']);assert emb['response_a'].shape==(len(both),256)
    devlookup={int(i):j for j,i in enumerate(frame.id)};dix=[devlookup[int(i)] for i in dev.id]
    counts=pd.concat([train.model_a,train.model_b]).value_counts();vocab=sorted(counts[counts>=cfg['source_min_training_answers']].index)+['__OTHER__']
    train_names=sorted(counts.index)
    cols=['id','prompt','response_a','response_b','model_a','model_b',*CLASSES]
    train[cols].to_parquet(out/'train.parquet',index=False);dev[cols].to_parquet(out/'dev.parquet',index=False)
    np.savez(out/'inputs.npz',ids=dev.id.to_numpy(),train_ids=train.id.to_numpy(),labels=y,train_labels=yt,groups=groups,train_groups=tg,
             probabilities=p,outer_fold=folds,source_fold=source_folds,
             dev_a=emb['response_a'][dix],dev_b=emb['response_b'][dix],
             train_a=emb['response_a'][parts['train']],train_b=emb['response_b'][parts['train']])
    vocab_path=out/'source_vocabulary.json';json_write(vocab_path,{'source_classes':vocab,'all_training_sources':train_names,'source_counts':counts.to_dict()})
    entries={k:{'path':str(p),'sha256':digest(p)} for k,p in files.items()}
    for name in ['train.parquet','dev.parquet','inputs.npz','source_vocabulary.json']:
        entries[name]={'path':str(out/name),'sha256':digest(out/name)}
    sources=['GOAL.md','configs/span_005.json','docs/SPAN_005_PROTOCOL.md','docs/SPAN_005_SOURCE_MODEL_PLAN.md',
             'scripts/prepare_span_005.py','scripts/run_span_005_e06.py','src/arena/source_models.py',
             'src/arena/source_evaluation.py','src/arena/span_heads.py','src/arena/span_calibration.py','src/arena/data.py','src/arena/joint_data.py']
    snapshots={str(p.relative_to(REPO)):digest(p) for p in (REPO/'backlog/versions').glob('*') if p.is_file()}
    result={'span':5,'started_at':cfg['started_at'],'deadline':cfg['deadline'],'clock_source':'First execution clock observation for explicit user instruction; not a new goal-service activation',
            'focus':cfg['focus'],'train_rows':len(train),'dev_rows':len(dev),'dev_groups':len(set(groups)),
            'source_models':len(train_names),'source_classes':len(vocab),'rare_source_models':int((counts<cfg['source_min_training_answers']).sum()),
            'embedding_text_order_digest_verified':True,'input_artifacts':entries,'source_sha256':{p:digest(REPO/p) for p in sources},
            'old_snapshot_sha256':snapshots,'user_notebook_sha256':{p:digest(REPO/p) for p in ['notebooks/07_error_poster.ipynb','notebooks/12_backlog_v1_span.ipynb']},
            'versions':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn']},
            'gpu_budget_seconds':0,'confirmation_used':False,'new_external_extraction':False,'prepared_at':datetime.now(timezone.utc).isoformat()}
    for folder in [out,pub]:json_write(folder/'artifact_inventory.json',result)
    print(json.dumps({k:result[k] for k in ['span','started_at','train_rows','dev_rows','dev_groups','source_models','source_classes','embedding_text_order_digest_verified']},indent=2))


if __name__=='__main__':main()
