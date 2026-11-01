"""Full-data refit for internet-disabled Kaggle inference; no API calls."""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from .data import CLASSES,TEXT_COLUMNS,SEED,parse_turns,structural_features,swapped,dense_pair_features,swapped_pair_features
from .models import augment,symmetric_predict
from .rnn import make_sequences,PreferenceRNN
from .vectorizers import tfidf_features


def train_full_rnn(frame,y,stats,epochs):
    torch.manual_seed(SEED);torch.set_num_threads(4)
    idx=np.arange(len(y))
    seq,vocab,_=make_sequences(frame,idx)
    swapped_stats=structural_features(swapped(frame))
    scaler=StandardScaler().fit(stats[idx])
    ss=scaler.transform(stats).astype(np.float32)
    sw=scaler.transform(swapped_stats).astype(np.float32)
    model=PreferenceRNN(len(vocab)+2,stats.shape[1])
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    rng=np.random.default_rng(SEED)
    for epoch in range(epochs):
        model.train();started=time.monotonic();total=0.
        order=rng.permutation(idx)
        for start in range(0,len(order),128):
            rows=order[start:start+128]
            x=torch.as_tensor(seq[rows].copy(),dtype=torch.long)
            s=torch.as_tensor(ss[rows].copy())
            labels=y[rows].copy()
            flip=rng.random(len(rows))<.5
            x[flip]=x[flip][:,[0,2,1]]
            s[flip]=torch.as_tensor(sw[rows[flip]])
            labels[flip]=np.array([1,0,2])[labels[flip]]
            target=torch.as_tensor(labels,dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(model(x,s),target,label_smoothing=.02)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            total+=loss.item()*len(rows)
        print(f'Full-data GRU epoch {epoch+1}/{epochs}: train loss {total/len(y):.5f}, {time.monotonic()-started:.1f}s',flush=True)
    model.eval();embeddings=[];predictions=[]
    with torch.inference_mode():
        for j in range(3):
            chunks=[]
            for start in range(0,len(frame),256):
                chunks.append(model.encode(torch.as_tensor(seq[start:start+256,j],dtype=torch.long)).numpy())
            embeddings.append(np.concatenate(chunks))
        for start in range(0,len(frame),256):
            x=torch.as_tensor(seq[start:start+256],dtype=torch.long)
            s=torch.as_tensor(ss[start:start+256]);swapstats=torch.as_tensor(sw[start:start+256])
            predictions.append(((model(x,s).softmax(1)+model(x[:,[0,2,1]],swapstats).softmax(1)[:,[1,0,2]])*.5).numpy())
    return tuple(embeddings),np.concatenate(predictions)


def train_submission(raw_dir,work_dir,config):
    raw_dir=Path(raw_dir);work_dir=Path(work_dir);work_dir.mkdir(parents=True,exist_ok=True)
    train=pd.read_csv(raw_dir/'train.csv');test=pd.read_csv(raw_dir/'test.csv')
    y=train[CLASSES].values.argmax(1)
    for frame in (train,test):
        for col in TEXT_COLUMNS:frame[col]=frame[col].map(parse_turns)
    frame=pd.concat([train,test],ignore_index=True);idx=np.arange(len(train))
    stats=structural_features(frame);swapstats=structural_features(swapped(frame))
    weights={k:v for k,v in config['weights'].items() if v>0}
    predictions={};features={}
    if any(n.startswith('tfidf') for n in weights):
        embeddings,pair,pairswap=tfidf_features(frame,idx,work_dir)
        if 'tfidf_sparse__logistic' in weights:
            features['tfidf_sparse']=(sparse.hstack([pair,sparse.csr_matrix(stats)],format='csr'),
                                      sparse.hstack([pairswap,sparse.csr_matrix(swapstats)],format='csr'))
        features['tfidf_svd']=(dense_pair_features(embeddings,stats),swapped_pair_features(embeddings,stats))
        del pair,pairswap
    if any(n.startswith('rnn') for n in weights):
        embeddings,p=train_full_rnn(frame,y,stats,config['rnn_best_epoch'])
        predictions['rnn__end_to_end']=p
        features['rnn']=(dense_pair_features(embeddings,stats),swapped_pair_features(embeddings,stats))
    features['structural']=(stats,swapstats)
    for name in weights:
        if name in predictions:continue
        representation,kind=name.split('__')
        x,xswap=features[representation]
        xx,yy=augment(x,xswap,y,idx)
        settings=config['settings'][name]
        if kind=='logistic':
            model=LogisticRegression(C=settings['C'],max_iter=250,solver='lbfgs',random_state=SEED)
        elif kind=='gbdt':
            model=lgb.LGBMClassifier(n_estimators=settings['best_iteration'],learning_rate=.03,num_leaves=15,
                min_child_samples=100,colsample_bytree=.8,subsample=.8,subsample_freq=1,reg_lambda=5.,
                verbosity=-1,n_jobs=4,random_state=SEED,deterministic=True,force_col_wise=True)
        else:raise ValueError(name)
        model.fit(xx,yy)
        predictions[name]=symmetric_predict(model,x,xswap)
        print('Refitted',name,flush=True)
        del xx,yy
    blend=sum(weights[n]*predictions[n][len(train):] for n in weights)
    logits=np.log(np.clip(blend,1e-12,1))/config['temperature'];logits-=logits.max(1,keepdims=True)
    p=np.exp(logits);p/=p.sum(1,keepdims=True)
    submission=pd.DataFrame(p,columns=CLASSES);submission.insert(0,'id',test.id.to_numpy())
    assert submission.id.is_unique and np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1)
    assert submission.columns.tolist()==pd.read_csv(raw_dir/'sample_submission.csv').columns.tolist()
    submission.to_csv(work_dir/'submission.csv',index=False)
    return submission
