from __future__ import annotations

import copy
import re
import time
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from torch import nn

from .data import SEED, TEXT_COLUMNS, json_write

TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text):
    return TOKEN.findall(text.casefold())


def make_sequences(frame, train_idx, max_tokens=128, vocab_size=30000):
    counts = Counter()
    for col in TEXT_COLUMNS:
        for text in frame.iloc[train_idx][col]:
            counts.update(tokenize(text))
    vocab = {word:i+2 for i,(word,count) in enumerate(counts.most_common(vocab_size-2)) if count>=2}
    arrays=[]
    diagnostics={}
    for col in TEXT_COLUMNS:
        array=np.zeros((len(frame),max_tokens),dtype=np.int32)
        truncated=unknown=total=0
        for i,text in enumerate(frame[col]):
            words=tokenize(text)
            truncated+=len(words)>max_tokens
            if len(words)>max_tokens: words=words[:max_tokens//2]+words[-max_tokens//2:]
            ids=[vocab.get(w,1) for w in words] or [1]
            array[i,:len(ids)]=ids
            unknown+=sum(t==1 for t in ids);total+=len(ids)
        arrays.append(array)
        diagnostics[col]={"truncation_rate":truncated/len(frame),"unknown_token_rate":unknown/total}
    return np.stack(arrays,axis=1),vocab,diagnostics


class PreferenceRNN(nn.Module):
    """Shared GRU, learned from this competition's training texts and labels."""
    def __init__(self,vocab_size,stats_dim,embedding_dim=64,hidden=48):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,embedding_dim,padding_idx=0)
        self.gru=nn.GRU(embedding_dim,hidden,batch_first=True)
        self.dropout=nn.Dropout(.2)
        self.head=nn.Sequential(nn.Linear(8*hidden*2+stats_dim,128),nn.ReLU(),nn.Dropout(.35),nn.Linear(128,3))

    def encode(self,ids):
        mask=ids.ne(0)
        outputs,_=self.gru(self.dropout(self.embedding(ids)))
        lengths=mask.sum(1).clamp(min=1)
        mean=(outputs*mask.unsqueeze(-1)).sum(1)/lengths[:,None]
        last=outputs[torch.arange(len(ids)),lengths-1]
        return torch.cat([mean,last],dim=1)

    def forward(self,seq,stats):
        n,fields,length=seq.shape
        enc=self.encode(seq.reshape(n*fields,length)).reshape(n,fields,-1)
        p,a,b=enc.unbind(1)
        features=torch.cat([p,a,b,a-b,torch.abs(a-b),a*b,p*a,p*b,stats],dim=1)
        return self.head(features)


def train_rnn(frame,labels,parts,stats,out,epochs=12,max_tokens=128,hidden=48):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(4)
    out.mkdir(parents=True,exist_ok=True)
    seq,vocab,diagnostics=make_sequences(frame,parts["train"],max_tokens=max_tokens)
    scaler=StandardScaler().fit(stats[parts["train"]])
    scaled=scaler.transform(stats).astype(np.float32)
    # Standardize after swapping raw structural features.
    swapped_stats=np.concatenate([stats[:,:9],stats[:,18:27],stats[:,9:18],-stats[:,27:36],stats[:,36:45]],axis=1)
    scaled_swap=scaler.transform(swapped_stats).astype(np.float32)
    model=PreferenceRNN(len(vocab)+2,stats.shape[1],hidden=hidden)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    loss_fn=nn.CrossEntropyLoss(label_smoothing=.02)
    rng=np.random.default_rng(SEED)
    best=100.;stale=0;history=[];best_state=None

    @torch.inference_mode()
    def predict(idx):
        model.eval();result=[]
        for start in range(0,len(idx),256):
            rows=idx[start:start+256]
            x=torch.as_tensor(seq[rows],dtype=torch.long)
            s=torch.as_tensor(scaled[rows])
            ss=torch.as_tensor(scaled_swap[rows])
            p=(model(x,s).softmax(1)+model(x[:,[0,2,1]],ss).softmax(1)[:,[1,0,2]])*.5
            result.append(p.numpy())
        return np.concatenate(result)

    for epoch in range(epochs):
        started=time.monotonic();model.train();total=0.
        order=rng.permutation(parts["train"])
        for start in range(0,len(order),128):
            rows=order[start:start+128]
            x=torch.as_tensor(seq[rows].copy(),dtype=torch.long)
            s=torch.as_tensor(scaled[rows].copy())
            y=torch.as_tensor(labels[rows].copy(),dtype=torch.long)
            flip=rng.random(len(rows))<.5
            x[flip]=x[flip][:,[0,2,1]]
            s[flip]=torch.as_tensor(scaled_swap[rows[flip]])
            y[flip]=torch.as_tensor(np.array([1,0,2])[y[flip].numpy()])
            optimizer.zero_grad(set_to_none=True)
            loss=loss_fn(model(x,s),y)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();total+=loss.item()*len(rows)
        score=log_loss(labels[parts["dev"]],predict(parts["dev"]),labels=[0,1,2])
        record={"epoch":epoch+1,"train_loss":total/len(order),"dev_log_loss":score,"seconds":time.monotonic()-started}
        history.append(record);print("RNN",record,flush=True)
        json_write(out/"rnn_history.json",history)
        if score<best-.0002:
            best=score;best_state=copy.deepcopy(model.state_dict());stale=0
            torch.save({"state_dict":best_state,"vocab":vocab,"max_tokens":max_tokens,"hidden":hidden,
                        "scaler_mean":scaler.mean_,"scaler_scale":scaler.scale_},out/"rnn_best.pt")
        else:
            stale+=1
            if stale>=3:break
    model.load_state_dict(best_state);model.eval()
    predictions=predict(np.arange(len(frame)))
    embeddings=[]
    with torch.inference_mode():
        for j in range(3):
            chunks=[]
            for start in range(0,len(frame),256):
                chunks.append(model.encode(torch.as_tensor(seq[start:start+256,j],dtype=torch.long)).numpy())
            embeddings.append(np.concatenate(chunks))
    json_write(out/"rnn_metadata.json",{"vocabulary_size":len(vocab)+2,"max_tokens":max_tokens,
        "hidden":hidden,"embedding_dimension":hidden*2,"best_dev_log_loss":best,"diagnostics":diagnostics,
        "epochs_run":len(history),"fit_partition":"train","early_stopping_partition":"dev"})
    return tuple(embeddings),predictions
