from __future__ import annotations

import copy
import time

import joblib
import lightgbm as lgb
import numpy as np
import torch
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from torch import nn

from .data import SEED, json_write


def scores(y,p):
    assert p.shape==(len(y),3) and np.isfinite(p).all()
    assert np.all(p>=0) and np.allclose(p.sum(1),1,atol=1e-5)
    return {"log_loss":float(log_loss(y,p,labels=[0,1,2])),
            "accuracy":float(accuracy_score(y,p.argmax(1))),
            "brier":float(np.mean(np.sum((p-np.eye(3)[y])**2,axis=1)))}


def symmetric_predict(model,x,xswap):
    return (model.predict_proba(x)+model.predict_proba(xswap)[:,[1,0,2]])*.5


def augment(x,xswap,y,idx):
    if sparse.issparse(x):
        xx=sparse.vstack([x[idx],xswap[idx]],format="csr")
    else:
        xx=np.concatenate([x[idx],xswap[idx]])
    return xx,np.concatenate([y[idx],np.array([1,0,2])[y[idx]]])


class NeuralClassifier:
    def __init__(self,input_dim):
        torch.manual_seed(SEED)
        self.model=nn.Sequential(nn.Linear(input_dim,128),nn.ReLU(),nn.Dropout(.4),nn.Linear(128,3))
        self.scaler=StandardScaler()
        self.history=[]

    def fit(self,x,y,dev,devswap,dev_y,epochs=30):
        torch.set_num_threads(4)
        xx=torch.as_tensor(self.scaler.fit_transform(x).astype(np.float32))
        yy=torch.as_tensor(y,dtype=torch.long)
        optimizer=torch.optim.AdamW(self.model.parameters(),lr=.0005,weight_decay=.05)
        best=100.;stale=0;state=None
        generator=torch.Generator().manual_seed(SEED)
        for epoch in range(epochs):
            self.model.train()
            order=torch.randperm(len(y),generator=generator)
            for batch in order.split(256):
                optimizer.zero_grad(set_to_none=True)
                loss=nn.functional.cross_entropy(self.model(xx[batch]),yy[batch],label_smoothing=.02)
                loss.backward();optimizer.step()
            p=symmetric_predict(self,dev,devswap)
            value=log_loss(dev_y,p,labels=[0,1,2])
            self.history.append({"epoch":epoch+1,"dev_log_loss":value})
            if value<best-.0001:
                best=value;state=copy.deepcopy(self.model.state_dict());stale=0
            else:
                stale+=1
                if stale>=4:break
        self.model.load_state_dict(state)
        return self

    @torch.inference_mode()
    def predict_proba(self,x):
        self.model.eval()
        result=[]
        for start in range(0,len(x),1024):
            xx=torch.as_tensor(self.scaler.transform(x[start:start+1024]).astype(np.float32))
            result.append(self.model(xx).softmax(1).numpy())
        return np.concatenate(result)


def fit_classifiers(name,x,xswap,y,parts,out,kinds=("logistic","gbdt","neural")):
    """Select settings on dev; do not read holdout labels here."""
    out.mkdir(parents=True,exist_ok=True)
    xx,yy=augment(x,xswap,y,parts["train"])
    dev=parts["dev"]
    results=[]
    for kind in kinds:
        started=time.monotonic();settings={}
        if kind=="logistic":
            # Sparse TF-IDF is naturally scaled. Dense vectors include structural features.
            scaler=StandardScaler(with_mean=not sparse.issparse(x),with_std=not sparse.issparse(x)).fit(xx)
            sx=scaler.transform(xx)
            best=100.;trials=[];model=None
            for c in ([.1,1.,3.] if sparse.issparse(x) else [.001,.01,.1]):
                candidate=LogisticRegression(C=c,max_iter=250,solver="lbfgs",random_state=SEED)
                candidate.fit(sx,yy)
                p=(candidate.predict_proba(scaler.transform(x[dev]))+candidate.predict_proba(scaler.transform(xswap[dev]))[:,[1,0,2]])*.5
                loss=log_loss(y[dev],p,labels=[0,1,2]);trials.append({"C":c,"dev_log_loss":loss})
                if loss<best:best=loss;model=candidate;settings={"C":c}
            from sklearn.pipeline import make_pipeline
            model=make_pipeline(scaler,model)
            settings["trials"]=trials
        elif kind=="gbdt":
            model=lgb.LGBMClassifier(n_estimators=1500,learning_rate=.03,num_leaves=15,
                min_child_samples=100,colsample_bytree=.8,subsample=.8,subsample_freq=1,
                reg_lambda=5.,verbosity=-1,n_jobs=4,random_state=SEED,deterministic=True,force_col_wise=True)
            model.fit(xx,yy,eval_set=[(x[dev],y[dev]),(xswap[dev],np.array([1,0,2])[y[dev]])],
                callbacks=[lgb.early_stopping(60,verbose=False)])
            settings={"best_iteration":model.best_iteration_,"num_leaves":15}
        elif kind=="neural":
            model=NeuralClassifier(x.shape[1]).fit(xx,yy,x[dev],xswap[dev],y[dev])
            settings={"history":model.history,"epochs":len(model.history)}
        else:raise ValueError(kind)
        # Save predictions for all rows; metrics for holdout are computed only at the final frozen comparison.
        p=symmetric_predict(model,x,xswap)
        model_name=f"{name}__{kind}"
        np.save(out/f"{model_name}_predictions.npy",p)
        joblib.dump(model,out/f"{model_name}.joblib")
        row={"model":model_name,"representation":name,"classifier":kind,
             "dev":scores(y[dev],p[dev]),"seconds":time.monotonic()-started,"settings":settings,
             "n_train":len(parts["train"]),"feature_count":x.shape[1]}
        json_write(out/f"{model_name}.json",row)
        results.append(row)
        print(model_name,row["dev"],f'{row["seconds"]:.1f}s',flush=True)
    return results
