"""Opponent-adjusted strength and swap-equivariant, fold-local residual heads."""
from __future__ import annotations

import hashlib
import re
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from .span_heads import normalize


def answer_hash(text):
    return hashlib.sha256(re.sub(r'\s+', ' ', text).strip().casefold().encode()).hexdigest()


def purge_answers(fit_hashes, held_hashes):
    held = set(held_hashes)
    return np.array([value not in held for value in fit_hashes], dtype=bool)


def softmax(logits):
    return np.exp(logits-logsumexp(logits, axis=1, keepdims=True))


class Strength:
    def __init__(self, penalty=30., maxiter=500):
        self.penalty, self.maxiter = penalty, maxiter

    def fit(self, a, b, y, names):
        self.names = list(names); self.index = {name:i for i,name in enumerate(names)}
        ia = np.array([self.index[x] for x in a]); ib = np.array([self.index[x] for x in b])
        n, m = len(y), len(names); target = np.eye(3)[y]
        def objective(theta):
            d = (theta[ia]-theta[ib])/2
            logits = np.column_stack([d,-d,np.full(n,theta[-1])])
            logp = logits-logsumexp(logits,axis=1,keepdims=True)
            error = np.exp(logp)-target
            direction = (error[:,0]-error[:,1])/2
            grad = np.r_[np.bincount(ia,weights=direction,minlength=m)-np.bincount(ib,weights=direction,minlength=m),error[:,2].sum()]
            grad[:-1] += self.penalty*theta[:-1]
            value = -logp[np.arange(n),y].sum()+self.penalty*np.dot(theta[:-1],theta[:-1])/2
            return value/n,grad/n
        theta = np.zeros(m+1); prior = np.mean(y==2);theta[-1]=np.log(2*prior/(1-prior))
        fit = minimize(objective,theta,jac=True,method='L-BFGS-B',options={'maxiter':self.maxiter,'ftol':1e-12,'gtol':1e-8})
        assert fit.success and np.isfinite(fit.fun),fit.message
        self.theta=fit.x;self.details={'iterations':int(fit.nit),'gradient_norm':float(np.linalg.norm(fit.jac))}
        # Prompt-cluster sandwich scores are retained for an explicitly conditional
        # normal approximation, not represented as refitted bootstrap intervals.
        p=self.predict(a,b);e=p-target;z=(e[:,0]-e[:,1])/2
        design=np.zeros((n,m+1));design[np.arange(n),ia]+=.5;np.add.at(design,(np.arange(n),ib),-.5)
        self.row_scores=2*z[:,None]*design;self.row_scores[:,-1]=e[:,2]
        variance=(p[:,0]+p[:,1]-(p[:,0]-p[:,1])**2)
        h=(design[:,:m].T*variance)@design[:,:m]
        h+=self.penalty*np.eye(m)
        cross=design[:,:m].T@(-(p[:,0]-p[:,1])*p[:,2])
        self.hessian=np.block([[h,cross[:,None]],[cross[None,:],np.array([[(p[:,2]*(1-p[:,2])).sum()]])]])
        return self

    def ratings(self,names):
        return np.array([self.theta[self.index[x]] if x in self.index else 0. for x in names])

    def predict(self,a,b):
        d=(self.ratings(a)-self.ratings(b))/2
        return normalize(softmax(np.column_stack([d,-d,np.full(len(d),self.theta[-1])])))


class ResidualHead:
    """Unpenalized symmetric calibration plus regularized auxiliary corrections."""
    def __init__(self,regularization=.01,slope_bounds=(1/3,2.),bias_bounds=(-1.,1.),maxiter=500):
        self.regularization=regularization;self.slope_bounds=slope_bounds;self.bias_bounds=bias_bounds;self.maxiter=maxiter

    def transform(self,p,d,s):
        lp=np.log(normalize(p));base_d=(lp[:,0]-lp[:,1])/2;base_s=lp[:,2]-(lp[:,0]+lp[:,1])/2
        return base_d,base_s,np.asarray(d)/self.dscale,(np.asarray(s)-self.smean)/self.sscale

    def objective(self,theta,features,y):
        bd,bs,d,s=features;k=d.shape[1]
        z=theta[0]*bd+d@theta[3:3+k];t=theta[1]*bs+theta[2]+s@theta[3+k:]
        logits=np.column_stack([z,-z,t]);lp=logits-logsumexp(logits,axis=1,keepdims=True)
        e=np.exp(lp);e[np.arange(len(y)),y]-=1;ed=e[:,0]-e[:,1];et=e[:,2]
        grad=np.r_[np.mean(ed*bd),np.mean(et*bs),np.mean(et),d.T@ed/len(y),s.T@et/len(y)]
        grad[3:]+=self.regularization*theta[3:]
        value=-lp[np.arange(len(y)),y].mean()+self.regularization*np.dot(theta[3:],theta[3:])/2
        return float(value),grad

    def fit(self,p,d,s,y):
        d=np.asarray(d,float);s=np.asarray(s,float);y=np.asarray(y,int)
        self.dscale=np.maximum(np.sqrt((d*d).mean(0)),.05)
        self.smean=s.mean(0);self.sscale=np.maximum(s.std(0),.05)
        features=self.transform(p,d,s);theta=np.r_[1.,1.,0.,np.zeros(d.shape[1]+s.shape[1])]
        bounds=[self.slope_bounds,self.slope_bounds,self.bias_bounds]+[(None,None)]*(len(theta)-3)
        fit=minimize(lambda x:self.objective(x,features,y),theta,jac=True,bounds=bounds,method='L-BFGS-B',
                     options={'maxiter':self.maxiter,'ftol':1e-11,'gtol':1e-7})
        assert fit.success and np.isfinite(fit.fun),fit.message
        self.theta=fit.x;self.details={'parameters':len(theta),'iterations':int(fit.nit),'gradient_norm':float(np.linalg.norm(fit.jac))}
        return self

    def predict_proba(self,p,d,s):
        bd,bs,d,s=self.transform(p,d,s);k=d.shape[1]
        z=self.theta[0]*bd+d@self.theta[3:3+k];t=self.theta[1]*bs+self.theta[2]+s@self.theta[3+k:]
        return normalize(softmax(np.column_stack([z,-z,t])))


def source_features(qa,qb,ratings,kind):
    qa=np.asarray(qa,float);qb=np.asarray(qb,float)
    assert qa.shape==qb.shape and np.allclose(qa.sum(1),1) and np.allclose(qb.sum(1),1)
    ea,eb=qa@ratings,qb@ratings
    va=np.maximum(0,qa@(ratings**2)-ea**2);vb=np.maximum(0,qb@(ratings**2)-eb**2)
    ha=-(qa*np.log(np.maximum(qa,1e-12))).sum(1);hb=-(qb*np.log(np.maximum(qb,1e-12))).sum(1)
    if kind=='full':
        return qa-qb,np.column_stack([(qa+qb)/2,np.abs(qa-qb),ha+hb,np.abs(ha-hb)])
    assert kind=='strength'
    return np.column_stack([ea-eb,ha-hb]),np.column_stack([np.abs(ea-eb),(ea+eb)/2,va+vb,ha+hb,np.abs(ha-hb)])


def style_features(a,b):
    def one(texts):
        return np.array([[np.log1p(len(t)),np.log1p(len(t.split())),np.log1p(t.count('\n')),
                          np.log1p(t.count('```')),float(bool(re.search(r'(?i)\b(sorry|cannot|unable|as an ai)\b',t)))] for t in texts])
    x,z=one(a),one(b)
    return x-z,np.column_stack([(x+z)/2,np.abs(x-z)])
