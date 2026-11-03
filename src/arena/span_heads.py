"""Fold-local symmetric probability heads and paired group uncertainty for span 001."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logsumexp
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


def normalize(p):
    p = np.asarray(p, dtype=np.float64)
    assert p.ndim == 2 and p.shape[1] == 3 and np.isfinite(p).all() and np.all(p >= 0)
    assert np.all(p <= 1+1e-6) and np.allclose(p.sum(1),1,atol=1e-6), 'Invalid probability rows; no prior fill or repair permitted'
    p = np.clip(p, 1e-12, 1)
    return p / p.sum(1, keepdims=True)


def loss_rows(y, p):
    return -np.log(normalize(p)[np.arange(len(y)), np.asarray(y, dtype=int)])


def metrics(y, p):
    p = normalize(p)
    y = np.asarray(y)
    guess = p.argmax(1)
    matrix = confusion_matrix(y, guess, labels=[0,1,2])
    return {'rows': len(y), 'log_loss': float(loss_rows(y,p).mean()),
            'accuracy': float(accuracy_score(y,guess)),
            'macro_f1': float(f1_score(y,guess,labels=[0,1,2],average='macro',zero_division=0)),
            'brier': float(np.mean(np.sum((p-np.eye(3)[y])**2,axis=1))),
            'class_counts': np.bincount(y,minlength=3).tolist(),
            'class_loss': [float(loss_rows(y,p)[y==i].mean()) if (y==i).any() else None for i in range(3)],
            'class_recall': (np.diag(matrix)/np.maximum(matrix.sum(1),1)).tolist(),
            'tie_precision': float(matrix[2,2]/max(matrix[:,2].sum(),1)),
            'mean_predictive_entropy': float(-(p*np.log(p)).sum(1).mean()),
            'mean_class_probability': p.mean(0).tolist(),
            'predicted_class_fraction': (np.bincount(guess,minlength=3)/len(y)).tolist(),
            'tie_auc': float(roc_auc_score(y==2,p[:,2])) if len(np.unique(y==2)) == 2 else None,
            'confusion_matrix': matrix.tolist()}


class SymmetricHead:
    """Same parameter count for coupled CE and factorized tie/conditional-winner CE."""
    def __init__(self, kind='coupled', regularization=.01):
        assert kind in ('coupled','factorized')
        self.kind, self.regularization = kind, regularization

    def _objective(self, theta, d, s, y):
        w, v = theta[:d.shape[1]], theta[d.shape[1]:]
        direction, tie = d@w, s@v
        is_tie = (y==2).astype(float)
        if self.kind == 'factorized':
            decisive = 1-is_tie
            target_a = (y==0).astype(float)
            value = (np.logaddexp(0,tie)-is_tie*tie +
                     decisive*(np.logaddexp(0,direction)-target_a*direction)).mean()
            gw = d.T@(decisive*(expit(direction)-target_a))/len(y)
            gv = s.T@(expit(tie)-is_tie)/len(y)
        else:
            logits = np.column_stack([direction/2,-direction/2,tie])
            logp = logits-logsumexp(logits,axis=1,keepdims=True)
            p = np.exp(logp)
            value = -logp[np.arange(len(y)),y].mean()
            target_direction = (y==0).astype(float)-(y==1).astype(float)
            gw = d.T@(.5*(p[:,0]-p[:,1]-target_direction))/len(y)
            gv = s.T@(p[:,2]-is_tie)/len(y)
        penalized = theta.copy()
        penalized[-1] = 0.  # symmetric intercept represents tie prevalence
        value += .5*self.regularization*np.dot(penalized,penalized)
        gradient = np.concatenate([gw,gv])+self.regularization*penalized
        return float(value), gradient

    def fit(self, d, s, y):
        d, s, y = np.asarray(d,float), np.asarray(s,float), np.asarray(y,int)
        self.direction_scale = np.maximum(np.sqrt(np.mean(d*d,axis=0)),1e-6)
        self.symmetric_mean, self.symmetric_scale = s.mean(0), np.maximum(s.std(0),1e-6)
        d, s = self.transform(d,s)
        theta = np.zeros(d.shape[1]+s.shape[1])
        prior = np.clip(np.mean(y==2),1e-4,1-1e-4)
        theta[-1] = np.log(prior/(1-prior)) + (np.log(2) if self.kind == 'coupled' else 0)
        result = minimize(lambda t:self._objective(t,d,s,y),theta,jac=True,method='L-BFGS-B',
                          options={'maxiter':500,'ftol':1e-12,'gtol':1e-7})
        assert np.isfinite(result.fun) and (result.success or np.linalg.norm(result.jac) < 1e-4), result.message
        self.theta = result.x
        self.diagnostics = {'kind': self.kind, 'regularization': self.regularization,
                            'parameters':len(theta),'iterations':int(result.nit),'objective':float(result.fun),
                            'gradient_norm':float(np.linalg.norm(result.jac))}
        return self

    def transform(self, d, s):
        return (np.asarray(d)/self.direction_scale,
                np.column_stack([(np.asarray(s)-self.symmetric_mean)/self.symmetric_scale,np.ones(len(s))]))

    def predict_proba(self, d, s):
        d, s = self.transform(d,s)
        direction, tie = d@self.theta[:d.shape[1]], s@self.theta[d.shape[1]:]
        if self.kind == 'factorized':
            q, a = expit(tie), expit(direction)
            p = np.column_stack([(1-q)*a,(1-q)*(1-a),q])
        else:
            logits = np.column_stack([direction/2,-direction/2,tie])
            p = np.exp(logits-logsumexp(logits,axis=1,keepdims=True))
        return normalize(p)


class FeatureMap:
    def __init__(self, kind, components=16, seed=20261118):
        self.kind, self.components, self.seed = kind, components, seed

    def fit(self, data, idx):
        if self.kind == 'rich':
            pair = np.vstack([data['answer_a'][idx],data['answer_b'][idx]]).astype(np.float32)
            self.pca = PCA(n_components=min(self.components,*pair.shape),svd_solver='randomized',
                           random_state=self.seed).fit(pair)
        return self

    def transform(self, data, idx):
        if self.kind == 'reward':
            scores = np.asarray(data['rewards'][idx],float)
            d = (scores[:,0]-scores[:,1])[:,None]
            return d, np.column_stack([np.abs(d[:,0]),scores.mean(1)])
        p = normalize(data['probabilities'][idx])
        d = (np.log(p[:,0])-np.log(p[:,1]))[:,None]
        s = np.column_stack([np.log(p[:,2])-np.log(p[:,:2].sum(1)),np.abs(d[:,0])])
        if self.kind == 'probability':
            return d, s
        a, b = data['answer_a'][idx].astype(np.float32),data['answer_b'][idx].astype(np.float32)
        ua, ub = self.pca.transform(a),self.pca.transform(b)
        joint = self.pca.transform(data['joint'][idx].astype(np.float32))
        cosine = (a*b).sum(1)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),1e-6)
        return np.column_stack([d,ua-ub]), np.column_stack([s,np.abs(ua-ub),(ua+ub)/2,joint,cosine])


def fit_with_inner_selection(data, y, groups, fit_idx, eval_idx, feature_kind, head_kind, config):
    """No eval labels are accessed; PCA/scaling are refit inside each training fold."""
    fit_idx = np.asarray(fit_idx)
    splits = StratifiedGroupKFold(config['head_inner_group_folds'],shuffle=True,random_state=config['seed'])
    folds = []
    for train, val in splits.split(fit_idx,y[fit_idx],groups[fit_idx]):
        train, val = fit_idx[train],fit_idx[val]
        assert not set(groups[train]) & set(groups[val])
        transform = FeatureMap(feature_kind,config['head_pca_components'],config['seed']).fit(data,train)
        folds.append((transform.transform(data,train),transform.transform(data,val),train,val))
    scores = []
    for regularization in config['head_regularization_grid']:
        values, counts = [], []
        for train_features,val_features,train,val in folds:
            head = SymmetricHead(head_kind,regularization).fit(*train_features,y[train])
            values.append(loss_rows(y[val],head.predict_proba(*val_features)).sum())
            counts.append(len(val))
        scores.append({'regularization':regularization,'inner_log_loss':float(sum(values)/sum(counts))})
    chosen = min(scores,key=lambda x:x['inner_log_loss'])['regularization']
    transform = FeatureMap(feature_kind,config['head_pca_components'],config['seed']).fit(data,fit_idx)
    head = SymmetricHead(head_kind,chosen).fit(*transform.transform(data,fit_idx),y[fit_idx])
    features = transform.transform(data,eval_idx)
    p = head.predict_proba(*features)
    swapped = head.predict_proba(-features[0],features[1])[:,[1,0,2]]
    assert np.allclose(p,swapped,atol=1e-10)
    summary = {'feature_kind':feature_kind,'head_kind':head_kind,'inner_selection':scores,
               'chosen_regularization':chosen,'fit_rows':len(fit_idx),
               'max_swap_probability_error':float(np.max(np.abs(p-swapped))),**head.diagnostics}
    return p, summary, (transform,head)


def temperature(p, t):
    logits = np.log(normalize(p))/t
    return np.exp(logits-logsumexp(logits,axis=1,keepdims=True))


def crossfit_temperature(y, p, groups, config):
    output = np.full_like(p,np.nan,dtype=float)
    temperatures = []
    cv = StratifiedGroupKFold(config['calibration_group_folds'],shuffle=True,random_state=config['seed'])
    for train,val in cv.split(p,y,groups):
        assert not set(groups[train]) & set(groups[val])
        fit = minimize_scalar(lambda t:loss_rows(y[train],temperature(p[train],t)).mean(),
                              bounds=config.get('temperature_bounds',[.5,3.]),method='bounded')
        output[val] = temperature(p[val],fit.x)
        temperatures.append(float(fit.x))
    assert np.isfinite(output).all()
    return output, temperatures


def paired_group_gain(y, baseline, candidate, groups, config):
    values = loss_rows(y,baseline)-loss_rows(y,candidate)
    return paired_group_difference(values, groups, config)


def paired_group_difference(values, groups, config):
    """Bootstrap paired row differences by prompt group; positive means reduction."""
    values = np.asarray(values, dtype=float)
    assert len(values) == len(groups) and np.isfinite(values).all()
    _, inverse = np.unique(groups,return_inverse=True)
    totals = np.bincount(inverse,weights=values)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(config['seed'])
    samples = rng.integers(0,len(totals),size=(config['bootstrap_group_replicates'],len(totals)))
    draws = totals[samples].sum(1)/counts[samples].sum(1)
    return {'gain':float(values.mean()),'interval_95':np.quantile(draws,[.025,.975]).tolist(),
            'groups':len(totals),'rows':len(values),'replicates':config['bootstrap_group_replicates']}


def order_diagnostics(original, aligned_swap):
    return {'decision_disagreement':float(np.mean(original.argmax(1)!=aligned_swap.argmax(1))),
            'mean_probability_l1':float(np.abs(original-aligned_swap).sum(1).mean()),
            'second_position_advantage':float(.5*((original[:,1]-aligned_swap[:,1])+
                                                   (aligned_swap[:,0]-original[:,0])).mean())}
