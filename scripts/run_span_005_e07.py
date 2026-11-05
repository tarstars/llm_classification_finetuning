"""Fixed cheap source-attribution heads; selection uses training source labels only."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,resource,signal,sys,time,warnings
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss,recall_score

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,json_write,prompt_groups
from arena.source_models import answer_hash,purge_answers
from arena.span_heads import paired_group_difference


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def packed(t,cap):
    return t if len(t)<=cap else t[:cap//2]+'\n[TRUNCATED]\n'+t[-cap//2:]


def source_metrics(y,p):
    assert np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1,atol=1e-6)
    pred=p.argmax(1);confidence=p.max(1);order=np.argsort(p,axis=1)
    ece=0
    bins=np.minimum((confidence*10).astype(int),9)
    for index in range(10):
        ix=bins==index
        if ix.any():ece+=ix.mean()*abs((pred[ix]==y[ix]).mean()-confidence[ix].mean())
    return {'rows':len(y),'log_loss':float(-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1)).mean()),
            'accuracy':float((pred==y).mean()),'top5_accuracy':float(np.mean(np.any(order[:,-5:]==y[:,None],axis=1))),
            'macro_recall':float(recall_score(y,pred,labels=np.arange(p.shape[1]),average='macro',zero_division=0)),
            'ece_10_bins':float(ece),'mean_entropy':float(-(p*np.log(np.maximum(p,1e-12))).sum(1).mean()),
            'class_counts':np.bincount(y,minlength=p.shape[1]).tolist(),
            'per_class_recall':recall_score(y,pred,labels=np.arange(p.shape[1]),average=None,zero_division=0).tolist()}


def aligned(model,x,size,prior):
    p=np.full((x.shape[0],size),1e-12);p[:,model.classes_]=model.predict_proba(x)
    if hasattr(x,'getnnz'):
        empty=x.getnnz(axis=1)==0;p[empty]=prior
    p/=p.sum(1,keepdims=True)
    return p.astype(np.float32)


def fit_head(family,fit,val,text,embedding,y,classes,cfg):
    start=time.monotonic();parts=[]
    if family=='tfidf':
        common={'min_df':2,'sublinear_tf':True,'dtype':np.float32}
        parts=[TfidfVectorizer(max_features=cfg['source_word_features'],ngram_range=tuple(cfg['source_word_ngram_range']),**common),
               TfidfVectorizer(analyzer='char_wb',max_features=cfg['source_char_features'],ngram_range=tuple(cfg['source_char_ngram_range']),**common)]
        x=hstack([v.fit_transform(text[fit].tolist()) for v in parts],format='csr',dtype=np.float32)
        transform=lambda idx:hstack([v.transform(text[idx].tolist()) for v in parts],format='csr',dtype=np.float32)
        model=LogisticRegression(C=cfg['source_logistic_C'],solver='saga',max_iter=cfg['source_max_iterations'],tol=cfg['source_tolerance'],random_state=cfg['seed'])
    else:
        scaler=StandardScaler().fit(embedding[fit]);parts=[scaler];x=scaler.transform(embedding[fit]);transform=lambda idx:scaler.transform(embedding[idx])
        model=LogisticRegression(C=cfg['source_embedding_C'],solver='lbfgs',max_iter=cfg['source_max_iterations'],tol=cfg['source_tolerance'])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');model.fit(x,y[fit])
    fit_seconds=time.monotonic()-start;counts=np.bincount(y[fit],minlength=classes);prior=(counts+1)/(len(fit)+classes)
    before=time.monotonic();p=aligned(model,transform(val),classes,prior);score_seconds=time.monotonic()-before
    bundle={'family':family,'parts':parts,'model':model,'prior':prior,'classes':classes,'max_chars':cfg['source_max_chars']}
    return p,bundle,{'fit_seconds':fit_seconds,'score_seconds':score_seconds,'fit_answers':len(fit),'evaluation_answers':len(val),
                     'iterations':model.n_iter_.tolist(),'warnings':[str(w.message) for w in caught],
                     'features':int(x.shape[1]),'rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2}


def main():
    started=time.monotonic();out=ROOT/'runs/span_005';pub=REPO/'results/span_005';directory=out/'E07';directory.mkdir(exist_ok=True)
    assert not (directory/'started.json').exists(),'E07 already started; inspect its actual outputs'
    cfg=json.loads((REPO/'configs/span_005.json').read_text());contract=json.loads((pub/'E07_contract.json').read_text())
    assert contract['runner_sha256']==digest(Path(__file__)) and contract['config_sha256']==digest(REPO/'configs/span_005.json')
    assert contract['E06_result_sha256']==digest(pub/'E06_results.json')
    assert json.loads((pub/'E06_results.json').read_text())['gate']['passed']
    inventory=json.loads((pub/'artifact_inventory.json').read_text())
    assert inventory['source_sha256']['src/arena/source_models.py']==digest(REPO/'src/arena/source_models.py')
    for name in ['train.parquet','dev.parquet','inputs.npz','source_vocabulary.json']:
        entry=inventory['input_artifacts'][name];assert digest(entry['path'])==entry['sha256'],name
    json_write(directory/'started.json',{'at':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'source_sha256':digest(Path(__file__))})
    def timeout(signum,frame):raise TimeoutError('Original E07 CPU elapsed cap reached')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(cfg['cpu_stage_seconds_caps']['E07'])
    train=pd.read_parquet(out/'train.parquet');dev=pd.read_parquet(out/'dev.parquet');arrays=np.load(out/'inputs.npz')
    vocabulary=json.loads((out/'source_vocabulary.json').read_text());classes=vocabulary['source_classes'];lookup={x:i for i,x in enumerate(classes)};other=lookup['__OTHER__'];m=len(classes)
    texts=np.r_[train.response_a.to_numpy(),train.response_b.to_numpy(),dev.response_a.to_numpy(),dev.response_b.to_numpy()]
    names=np.r_[train.model_a.to_numpy(),train.model_b.to_numpy(),dev.model_a.to_numpy(),dev.model_b.to_numpy()]
    source_y=np.array([lookup.get(x,other) for x in names]);hashes=np.array([answer_hash(t) for t in texts])
    text=np.array([packed(t,cfg['source_max_chars']) for t in texts],dtype=object)
    ntrain=2*len(train);ndev=2*len(dev);trange=np.arange(ntrain);drange=np.arange(ntrain,ntrain+ndev)
    folds=np.tile(arrays['source_fold'],2);groups=np.tile(prompt_groups(train).to_numpy(dtype=str),2)
    embedding=np.vstack([arrays['train_a'],arrays['train_b'],arrays['dev_a'],arrays['dev_b']])
    outputs={};all_details={};dev_predictions={};bundles={}
    for family in cfg['source_families']:
        probabilities=np.full((ntrain,m),np.nan,dtype=np.float32);frequency=np.full_like(probabilities,np.nan);records=[]
        for fold in range(cfg['source_group_folds']):
            fit=trange[folds!=fold];val=trange[folds==fold];keep=purge_answers(hashes[fit],hashes[val]);purged=int((~keep).sum());fit=fit[keep]
            assert not set(groups[fit])&set(groups[val]);assert not set(hashes[fit])&set(hashes[val])
            p,bundle,record=fit_head(family,fit,val,text,embedding,source_y,m,cfg)
            probabilities[val]=p;frequency[val]=bundle['prior'];record.update(fold=fold,purged_fit_answers=purged)
            records.append(record)
            np.savez(directory/(family+f'_fold{fold}.npz'),fit_indices=fit,evaluation_indices=val,probabilities=p)
            assert record['rss_gib']<cfg['rss_guard_gib'],'Resident memory guard'
            print(json.dumps({'family':family,**record}),flush=True)
        assert np.isfinite(probabilities).all()
        loss=-np.log(np.maximum(probabilities[np.arange(ntrain),source_y[:ntrain]],1e-12));base=-np.log(frequency[np.arange(ntrain),source_y[:ntrain]])
        stats={'oof':source_metrics(source_y[:ntrain],probabilities),'oof_prior':source_metrics(source_y[:ntrain],frequency),
               'gain_vs_prior':paired_group_difference(base-loss,groups,cfg)}
        fit=trange[purge_answers(hashes[trange],hashes[drange])]
        assert not set(hashes[fit])&set(hashes[drange])
        p,bundle,final=fit_head(family,fit,drange,text,embedding,source_y,m,cfg)
        final['purged_fit_answers']=len(trange)-len(fit);dev_predictions[family]=p;bundles[family]=bundle
        stats['final_fit']=final;stats['development_source_diagnostic']=source_metrics(source_y[drange],p)
        stats['development_labels_used_to_select_or_fit']=False
        path=directory/(family+'_bundle.joblib');joblib.dump(bundle,path,compress=3)
        before=time.monotonic();loaded=joblib.load(path);stats['reload_seconds']=time.monotonic()-before
        stats['bundle_bytes']=path.stat().st_size;stats['bundle_sha256']=digest(path)
        np.savez(directory/(family+'_predictions.npz'),oof=probabilities,oof_prior=frequency,dev_a=p[:len(dev)],dev_b=p[len(dev):],source_labels=source_y[:ntrain])
        np.savez(directory/(family+'_final_fit_indices.npz'),fit_indices=fit,evaluation_indices=drange)
        stats['prediction_sha256']=digest(directory/(family+'_predictions.npz'));outputs[family]=stats;all_details[family]=records
        print(json.dumps({'family':family,'oof_loss':stats['oof']['log_loss'],'prior_loss':stats['oof_prior']['log_loss'],'final_fit':final}),flush=True)
    # Portable family is fixed by the zero-GPU/offline contract, not dev votes.
    chosen=outputs['tfidf'];seconds_per_answer=chosen['final_fit']['score_seconds']/ndev
    classifier_rate=contract['frozen_classifier_seconds_per_row'];projection=25000*(classifier_rate+2*seconds_per_answer)*1.2+600+chosen['reload_seconds']
    gate=bool(chosen['gain_vs_prior']['gain']>0 and projection<=25200 and chosen['final_fit']['rss_gib']<cfg['rss_guard_gib'])
    result={'task':'E07','rows':len(train),'source_answers':ntrain,'source_classes':m,'source_vocabulary':classes,'families':outputs,
            'portable_family':'tfidf','embedding_head_research_only':True,'selection_uses_development_preference_labels':False,
            'exact_normalized_answer_hashes_purged':True,'raw_calibration_diagnostics_only':True,
            'source_schema':'Fixed original-training vocabulary; rare-count mapping defines the label schema, while learned parameters/IDF/scaling/prior use each fitting fold only.',
            'cpu_elapsed_seconds':time.monotonic()-started,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
            'source_seconds_per_answer':seconds_per_answer,'projected_classifier_plus_source_seconds':projection,
            'projection_note':'Conditional25000hidden rows, sequential classifier+CPU source stage,20%margin,600sreserve; classifier rate from frozen prior profile. Not a new end-to-end Kaggle run.',
            'gate':{'passed':gate,'rule':'Portable TF-IDF OOF gain over prior >0; finite full coverage; measured conditional path<=25200s and RSS<6GiB.'},
            'confirmation_used':False,'gpu_seconds':0,'source_sha256':digest(Path(__file__)),'config_sha256':digest(REPO/'configs/span_005.json')}
    json_write(directory/'folds.json',all_details);result['folds_sha256']=digest(directory/'folds.json')
    for folder in [directory,pub]:json_write(folder/'E07_results.json',result)
    json_write(directory/'summary.json',{'complete':True,'gate_passed':gate,'cpu_elapsed_seconds':result['cpu_elapsed_seconds']});signal.alarm(0)
    print(json.dumps({'task':'E07','gate':result['gate'],'cpu_seconds':result['cpu_elapsed_seconds'],'projection_seconds':projection},indent=2),flush=True)


if __name__=='__main__':main()
