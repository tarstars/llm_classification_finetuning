"""Run independent stages; final comparison reads holdout only after model selection."""
import argparse
import hashlib
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS","4")
os.environ.setdefault("OPENBLAS_NUM_THREADS","4")
os.environ.setdefault("MKL_NUM_THREADS","4")
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import pandas as pd
from scipy import sparse
from arena.data import ROOT,CLASSES,TEXT_COLUMNS,load_data,split_data,prompt_groups,structural_features,swapped,dense_pair_features,swapped_pair_features,json_write
from arena.models import fit_classifiers,scores

parser=argparse.ArgumentParser()
parser.add_argument("stage",choices=["prepare","tfidf","rnn","rnn_long","qwen","structural","hybrid"])
args=parser.parse_args()
out=ROOT/"runs/attempt_001"
out.mkdir(parents=True,exist_ok=True)
if (out/'final_evaluation.json').exists():
    raise SystemExit('Attempt 001 is frozen. Use a fresh ARENA_DATA_DIR to reproduce or start another experiment.')
train,test=load_data()
frame=pd.concat([train,test],ignore_index=True)
if args.stage!='prepare' and (out/'split_manifest.parquet').exists():
    manifest=pd.read_parquet(out/'split_manifest.parquet')
    assert np.array_equal(manifest.id.to_numpy(),train.id.to_numpy())
    assert np.array_equal(manifest.prompt_group.to_numpy(),prompt_groups(train).to_numpy())
    y=np.load(out/'labels.npy');fold=np.load(out/'fold.npy')
    assert np.array_equal(y,train[CLASSES].values.argmax(1))
    parts={k:np.load(out/f'{k}_idx.npy') for k in ['train','dev','holdout']}
else:
    y,parts,fold=split_data(train)
    np.save(out/"labels.npy",y)
    np.save(out/"fold.npy",fold)
    for name,idx in parts.items():np.save(out/f"{name}_idx.npy",idx)
stats=structural_features(frame)

if args.stage=="prepare":
    from arena.data import prompt_groups
    group=prompt_groups(train)
    summary={"train_rows":len(train),"visible_test_rows":len(test),"classes":CLASSES,
        "counts":train[CLASSES].sum().to_dict(),"unique_prompt_groups":group.nunique(),
        "split_counts":{k:len(v) for k,v in parts.items()},"seed":20261101,
        "split":"StratifiedGroupKFold(10): folds 2..9 train; 1 dev; 0 holdout",
        "class_proportions_by_split":{k:np.bincount(y[v],minlength=3).tolist() for k,v in parts.items()},
        "text_lengths":{c:train[c].str.len().describe(percentiles=[.5,.9,.95,.99]).to_dict() for c in TEXT_COLUMNS},
        "raw_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/"raw").glob("*.csv")},
        "excluded_features":["id","model_a","model_b"],"prompt_overlap_across_splits":0}
    json_write(out/"data_summary.json",summary)
    train[["id"]].assign(fold=fold,prompt_group=group).to_parquet(out/"split_manifest.parquet",index=False)
    print(summary,flush=True)
elif args.stage=="structural":
    fit_classifiers("structural",stats,structural_features(swapped(frame)),y,parts,out)
elif args.stage=="tfidf":
    from arena.vectorizers import tfidf_features
    embeddings,pair,pairswap=tfidf_features(frame,parts["train"],out)
    # Append unscaled structural features; sparse classifier scales on training rows.
    pair=sparse.hstack([pair,sparse.csr_matrix(stats)],format="csr")
    pairswap=sparse.hstack([pairswap,sparse.csr_matrix(structural_features(swapped(frame)))],format="csr")
    fit_classifiers("tfidf_sparse",pair,pairswap,y,parts,out,kinds=("logistic",))
    del pair,pairswap
    np.savez(out/"tfidf_embeddings.npz",**dict(zip(TEXT_COLUMNS,embeddings)))
    fit_classifiers("tfidf_svd",dense_pair_features(embeddings,stats),swapped_pair_features(embeddings,stats),y,parts,out)
elif args.stage in ("rnn","rnn_long"):
    from arena.rnn import train_rnn
    representation=args.stage
    rnn_out=out if representation=='rnn' else out/'rnn_long'
    embeddings,p=train_rnn(frame,y,parts,stats,rnn_out,max_tokens=128 if representation=='rnn' else 256)
    np.save(out/f"{representation}__end_to_end_predictions.npy",p)
    json_write(out/f"{representation}__end_to_end.json",{"model":f"{representation}__end_to_end","representation":representation,
        "classifier":"end_to_end","dev":scores(y[parts["dev"]],p[parts["dev"]])})
    np.savez(out/f"{representation}_embeddings.npz",**dict(zip(TEXT_COLUMNS,embeddings)))
    fit_classifiers(representation,dense_pair_features(embeddings,stats),swapped_pair_features(embeddings,stats),y,parts,out)
elif args.stage=="qwen":
    with np.load(ROOT/"embeddings/qwen3_256.npz") as f:embeddings=tuple(f[c] for c in TEXT_COLUMNS)
    fit_classifiers("qwen3_256",dense_pair_features(embeddings,stats),swapped_pair_features(embeddings,stats),y,parts,out)
elif args.stage=="hybrid":
    features=[];swaps=[]
    for path in [out/"tfidf_embeddings.npz",out/"rnn_embeddings.npz",ROOT/"embeddings/qwen3_256.npz"]:
        with np.load(path) as f:embeddings=tuple(f[c] for c in TEXT_COLUMNS)
        features.append(dense_pair_features(embeddings,stats)[:,:-stats.shape[1]])
        swaps.append(swapped_pair_features(embeddings,stats)[:,:-stats.shape[1]])
    features.append(stats);swaps.append(structural_features(swapped(frame)))
    fit_classifiers("hybrid",np.column_stack(features),np.column_stack(swaps),y,parts,out,kinds=("gbdt",))
