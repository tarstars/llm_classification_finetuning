from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
import requests
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .data import ROOT, SEED, TEXT_COLUMNS, json_write


def tfidf_features(frame, train_idx, out, dimensions=96):
    """Fit vocabulary, IDF and SVD exclusively on the training partition."""
    texts = frame[TEXT_COLUMNS].values.T.ravel().tolist()
    n = len(frame)
    train_texts = frame.iloc[train_idx][TEXT_COLUMNS].values.T.ravel().tolist()
    vectorizer = TfidfVectorizer(max_features=40000, ngram_range=(1, 2), min_df=3,
                                 sublinear_tf=True, strip_accents="unicode", dtype=np.float32)
    print("Fitting TF-IDF on training texts", flush=True)
    fit_matrix = vectorizer.fit_transform(train_texts)
    all_matrix = vectorizer.transform(texts)
    svd = TruncatedSVD(n_components=dimensions, n_iter=4, random_state=SEED)
    print("Fitting training-only SVD", flush=True)
    svd.fit(fit_matrix)
    dense = normalize(svd.transform(all_matrix)).astype(np.float32)
    embeddings = tuple(dense[i*n:(i+1)*n] for i in range(3))
    p, a, b = [all_matrix[i*n:(i+1)*n].tocsr() for i in range(3)]
    difference = a-b
    absolute = difference.copy()
    absolute.data = np.abs(absolute.data)
    pair = sparse.hstack([p, a, b, difference, absolute], format="csr")
    swap = sparse.hstack([p, b, a, -difference, absolute], format="csr")
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump((vectorizer, svd), out / "tfidf_svd.joblib")
    json_write(out / "tfidf_metadata.json", {"vocabulary": len(vectorizer.vocabulary_),
               "dimensions": dimensions, "explained_variance": float(svd.explained_variance_ratio_.sum()),
               "fit_partition": "train", "ngram_range": [1, 2]})
    return embeddings, pair, swap


def clip_text(text, max_chars=6000):
    """Bound API context while retaining both start and end; same rule for every split."""
    text = text or "[EMPTY]"
    if len(text) <= max_chars:
        return text
    return text[:max_chars//2] + "\n[TRUNCATED]\n" + text[-max_chars//2:]


def eliza_embeddings(frame, model="qwen/qwen3-embedding-8b", dimensions=256,
                     batch_size=32, workers=4):
    url = os.environ.get("ELIZA_EMBEDDING_URL", "https://api.eliza.yandex.net/raw/openrouter/v1/embeddings")
    token_path = Path(os.environ.get("ELIZA_TOKEN_PATH", Path.home()/".soy/token"))
    token = os.environ.get("ELIZA_TOKEN") or token_path.read_text().strip()
    all_texts = [clip_text(t) for col in TEXT_COLUMNS for t in frame[col]]
    unique = list(dict.fromkeys(all_texts))
    text_to_idx = {t:i for i,t in enumerate(unique)}
    digest = hashlib.sha256((model + str(dimensions) + json.dumps(unique, ensure_ascii=False)).encode()).hexdigest()
    out = ROOT / "embeddings" / digest[:16]
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"model": model, "dimensions": dimensions, "unique_texts": len(unique),
                "total_texts": len(all_texts), "sha256": digest, "batch_size": batch_size,
                "max_chars": 6000, "truncated_occurrences": int(sum(len(t)>6000 for col in TEXT_COLUMNS for t in frame[col])),
                "endpoint": url, "created_with": "clip-head-tail-v1"}
    if (out/"manifest.json").exists():
        assert json.loads((out/"manifest.json").read_text()) == manifest
    json_write(out/"manifest.json", manifest)
    batches = [(i,unique[i:i+batch_size]) for i in range(0,len(unique),batch_size)]

    def encode(item):
        i, texts = item
        path = out/f"batch_{i:07d}.npy"
        if path.exists():
            return i, "cached"
        for attempt in range(6):
            try:
                response = requests.post(url, json={"model":model,"input":texts,"dimensions":dimensions},
                    headers={"Authorization":"OAuth "+token}, timeout=(15,180))
                if response.status_code in (429,500,502,503,504):
                    time.sleep(min(2**attempt,30))
                    continue
                if not response.ok:
                    raise RuntimeError(f"Eliza HTTP {response.status_code}: {response.text[:300]}")
                result = response.json()
                vectors = np.asarray([d["embedding"] for d in sorted(result["data"],key=lambda x:x["index"])],dtype=np.float32)
                assert vectors.shape[0]==len(texts) and vectors.shape[1]>=dimensions
                vectors=normalize(vectors[:,:dimensions]).astype(np.float32)
                assert np.isfinite(vectors).all()
                with path.with_suffix(".tmp").open("wb") as f: np.save(f,vectors)
                path.with_suffix(".tmp").replace(path)
                json_write(path.with_suffix(".json"), {"usage":result.get("usage"),"model":result.get("model")})
                return i, "ok"
            except (requests.Timeout, requests.ConnectionError):
                time.sleep(min(2**attempt,30))
        raise RuntimeError(f"Embedding batch {i} failed after six attempts")

    done=0
    print(f"Embedding {len(unique)} unique texts in {len(batches)} batches; cache {out}",flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(encode,b) for b in batches]
        for future in as_completed(futures):
            future.result()
            done+=1
            if done%25==0 or done==len(batches):print(f"Embeddings {done}/{len(batches)} batches",flush=True)
    matrix=np.concatenate([np.load(out/f"batch_{i:07d}.npy") for i,_ in batches])
    arrays=[matrix[[text_to_idx[clip_text(t)] for t in frame[col]]] for col in TEXT_COLUMNS]
    return tuple(arrays), manifest
