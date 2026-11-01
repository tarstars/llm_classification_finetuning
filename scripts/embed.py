import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import pandas as pd
from arena.data import ROOT, load_data, json_write
from arena.vectorizers import eliza_embeddings

train,test=load_data()
frame=pd.concat([train,test],ignore_index=True)
embeddings,metadata=eliza_embeddings(frame)
np.savez(ROOT/"embeddings/qwen3_256.npz", **dict(zip(["prompt","response_a","response_b"],embeddings)))
json_write(ROOT/"embeddings/qwen3_256.json",metadata)
print("Saved complete Qwen3 embeddings",flush=True)
