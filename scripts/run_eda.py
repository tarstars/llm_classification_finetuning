"""Run local dataset/model EDA without modifying the frozen models."""
import argparse,os,sys
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('RAYON_NUM_THREADS','4')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','analyze']);p.add_argument('--force',action='store_true')
a=p.parse_args()
if a.stage=='prepare':
    from arena.eda_features import prepare
    prepare(a.force)
else:
    from arena.eda_analysis import analyze
    analyze()
