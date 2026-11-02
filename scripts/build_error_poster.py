import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from arena.error_poster import build
p=argparse.ArgumentParser();p.add_argument('--per-category',type=int,default=12)
p.add_argument('--allow-missing-train',action='store_true')
a=p.parse_args();build(a.per_category,a.allow_missing_train)
