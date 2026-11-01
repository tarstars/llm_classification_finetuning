import argparse,json,os,sys
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from arena.data import ROOT
from arena.offline import train_submission

parser=argparse.ArgumentParser()
parser.add_argument('--raw-dir',type=Path,default=ROOT/'raw')
parser.add_argument('--work-dir',type=Path,default=ROOT/'runs/full_refit')
parser.add_argument('--config',type=Path,default=ROOT/'runs/attempt_001/offline_config.json')
args=parser.parse_args()
print(train_submission(args.raw_dir,args.work_dir,json.loads(args.config.read_text())))
