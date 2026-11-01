"""Create new, explicit prompt-grouped split IDs without touching frozen attempt 001."""
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from arena.data import ROOT,CLASSES,json_write
from arena.joint_data import JOINT_SEED,read_competition,phase_two_split

out=ROOT/'runs/attempt_002'
out.mkdir(parents=True,exist_ok=True)
if (out/'split_ids.json').exists():
    raise SystemExit('Attempt-002 split already exists; preserve it.')
train,_=read_competition(ROOT/'raw')
parts,folds=phase_two_split(train)
manifest={k:train.id.iloc[parts[k]].astype(int).tolist() for k in ['dev','confirmation']}
json_write(out/'split_ids.json',manifest)
summary={'seed':JOINT_SEED,'rows':len(train),'split_counts':{k:len(v) for k,v in parts.items()},
    'class_counts':{k:train.iloc[v][CLASSES].sum().astype(int).to_dict() for k,v in parts.items()},
    'split_ids_sha256':hashlib.sha256((out/'split_ids.json').read_bytes()).hexdigest(),
    'note':'New split for attempt 002; previous project results have been examined, so this is not an entirely unseen project-wide test set.'}
json_write(out/'split_summary.json',summary)
repo=Path(__file__).resolve().parents[1]
json_write(repo/'results/attempt_002/split_summary.json',summary)
train[['id']].assign(fold=folds).to_parquet(out/'split_manifest.parquet',index=False)
print(json.dumps(summary,indent=2))
