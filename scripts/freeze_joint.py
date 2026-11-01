"""Freeze a non-pilot training result before confirmation and hidden-test inference."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from arena.data import ROOT,json_write

parser=argparse.ArgumentParser()
parser.add_argument('--job',required=True)
parser.add_argument('--folder',type=Path,required=True)
args=parser.parse_args()
out=ROOT/'runs/attempt_002'
job=out/'jobs'/args.job
summary=json.loads((job/'output/joint_run/summary.json').read_text())
assert not summary['pilot'],'A hardware pilot is not a final model'
assert not summary['confirmation_evaluated']
selection={'job':args.job,'kernel':json.loads((args.folder/'kernel-metadata.json').read_text())['id'],
    'version_number':json.loads((job/'push.json').read_text())['version_number'],
    'adapter_sha256':summary['adapter_sha256'],'temperature':summary['temperature'],
    'config':summary['config'],'development':summary['development'],
    'calibrated_development':summary['calibrated_development'],
    'epochs_completed':summary['epochs_completed'],'stopped_for_budget':summary['stopped_for_budget'],
    'confirmation_used_for_selection':False}
path=out/'model_selection.json'
if path.exists():
    assert json.loads(path.read_text())==selection,'The model choice is already frozen'
else:
    json_write(path,selection)
json_write(Path(__file__).resolve().parents[1]/'results/attempt_002/model_selection.json',selection)
print(json.dumps(selection,indent=2))
