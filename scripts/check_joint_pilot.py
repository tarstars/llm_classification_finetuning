"""Require real GPU loss/memory checks before a longer training execution."""
import argparse,json,math,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from arena.data import ROOT,json_write

parser=argparse.ArgumentParser()
parser.add_argument('--job',required=True)
args=parser.parse_args()
job=ROOT/'runs/attempt_002/jobs'/args.job
summary=json.loads((job/'output/joint_run/summary.json').read_text())
progress=[json.loads(line) for line in (job/'output/joint_run/training_progress.jsonl').read_text().splitlines()]
assert summary['pilot'] and summary['world_size']==2
assert summary['global_steps']>=10
assert math.isfinite(summary['train_result']['train_loss'])
checks=[p for p in progress if 'eval_log_loss' in p]
assert checks,'No independent development metrics were recorded'
ratios=[p['eval_loss']/p['eval_log_loss'] for p in checks]
assert np.allclose(ratios,1,rtol=1e-4,atol=1e-4),f'DDP loss normalization is still wrong: {ratios}'
late_gradients=[p['grad_norm'] for p in progress if p['step']>=5 and 'grad_norm' in p]
assert late_gradients and np.isfinite(late_gradients[-1]),f'GradScaler has not stabilized: {late_gradients}'
assert summary['peak_gpu_bytes']<15_000_000_000,'Leave memory headroom for real batches'
eval_seconds=sum(p['eval_runtime'] for p in checks)
step_seconds=(summary['training_seconds']-eval_seconds)/summary['global_steps']
speed=float(np.mean([p['eval_samples_per_second'] for p in checks]))
result={'passed':True,'pilot_job':args.job,'global_steps':summary['global_steps'],
    'loss_to_independent_log_loss_ratios':ratios,'late_gradient_norms':late_gradients,
    'peak_gpu_bytes':summary['peak_gpu_bytes'],'train_seconds_per_step_excluding_evaluation':step_seconds,
    'evaluation_rows_per_second_both_GPUs':speed,
    'estimated_hidden_25k_swap_inference_seconds':2*25000/speed,
    'note':'Speed estimates use a small pilot; long-run timing and quality must be measured.'}
json_write(job/'pilot_checks.json',result)
print(json.dumps(result,indent=2))
