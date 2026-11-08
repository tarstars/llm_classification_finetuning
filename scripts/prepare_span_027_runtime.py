"""Freeze retained startup overhead separately from the throughput being changed."""
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,read_json,sha256,utc_now


def main():
    started=time.monotonic();pub=REPO/'results/span_027';assert not (pub/'runtime_reference.json').exists()
    roles=['m0','m3','m0_reverse'];folders=[ROOT/'runs/span_013/jobs/gemma_order/output/order_completion',
        ROOT/'runs/span_026/jobs/context_pilot/output/context_audit',ROOT/'runs/span_026/jobs/context_qualification/output/context_candidate']
    runs=[{r:read_json(p/(r+'_summary.json')) for r in roles} for p in folders]
    reference_paths=[p/(r+'_summary.json') for p in folders for r in roles]
    parent_path=REPO/'results/span_013/gemma_order_qualification.json';parent=read_json(parent_path)['runtime']
    other=[parent['other_seconds']]
    for folder,run,selector_file,key in [(folders[1],runs[1],'runtime_spec.json','runtime_selector_seconds'),
        (folders[2],runs[2],'qualification.json','selector_seconds')]:
        measured=sum(s['cold_seconds']+sum(c['seconds'] for c in s['cohorts'].values()) for s in run.values())
        session_path=folder.parent/'session_summary.json';selector_path=folder/selector_file
        other.append(max(0,read_json(session_path)['seconds']-measured-read_json(selector_path)[key]))
        reference_paths.extend([session_path,selector_path])
    floor=dict(cold_by_role={r:max(run[r]['cold_seconds'] for run in runs) for r in roles},
        warm_by_role={r:max(run[r]['cohorts']['warmup']['seconds'] for run in runs) for r in roles},other_seconds=max(other))
    result=dict(at=utc_now(),uses_targets=False,overhead_floor=floor,
        retained8192_uniform_seconds=[{r:run[r]['cohorts']['routed_uniform' if r=='m0_reverse' else 'uniform']['seconds'] for r in roles}
            for run in [runs[0],runs[2]]],
        old8192throughput_is_comparator_not_new_mode_floor=True,
        database_reference_sha256={str(p.relative_to(ROOT)):sha256(p) for p in reference_paths},
        project_reference_sha256={str(parent_path.relative_to(REPO)):sha256(parent_path)},
        elapsed_cpu_seconds=time.monotonic()-started)
    atomic_json(pub/'runtime_reference.json',result);print(result['overhead_floor'])


if __name__=='__main__':main()
