"""Freeze two-role qualification references and a hidden spec containing no controls."""
from pathlib import Path
import shutil
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,read_json,sha256,utc_now,verify_sources


def main():
    started=time.monotonic();pub=REPO/'results/span_030';root=ROOT/'runs/span_030'
    assert not (pub/'candidate_preparation.json').exists()
    quality=read_json(pub/'quality_replay.json')
    assert quality['gross_regression_guard_passed'] and quality['expert_weights']==[.4,.6]
    assert quality['quality_variants_tested']==1 and quality['cumulative_twoexpert_priors_tested']==[[.6,.4],[.4,.6]]
    verify_sources(REPO,quality['source_sha256'])
    old=ROOT/'runs/span_028';previous=read_json(REPO/'results/span_028/profile_preparation.json')
    assert sha256(old/'profile_spec.json')==previous['profile_spec_sha256']
    base=read_json(old/'profile_spec.json');roles=['m0','m3']
    refs={r:base['token_references'][r] for r in roles};batches={r:base['batch_references'][r] for r in roles}
    ids={n:base['frame_ids'][n] for n in ['warmup','control','uniform','long']}
    ids['test']=refs['m0']['test']['ids']
    spec=dict(mode='qualification',uses_targets=False,packing_tokens=8192,conversation_tokens=4096,
        execution_mode='cpu_prefetch',queue_depth=2,producer_threads=1,frame_ids=ids,
        train_csv_sha256=base['train_csv_sha256'],checkpoint_references={r:base['checkpoint_references'][r] for r in roles},
        token_references=refs,batch_references=batches,quality_replay_sha256=sha256(pub/'quality_replay.json'))
    hidden={k:spec[k] for k in ['uses_targets','packing_tokens','checkpoint_references']};hidden['mode']='hidden'
    controls=root/'controls';controls.mkdir(exist_ok=True);control_hashes={}
    for r in roles:
        name=r+'.npz';source=old/'controls'/name;assert sha256(source)==previous['canonical_control_sha256'][name]
        assert not (controls/name).exists();shutil.copyfile(source,controls/name);control_hashes[name]=sha256(source)
    for name,value in [('qualification_spec.json',spec),('hidden_spec.json',hidden)]:
        assert not (root/name).exists();atomic_json(root/name,value)
    timing=read_json(REPO/'results/span_028/runtime_reference.json')
    parent_paths={r:old/'jobs/prefetch_profile/output/prefetch_profile'/(r+'_summary.json') for r in roles}
    timing.update(at=utc_now(),parent_summary_sha256={str(p.relative_to(ROOT)):sha256(p) for p in parent_paths.values()},
        parent_summary_paths={r:str(p.relative_to(ROOT)) for r,p in parent_paths.items()},
        quality_recipe='normalized(.4 Gemma original + .6 aligned Llama swapped)',roles=roles)
    atomic_json(pub/'runtime_reference.json',timing)
    result=dict(at=utc_now(),prepared=True,qualification_spec_sha256=sha256(root/'qualification_spec.json'),
        hidden_spec_sha256=sha256(root/'hidden_spec.json'),canonical_control_sha256=control_hashes,
        quality_replay_sha256=sha256(pub/'quality_replay.json'),mixture_sha256=sha256(REPO/'src/arena/span030_mixture.py'),
        elapsed_cpu_seconds=time.monotonic()-started,new_gpu_executions=0)
    atomic_json(pub/'candidate_preparation.json',result);print(result)


if __name__=='__main__':main()
