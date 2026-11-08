"""Freeze H74's unchanged H60 frames/tokens/checkpoints and strict raw controls."""
from pathlib import Path
import shutil
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,read_json,sha256,utc_now


def main():
    pub=REPO/'results/span_027';root=ROOT/'runs/span_027'
    assert not (pub/'profile_preparation.json').exists(),'Preserve initial profile references'
    proof=read_json(pub/'packing_cpu_audit.json');assert proof['cpu_packing_proof_passed']
    old=ROOT/'runs/span_026';preparation=read_json(REPO/'results/span_026/candidate_preparation.json')
    assert sha256(old/'candidate_spec.json')==preparation['spec_sha256'];base=read_json(old/'candidate_spec.json')
    refs={}
    for role in ['m0','m3','m0_reverse']:
        names=['warmup',*(['control','uniform','long','test'] if role!='m0_reverse' else ['routed_control','routed_uniform','routed_long','routed_test'])]
        refs[role]={n:base['token_references'][role][n] for n in names}
        for n in names:assert proof['actual_native_reference_cohorts'][role][n]['10240']['token_stream_sha256']==refs[role][n]['token_ids_sha256']
    spec=dict(at=utc_now(),uses_targets=False,packing_tokens=10240,conversation_tokens=4096,
        train_csv_sha256=base['train_csv_sha256'],frame_ids=dict(base['frame_ids'],warmup=refs['m0']['warmup']['ids']),
        checkpoint_references=base['checkpoint_references'],token_references=refs,canonical_raw_tolerance=1e-5,
        memory_allocated_fraction_cap=.95,source_proof_sha256=sha256(pub/'packing_cpu_audit.json'))
    root.mkdir(parents=True,exist_ok=True);controls=root/'controls';controls.mkdir(exist_ok=True)
    original=read_json(REPO/'results/span_026/pilot_preparation.json')
    for n,h in original['control_sha256'].items():
        source=old/'controls'/n;assert sha256(source)==h
        target=controls/n;assert not target.exists();shutil.copyfile(source,target)
    target=root/'profile_spec.json';assert not target.exists();atomic_json(target,spec)
    atomic_json(pub/'profile_preparation.json',dict(at=utc_now(),prepared=True,uses_targets=False,
        profile_spec_sha256=sha256(target),canonical_control_sha256=original['control_sha256'],
        cohorts=['control','uniform','long','test'],roles=['m0','m3','m0_reverse'],
        packing_tokens=10240,conversation_tokens=4096,new_gpu_executions=0,gpu_admitted=False))
    print('H74 exact profile references and canonical controls prepared; no GPU admitted.')


if __name__=='__main__':main()
