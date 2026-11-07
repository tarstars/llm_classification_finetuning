"""Audit exact E16source, matched raw controls, packing eligibility and base runtime."""
from pathlib import Path
import argparse
import json
import sys

import nbformat
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,sha256,verify_sources,utc_now
from arena.span004_support import api_client,read_call
from arena.span012_profile import decide_packing,validate_probabilities,PROFILE_NAMES


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-only',action='store_true');args=parser.parse_args()
    root=ROOT/'runs/span_012';pub=REPO/'results/span_012';role='packing_profile';job=root/'jobs'/role
    bundle=read_json(pub/(role+'_bundle.json'));verify_sources(REPO,bundle['source_sha256'])
    push=read_json(job/'push.json');exact=push['ref'].removeprefix('/code/')+'/'+str(push['version_number'])
    original=root/'notebooks'/role/'job.ipynb';assert sha256(original)==bundle['notebook_sha256']
    api=api_client();folder=root/'audited_sources'/role;folder.mkdir(parents=True,exist_ok=True)
    read_call(lambda:api.kernels_pull(exact,path=str(folder),metadata=True,quiet=True))
    copies=list(folder.glob('*.ipynb'));assert len(copies)==1
    def cells(path):return [(c.cell_type,c.source) for c in nbformat.read(path,4).cells]
    assert cells(original)==cells(copies[0]),'Accepted source differs from frozen engineering notebook'
    metadata=read_json(folder/'kernel-metadata.json');local=read_json(root/'notebooks'/role/'kernel-metadata.json')
    assert metadata['enable_internet'] is False and metadata['docker_image']==local['docker_image']
    assert sorted(x.removesuffix('/1') for x in local['dataset_sources'])==sorted(metadata['dataset_sources'])
    source=dict(at=utc_now(),passed=True,kernel_version=exact,notebook_sha256=bundle['notebook_sha256'],
                cell_sources_identical=True,internet_enabled=False,engineering_only=True)
    atomic_json(pub/'packing_profile_source_verification.json',source)
    if args.source_only:print(json.dumps(source));return
    assert str(read_call(lambda:api.kernels_status(exact)).status).split('.')[-1]=='COMPLETE'
    receipt=read_json(job/'retrieval.json')
    for name,digest in receipt['artifact_sha256'].items():assert sha256(job/'output'/name)==digest
    out=job/'output/packing_profile';done=read_json(out/'profile_complete.json')
    assert done['complete'] and not done['uses_targets'] and done['pack_sequence']==[8192,6144,8192] and done['warmup_rows']==80
    previous=ROOT/'runs/span_009/jobs/public_ensemble_resume/output/public_ensemble'
    expected=read_json(REPO/'results/span_009/profile_selection.json')['uniform_ids']
    selected=read_json(out/'profile_selection.json')
    assert selected['control_ids']==expected and not selected['uses_targets']
    assert selected['train_csv_sha256']==read_json(REPO/'results/span_009/profile_selection.json')['train_csv_sha256']
    summaries={};differences={};telemetry=True
    for model in ['m0','m3']:
        assert read_json(out/(model+'_checkpoint.json'))==read_json(previous/(model+'_checkpoint.json'))
        reference=np.load(previous/(model+'_uniform.npz'));old=read_json(previous/(model+'_summary.json'))
        summary=read_json(out/(model+'_summary.json'));summaries[model]=summary;assert summary['loading_exact']
        assert list(summary['cohorts'])==PROFILE_NAMES
        differences[model]={}
        for name in PROFILE_NAMES:
            value=summary['cohorts'][name];actual=np.load(out/(model+'_'+name+'.npz'))
            count=80 if name.startswith('warmup_') else 256
            assert value['rows']==count and np.array_equal(actual['ids'],reference['ids'][:count])
            assert np.array_equal(actual['lengths'],reference['lengths'][:count])
            validate_probabilities(actual['probabilities'],count)
            assert value['packing_tokens']==int(name.split('_')[1])
            assert all(x>0 for x in value['peak_allocated_bytes']) and len(value['peak_allocated_bytes'])==2
            if count==256:
                assert value['token_ids_sha256']==old['cohorts']['uniform']['token_ids_sha256']
                differences[model][name]=float(np.max(np.abs(actual['probabilities']-reference['probabilities'])))
            for when in ['hardware_before','hardware_after']:
                view=value[when];telemetry&=view['available']
                if view['available']:
                    telemetry&=len(view['devices'])==2 and all('T4' in str(g['name']) and
                        all(g[k] is not None for k in ['sm_clock','memory_clock','utilization']) for g in view['devices'])
    try:selection=decide_packing(summaries,differences)
    except ValueError:
        selection=dict(selected_packing=8192,base_controls_passed=False,selection_uses_labels=False,
                       failure='Canonical raw control or measured duration invalid; no production admission')
    # The frozen selector compares a NumPy mean, producing np.bool_ for its
    # speed decision. Convert output scalars without changing any gate or input.
    selection={key:value.item() if isinstance(value,np.generic) else value for key,value in selection.items()}
    packing=selection['selected_packing']
    times=[]
    for summary in summaries.values():
        names=['profile_6144'] if packing==6144 else ['profile_8192_first','profile_8192_last']
        times.append(max(summary['cohorts'][n]['seconds'] for n in names))
    session=read_json(job/'output/session_summary.json');assert session['exit_code']==0 and session['seconds']<=1800
    cold=sum(s['cold_seconds'] for s in summaries.values())
    measured=cold+sum(v['seconds'] for s in summaries.values() for v in s['cohorts'].values())
    assert session['seconds']>=measured-1
    other=max(0.,session['seconds']-measured)
    warm=sum(max(v['seconds'] for n,v in s['cohorts'].items() if n.startswith('warmup_')) for s in summaries.values())
    projection=25000*sum(times)/256*1.2+cold+warm+other+600
    hardware=read_json(job/'output/hardware.json')
    assert len(hardware['gpus'])==2 and all('T4' in name for name in hardware['gpus']) and hardware['python'].startswith('3.10.')
    gate=bool(selection['base_controls_passed'] and telemetry and projection<32400)
    record=dict(at=utc_now(),complete=True,source=source,selection=selection,
        engineering_gate_passed=gate,selected_packing=packing,raw_control_differences=differences,
        hardware_telemetry_passed=bool(telemetry),both_t4_verified=True,input_token_identity=True,
        base_only_projection_seconds=projection,startup_components=dict(cold=cold,one_max_full_warmup_per_model=warm,other=other),
        artifact_files_verified=len(receipt['artifact_sha256']),targets_used=False,quality_evaluated=False,
        measured_roles=summaries,session_seconds=session['seconds'],
        production_still_requires_actual_three_pass_qualification=True)
    atomic_json(pub/'E16_qualification.json',record)
    print(json.dumps({k:record[k] for k in ['complete','engineering_gate_passed','selected_packing','raw_control_differences','base_only_projection_seconds']}))


if __name__=='__main__':main()
