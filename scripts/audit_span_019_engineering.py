"""Audit the exact public engineering artifacts without reading human votes."""
from pathlib import Path
import json
import sys

import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,read_json,sha256,utc_now,verify_sources
from arena.span019_runtime import projection


def main():
    pub,root=REPO/'results/span_019',ROOT/'runs/span_019'
    job=root/'jobs/classifier_pilot';out=job/'output/mirror_experiment'
    assert not (pub/'engineering_qualification.json').exists(),'Preserve first engineering decision'
    bundle=read_json(pub/'classifier_pilot_bundle.json');verify_sources(REPO,bundle['source_sha256'])
    assert read_json(job/'status.json')['status']=='COMPLETE'
    retrieval=read_json(job/'retrieval.json')
    for name,digest in retrieval['artifact_sha256'].items():assert sha256(job/'output'/name)==digest
    assert read_json(pub/'jobs/classifier_pilot/source_verification.json')['passed']
    assets=read_json(job/'output/asset_verification.json')
    manifest=read_json(pub/'asset_manifest.json')
    assert assets['passed'] and assets['files']=={r['filename']:r['sha256'] for r in manifest['entries']}
    hardware=read_json(job/'output/hardware.json');assert len(hardware['gpus'])==2 and all('T4' in g for g in hardware['gpus'])
    selection=read_json(pub/'evaluation_selection.json');tokens=read_json(pub/'token_reference.json')
    g,m,n=[read_json(out/name) for name in ['m0_summary.json','mirror_summary.json','native_summary.json']]
    assert n['loading_exact'] and n['devices']==[0] and n['head_bytes_verified']
    assert g['loading_exact'] and m['loading_exact'] and m['devices']==[0,1] and m['head_bytes_verified']
    for summary in [g,m,n]:
        assert summary['versions']['python'].startswith('3.10.')
        assert summary['versions']['torch'].split('+')[0]=='2.1.2'
        assert summary['versions']['transformers']=='4.42.3'
    assert m['head_sha256']==n['head_sha256']==manifest['classifier_head_sha256']
    controls=read_json(out/'mirror_native_controls.json')
    assert controls['passed'] and all(v<=1e-5 for k,v in controls.items() if k.endswith('_max_abs'))
    for name in ['warmup','control','uniform','long','test']:
        a=np.load(out/('mirror_'+name+'.npz'))
        expected=np.load(ROOT/next(k for k in tokens['artifact_sha256'] if k.endswith('/'+name+'.npz')))
        assert np.array_equal(a['ids'],expected['ids']) and np.array_equal(a['lengths'],expected['lengths'])
        assert a['token_fingerprint'].item()==tokens['cohorts'][name]['token_fingerprint']
    canonical=np.load(out/'m0_control.npz')
    old=np.load(ROOT/'runs/span_013/jobs/gemma_order/output/order_completion/m0_control.npz')
    assert canonical['ids'].tolist()==selection['control_ids'] and np.array_equal(canonical['ids'],old['ids'])
    difference=float(np.max(np.abs(canonical['probabilities']-old['probabilities'])))
    assert difference<=1e-5
    parent=read_json(pub/'parent_runtime_reference.json')
    assert sha256(Path(parent['source_path']))==parent['source_sha256']
    engineering=read_json(out/'engineering.json');setup=read_json(job/'output/setup_summary.json')
    runtime=projection(g,m,parent,engineering['retained_process_seconds'],engineering['mirror_process_seconds'],setup['seconds'])
    result=dict(at=utc_now(),engineering_checks_passed=True,native_controls=controls,
        retained_canonical_probability_max_abs=difference,runtime=runtime,
        engineering_gate_passed=runtime['runtime_gate_passed'],
        quality_evaluated=False,official_submission_eligible=False,human_votes_read=False,
        actual_uniform_rows=512,actual_warmup_rows=80,
        source_sha256={str(Path(__file__).relative_to(REPO)):sha256(Path(__file__)),
            'src/arena/span019_runtime.py':sha256(REPO/'src/arena/span019_runtime.py')})
    atomic_json(pub/'engineering_qualification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
