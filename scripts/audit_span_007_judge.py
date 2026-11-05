"""Replay native request/probability/stability evidence without labels or GPU work."""
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,sha256,verify_sources,utc_now
from arena.judge import label_probabilities
from arena.span_repeatability import difference,validate


def main():
    started=time.monotonic();pub=REPO/'results/span_007';run=ROOT/'runs/span_007';job=run/'jobs/judge_stability';out=job/'output/judge_stability'
    assert not (pub/'judge_audit.json').exists()
    assert read_json(job/'controller_state.json')['stage']=='complete'
    verify_sources(REPO,read_json(pub/'judge_stability_bundle.json')['source_sha256'])
    assert read_json(pub/'jobs/judge_stability/source_verification.json')['passed']
    receipt=read_json(job/'retrieval.json')
    for file,digest in receipt['artifact_sha256'].items():assert sha256(job/'output'/file)==digest
    result=read_json(pub/'E05_results.json');original=read_json(out/'diagnostic_summary.json')
    assert original['model_identity']['sha256']==result['model_sha256']=='4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5'
    hw=read_json(job/'output/hardware.json');assert 'not found' not in hw['ldd'] and all(v in hw['devices'] for v in ['CUDA0: Tesla T4','CUDA1: Tesla T4'])
    ids=read_json(run/'sample_ids.json')['judge']['sentinel'];assert len(ids)==len(set(ids))==64
    sessions=[read_json(out/s/'session.json') for s in ['server_initial','server_restart']]
    assert sessions[0]['server_command']==sessions[1]['server_command']
    events=[json.loads(line) for line in (out/'stages/events.jsonl').read_text().splitlines()]
    all_tokens={};passed={};requests=0
    for name,concurrency in [('serial',1),('parallel',16)]:
        reference=np.load(out/name/'initial/predictions.npz');np.testing.assert_array_equal(reference['ids'],ids)
        case=original['configurations'][name];public=result['configurations'][name]
        assert case['cache_prompt'] is False and case['native_both_devices_verified']
        assert len(case['sampled_device_used_peak_mib'])==2 and min(case['sampled_device_used_peak_mib'])>1024
        assert public['concurrency']==concurrency
        tokens_by_pass={}
        for stage in ['initial','repeat','permuted','restart']:
            folder=out/name/stage;raw=np.load(folder/'predictions.npz');validate(raw['original'])
            assert len(raw['ids'])==64 and set(map(int,raw['ids']))==set(ids)
            assert sha256(folder/'predictions.npz')==public['passes'][stage]['prediction_sha256']
            assert sha256(folder/'requests.jsonl')==public['passes'][stage]['requests_sha256']
            traces=[json.loads(line) for line in (folder/'requests.jsonl').read_text().splitlines()]
            assert len(traces)==64 and {t['id'] for t in traces}==set(ids);requests+=len(traces)
            lookup={int(i):p for i,p in zip(raw['ids'],raw['original'])};tokens_by_pass[stage]={}
            for t in traces:
                assert t['cache_prompt'] is False and t['pass']==stage and not t['response'].get('truncated',False)
                assert 0<len(t['prepared_tokens'])<1024 and all(isinstance(v,int) for v in t['prepared_tokens'])
                assert hashlib.sha256(json.dumps(t['prepared_tokens']).encode()).hexdigest()==t['prepared_tokens_sha256']
                np.testing.assert_allclose(label_probabilities(t['response'],case['label_tokens']),lookup[t['id']],rtol=0,atol=1e-12)
                tokens_by_pass[stage][t['id']]=t['prepared_tokens_sha256']
            assert tokens_by_pass[stage]==tokens_by_pass['initial']
            log=[e for e in events if e['stage']==f'judge_stability/{name}/{stage}' and e['event']=='pass_completed']
            assert len(log)==1 and log[0]['rows']==64 and log[0]['metrics']['client_concurrency']==concurrency
            if stage!='initial':
                computed=difference(reference['ids'],reference['original'],raw['ids'],raw['original'],read_json(REPO/'configs/span_007.json')['repeatability_limits'])
                saved=public['comparisons'][stage]
                for key,value in computed.items():assert value==saved[key]
                assert saved['prepared_tokens_identical']
        all_tokens[name]=tokens_by_pass['initial'];passed[name]=public['repeatability_passed']
        assert passed[name]==all(c['passed'] and c['prepared_tokens_identical'] for c in public['comparisons'].values())
        assert public['seconds_per_row']==max(v['seconds'] for v in public['passes'].values())/64
    assert all_tokens['serial']==all_tokens['parallel'] and requests==512
    serial=result['configurations']['serial'];scenario=serial['projected_cascade_seconds']-25000*.1*serial['seconds_per_row']*1.2
    record={'passed':True,'checked_at':utc_now(),'competition_requests_replayed':requests,'labels_read':False,
            'case_stability':passed,'token_inputs_identical':True,'cache_off_verified':True,'native_two_t4_verified':True,
            'serial_ten_percent_route_seconds_scenario':scenario,
            'scenario_note':'Timing-only extrapolation with the existing formula; not a qualified end-to-end recipe or a change to the failed20%gate.',
            'audit_seconds':time.monotonic()-started,'source_sha256':sha256(Path(__file__)),
            'E05_results_sha256':sha256(pub/'E05_results.json'),'retrieval_sha256':sha256(job/'retrieval.json')}
    atomic_json(pub/'judge_audit.json',record);print(record)


if __name__=='__main__':main()
