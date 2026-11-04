"""Verify v4 evidence and fresh service state; closure mode checks the sealed boundary."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import nbformat
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,CLASSES,json_write,prompt_groups
from arena.joint_data import read_competition
from arena.span004_support import api_client,inventory,read_call
from arena.span_cascade import RecipeSet

AGG=REPO/'results/span_004';HEAVY=ROOT/'runs/span_004'
ROLES=['repeatability','repeatability_retry']


def read(p):return json.loads(Path(p).read_text())
def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def close(a,b):np.testing.assert_allclose(a,b,atol=1e-10,rtol=0)
def norm(p):
    p=np.asarray(p,float);assert np.isfinite(p).all() and np.all(p>=0)
    np.testing.assert_allclose(p.sum(1),1,atol=1e-8)
    p=np.clip(p,1e-12,1);return p/p.sum(1,keepdims=True)
def losses(y,p):return -np.log(norm(p)[np.arange(len(y)),y])


def local():
    activation,plan=read(AGG/'activation.json'),read(AGG/'span_plan.json')
    assert activation['goal_service_created_at']==1793802572
    start=datetime.fromtimestamp(1793802572,timezone.utc)
    assert plan['started_at']==activation['started_at']==start.isoformat()
    assert plan['deadline']==(start+timedelta(hours=48)).isoformat()
    assert digest(REPO/'GOAL.md')==activation['goal_contract_sha256']
    initial=read(HEAVY/'resource_inventory_initial.json')
    close(plan['gpu_session_seconds_cap'],min(43200,initial['quota']['remaining_seconds'],29337.999))
    config=read(REPO/'configs/span_004.json')
    original=json.loads(subprocess.check_output(['git','show','77fdda2:configs/span_004.json'],cwd=REPO))
    admin={'job_seconds_caps','diagnostic_stage_seconds_cap'}
    assert {k:v for k,v in config.items() if k not in admin}=={k:v for k,v in original.items() if k not in admin}
    for name,revision in [('premeasurement_contract.json','77fdda2'),('repeatability_retry_contract.json','727615e')]:
        contract=read(AGG/name)
        for path,sha in contract['source_sha256'].items():
            blob=subprocess.check_output(['git','show',revision+':'+path],cwd=REPO)
            assert hashlib.sha256(blob).hexdigest()==sha,path
    worker=REPO/'src/arena/span_repeatability.py'
    assert worker.read_bytes()==subprocess.check_output(['git','show','77fdda2:src/arena/span_repeatability.py'],cwd=REPO)
    inputs=read(AGG/'artifact_inventory.json')
    for value in inputs.values():
        if isinstance(value,dict) and {'path','sha256'}<=set(value):assert digest(value['path'])==value['sha256']
    assert digest(HEAVY/'sample_ids.json')==inputs['sample_ids_sha256']
    sample=read(HEAVY/'sample_ids.json');ids=np.asarray(sample['sentinel'])
    np.testing.assert_array_equal(sample['permutation_indices'],np.random.default_rng(20261121).permutation(64))
    assert sample['sentinel']==read(ROOT/'runs/span_003/sample_ids.json')['native_profile']
    assert sample['followup']==read(ROOT/'runs/span_003/sample_ids.json')['cascade_followup']
    frame,_=read_competition(ROOT/'raw');frame=frame.set_index('id',drop=False)
    groups=prompt_groups(frame)
    assert not set(groups.loc[sample['fitting']]) & set(groups.loc[sample['followup']])
    assert all(not (HEAVY/'jobs'/role/'push_intent.json').exists() for role in ['consistent_cache','cascade_followup','offline_profile'])
    preserved={}
    for p in sorted((REPO/'backlog/versions').glob('*')):
        if p.name.startswith(('v1.','v2.','v3.','v4.start.')):
            relative=str(p.relative_to(REPO));assert p.read_bytes()==subprocess.check_output(['git','show','9aef67c:'+relative],cwd=REPO)
            preserved[relative]=digest(p)
    assert len(preserved)==14
    out=HEAVY/'jobs/repeatability_retry/output'
    summary=read(out/'repeatability/diagnostic_summary.json');result=read(AGG/'repeatability_results.json')
    assert digest(REPO/'scripts/analyze_span_004_repeatability.py')==result['source_sha256']
    assert digest(out/'repeatability/diagnostic_summary.json')==result['private_summary_sha256']
    manifests=[read(Path(inputs['portable_manifest']['path'])),read(HEAVY/'jobs/repeatability/output/offline_native_runtime/manifest.json'),read(out/'offline_native_runtime/manifest.json')]
    assert manifests[0]==manifests[1]==manifests[2] and len(manifests[0]['files'])==175
    assert digest(HEAVY/'jobs/repeatability/output/offline_native_runtime/manifest.json')!=inputs['portable_manifest']['sha256']
    recovery=read(HEAVY/'jobs/repeatability_retry/retrieval_recovery.json')
    assert recovery['gpu_jobs_launched']==0 and recovery['stopped_observer_exit']==143
    assert digest(REPO/'scripts/recover_span_004_outputs.py')==recovery['source_sha256']
    for name,sha in recovery['artifact_sha256'].items():assert digest(out/name)==sha,name
    initial_server,restarted=[read(out/'repeatability'/name/'session.json') for name in ['server_initial','server_restart']]
    assert initial_server['server_command']==restarted['server_command']
    command=initial_server['server_command']
    for key,value in [('--parallel','16'),('--batch-size','4096'),('--ubatch-size','1024'),('--ctx-size','16384'),('--tensor-split','1,1')]:assert command[command.index(key)+1]==value
    y=frame.loc[ids,CLASSES].values.argmax(1);g=groups.loc[ids].to_numpy()
    _,inverse=np.unique(g,return_inverse=True);counts=np.bincount(inverse)
    draws=np.random.default_rng(config['seed']).integers(0,len(counts),size=(2000,len(counts)))
    old=np.load(inputs['previous_native_reference']['path'])['original']
    frozen=read(ROOT/'runs/span_003/frozen_cached_recipes.json');cal=RecipeSet.from_dict(frozen['parameters'],frozen['config']).fits[1]
    old_cal=cal.predict_proba(old);common_tokens=None;requests_checked=0;passes={}
    for name,value in result['cases'].items():
        arrays={}
        token_ids=summary['configurations'][name]['label_tokens']
        for phase,metric in value['passes'].items():
            folder=out/'repeatability'/name/phase;a=np.load(folder/'predictions.npz')
            wanted=ids[np.asarray(sample['permutation_indices'])] if phase=='permuted' else ids
            np.testing.assert_array_equal(a['ids'],wanted)
            lookup={int(r):index for index,r in enumerate(a['ids'])};p=a['original'][[lookup[int(r)] for r in ids]];arrays[phase]=p
            traces=[json.loads(line) for line in (folder/'requests.jsonl').read_text().splitlines()]
            assert len(traces)==64 and {r['id'] for r in traces}==set(map(int,ids))
            tokens={}
            for row in traces:
                assert row['cache_prompt'] is value['cache_prompt']
                token_hash=hashlib.sha256(json.dumps(row['prepared_tokens']).encode()).hexdigest()
                assert token_hash==row['prepared_tokens_sha256'];tokens[row['id']]=token_hash
                response=row['response'];prob_rows=response.get('completion_probabilities',response.get('probs'))
                assert len(prob_rows)==1
                choices=prob_rows[0].get('top_probs',prob_rows[0].get('top_logprobs'))
                by_id={int(c['id']):float(c['prob'] if 'prob' in c else np.exp(c['logprob'])) for c in choices}
                raw=np.array([by_id[t] for t in token_ids]);raw/=raw.sum()
                close(raw,p[list(ids).index(row['id'])]);requests_checked+=1
            if common_tokens is None:common_tokens=tokens
            assert common_tokens==tokens
            current=cal.predict_proba(p)
            for probability,key in [(p,'raw_metrics'),(current,'fixed_calibration_metrics')]:
                m=metric[key];close(losses(y,probability).mean(),m['log_loss']);close((probability.argmax(1)==y).mean(),m['accuracy'])
                matrix=np.zeros((3,3),int);np.add.at(matrix,(y,probability.argmax(1)),1)
                np.testing.assert_array_equal(matrix,m['confusion_matrix'])
                close([losses(y,probability)[y==k].mean() for k in range(3)],m['class_loss'])
                close(np.diag(matrix)/matrix.sum(1),m['class_recall'])
            for baseline,observed,key in [(old,p,'raw_loss_effect'),(old_cal,current,'fixed_calibration_loss_effect')]:
                delta=losses(y,baseline)-losses(y,observed);sums=np.bincount(inverse,weights=delta)
                close(delta.mean(),metric[key]['gain'])
                close(np.quantile(sums[draws].sum(1)/counts[draws].sum(1),[.025,.975]),metric[key]['interval_95'])
        for phase in ['repeat','permuted','restart']:
            delta=np.abs(arrays['initial']-arrays[phase]);c=value['comparisons'][phase]
            close([delta.mean(),delta.max()],[c['mean_probability_difference'],c['max_probability_difference']])
            assert not c['passed'] and (delta.mean()>.001 or delta.max()>.01)
        passes[name]=len(arrays)
    assert requests_checked==512 and result['selected_recipe'] is None and not result['new_uniform_cache_admitted']
    controls=read(AGG/'diagnostic_controls.json');assert controls['rows']==64
    assert digest(REPO/'scripts/build_span_004_diagnostics.py')==controls['source_sha256']
    close(controls['controls']['uniform']['log_loss'],np.log(3))
    close(controls['controls']['majority_clipped_1e-12']['accuracy'],np.mean(y==0))
    charges=[]
    for role in ROLES:
        d=HEAVY/'jobs'/role
        charges.append(read(d/'quota.json')['used_seconds']-read(d/'push_intent.json')['quota_before']['used_seconds'])
    close(charges,[37.403,801.85]);close(sum(charges),result['span_charged_seconds'])
    assert sum(charges)<1800 and sum(charges)<plan['gpu_session_seconds_cap']
    return {'passed':True,'fresh_requests_reconstructed':requests_checked,'passes_by_case':passes,
            'all_prepared_tokens_identical':True,'fixed_calibration_effect_intervals_recomputed':16,
            'old_snapshot_sha256':preserved,'preserved_snapshot_count':14,'source_recipe_unchanged':True,
            'hardware_and_175_file_manifest_verified':True,'charges_seconds':charges,
            'total_charged_seconds':sum(charges),'no_optional_inference':True,'confirmation_used':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--closure',action='store_true');args=parser.parse_args()
    if not args.closure:
        json_write(AGG/'terminal_audit.json',{'passed':False,'stage':'verification_running'})
        evidence=local();api=api_client();plan=read(AGG/'span_plan.json');jobs=[]
        for role in ROLES:
            d=HEAVY/'jobs'/role;pushed=read(d/'push.json');exact=pushed['ref'].removeprefix('/code/')+'/'+str(pushed['version_number'])
            status=str(read_call(lambda:api.kernels_status(exact)).status).split('.')[-1]
            assert status==('ERROR' if role=='repeatability' else 'COMPLETE')
            dest=HEAVY/'audited_sources'/role;dest.mkdir(parents=True,exist_ok=True)
            read_call(lambda:api.kernels_pull(exact,path=str(dest),metadata=True,quiet=True))
            pulled=list(dest.glob('*.ipynb'));assert len(pulled)==1
            source=HEAVY/'notebooks'/role/'job.ipynb'
            cells=lambda p:[(c.cell_type,c.source) for c in nbformat.read(p,as_version=4).cells]
            assert cells(source)==cells(pulled[0]) and read(dest/'kernel-metadata.json')['enable_internet'] is False
            assert digest(source)==read(d/'push_intent.json')['notebook_sha256']
            jobs.append({'role':role,'kernel_version':exact,'status':status,'source_matches':True,'internet_disabled':True,'notebook_sha256':digest(source)})
        observed=inventory(api,plan['started_at']);assert not observed['nonterminal_jobs']
        q=observed['quota'];assert q['reserved_seconds']==0 and not q['paid_scaling']
        assert q['refresh_time']==plan['initial_quota_refresh_time']
        close(q['used_seconds']-plan['quota_used_at_start'],evidence['total_charged_seconds'])
        submissions=list(read_call(lambda:api.competition_submissions('llm-classification-finetuning')) or [])
        baseline=next(row for row in submissions if int(row.ref)==57979825)
        assert str(baseline.status).endswith('COMPLETE');close(float(baseline.public_score),.95481)
        start=datetime.fromisoformat(plan['started_at']);recent=[]
        for row in submissions:
            date=datetime.fromisoformat(str(row.date))
            if date.tzinfo is None:date=date.replace(tzinfo=timezone.utc)
            if date>=start:recent.append(int(row.ref))
        assert not recent
        now=datetime.now(timezone.utc);assert now<datetime.fromisoformat(plan['deadline'])
        result={**evidence,'checked_at':now.isoformat(),'jobs':jobs,'owned_notebooks':observed['owned_notebook_count'],
                'unobserved_old_notebooks':observed['unobserved_old_notebooks'],'no_nonterminal_owned_notebooks':True,
                'quota':q,'original_start':plan['started_at'],'original_deadline':plan['deadline'],
                'within_original_caps':True,'official_score':.95481,'new_official_submissions':recent,
                'audit_program_sha256':digest(Path(__file__)),'repeatability_results_sha256':digest(AGG/'repeatability_results.json')}
        for p in [AGG,HEAVY]:json_write(p/'terminal_audit.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in {'old_snapshot_sha256','unobserved_old_notebooks'}},indent=2));return
    terminal=read(AGG/'terminal_audit.json');assert terminal['passed']
    assert terminal['audit_program_sha256']==digest(Path(__file__))
    assert terminal['repeatability_results_sha256']==digest(AGG/'repeatability_results.json')
    assert (datetime.now(timezone.utc)-datetime.fromisoformat(terminal['checked_at'])).total_seconds()<3600
    registry=read(REPO/'backlog/hypotheses.json');versions={v['number']:v for v in registry['versions']};items={i['id']:i for i in registry['items']}
    assert registry['active_version']==5 and versions[4]['state']=='closed'
    assert versions[5]['state']=='active' and not versions[5]['focus_ids']
    assert versions[6]['state']=='draft' and not versions[6]['item_ids']
    assert items['E04']['status']=='done' and items['H44']['status']=='deferred' and items['H44']['evidence']
    assert not any(i['status']=='running' for i in items.values())
    carry={key for key in versions[4]['item_ids'] if items[key]['status'] not in {'done','rejected'}}
    new={key for key in versions[5]['item_ids'] if items[key]['introduced_in']==5}
    assert new=={'E05'} and set(versions[5]['item_ids'])==carry|new and len(carry)==34
    assert all(items[key]['status'] in {'planned','deferred'} for key in versions[5]['item_ids'])
    for p,sha in terminal['old_snapshot_sha256'].items():assert digest(REPO/p)==sha
    notebook=REPO/'notebooks/15_backlog_v4_span.ipynb';nb=nbformat.read(notebook,as_version=4)
    code=[c for c in nb.cells if c.cell_type=='code'];outputs=[o for c in code for o in c.outputs]
    assert len(code)==5 and all(c.execution_count is not None for c in code)
    assert not any(o.output_type=='error' for o in outputs)
    assert sum('image/png' in o.get('data',{}) for o in outputs)==3
    text='\n'.join(o.get('text','') for o in outputs);assert 'v4 state: closed' in text and '0.95481' in text
    assert (Path.home()/'notebooks/llm_classification_finetuning').resolve()==REPO/'notebooks'
    reports=[REPO/'docs'/('SPAN_004_'+name+'.md') for name in ['RESULTS','E04','H44','COMPLETION']]
    files=reports+[notebook,REPO/'backlog/hypotheses.json',REPO/'backlog/versions/v4.final.json',REPO/'backlog/versions/v4.final.md',REPO/'backlog/versions/v5.start.json',REPO/'backlog/versions/v5.start.md']
    files += [p for p in AGG.glob('*.json') if p.name!='closure_audit.json']
    result={'checked_at':datetime.now(timezone.utc).isoformat(),'passed':True,'v4_closed':True,'active_version':5,
            'v5_carryovers':len(carry),'v5_new':sorted(new),'v5_tasks':len(versions[5]['item_ids']),'v6_empty_draft':True,
            'no_v5_execution':True,'no_running_tasks':True,'notebook_code_cells':5,'notebook_figures':3,'notebook_errors':0,
            'terminal_audit_at':terminal['checked_at'],'total_charged_seconds':terminal['total_charged_seconds'],
            'artifact_sha256':{str(p.relative_to(REPO)):digest(p) for p in files},
            'publication':'Verify remote main equals closing local HEAD after commit/push.'}
    json_write(AGG/'closure_audit.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
