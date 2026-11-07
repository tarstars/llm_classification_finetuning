"""Shared exact-version controller and artifact recovery for future research runs."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import os
import re
import subprocess
import sys
import time

from .experiment import (read_json, atomic_json, write_intent, exclusive_lock, sha256,
                         verify_sources, admit, output_target, collect_pages, download_output,
                         EventJournal, utc_now)
from .span004_support import api_client, inventory, public_inventory, quota, read_call, TERMINAL


def output_catalog(api, exact):
    from kaggle.api.kaggle_api_extended import ApiListKernelSessionOutputRequest
    owner, slug, version = api.parse_kernel_string(exact)
    def fetch(cursor):
        request = ApiListKernelSessionOutputRequest()
        request.user_name, request.kernel_slug = owner, slug
        request.version_label = api._kernel_version_label(version)
        api._set_paging(request, 100, cursor)
        with api.build_kaggle_client() as client:
            return read_call(lambda: client.kernels.kernels_api_client.list_kernel_session_output(request))
    return collect_pages(fetch)


def retrieve(api, exact, directory):
    directory = Path(directory); output = directory / 'output'
    files = output_catalog(api, exact)
    final_checkpoint = max((int(m[1]) for item in files if (m := re.search(r'checkpoint-(\d+)/',item.file_name))),default=-1)
    def selected(name):
        if name.startswith('bundled_code/'): return False
        if '/checkpoints/' in name:
            return f'checkpoint-{final_checkpoint}/' in name
        return bool(re.search(r'\.(json|jsonl|npz|log|csv|safetensors)$',name))
    items = [f for f in files if selected(f.file_name)]
    previous = read_json(directory/'retrieval.json').get('artifact_sha256',{}) if (directory/'retrieval.json').exists() else {}
    def fetch(item):
        target = output_target(output,item.file_name)
        if target.exists() and previous.get(item.file_name)==sha256(target):
            return item.file_name,previous[item.file_name]
        return item.file_name,download_output(item.url,target)['sha256']
    with ThreadPoolExecutor(max_workers=3) as pool: hashes=dict(pool.map(fetch,items))
    receipt={'files':len(hashes),'artifact_sha256':hashes,'final_checkpoint':final_checkpoint,
             'new_gpu_executions':0,'signed_urls_persisted':False,'retrieved_at':utc_now()}
    atomic_json(directory/'retrieval.json',receipt)
    return receipt


def exact_model_version(source, versions):
    """Select one fully identified version, never an unrelated version number.

    Kaggle can return variations/frameworks other than the requested instance.
    Missing or contradictory identity is insufficient evidence of availability.
    """
    parts = source.strip('/').split('/')
    if len(parts) != 5 or not parts[-1].isdigit():
        raise ValueError('Expected pinned owner/model/framework/variation/version')
    def framework(value):
        return str(value).lower().removeprefix('model_framework_').replace('_', '').replace('-', '')
    wanted = (parts[0].lower(), parts[1].lower(), framework(parts[2]),
              parts[3].lower(), int(parts[4]))
    matches = []
    for row in versions:
        def field(camel, snake):
            return row.get(camel, row.get(snake))
        values = [field('ownerSlug', 'owner_slug'), field('modelSlug', 'model_slug'),
                  row.get('framework'), field('variationSlug', 'variation_slug'),
                  field('versionNumber', 'version_number')]
        if any(value is None for value in values):
            continue
        try:
            identity = (str(values[0]).lower(), str(values[1]).lower(), framework(values[2]),
                        str(values[3]).lower(), int(values[4]))
        except (TypeError, ValueError):
            continue
        if identity != wanted:
            continue
        if row.get('url'):
            url_parts = row['url'].strip('/').split('/')
            expected = ['models', *parts]
            if [p.lower() for p in url_parts] != [p.lower() for p in expected]:
                raise ValueError('Conflicting model identity URL')
        matches.append(row)
    if len(matches) != 1:
        raise ValueError(f'Exact model version unavailable or ambiguous: {source} ({len(matches)} matches)')
    return matches[0]


def verify_dependencies(api, metadata):
    result={'observed_at':utc_now(),'parents':[],'models':[]}
    for source in metadata['kernel_sources']:
        if not source.split('/')[-1].isdigit(): raise ValueError('Unpinned parent')
        status=str(read_call(lambda:api.kernels_status(source)).status).split('.')[-1]
        if status!='COMPLETE': raise ValueError('Parent is not complete')
        result['parents'].append({'source':source,'status':status,'output_files':len(output_catalog(api,source))})
    for source in metadata['model_sources']:
        instance,version=source.rsplit('/',1)
        if not version.isdigit(): raise ValueError('Unpinned model')
        response=read_call(lambda:api.model_instance_versions_list(instance,page_size=100))
        versions=[x.to_dict() for x in response.version_list.versions]
        if len(versions) >= 100:
            raise ValueError('Model listing may be truncated; paginate before admission')
        matched=exact_model_version(source,versions)
        result['models'].append({'source':source,'available':True,
            'model_instance_id':matched.get('modelInstanceId',matched.get('model_instance_id')),
            'model_version_id':matched.get('id')})
    return result


def sanitize_event(record):
    """Keep aggregate telemetry and bounded worker identifiers, never raw text."""
    allowed={'at','stage','event','step','elapsed_seconds','rank','counts','metrics','gpu',
             'training_seconds','rows','seconds','passed','complete','new_cycles','stopped_for_budget'}
    result={key:value for key,value in record.items() if key in allowed}
    for key in ['role','cohort']:
        value=record.get(key)
        if isinstance(value,str) and re.fullmatch(r'[A-Za-z0-9_.-]{1,64}',value):
            result[key]=value
    value=record.get('batch')
    if isinstance(value,int) and not isinstance(value,bool) and value>=0:
        result['batch']=value
    return result


def observe_logs(repo, root, role, public):
    """Reuse the existing bounded SSE reader; retain raw streams privately."""
    command=[sys.executable,str(repo/'scripts/kaggle_job.py'),'logs','--run-directory',str(root),
             '--attempt',root.name,
             '--folder',str(root/'notebooks'/role),'--job',role,'--seconds','8']
    try:
        result=subprocess.run(command,cwd=repo,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=25)
        (root/'jobs'/role/'log_observer.txt').write_bytes(result.stdout)
        path=root/'jobs'/role/'live_events.jsonl'
        if not path.exists(): return
        records=[]
        for line in path.read_text().splitlines():
            for message in str(json.loads(line).get('data','')).splitlines():
                if not message.startswith('ARENA_EVENT '): continue
                try:record=json.loads(message.removeprefix('ARENA_EVENT '))
                except json.JSONDecodeError:continue
                records.append(sanitize_event(record))
        if records:
            atomic_json(public/'progress.json',{'observed_at':utc_now(),'latest':records[-1],'events':records})
    except (OSError,subprocess.TimeoutExpired,ValueError):
        # Observation failure never causes a mutation or invalidates a running job.
        atomic_json(public/'log_observation.json',{'at':utc_now(),'available':False,'job_execution_unchanged':True})


def run_job(repo,root,pub,config,role,dry_run=False,precondition=None):
    repo,root,pub=map(Path,[repo,root,pub]);directory=root/'jobs'/role;public=pub/'jobs'/role
    public.mkdir(parents=True,exist_ok=True);directory.mkdir(parents=True,exist_ok=True)
    with exclusive_lock(directory/'controller.lock'):
        cfg=read_json(config);plan=read_json(pub/'span_plan.json');bundle=read_json(pub/(role+'_bundle.json'))
        folder=root/'notebooks'/role;metadata=read_json(folder/'kernel-metadata.json');cap=cfg['job_seconds_caps'][role]
        verify_sources(repo,bundle['source_sha256'])
        assert sha256(folder/'job.ipynb')==bundle['notebook_sha256'] and sha256(folder/'kernel-metadata.json')==bundle['metadata_sha256']
        assert metadata['enable_internet'] is False
        api=api_client();journal=EventJournal(public)
        def save(name,value):
            for target in [directory,public]:atomic_json(target/name,value)
        def state(stage,**values):
            value={'stage':stage,'at':utc_now(),'pid':os.getpid(),'role':role,**values}
            save('controller_state.json',value)
            if stage!='monitoring':journal.emit(role,stage,**values)
        if not (directory/'push.json').exists():
            assert not (directory/'push_intent.json').exists(),'Ambiguous mutation: reconcile the existing intent, never re-push'
            with exclusive_lock(root/'admission.lock'):
                fresh=inventory(api,cfg['started_at']);admit(plan,fresh['quota'],cap,fresh['nonterminal_jobs'])
                dependencies=verify_dependencies(api,metadata)
                save('dependencies.json',dependencies);save('admission_inventory.json',public_inventory(fresh))
                if dry_run:state('dry_run_admissible',cap_seconds=cap);return
                if precondition is not None:precondition(role)
                intent={'at':utc_now(),'kernel':metadata['id'],'quota_before':fresh['quota'],'max_session_seconds':cap,
                        'notebook_sha256':bundle['notebook_sha256'],'metadata_sha256':bundle['metadata_sha256']}
                write_intent(directory/'push_intent.json',intent);atomic_json(public/'push_intent.json',intent)
                state('pushing_one_version')
                response=api.kernels_push(str(folder),timeout=str(cap),acc='NvidiaTeslaT4')
                pushed={k:getattr(response,k) for k in ['ref','version_number','url','error','invalid_model_sources','invalid_dataset_sources'] if hasattr(response,k)}
                save('push.json',pushed)
                assert pushed.get('version_number') and not any(pushed.get(k) for k in ['error','invalid_model_sources','invalid_dataset_sources'])
        pushed=read_json(directory/'push.json');exact=pushed['ref'].removeprefix('/code/')+'/'+str(pushed['version_number'])
        last,last_logs=None,0
        while time.time()<__import__('datetime').datetime.fromisoformat(cfg['deadline']).timestamp():
            try:status=str(read_call(lambda:api.kernels_status(exact)).status).split('.')[-1]
            except Exception as error:
                state('observation_retry',error_type=type(error).__name__,kernel_version=exact);time.sleep(30);continue
            save('status.json',{'status':status,'kernel_version':exact,'observed_at':utc_now()})
            state('monitoring',status=status,kernel_version=exact)
            if status!=last:journal.emit(role,'remote_status',status=status,kernel_version=exact);last=status
            if status in TERMINAL:
                if not (directory/'quota.json').exists():save('quota.json',quota(api))
                receipt=retrieve(api,exact,directory);atomic_json(public/'retrieval.json',receipt)
                charge=read_json(directory/'quota.json')['used_seconds']-read_json(directory/'push_intent.json')['quota_before']['used_seconds']
                state('complete' if status=='COMPLETE' else 'failed',status=status,kernel_version=exact,charged_seconds=charge,files=receipt['files'])
                assert charge<=cap+.1
                if status!='COMPLETE':raise RuntimeError('Remote execution failed; preserved exact source and outputs')
                return
            if time.monotonic()-last_logs>120:
                observe_logs(repo,root,role,public);last_logs=time.monotonic()
            time.sleep(45)
        state('original_deadline_reached',kernel_version=exact)
        raise TimeoutError('Observe/stop the existing version; never duplicate it')
