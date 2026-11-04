"""Recover bounded reads from the same COMPLETE E04 version; never launch a job."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import requests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT, json_write
from arena.span004_support import api_client, quota, read_call
from kaggle.api.kaggle_api_extended import ApiListKernelSessionOutputRequest


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    root = ROOT / 'runs/span_004'
    directory, public = root / 'jobs/repeatability_retry', REPO / 'results/span_004/jobs/repeatability_retry'
    lock = (directory / 'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    push = read(directory / 'push.json')
    exact = push['ref'].removeprefix('/code/') + '/' + str(push['version_number'])
    api = api_client()
    assert str(read_call(lambda: api.kernels_status(exact)).status).split('.')[-1] == 'COMPLETE'
    owner,slug,version = api.parse_kernel_string(exact)
    cursor, files, log = None, [], None
    for _ in range(20):
        def listing():
            request = ApiListKernelSessionOutputRequest()
            request.user_name,request.kernel_slug = owner,slug
            request.version_label = api._kernel_version_label(version)
            api._set_paging(request,100,cursor)
            with api.build_kaggle_client() as client:
                return client.kernels.kernels_api_client.list_kernel_session_output(request)
        response = read_call(listing)
        files.extend(response.files)
        log = log or response.log
        cursor = response.next_page_token
        if not cursor:break
    assert not cursor
    verified, output = directory / 'verified_output', directory / 'output'
    verified.mkdir(exist_ok=True)
    def selected(name):
        return name in {'session_summary.json','hardware.json','native_reference.npz','offline_native_runtime/manifest.json'} or (
            name.startswith('repeatability/') and Path(name).suffix in {'.json','.jsonl','.npz','.log'})
    wanted = [item for item in files if selected(item.file_name)]
    by_name = {item.file_name:item for item in wanted}
    def fetch(name, expected_sha=None):
        target = verified / name
        assert target.resolve().is_relative_to(verified.resolve())
        target.parent.mkdir(parents=True,exist_ok=True)
        old = output / name
        if expected_sha and old.exists() and digest(old) == expected_sha:
            shutil.copy2(old,target)
            return name
        item = by_name[name]
        for attempt in range(3):
            try:
                response = requests.get(item.url,headers={'Accept-Encoding':'identity'},timeout=(10,45))
                response.raise_for_status()
                data = response.content
                if response.headers.get('Content-Length') and not response.headers.get('Content-Encoding'):
                    assert len(data) == int(response.headers['Content-Length'])
                if expected_sha:
                    assert hashlib.sha256(data).hexdigest() == expected_sha
                temporary = target.with_suffix(target.suffix+'.download')
                temporary.write_bytes(data)
                temporary.replace(target)
                return name
            except (requests.exceptions.RequestException,AssertionError):
                if attempt == 2:
                    raise RuntimeError('Artifact read failed: '+name) from None
                time.sleep(2*(attempt+1))
    fetch('repeatability/diagnostic_summary.json')
    summary = read(verified / 'repeatability/diagnostic_summary.json')
    expected = {}
    for case,record in summary['configurations'].items():
        for phase,values in record['passes'].items():
            prefix = 'repeatability/'+case+'/'+phase+'/'
            expected[prefix+'predictions.npz'] = values['prediction_sha256']
            expected[prefix+'requests.jsonl'] = values['requests_sha256']
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda item:fetch(item.file_name,expected.get(item.file_name)),
                      [item for item in wanted if item.file_name != 'repeatability/diagnostic_summary.json']))
    for name,sha in expected.items():assert digest(verified / name)==sha,name
    assert read(verified / 'offline_native_runtime/manifest.json') == read(ROOT / 'runs/span_003/jobs/native_profile_retry/output/offline_native_runtime/manifest.json')
    for item in wanted:
        source, target = verified / item.file_name, output / item.file_name
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary = target.with_suffix(target.suffix+'.verified')
        shutil.copy2(source,temporary)
        temporary.replace(target)
    if log:(output / (slug+'.log')).write_text(log)
    q = quota(api)
    result = {'observed_at':datetime.now(timezone.utc).isoformat(),'kernel_version':exact,
        'kernel_status':'COMPLETE','gpu_jobs_launched':0,'verified_files':len(wanted),
        'all_prediction_and_request_hashes_match_gpu_summary':True,
        'transport':'Separate bounded GETs with4workers; signed URLs never printed/persisted.',
        'stopped_read_only_observer_pid':2341404,'stopped_observer_exit':143,
        'source_sha256':digest(Path(__file__).resolve()),
        'artifact_sha256':{item.file_name:digest(verified / item.file_name) for item in wanted}}
    state = {'stage':'complete','utc':result['observed_at'],'job':'repeatability_retry','kernel_version':exact,
        'status':'COMPLETE','observed_charge_seconds':q['used_seconds']-read(directory / 'push_intent.json')['quota_before']['used_seconds'],
        'transport_recovered':True}
    for destination in [directory,public]:
        json_write(destination / 'retrieval_recovery.json',result)
        json_write(destination / 'quota.json',q)
        json_write(destination / 'controller_state.json',state)
    print(json.dumps({key:value for key,value in result.items() if key!='artifact_sha256'},indent=2))


if __name__ == '__main__':main()
