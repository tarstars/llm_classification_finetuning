"""Read one bounded exact-version stream; never mutate or restart GPU work."""
from pathlib import Path
import argparse
import json
import signal
import sys

import requests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT
from arena.experiment import atomic_json, read_json, sha256, utc_now
from arena.span004_support import api_client, read_call


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=int, default=25)
    args = parser.parse_args()
    assert 1 <= args.seconds <= 45
    observer_source = sha256(Path(__file__))
    root, pub = ROOT/'runs/span_016', REPO/'results/span_016'
    pushed = read_json(root/'jobs/reward_pilot/push.json')
    exact = pushed['ref'].removeprefix('/code/')+'/'+str(pushed['version_number'])
    api = api_client()
    owner, slug, version = api.parse_kernel_string(exact)
    events, limited = [], None
    def deadline(*unused):
        raise TimeoutError('Bounded log observation finished')
    previous = signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.seconds)
    try:
        with api.build_kaggle_client() as client:
            http = client._http_client
            http._init_session()
            headers = dict(http._session.headers)
            headers.update(Accept='text/event-stream, */*', **{'Accept-Encoding':'identity'})
            headers.pop('Content-Type', None)
            with http._session.get(http._endpoint+f'/v1/kernels/logs/stream/{owner}/{slug}',
                    stream=True, headers=headers, auth=http._session.auth,
                    params={'versionLabel':api._kernel_version_label(version)}, timeout=(10,args.seconds)) as response:
                response.raise_for_status()
                if response.headers.get('Content-Type','').startswith('text/event-stream'):
                    for line in response.iter_lines(chunk_size=1,decode_unicode=True):
                        if not line.startswith('data:'):
                            continue
                        payload = line[5:].strip()
                        if payload == api._LOG_STREAM_END_SENTINEL:
                            break
                        try:
                            events.append(json.loads(payload))
                        except json.JSONDecodeError:
                            events.append({'data':payload})
                else:
                    events.extend(api._iter_blob_lines(response))
    except (TimeoutError, requests.exceptions.RequestException) as error:
        limited = type(error).__name__
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM,previous)
    observation = root/'observations'
    observation.mkdir(exist_ok=True)
    atomic_json(observation/'latest_log_snapshot.json',dict(at=utc_now(),exact=exact,
        events=events,observation_limited_by=limited))
    status = str(read_call(lambda:api.kernels_status(exact)).status).split('.')[-1]
    fields = {'stage','role','size','cohort','seconds','rows','batch','completed_sequences','total_sequences'}
    stages = {'reward_progress','inference_progress','cohort_complete','model_loading','model_ready','complete'}
    progress = []
    errors = 0
    for event in events:
        for line in str(event.get('data','')).splitlines():
            errors += int('Traceback (most recent call last)' in line)
            if not line.startswith('ARENA_EVENT '):
                continue
            try:
                value = json.loads(line.removeprefix('ARENA_EVENT '))
            except json.JSONDecodeError:
                continue
            if value.get('stage') in stages:
                progress.append({k:v for k,v in value.items() if k in fields})
    latest = {}
    for value in progress:
        latest[(value.get('role',value.get('size')),value.get('cohort'))] = value
    result = dict(at=utc_now(),kernel_version=exact,status=status,stream_events=len(events),
        observation_limited_by=limited,tracebacks_in_stream=errors,
        latest_stage=progress[-1] if progress else None,latest_worker_cohorts=list(latest.values()),
        quality_audited_by_this_observer=False,target_runtime_profiled_by_this_observer=False,
        observer_sha256=observer_source,new_gpu_executions=0)
    atomic_json(pub/'pilot_live_observation.json',result)
    print(json.dumps({k:result[k] for k in ['at','kernel_version','status','stream_events',
        'observation_limited_by','tracebacks_in_stream','latest_stage']},indent=2))


if __name__ == '__main__':
    main()
