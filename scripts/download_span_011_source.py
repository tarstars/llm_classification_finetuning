"""Download only the seven pinned public source shards with size/hash/deadline guards."""
from datetime import datetime,timedelta,timezone
from pathlib import Path
import json
import signal
import sys

import requests

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,write_intent,exclusive_lock,download_output,sha256,utc_now


def main():
    root=ROOT/'runs/span_011';private=root/'source_audit';data=root/'source_data';pub=REPO/'results/span_011'
    private.mkdir(parents=True,exist_ok=True);data.mkdir(parents=True,exist_ok=True)
    registry=read_json(REPO/'backlog/hypotheses.json')
    assert registry['active_version']==11
    version=next(v for v in registry['versions'] if v['number']==11)
    assert version['focus_ids']==['H59']
    audit=read_json(pub/'source_metadata_audit.json');assert audit['metadata_stage_passed']
    source=audit['source'];revision=source['revision'];name=source['dataset']
    expected={f'data/train-{i:05d}-of-00007.parquet' for i in range(7)}
    assert {p for p in source['files'] if p.endswith('.parquet')}==expected
    with exclusive_lock(private/'download.lock'):
        intent_path=private/'download_intent.json'
        if not intent_path.exists():
            write_intent(intent_path,dict(started_at=utc_now(),revision=revision,
                deadline=(datetime.now(timezone.utc)+timedelta(minutes=30)).isoformat(),
                maximum_compressed_bytes=2_000_000_000))
        intent=read_json(intent_path);assert intent['revision']==revision
        remaining=(datetime.fromisoformat(intent['deadline'])-datetime.now(timezone.utc)).total_seconds()
        assert remaining>0,'Review expired source-preparation budget; never reset it implicitly'
        def expired(*args):raise TimeoutError('Original source-download deadline reached')
        signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,remaining)
        record=dict(at=utc_now(),complete=False,revision=revision,files=[],ready_for_gpu=False,
            source_rows_parsed=0,source_rows_scored=0,new_gpu_executions=0,deadline=intent['deadline'])
        def save():
            record['at']=utc_now();atomic_json(pub/'source_download.json',record)
            atomic_json(private/'download_receipt.json',record)
        try:
            url=f'https://huggingface.co/api/datasets/{name}/tree/{revision}/data'
            response=requests.get(url,timeout=(10,45));response.raise_for_status();items=response.json()
            assert {x['path'] for x in items if x['type']=='file'}==expected
            total=sum(x['size'] for x in items);assert total<=intent['maximum_compressed_bytes']
            record['expected_bytes']=total;save()
            for item in sorted(items,key=lambda x:x['path']):
                assert item['type']=='file' and item['path'] in expected
                digest=item['lfs']['oid'].removeprefix('sha256:')
                assert len(digest)==64 and all(c in '0123456789abcdef' for c in digest)
                target=data/Path(item['path']).name
                reused=target.exists() and target.stat().st_size==item['size'] and sha256(target)==digest
                if not reused:
                    stable=f'https://huggingface.co/datasets/{name}/resolve/{revision}/{item["path"]}?download=true'
                    result=download_output(stable,target,expected_size=item['size'])
                    assert result['sha256']==digest,'Pinned source shard hash mismatch'
                record['files'].append(dict(name=target.name,bytes=item['size'],sha256=digest,reused=reused))
                save();print(json.dumps(dict(files=len(record['files']),total_files=7,
                    verified_bytes=sum(x['bytes'] for x in record['files']),row_scoring_started=False)),flush=True)
            record.update(complete=True,total_bytes=sum(x['bytes'] for x in record['files']));save()
        except Exception as error:
            record.update(error_type=type(error).__name__,complete=False);save()
            raise RuntimeError('Pinned source download failed; preserve partial files and resume original intent after inspection') from None
        finally:signal.setitimer(signal.ITIMER_REAL,0)


if __name__=='__main__':main()
