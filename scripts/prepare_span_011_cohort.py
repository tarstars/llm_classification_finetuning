"""Stream later votes into fixed, label-blind length-matched calibration groups."""
from collections import Counter
from datetime import datetime,timezone,timedelta
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import resource
import sqlite3
import sys
import time

os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,write_intent,exclusive_lock,sha256,utc_now
from arena.span011_data import parse_pair,pair_key,Groups,digest,split_selected,write_inference_inputs,LABELS

TEXT=['prompt','response_a','response_b']


def main():
    torch.set_num_threads(2);pa.set_cpu_count(2);pa.set_io_thread_count(2)
    root=ROOT/'runs/span_011';pub=REPO/'results/span_011';out=root/'cohort';out.mkdir(parents=True,exist_ok=True)
    download=read_json(pub/'source_download.json');assert download['complete'] and len(download['files'])==7
    meta=read_json(pub/'source_metadata_audit.json');assert meta['metadata_stage_passed']
    assert not list((root/'jobs').glob('*/push_intent.json')),'Preserve the cohort after any accepted GPU source'
    registry=read_json(REPO/'backlog/hypotheses.json');assert registry['active_version']==11
    assert next(i for i in registry['items'] if i['id']=='H59')['status']=='running'
    with exclusive_lock(out/'preparation.lock'):
        if (pub/'cohort_manifest.json').exists() and read_json(pub/'cohort_manifest.json').get('complete'):
            prior=read_json(pub/'cohort_manifest.json')
            for name,d in prior['artifact_sha256'].items():assert sha256(out/name)==d
            print('Verified existing immutable cohort; no reselection');return
        assert not (out/'rows.sqlite').exists(),'Inspect partial preparation before replay; do not reset its budget'
        intent=out/'preparation_intent.json'
        write_intent(intent,dict(started_at=utc_now(),deadline=(datetime.now(timezone.utc)+timedelta(minutes=30)).isoformat(),
            source_revision=download['revision'],script_sha256=sha256(Path(__file__))))
        deadline=datetime.fromisoformat(read_json(intent)['deadline']).timestamp()
        start=time.monotonic();stats=Counter();groups=Groups()
        def bounded():
            assert time.time()<deadline,'Original preparation deadline exceeded; review before another attempt'
            assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<6*1024**3,'Memory target exceeded'
        record=dict(at=utc_now(),complete=False,ready_for_gpu=False,source_revision=download['revision'],uses_targets_for_selection=False)
        def progress(stage,**values):
            atomic_json(pub/'cohort_progress.json',dict(at=utc_now(),stage=stage,seconds=time.monotonic()-start,
                counts=dict(stats),**values));print(json.dumps(dict(stage=stage,counts=dict(stats),**values)),flush=True)
        try:
            known=set();competition_ids=set()
            for name in ['train','test']:
                frame=pd.read_csv(ROOT/'raw'/(name+'.csv'),usecols=['id',*TEXT]);competition_ids.update(frame.id.tolist())
                for p,a,b in frame[TEXT].itertuples(index=False,name=None):known.add(pair_key(json.loads(p),json.loads(a),json.loads(b)))
            run9=ROOT/'runs/span_009';model=run9/'research/lmsys-checkpoints-0-0805'
            module=run9/'research/lmsys-modules-0805/data/processors.py'
            expected=read_json(run9/'jobs/public_ensemble_resume/output/public_ensemble/m0_checkpoint.json')
            length_audit=read_json(REPO/'results/span_009/full_pab_length_audit.json')
            assert sha256(model/'tokenizer.json')==expected['tokenizer.json']
            assert sha256(module)==length_audit['processor_sha256']
            spec=importlib.util.spec_from_file_location('original_pab_for_h59',module)
            author=importlib.util.module_from_spec(spec);spec.loader.exec_module(author)
            processor=author.ProcessorPAB(AutoTokenizer.from_pretrained(model,local_files_only=True),4096,False)
            def encode(row):
                return processor.build_input({**{c:row[c] for c in TEXT},'winner_model_a':1,'winner_model_b':0,'winner_tie':0})['input_ids'].numpy()
            control_ids=read_json(REPO/'results/span_009/profile_selection.json')['uniform_ids']
            control=pd.read_csv(ROOT/'raw/train.csv',usecols=['id',*TEXT]).set_index('id').loc[control_ids]
            fingerprint=hashlib.sha256()
            for row in control.to_dict('records'):fingerprint.update(encode(row).tobytes())
            expected_fp=read_json(run9/'jobs/public_ensemble_resume/output/public_ensemble/m0_summary.json')['cohorts']['uniform']['token_ids_sha256']
            assert fingerprint.hexdigest()==expected_fp,'CPU input tokens differ from accepted GPU control'
            record.update(cpu_gpu_token_identity=True,processor_sha256=sha256(module),tokenizer_sha256=expected['tokenizer.json'])
            connection=sqlite3.connect(out/'rows.sqlite')
            connection.execute('CREATE TABLE rows (uid TEXT PRIMARY KEY, numeric_id INTEGER UNIQUE, priority TEXT, prompt_group TEXT, session TEXT, payload TEXT, winner TEXT)')
            columns=['id','evaluation_session_id','evaluation_order','language','timestamp','conversation_a','conversation_b','winner']
            for item in download['files']:
                path=root/'source_data'/item['name'];assert sha256(path)==item['sha256']
                file=pq.ParquetFile(path)
                for batch in file.iter_batches(batch_size=256,columns=columns):
                    for raw in batch.to_pylist():
                        stats['source_rows']+=1
                        row,reason=parse_pair(raw)
                        if reason:stats['reject_'+reason]+=1;continue
                        if row['pair_hash'] in known:stats['reject_known_pair']+=1;continue
                        assert row['id'] not in competition_ids,'Synthetic ID collides with competition IDs'
                        groups.join('p:'+row['prompt_group'],'s:'+row['session'])
                        connection.execute('INSERT INTO rows VALUES (?,?,?,?,?,?,?)',
                            (row['source_uuid'],row['id'],row['row_priority'],row['prompt_group'],row['session'],
                             json.dumps(row,ensure_ascii=False),raw.get('winner')))
                        stats['eligible_rows']+=1
                    connection.commit();bounded()
                progress('source_shard_audited',shard=item['name'])
            assert stats['source_rows']==135634
            connection.execute('CREATE TABLE representatives (group_id TEXT PRIMARY KEY, uid TEXT UNIQUE, sample_priority TEXT)')
            seen=set()
            for uid,prompt in connection.execute('SELECT uid,prompt_group FROM rows ORDER BY priority,uid'):
                group=groups.find('p:'+prompt)
                if group in seen:continue
                seen.add(group)
                connection.execute('INSERT INTO representatives VALUES (?,?,?)',(group,uid,digest('42:sample:'+group)))
            connection.commit();stats['unique_groups']=len(seen)
            bins=np.asarray(meta['target_length_bins']);quota=np.asarray(meta['cohort_bin_quotas']);filled=np.zeros(len(quota),dtype=int)
            selected=[]
            query='SELECT r.payload,r.winner,p.group_id FROM representatives p JOIN rows r ON r.uid=p.uid ORDER BY p.sample_priority,p.uid'
            for payload,winner,group in connection.execute(query):
                row=json.loads(payload);length=len(encode(row));stats['groups_tokenized']+=1
                index=int(np.searchsorted(bins,length,side='right')-1)
                assert 0<=index<len(quota),'Prepared input length outside frozen bins'
                if filled[index]<quota[index]:
                    row.update(group_id=group,length_bin=index,prepared_tokens=length,private_winner=winner)
                    selected.append(row);filled[index]+=1
                if stats['groups_tokenized']%1000==0:progress('matching_length_bins',selected_rows=len(selected),bin_counts=filled.tolist())
                bounded()
                if np.array_equal(filled,quota):break
            assert np.array_equal(filled,quota),'Insufficient source support; revise prospectively before scoring'
            selected=split_selected(selected,meta['learning_bin_quotas'])
            assert len({r['group_id'] for r in selected})==2048 and len({r['id'] for r in selected})==2048
            artifacts=[];split_ids={};label_hashes={}
            for name in ['learning','confirmation']:
                subset=sorted([r for r in selected if r['split']==name],key=lambda r:r['id']);assert len(subset)==1024
                inputs=pd.DataFrame([{k:r[k] for k in ['id',*TEXT]} for r in subset])
                write_inference_inputs(out/(name+'.parquet'),subset)
                # Targets are first interpreted only after fixed selection/splitting.
                labels=np.asarray([LABELS[r['private_winner']] for r in subset],dtype=np.int8)
                np.savez_compressed(out/(name+'_labels.npz'),ids=inputs.id.to_numpy(),labels=labels,
                    groups=np.asarray([r['group_id'] for r in subset]))
                mapping=pd.DataFrame([{k:r[k] for k in ['id','source_uuid','session','prompt_group','group_id','timestamp','prepared_tokens','length_bin']} for r in subset])
                mapping.to_parquet(out/(name+'_mapping.parquet'),index=False)
                artifacts.extend([name+'.parquet',name+'_labels.npz',name+'_mapping.parquet'])
                split_ids[name]=inputs.id.tolist();label_hashes[name]=sha256(out/(name+'_labels.npz'))
            connection.close()
            stamps=[r['timestamp'] for r in selected]
            record.update(complete=True,data_ready=True,ready_for_gpu=False,rows=2048,learning_rows=1024,confirmation_rows=1024,
                counts=dict(stats),selected_bin_counts=filled.tolist(),learning_bin_counts=meta['learning_bin_quotas'],
                confirmation_bin_counts=meta['confirmation_bin_quotas'],split_ids=split_ids,label_sha256=label_hashes,
                confirmation_labels_inspected=False,selected_timestamp_range=[min(stamps),max(stamps)],
                selected_mean_prepared_tokens=float(np.mean([r['prepared_tokens'] for r in selected])),
                source_download_manifest_sha256=sha256(pub/'source_download.json'),
                train_input_sha256=sha256(ROOT/'raw/train.csv'),visible_test_sha256=sha256(ROOT/'raw/test.csv'),
                script_sha256=sha256(Path(__file__)),helpers_sha256=sha256(REPO/'src/arena/span011_data.py'),
                elapsed_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                artifact_sha256={name:sha256(out/name) for name in artifacts})
            atomic_json(pub/'cohort_manifest.json',record);atomic_json(out/'manifest.json',record)
            progress('cohort_frozen',rows=2048,learning=1024,confirmation=1024,labels_inspected=False)
        except Exception as error:
            record.update(complete=False,error_type=type(error).__name__,counts=dict(stats))
            atomic_json(pub/'cohort_failure.json',record)
            raise RuntimeError('Cohort preparation failed; inspect private state before replay') from None


if __name__=='__main__':main()
