"""Freeze v6 inputs while loading no confirmation votes or source identities."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys
import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,TEXT_COLUMNS,parse_turns,prompt_groups,json_write
from arena.span006_inputs import read_inputs,text_digest


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    out=ROOT/'runs/span_006';pub=REPO/'results/span_006';out.mkdir(parents=True,exist_ok=True)
    assert not (out/'confirmation_inputs.parquet').exists(),'Preparation is immutable'
    cfg=json.loads((REPO/'configs/span_006.json').read_text());old=json.loads((REPO/'results/span_005/artifact_inventory.json').read_text())
    fields=['classifier','adapter','split','raw_train','candidate','train.parquet','dev.parquet','source_vocabulary.json']
    entries={key:old['input_artifacts'][key] for key in fields}
    for key,item in entries.items():assert sha(item['path'])==item['sha256'],key
    split=json.loads(Path(entries['split']['path']).read_text())
    confirm=read_inputs(entries['raw_train']['path'],split['confirmation'])
    assert len(confirm)==5746 and confirm.id.is_unique and set(confirm.columns)=={'id',*TEXT_COLUMNS}
    train=pd.read_parquet(entries['train.parquet']['path']);dev=pd.read_parquet(entries['dev.parquet']['path'])
    assert not set(prompt_groups(confirm))&(set(prompt_groups(train))|set(prompt_groups(dev)))
    confirm.to_parquet(out/'confirmation_inputs.parquet',index=False)
    samples=json.loads((ROOT/'runs/span_003/sample_ids.json').read_text())['classifier_smoke']
    cache=np.load(entries['classifier']['path']);lookup={int(i):j for j,i in enumerate(cache['ids'])};idx=[lookup[int(i)] for i in samples]
    np.savez(out/'smoke_reference.npz',ids=np.array(samples),original=cache['original'][idx],aligned_swap=cache['aligned_swap'][idx])
    ids={'smoke':samples,'confirmation':list(map(int,confirm.id))};json_write(out/'sample_ids.json',ids)
    for key,path in {'strength':ROOT/'runs/span_005/E06/strength.npz','classifier_manifest':REPO/'results/span_002/research_candidate.json',
                     'training_selection':ROOT/'runs/attempt_002/model_selection.json','prior_source_config':REPO/'configs/span_005.json',
                     'confirmation_inputs':out/'confirmation_inputs.parquet','smoke_reference':out/'smoke_reference.npz','sample_ids':out/'sample_ids.json'}.items():
        entries[key]={'path':str(path),'sha256':sha(path)}
    confirm_digest=text_digest(confirm)
    initial=json.loads((pub/'resource_inventory_initial.json').read_text());assert not initial['nonterminal_jobs'] and initial['quota']['reserved_seconds']==0
    cap=min(cfg['gpu_session_seconds_cap'],initial['quota']['remaining_seconds'])
    assert cap>=sum(cfg['job_seconds_caps'].values())
    sources=['GOAL.md','configs/span_006.json','docs/SPAN_006_PROTOCOL.md','scripts/prepare_span_006.py','scripts/freeze_span_006.py',
             'src/arena/span006_source.py','src/arena/span006_inputs.py','src/arena/source_models.py','src/arena/source_evaluation.py','src/arena/span_heads.py','src/arena/data.py','src/arena/joint_data.py',
             'scripts/run_span_005_e07.py']
    result={'span':6,'started_at':cfg['started_at'],'deadline':cfg['deadline'],'clock_source':'First clock observation executing explicit user task; no new goal-service activation',
            'focus':cfg['focus'],'train_rows':len(train),'dev_rows':len(dev),'confirmation_rows':len(confirm),'confirmation_groups':int(prompt_groups(confirm).nunique()),
            'confirmation_votes_read':False,'confirmation_source_names_read':False,'confirmation_text_digest':confirm_digest,
            'input_artifacts':entries,'source_sha256':{p:sha(REPO/p) for p in sources},
            'old_snapshot_sha256':{str(p.relative_to(REPO)):sha(p) for p in (REPO/'backlog/versions').glob('*') if p.is_file()},
            'user_notebook_sha256':{p:sha(REPO/p) for p in ['notebooks/07_error_poster.ipynb','notebooks/12_backlog_v1_span.ipynb']},
            'gpu_session_seconds_cap':cap,'quota_used_at_start':initial['quota']['used_seconds'],'initial_quota_refresh_time':initial['quota']['refresh_time'],
            'prepared_at':datetime.now(timezone.utc).isoformat()}
    for folder in [pub,out]:json_write(folder/'artifact_inventory.json',result)
    for folder in [pub,out]:json_write(folder/'span_plan.json',{k:result[k] for k in ['span','started_at','deadline','focus','gpu_session_seconds_cap','quota_used_at_start','initial_quota_refresh_time']})
    json_write(pub/'activation.json',{'span':6,'started_at':cfg['started_at'],'deadline':cfg['deadline'],'clock_source':result['clock_source'],'goal_contract_sha256':sha(REPO/'GOAL.md')})
    print(json.dumps({k:result[k] for k in ['span','train_rows','dev_rows','confirmation_rows','confirmation_groups','gpu_session_seconds_cap','confirmation_votes_read']},indent=2))


if __name__=='__main__':main()
