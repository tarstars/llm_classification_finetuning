"""Seal observed source facts, preserving absence/access limitations."""
from datetime import datetime,timezone
from pathlib import Path
import json
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,read_json,sha256,utc_now


def main():
    pub=REPO/'results/span_023';root=ROOT/'runs/span_023/source_audit'
    c=read_json(pub/'audit_contract.json');now=datetime.now(timezone.utc)
    assert now<datetime.fromisoformat(c['deadline'])
    for name,digest in {**c['snapshot_sha256'],**c['user_notebook_sha256']}.items():assert sha256(REPO/name)==digest
    a=read_json(pub/'source_listing.json');b=read_json(pub/'candidate_inventory.json');d=read_json(pub/'base_observation.json')
    assert a['complete_author_listing'] and a['listing_pages']==1
    lmsys=[x for x in a['author_datasets'] if '/lmsys-' in x]
    assert sorted(lmsys)==sorted('tascj0/'+s for s in ['lmsys-checkpoints-0-0805',
        'lmsys-checkpoints-3-0805','lmsys-modules-0805','lmsys-packages'])
    hf='rbiswasfc/lmsys-r-gemma-s1-mix110-merged-headless'
    assert b['errors'][hf]['http_status']==401 and not d['model_search']['base'].get('models')
    assert not b['listings']['fourth_checkpoint'] and not d['inaccessible_hf_base_retried']
    runtime=root/'utils-lmsys-v10/r_gemma/lmsys_model.py'
    related=read_json(root/'github_related.json')
    reference=related['third_gemma_model']['structuredContent']['content'].encode()
    adapter=read_json(root/'lmsys-r-gemma-m205-21000-lb8741/adapter_config.json')
    assert adapter['base_model_name_or_path']==hf and 'classification_head' in adapter['modules_to_save']
    files=sorted(p for p in root.rglob('*') if p.is_file());size=sum(p.stat().st_size for p in files)
    assert size<c['bytes_cap']
    r=dict(at=utc_now(),decision='no_qualifying_new_checkpoint_in_audited_sources',
        exhaustive_public_model_search=False,new_classifier_ready=False,
        tascj_additional_final_checkpoint_found=False,third_base_http_status=401,
        third_public_kaggle_base_matches=0,fourth_exact_checkpoint_search_matches=0,
        actual_runtime_equals_repository_reference=runtime.read_bytes()==reference,
        source_archive_bytes=size,source_archive_sha256={str(p.relative_to(root)):sha256(p) for p in files},
        elapsed_audit_seconds=(now-datetime.fromisoformat(c['started_at'])).total_seconds(),
        cap_seconds=3600,cap_bytes=c['bytes_cap'],weights_downloaded=False,new_gpu_executions=0,
        external_text_inference=0,official_submissions=0,hidden_prediction_outputs_used=False,
        next_discovery='Published fixed empty/identical-answer postprocessing, untested against H60',
        source_references=dict(tascj='83cd93d50b9283c18711e8c63e4e1c6399c7b9ce',
            third='4007b8dfdb4c19899252962ff7f690e5f7038c86',fourth='f4d2a3fbd0497e28673010787612d3dac04e7668'))
    assert not r['actual_runtime_equals_repository_reference']
    atomic_json(pub/'source_audit.json',r);print({k:v for k,v in r.items() if k!='source_archive_sha256'})


if __name__=='__main__':main()
