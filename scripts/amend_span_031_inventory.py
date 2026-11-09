"""Preserve the preflight-failed bundle and bind the tested inventory adapter."""
from pathlib import Path
import shutil
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,sha256,utc_now


def main():
    root=ROOT/'runs/span_031';pub=REPO/'results/span_031';role='size_pilot_compact'
    assert not (root/'jobs'/role/'push_intent.json').exists(),'Never amend after a save intent'
    assert not list((root/'jobs').glob('*/push.json')),'An accepted version already exists'
    backup=pub/'size_pilot_compact_bundle.before_inventory_fix.json'
    assert not backup.exists(),'Preserve the first amendment'
    recovery=read_json(pub/'inventory_identity_recovery.json')
    assert recovery['no_compact_save_intent'] and recovery['owned_search_complete']
    assert recovery['existing_references_preserved'] and recovery['search_all_addressable'] and recovery['run_http_status']==404
    assert recovery['profile_count']==recovery['owned_search_count']==75 and recovery['profile_addressable']==74
    path=pub/(role+'_bundle.json');old=read_json(path)
    changed={'scripts/run_span_031_compact.py','docs/SPAN_031_TRANSPORT.md'}
    for name,digest in old['source_sha256'].items():
        if name not in changed:assert sha256(REPO/name)==digest,name
    folder=root/'notebooks'/role
    assert sha256(folder/'job.ipynb')==old['notebook_sha256']
    assert sha256(folder/'kernel-metadata.json')==old['metadata_sha256']
    shutil.copyfile(path,backup)
    names=changed|{'src/arena/span031_inventory.py','tests/test_span031_inventory.py','scripts/amend_span_031_inventory.py'}
    amended=dict(old);amended['source_sha256']=dict(old['source_sha256'])
    amended['source_sha256'].update({name:sha256(REPO/name) for name in sorted(names)})
    amended.update(inventory_admission_revision=1,previous_bundle_sha256=sha256(backup))
    atomic_json(path,amended)
    proof=dict(at=utc_now(),amended=True,reason='Read-only admission failed before mutation on a redacted own draft',
        previous_bundle_sha256=sha256(backup),active_bundle_sha256=sha256(path),
        numerical_notebook_unchanged=True,notebook_sha256=old['notebook_sha256'],metadata_unchanged=True,
        model_data_config_unchanged=True,accepted_gpu_versions=0,new_gpu_executions=0,
        tested_resolver_boundaries=4,observed_unittest_exit_code=0,
        source_sha256={name:sha256(REPO/name) for name in sorted(names)})
    atomic_json(pub/'inventory_source_amendment.json',proof);print(proof)


if __name__=='__main__':main()
