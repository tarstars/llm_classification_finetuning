"""Admit or observe one exact H56 public version; never repeat accepted mutations."""
from pathlib import Path
import argparse
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT
from arena.experiment import atomic_json, read_json, sha256, utc_now, verify_sources
from arena.kaggle_control import run_job
from arena.span004_support import api_client, read_call
from submit_span_013_order import submissions, TERMINAL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root, pub = ROOT / 'runs/span_016', REPO / 'results/span_016'
    def precondition(role):
        registry = read_json(REPO / 'backlog/hypotheses.json')
        assert registry['active_version'] == 16
        assert next(v for v in registry['versions'] if v['number'] == 16)['focus_ids'] == ['H56']
        assert next(i for i in registry['items'] if i['id'] == 'H56')['status'] == 'running'
        assert role == 'reward_pilot'
        bundle = read_json(pub / 'reward_pilot_bundle.json')
        assert bundle['public_stage'] == 'information_only' and not bundle['official_submission_eligible']
        assert read_json(pub / 'source_audit.json')['source_stage_passed']
        assert read_json(pub / 'reward_pilot_cell_rehearsals.json')['passed']
        asset = read_json(pub / 'asset_upload.json')
        assert asset['ready'] and asset['private_verified'] and asset['all_weight_hashes_verified']
        assert asset['asset_manifest_sha256'] == sha256(pub / 'asset_manifest.json')
        inputs = read_json(pub / 'matched_input_upload.json')
        assert inputs['ready'] and inputs['private_verified']
        assert inputs['input_manifest_sha256'] == sha256(pub / 'matched_input_manifest.json')
        assert read_json(pub / 'asset_preparation.json')['all_expected_weight_hashes_verified']
        assert len(list((root / 'jobs').glob('*/push.json'))) < 2
        verify_sources(REPO, {'scripts/run_span_016_pilot.py': sha256(Path(__file__)),
            'src/arena/kaggle_control.py': sha256(REPO / 'src/arena/kaggle_control.py')})
        api = api_client()
        from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
        sources = ['tascj0/' + s + '/1' for s in ['lmsys-checkpoints-0-0805',
            'lmsys-checkpoints-3-0805', 'lmsys-modules-0805', 'lmsys-packages']]
        sources.append(asset['dataset'])
        sources.append(inputs['dataset'])
        dependencies = []
        for reference in sources:
            owner, slug, version = reference.split('/')
            assert version == '1'
            request = ApiGetDatasetRequest()
            request.owner_slug, request.dataset_slug = owner, slug
            with api.build_kaggle_client() as client:
                dataset = read_call(lambda: client.datasets.dataset_api_client.get_dataset(request))
            assert dataset.current_version_number == 1
            assert dataset.is_private == (owner == 'arseniytrushin')
            dependencies.append(dict(source=reference, private=dataset.is_private, version=1))
        history = submissions(api)
        pending = [str(r.ref) for r in history if str(r.status).split('.')[-1] not in TERMINAL]
        assert not pending, 'Official scoring is pending'
        inventory = read_json(pub / 'jobs/reward_pilot/admission_inventory.json')
        assert inventory['quota']['remaining_seconds'] >= 3600 + 32400
        assert not inventory['nonterminal_jobs'] and not inventory['quota']['paid_scaling']
        atomic_json(pub / 'reward_pilot_admission.json', dict(at=utc_now(), admitted=True,
            dependencies=dependencies, all_official_receipts=len(history), pending_official_receipts=pending,
            controller_sha256=sha256(Path(__file__)), bundle_sha256=sha256(pub/'reward_pilot_bundle.json'),
            public_seconds_cap=3600, phase_seconds_cap=18000, quality_recipes=1))
    run_job(REPO, root, pub, REPO / 'configs/span_016.json', 'reward_pilot',
        dry_run=args.dry_run, precondition=precondition)


if __name__ == '__main__':
    main()
