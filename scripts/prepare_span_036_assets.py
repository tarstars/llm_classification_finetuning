"""Prepare exact native inputs, acquire one pinned model, and stage one asset."""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import requests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT
from arena.experiment import atomic_json, read_json, sha256, utc_now, download_output, write_intent, exclusive_lock


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['small', 'weights', 'upload', 'verify'])
    args = parser.parse_args(); start = time.monotonic()
    run, pub = ROOT / 'runs/span_036', REPO / 'results/span_036'
    cfg = read_json(REPO / 'configs/span_036.json')
    audit = run / 'source_audit'; assets = run / 'input_asset'; upload = assets / 'upload'
    assets.mkdir(parents=True, exist_ok=True)
    assert read_json(pub / 'source_audit.json')['source_stage_passed']
    ref = 'arseniytrushin/llm-span036-pairrm-inputs'
    with exclusive_lock(assets / 'controller.lock'):
        if args.action == 'small':
            import numpy as np
            import pandas as pd
            import torch
            import transformers
            from transformers import AutoTokenizer
            from arena.span036_pairrm import native_definitions, records, token_proof
            assert not (pub / 'preparation.json').exists()
            assert transformers.__version__ == '4.42.3'
            torch.set_num_threads(2); upload.mkdir(exist_ok=True)
            source = read_json(pub / 'source_retrieval.json')
            for f in source['files']: assert sha256(Path(f['path'])) == f['sha256']
            assert source['model_revision'] == cfg['model_revision']
            for name in ['tokenizer.json', 'spm.model', 'added_tokens.json', 'tokenizer_config.json', 'special_tokens_map.json']:
                target = upload / name
                if not target.exists():
                    response = requests.get(f"https://huggingface.co/llm-blender/PairRM/resolve/{cfg['model_revision']}/{name}", timeout=60)
                    response.raise_for_status(); target.write_bytes(response.content)
            for origin, target in [('code/llm_blender/pair_ranker/ranker.py', 'native_ranker.py'),
                ('code/llm_blender/pair_ranker/collator.py', 'native_collator.py'),
                ('code/llm_blender/blender/blender_utils.py', 'native_utils.py'),
                ('base/config.json', 'base_config.json'), ('model/config.json', 'ranker_config.json'),
                ('model/README.md', 'MODEL_CARD.md'), ('code/LICENSE', 'LICENSE')]:
                shutil.copyfile(audit / origin, upload / target)
            source_dir = ROOT / 'runs/span_016/cohort_matched'
            manifest = read_json(REPO / 'results/span_016/matched_cohort_manifest.json')
            assert sha256(source_dir / 'source.parquet') == manifest['artifact_sha256']['source.parquet']
            shutil.copyfile(source_dir / 'source.parquet', upload / 'source.parquet')
            frame = pd.read_parquet(upload / 'source.parquet')
            assert set(frame) == {'id', 'prompt', 'response_a', 'response_b'} and len(frame) == 1024
            old = read_json(REPO / 'results/span_034/preparation.json')
            assert sha256(ROOT / 'runs/span_034/inputs.npz') == old['inputs_sha256']
            original = np.load(ROOT / 'runs/span_034/inputs.npz')
            np.testing.assert_array_equal(original['ids'], frame.id)
            np.savez_compressed(run / 'baseline.npz', ids=frame.id, baseline=original['baseline'])
            _, collator_cls, get_pair = native_definitions(upload)
            tokenizer = AutoTokenizer.from_pretrained(upload, local_files_only=True, use_fast=True)
            collator = collator_cls(1224, tokenizer, 412)
            items = records(frame, get_pair)
            spec = dict(ids=frame.id.tolist(), batch_size=2, frame_sha256=sha256(upload / 'source.parquet'),
                ab=token_proof(items, collator, tokenizer), ba=token_proof(items, collator, tokenizer, True))
            assert spec['ab']['lengths'] == spec['ba']['lengths']
            atomic_json(upload / 'token_spec.json', spec)
            snapshots = sorted((REPO / 'backlog/versions').glob('*')); assert len(snapshots) == 142
            prior = read_json(REPO / 'results/span_035/preservation.json')
            for n, h in prior['user_notebook_sha256'].items(): assert sha256(REPO / n) == h
            atomic_json(pub / 'preservation.json', dict(at=utc_now(), snapshot_sha256={str(p.relative_to(REPO)):sha256(p) for p in snapshots}, user_notebook_sha256=prior['user_notebook_sha256']))
            quota = read_json(REPO / 'results/span_035/resource_close.json')['quota']
            atomic_json(pub / 'span_plan.json', dict(started_at=cfg['started_at'], deadline=cfg['deadline'],
                quota_used_at_start=quota['used_seconds'], initial_quota_refresh_time=quota['refresh_time'],
                last_verified_quota=quota, gpu_session_seconds_cap=5400, public_version_limit=2,
                official_candidate_limit=1, hidden_cap_seconds=32400, parent_span=35))
            paths = ['GOAL.md','docs/SPAN_036_PLAN.md','docs/SPAN_036_INFORMATION_CONTRACT.md','configs/span_036.json',
                'src/arena/span036_pairrm.py','scripts/prepare_span_036_assets.py','results/span_036/source_audit.json']
            result = dict(at=utc_now(), prepared=True, rows=1024, targets_loaded=False,
                native_token_proof_passed=True, native_input_max=max(spec['ab']['lengths']),
                mean_tokens=float(np.mean(spec['ab']['lengths'])), baseline_sha256=sha256(run / 'baseline.npz'),
                token_spec_sha256=sha256(upload / 'token_spec.json'),
                small_files_sha256={p.name:sha256(p) for p in upload.iterdir() if p.is_file()},
                source_sha256={p:sha256(REPO / p) for p in paths}, elapsed_seconds=time.monotonic()-start,
                runtime=dict(transformers=transformers.__version__,tokenizers=__import__('tokenizers').__version__))
            assert result['elapsed_seconds'] < cfg['cpu_preparation_seconds_cap']
            atomic_json(pub / 'preparation.json', result); print({k:v for k,v in result.items() if 'sha256' not in k}); return
        prep = read_json(pub / 'preparation.json')
        for n, h in prep['small_files_sha256'].items(): assert sha256(upload / n) == h
        if args.action == 'weights':
            assert not (pub / 'asset_preparation.json').exists()
            for n, h in prep['source_sha256'].items():
                assert sha256(REPO / n) == h
                import hashlib
                assert hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+n],cwd=REPO)).hexdigest() == h, 'Publish method before weights'
            assert shutil.disk_usage(run).free > 6 * 1024**3
            metadata = read_json(audit / 'model_metadata.json')
            item = next(x for x in metadata['siblings'] if x['rfilename'] == 'model.safetensors')
            target = upload / 'model.safetensors'
            if not target.exists():
                try:
                    receipt = download_output(f"https://huggingface.co/llm-blender/PairRM/resolve/{cfg['model_revision']}/model.safetensors",target,expected_size=item['size'])
                except requests.RequestException as error:
                    atomic_json(pub/'asset_download_failure.json',dict(at=utc_now(),error_type=type(error).__name__,http_status=getattr(getattr(error,'response',None),'status_code',None)))
                    raise SystemExit('Pinned acquisition failed; inspect sanitized failure and reconcile same partial file')
            assert target.stat().st_size == item['size'] and sha256(target) == item['lfs']['sha256']
            (upload / 'README.md').write_text('# Private frozen joint-comparator inputs\n\nPairRM: https://huggingface.co/llm-blender/PairRM (MIT); authors code: https://github.com/yuchenlin/LLM-Blender (MIT). Exact pinned files and original license/model card included. Base architecture config only; no base weights. Source texts are the existing fixed1024known external post2024Arena training conversations, CC BY4.0, https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k . No labels, model identities, hidden test data or predictions. Private transport; component terms retained.\n')
            atomic_json(upload/'dataset-metadata.json',dict(id=ref,title='LLM Span036 PairRM Inputs',licenses=[dict(name='other')],description='Private pinned MIT PairRM/model source and CC BY4 external text inputs; no target labels or hidden data.'))
            result = dict(at=utc_now(),prepared=True,dataset=ref+'/1',private_requested=True,
                files_sha256={p.name:sha256(p) for p in upload.iterdir()},weight_sha256=item['lfs']['sha256'],
                total_bytes=sum(p.stat().st_size for p in upload.iterdir()),new_gpu_executions=0)
            atomic_json(pub/'asset_preparation.json',result);print({k:v for k,v in result.items() if k!='files_sha256'});return
        from arena.span004_support import api_client, read_call
        api=api_client(); prepared=read_json(pub/'asset_preparation.json')
        for n,h in prepared['files_sha256'].items():assert sha256(upload/n)==h
        if args.action=='upload':
            assert not (assets/'upload_intent.json').exists(),'Reconcile same accepted/ambiguous upload'
            write_intent(assets/'upload_intent.json',dict(at=utc_now(),dataset=ref,private=True,manifest_sha256=sha256(pub/'asset_preparation.json')))
            response=api.dataset_create_new(str(upload),public=False,quiet=True,convert_to_csv=False)
            result={k:getattr(response,k) for k in ['ref','url','status','error'] if hasattr(response,k)}
            atomic_json(pub/'input_asset_response.json',result);assert not result.get('error');print(result);return
        from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
        request=ApiGetDatasetRequest();request.owner_slug,request.dataset_slug=ref.split('/')
        with api.build_kaggle_client() as client:d=read_call(lambda:client.datasets.dataset_api_client.get_dataset(request))
        status=json.loads(read_call(lambda:api.dataset_status(ref,format='json')))
        assert d.is_private and d.current_version_number==status['current_version_number']==1
        result=dict(at=utc_now(),dataset=ref+'/1',ready=status['status']=='ready',status=status['status'],private_verified=True)
        atomic_json(pub/'input_asset_ready.json',result);print(result)


if __name__=='__main__':main()
