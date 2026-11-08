"""Acquire pinned H65 weights and expand their fixed quantization, without GPUs."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import shutil
import sys

import numpy as np
import requests
from safetensors import safe_open

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT
from arena.dense_checkpoint import expand_checkpoint
from arena.experiment import atomic_json, download_output, exclusive_lock, read_json, sha256, utc_now, verify_sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['acquire', 'expand'])
    args = parser.parse_args()
    pub, root = REPO / 'results/span_019', ROOT / 'runs/span_019'
    assets, audit = root / 'assets', root / 'source_audit'
    origin, dense = assets / 'gptq_origin', assets / 'dense'
    assets.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(assets / 'controller.lock'):
        plan = read_json(pub / 'span_plan.json')
        assert datetime.now(timezone.utc) < datetime.fromisoformat(plan['deadline'])
        observation = read_json(pub / 'source_observation.json')
        controls = read_json(pub / 'native_cpu_controls.json')
        assert observation['structural_source_checks_passed'] and controls['passed']
        assert controls['selection_sha256'] == sha256(pub / 'evaluation_selection.json')
        sources = {name:sha256(REPO / name) for name in ['scripts/stage_span_019_weights.py',
            'src/arena/dense_checkpoint.py', 'src/arena/gptq8.py', 'docs/SPAN_019_ASSET_PROTOCOL.md',
            'results/span_019/native_cpu_controls.json', 'results/span_019/evaluation_selection.json']}
        verify_sources(REPO, sources)
        for name, digest in observation['source_files_sha256'].items():
            assert sha256(audit / name) == digest
        index = read_json(audit / 'model.safetensors.index.json')
        headers = {}
        for name in sorted(set(index['weight_map'].values())):
            headers.update({k:v for k,v in read_json(audit/'headers'/(name+'.json')).items() if k != '__metadata__'})
        projections = [k.removesuffix('.qweight') for k in headers if k.endswith('.qweight')]
        consumed = {p+s for p in projections for s in ['.qweight','.qzeros','.scales','.g_idx']}
        dense_bytes = sum(v['data_offsets'][1]-v['data_offsets'][0] for k,v in headers.items() if k not in consumed)
        dense_bytes += sum(headers[p+'.qweight']['shape'][0]*headers[p+'.qweight']['shape'][1]*8 for p in projections)
        assert len(projections) == 294 and dense_bytes < 19*10**9
        reserve = 10*1024**3
        if args.action == 'acquire':
            assert not (pub / 'weight_acquisition.json').exists(), 'Preserve completed acquisition'
            origin.mkdir(exist_ok=True)
            additional = 0
            for item in observation['weight_files']:
                target = origin / item['name']
                available = target.stat().st_size if target.exists() else (
                    target.with_name(target.name+'.part').stat().st_size if target.with_name(target.name+'.part').exists() else 0)
                additional += max(0, item['bytes']-available)
            free = shutil.disk_usage(root).free
            assert free >= additional + dense_bytes + reserve, 'Insufficient room for both copies and10GiBreserve'
            atomic_json(pub / 'storage_preflight.json', dict(at=utc_now(), free_bytes=free,
                raw_remaining_bytes=additional, dense_tensor_bytes=dense_bytes, required_reserve_bytes=reserve))
            records = []
            for item in observation['weight_files']:
                target = origin / item['name']
                print(json.dumps(dict(stage='acquiring_weight', file=item['name'], expected_bytes=item['bytes'])), flush=True)
                if not target.exists():
                    url = f"https://huggingface.co/{observation['repository']}/resolve/{observation['revision']}/{item['name']}"
                    try:
                        download_output(url, target, expected_size=item['bytes'])
                    except requests.exceptions.RequestException as error:
                        code = getattr(getattr(error, 'response', None), 'status_code', None)
                        atomic_json(pub / 'weight_download_failure.json', dict(at=utc_now(), file=item['name'],
                            error_type=type(error).__name__, http_status=code, resumable_read_only=True))
                        raise SystemExit('Weight read failed; exact partial file retained. No Kaggle mutation.') from None
                assert target.stat().st_size == item['bytes'] and sha256(target) == item['sha256']
                records.append(item)
                atomic_json(assets / 'acquisition_progress.json', dict(at=utc_now(), verified=records))
                print(json.dumps(dict(stage='weight_verified', file=item['name'])), flush=True)
            for name in ['config.json','model.safetensors.index.json','tokenizer.json','tokenizer.model',
                         'tokenizer_config.json','special_tokens_map.json','README.md','Readme.md']:
                target = origin / name
                if not target.exists(): os.link(audit/name,target)
                assert sha256(target) == observation['source_files_sha256'][name]
            atomic_json(pub / 'weight_acquisition.json', dict(at=utc_now(), complete=True,
                repository=observation['repository'], revision=observation['revision'], weights=records,
                total_weight_file_bytes=sum(i['bytes'] for i in records), dense_tensor_bytes=dense_bytes,
                source_sha256=sources, new_gpu_executions=0, quality_measured=False))
            return
        assert read_json(pub / 'weight_acquisition.json')['complete']
        assert not (pub / 'dense_expansion.json').exists(), 'Preserve completed expansion'
        assert shutil.disk_usage(root).free >= dense_bytes + reserve
        for item in observation['weight_files']:
            assert sha256(origin/item['name']) == item['sha256']
        print(json.dumps(dict(stage='expanding_fixed_GPTQ8', dense_tensor_bytes=dense_bytes)), flush=True)
        result = expand_checkpoint(origin, dense)
        assert result['total_tensor_bytes'] == dense_bytes and result['tensor_keys'] == 465
        for name in ['tokenizer.json','tokenizer.model','tokenizer_config.json','special_tokens_map.json','README.md','Readme.md']:
            os.link(origin/name,dense/name)
        config = read_json(origin / 'config.json')
        config.pop('quantization_config')
        config['pad_token_id'] = 0
        config['_name_or_path'] = observation['repository']
        atomic_json(dense / 'config.json', config)
        atomic_json(dense / 'conversion.json', dict(repository=observation['repository'], revision=observation['revision'],
            source_quantization='GPTQ8fixed', newly_quantized=False, runtime_native_layout_required=True,
            projection_names=result['projection_names'], original_head_unchanged=True))
        converted_index = read_json(dense/'model.safetensors.index.json')
        with safe_open(dense/converted_index['weight_map']['score.weight'], framework='np') as tensors:
            head = tensors.get_tensor('score.weight')
            assert head.tobytes() == (audit/'score.weight.f16').read_bytes()
        file_hashes = {p.name:dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in sorted(dense.iterdir()) if p.is_file()}
        result.update(at=utc_now(), complete=True, source_sha256=sources, files=file_hashes,
            head_bytes_identical=True, new_gpu_executions=0, quality_measured=False, gpu_admitted=False)
        atomic_json(pub / 'dense_expansion.json', result)
        print(json.dumps({k:result[k] for k in ['complete','tensor_keys','total_tensor_bytes','head_bytes_identical','gpu_admitted']}))


if __name__ == '__main__':
    main()
