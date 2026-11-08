"""Pinned H65 small-file/tensor-header audit, with no complete weight download."""
from datetime import datetime, timezone
from pathlib import Path
import json
import struct
import sys
import time

import numpy as np
import requests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT
from arena.experiment import atomic_json, exclusive_lock, read_json, sha256, utc_now, verify_sources
from arena.remote_tensor import read_range


def main():
    pub, root = REPO / 'results/span_019', ROOT / 'runs/span_019/source_audit'
    plan = read_json(pub / 'span_plan.json')
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(root / 'gather.lock'):
        assert not (pub / 'source_observation.json').exists(), 'Preserve the original source observation'
        source_names = ['GOAL.md', 'docs/SPAN_019_PLAN.md', 'results/span_019/span_plan.json',
                        'scripts/gather_span_019_source.py', 'src/arena/remote_tensor.py']
        sources = {name: sha256(REPO / name) for name in source_names}
        verify_sources(REPO, sources)
        started = time.monotonic()
        total, files = 0, {}

        def check():
            assert total <= plan['source_audit_bytes_cap'], 'Small-source byte cap'
            assert datetime.now(timezone.utc) < datetime.fromisoformat(plan['source_audit_deadline'])

        def store(name, data):
            nonlocal total
            total += len(data)
            check()
            path = root / name
            assert not path.exists(), 'Reconcile partial audit, never overwrite its inputs'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            files[name] = sha256(path)
            return path

        def fetch(url, name):
            check()
            with requests.get(url, stream=True, timeout=(10, 45)) as response:
                response.raise_for_status()
                blocks, length = [], 0
                for block in response.iter_content(65536):
                    length += len(block)
                    assert length <= 20*1024**2 and total + length <= plan['source_audit_bytes_cap']
                    blocks.append(block)
            return store(name, b''.join(blocks))

        model, revision = plan['repository'], plan['revision']
        meta = read_json(fetch(f'https://huggingface.co/api/models/{model}/revision/{revision}?blobs=true', 'metadata.json'))
        assert meta['sha'] == revision and not meta.get('private') and not meta.get('gated')
        siblings = {f['rfilename']: f for f in meta['siblings']}
        for name in ['README.md', 'Readme.md', 'config.json', 'model.safetensors.index.json',
                     'tokenizer.json', 'tokenizer.model', 'tokenizer_config.json', 'special_tokens_map.json']:
            fetch(f'https://huggingface.co/{model}/resolve/{revision}/{name}', name)
        config, index = [read_json(root / name) for name in ['config.json', 'model.safetensors.index.json']]
        assert config['architectures'] == ['Gemma2ForSequenceClassificationV1']
        assert len(config['id2label']) == 3 and config['hidden_size'] == 3584 and config['num_hidden_layers'] == 42
        quant = config['quantization_config']
        assert quant['bits'] == 8 and quant['group_size'] == 128 and not quant['desc_act']
        observed_headers, remote_bytes = {}, []
        for name in sorted(set(index['weight_map'].values())):
            size = siblings[name]['size']
            assert siblings[name]['lfs']['size'] == size
            url = f'https://huggingface.co/{model}/resolve/{revision}/{name}'
            def part(start, end):
                data = read_range(url + f'?arena_range={start}-{end}', start, end, size)
                remote_bytes.append(dict(file=name, start=start, end=end, bytes=len(data)))
                return data
            length = struct.unpack('<Q', part(0, 7))[0]
            assert 0 < length <= 2**20
            header = json.loads(part(8, 7 + length))
            store('headers/' + name + '.json', json.dumps(header, sort_keys=True).encode())
            for tensor, info in header.items():
                if tensor != '__metadata__':
                    assert tensor not in observed_headers
                    assert index['weight_map'][tensor] == name
                    observed_headers[tensor] = info
            if name == index['weight_map']['score.weight']:
                head = header['score.weight']
                assert head['dtype'] == 'F16' and head['shape'] == [3, 3584]
                begin, end = head['data_offsets']
                raw = part(8 + length + begin, 8 + length + end - 1)
                store('score.weight.f16', raw)
        assert set(observed_headers) == set(index['weight_map'])
        for layer in range(42):
            for module in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                           'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']:
                prefix = f'model.layers.{layer}.{module}.'
                assert all(prefix + suffix in observed_headers for suffix in ['qweight', 'qzeros', 'scales', 'g_idx'])
        head = np.fromfile(root / 'score.weight.f16', dtype='<f2').reshape(3, 3584).astype(float)
        assert np.isfinite(head).all() and np.all(np.std(head, axis=1) > 0)
        assert not np.array_equal(head[0], head[1]) and not np.array_equal(head[0], head[2])
        for url, name in [
            ('https://raw.githubusercontent.com/AutoGPTQ/AutoGPTQ/v0.7.1/auto_gptq/nn_modules/qlinear/qlinear_cuda.py', 'native/qlinear_cuda_v0.7.1.py'),
            ('https://raw.githubusercontent.com/huggingface/transformers/v4.42.3/src/transformers/quantizers/quantizer_gptq.py', 'native/quantizer_gptq_v4.42.3.py'),
        ]:
            fetch(url, name)
        check()
        result = dict(at=utc_now(), repository=model, revision=revision,
            structural_source_checks_passed=True, tensor_keys=len(observed_headers),
            classifier_head_shape=list(head.shape), classifier_head_dtype='F16',
            classifier_head_finite=True, classifier_head_row_L2=np.linalg.norm(head, axis=1).tolist(),
            classifier_head_training_authenticated=False, exact_author_run_lineage_verified=False,
            declared_checkpoint_license=meta.get('cardData', {}).get('license'),
            config_parent=config['_name_or_path'], native_card_max_length=2000,
            pad_token_id=config['pad_token_id'], eos_token_id=config['eos_token_id'],
            quantization={k:v for k,v in quant.items() if k != 'dataset'},
            quantization_calibration_text_rows=len(quant.get('dataset', [])),
            declared_tensor_bytes=index['metadata']['total_size'],
            weight_files=[dict(name=name, bytes=siblings[name]['size'], sha256=siblings[name]['lfs']['sha256'])
                          for name in sorted(set(index['weight_map'].values()))],
            partial_weight_reads=remote_bytes, source_bytes_stored=total,
            elapsed_seconds=time.monotonic()-started, source_files_sha256=files, source_sha256=sources,
            full_weights_downloaded=False, model_inference_executed=False, new_gpu_executions=0,
            quality_measured=False, gpu_admitted=False,
            decision='Pinned source/head available; inspect declared native execution and freeze follow-up before any weights or GPU.')
        atomic_json(pub / 'source_observation.json', result)
        print(json.dumps({k:result[k] for k in ['structural_source_checks_passed', 'tensor_keys',
            'classifier_head_shape', 'classifier_head_row_L2', 'source_bytes_stored', 'elapsed_seconds', 'gpu_admitted']}))


if __name__ == '__main__':
    main()
