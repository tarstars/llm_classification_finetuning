"""Validate native reward rendering using only the small pinned tokenizer files."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import sys

import requests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT
from arena.experiment import atomic_json, exclusive_lock, read_json, sha256, utc_now
from arena.span016_reward import conversation, encode_conversation


def main():
    from transformers import AutoTokenizer
    pub = REPO / 'results/span_016'
    plan = read_json(pub / 'span_plan.json')
    root = ROOT / 'runs/span_016/source_audit'
    target = root / '8B'
    with exclusive_lock(root / 'tokenizer.lock'):
        assert not (pub / 'tokenizer_validation.json').exists(), 'Preserve validation'
        assert datetime.now(timezone.utc) < datetime.fromisoformat(plan['source_audit_deadline'])
        metadata = read_json(target / 'metadata.json')
        other = read_json(root / '0.6B/metadata.json')
        siblings = {s['rfilename']: s for s in metadata['siblings']}
        other_siblings = {s['rfilename']: s for s in other['siblings']}
        spec = plan['models']['8B']
        for name in ['added_tokens.json', 'merges.txt', 'vocab.json', 'tokenizer.json']:
            info = siblings[name]
            assert info['blobId'] == other_siblings[name]['blobId'], 'Tokenizer differs by size arm'
            assert info['size'] < 16 * 1024**2
            path = target / name
            if path.exists():
                assert path.stat().st_size == info['size']
                if info.get('lfs'):
                    assert sha256(path) == info['lfs']['sha256']
                else:
                    raw = path.read_bytes()
                    assert hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == info['blobId']
                continue
            url = f"https://huggingface.co/{spec['repository']}/resolve/{spec['revision']}/{name}"
            part = path.with_suffix(path.suffix + '.part')
            with requests.get(url, stream=True, timeout=(10, 45)) as response:
                response.raise_for_status()
                size = 0
                with part.open('xb') as stream:
                    for block in response.iter_content(65536):
                        size += len(block)
                        assert size <= info['size']
                        stream.write(block)
            assert size == info['size']
            if info.get('lfs'):
                assert sha256(part) == info['lfs']['sha256']
            else:
                raw = part.read_bytes()
                assert hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == info['blobId']
            part.rename(path)
        tokenizer = AutoTokenizer.from_pretrained(target, local_files_only=True)
        assert tokenizer.pad_token_id == 151654 and tokenizer.eos_token_id == 151645
        row = dict(prompt=json.dumps(['What is two plus two?', 'Explain briefly.']),
            response_a=json.dumps(['Four.', 'Two pairs make four.']),
            response_b=json.dumps(['Five.', 'This is an incorrect count.']))
        checks = []
        for side in ['a', 'b']:
            messages = conversation(row, side)
            ids, meta = encode_conversation(tokenizer, messages)
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            # Qwen's native tokenizer adds no BOS, matching the card's two-step example.
            reference = tokenizer(rendered)['input_ids']
            assert ids == reference and tokenizer.bos_token is None
            assert ids[-2] == tokenizer.eos_token_id
            shortened, short_meta = encode_conversation(tokenizer, messages, max_length=16)
            assert shortened == ids[:14] + ids[-2:] and short_meta['truncated']
            checks.append(dict(side=side, native_reference_identical=True, suffix_preserved=True, **meta))
        files = [p for p in root.rglob('*') if p.is_file() and p.suffix != '.lock']
        total = sum(p.stat().st_size for p in files)
        assert total <= plan['source_audit_bytes_cap']
        assert datetime.now(timezone.utc) < datetime.fromisoformat(plan['source_audit_deadline'])
        result = dict(at=utc_now(), passed=True, synthetic_conversations_only=True,
            checks=checks, common_tokenizer_verified=True, native_pad_token_id=tokenizer.pad_token_id,
            native_eos_token_id=tokenizer.eos_token_id, local_transformers_version=__import__('transformers').__version__,
            source_files_bytes=total, source_files_sha256={str(p.relative_to(root)): sha256(p) for p in files},
            model_weights_downloaded=False, new_gpu_executions=0, source_sha256=sha256(Path(__file__)))
        atomic_json(pub / 'tokenizer_validation.json', result)
        print(json.dumps(dict(passed=True, native_reference_identical=True,
            common_tokenizer_verified=True, source_files_bytes=total, new_gpu_executions=0)))


if __name__ == '__main__':
    main()
