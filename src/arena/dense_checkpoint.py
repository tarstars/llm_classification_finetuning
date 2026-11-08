"""Stream a fixed GPTQ8 checkpoint into standard dense safetensors shards."""
from contextlib import ExitStack
from pathlib import Path
import json
import struct

import numpy as np
from safetensors import safe_open

from .gptq8 import dense_weight


def expand_checkpoint(source, destination, max_shard_bytes=2*1024**3):
    source, destination = Path(source), Path(destination)
    index = json.loads((source / 'model.safetensors.index.json').read_text())
    mapping = index['weight_map']
    headers, header_lengths = {}, {}
    for name in sorted(set(mapping.values())):
        with (source / name).open('rb') as stream:
            size = struct.unpack('<Q', stream.read(8))[0]
            header = json.loads(stream.read(size))
        header_lengths[name] = size
        for key, info in header.items():
            if key != '__metadata__':
                assert mapping[key] == name and key not in headers
                headers[key] = info
    assert set(headers) == set(mapping)
    projections = sorted(k.removesuffix('.qweight') for k in mapping if k.endswith('.qweight'))
    consumed = {prefix + suffix for prefix in projections for suffix in ['.qweight','.qzeros','.scales','.g_idx']}
    assert consumed <= set(mapping)
    entries = []
    for name in sorted(set(mapping)-consumed):
        info = headers[name]
        begin, end = info['data_offsets']
        entries.append(dict(name=name, dtype=info['dtype'], shape=info['shape'], bytes=end-begin, original=name))
    for prefix in projections:
        shape = headers[prefix+'.qweight']['shape']
        entries.append(dict(name=prefix+'.weight', dtype='F16', shape=[shape[1],shape[0]*4],
                            bytes=shape[1]*shape[0]*8, projection=prefix))
    shards, current, size = [], [], 0
    for entry in entries:
        if current and size + entry['bytes'] > max_shard_bytes:
            shards.append(current)
            current, size = [], 0
        current.append(entry)
        size += entry['bytes']
    if current: shards.append(current)
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {}
    with ExitStack() as stack:
        tensors = {name:stack.enter_context(safe_open(source/name, framework='np')) for name in header_lengths}
        def get(name): return tensors[mapping[name]].get_tensor(name)
        for number, group in enumerate(shards, 1):
            name = f'model-{number:05d}-of-{len(shards):05d}.safetensors'
            path = destination / name
            assert not path.exists() and not path.with_suffix('.part').exists(), 'Reconcile partial conversion first'
            header, offset = {'__metadata__': {'format': 'pt'}}, 0
            for entry in group:
                header[entry['name']] = dict(dtype=entry['dtype'], shape=entry['shape'],
                                              data_offsets=[offset, offset+entry['bytes']])
                offset += entry['bytes']
            encoded = json.dumps(header, separators=(',', ':')).encode()
            encoded += b' ' * (-len(encoded) % 8)
            temporary = path.with_suffix('.part')
            with temporary.open('xb') as out:
                out.write(struct.pack('<Q', len(encoded)))
                out.write(encoded)
                for entry in group:
                    if 'original' in entry:
                        original = entry['original']
                        begin = headers[original]['data_offsets'][0]
                        with (source / mapping[original]).open('rb') as stream:
                            stream.seek(8 + header_lengths[mapping[original]] + begin)
                            remaining = entry['bytes']
                            while remaining:
                                block = stream.read(min(8*1024**2, remaining))
                                if not block: raise ValueError('Truncated original tensor')
                                out.write(block)
                                remaining -= len(block)
                    else:
                        prefix = entry['projection']
                        dense = dense_weight(*(get(prefix+suffix) for suffix in ['.qweight','.qzeros','.scales','.g_idx']))
                        if list(dense.shape) != entry['shape'] or not np.isfinite(dense).all():
                            raise ValueError('Invalid expanded projection')
                        assert dense.nbytes == entry['bytes']
                        out.write(memoryview(dense).cast('B'))
                        del dense
                    outputs[entry['name']] = name
            assert temporary.stat().st_size == 8 + len(encoded) + offset
            temporary.replace(path)
    result = dict(metadata=dict(total_size=sum(e['bytes'] for e in entries)), weight_map=outputs)
    target = destination / 'model.safetensors.index.json'
    assert not target.exists()
    target.write_text(json.dumps(result, indent=2)+'\n')
    return dict(projection_names=projections, tensor_keys=len(outputs), total_tensor_bytes=result['metadata']['total_size'],
                shards=sorted(set(outputs.values())), runtime_native_layout_required=True)
