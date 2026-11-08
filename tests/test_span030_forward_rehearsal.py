"""Exercise the actual new forward-loop source with CPU tensors and fake GPUs."""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from threading import current_thread
import textwrap
import time
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from arena.cpu_prefetch import CpuPrefetch


class ForwardRehearsal(unittest.TestCase):
    def test_actual_loop_preserves_order_and_keeps_device_work_on_consumer(self):
        owner = current_thread().ident
        producer_owners, calls = [], []
        def micro(rows):
            lengths = [n for _, n in rows]
            return dict(input_ids=torch.tensor([[v for v, n in rows for _ in range(n)]]),
                seq_lens=lengths, cu_seqlens=torch.tensor([0, *np.cumsum(lengths)], dtype=torch.int32),
                position_ids=torch.cat([torch.arange(n) for n in lengths]).unsqueeze(0),
                max_seq_len=max(lengths), label=torch.tensor([[1., 0., 0.]]*len(rows)))
        batches = [[micro([(11, 3)]), micro([(22, 2)])], [micro([(33, 1), (44, 4)])]]
        def source():
            for batch in batches:
                producer_owners.append(current_thread().ident)
                yield batch
        def device(value, name):
            self.assertEqual(current_thread().ident, owner)
            self.assertIn(name, ['cuda:0', 'cuda:1'])
            return value
        def synchronize(gpu):
            self.assertEqual(current_thread().ident, owner)
            self.assertIn(gpu, [0, 1])
        class Model:
            def forward_part1(inner, values, seq, freq):
                self.assertEqual(current_thread().ident, owner)
                ends = seq['cu_seqlens'][1:].long()-1
                ids = values[0, ends].float()
                calls.append(('first', ids.tolist()))
                return torch.stack([ids/100, -ids/100, ids*0], dim=1)
            def forward_part2(inner, hidden, seq, freq):
                self.assertEqual(current_thread().ident, owner)
                calls.append(('second', hidden[:, 0].tolist()))
                return hidden
        real = Path(__file__).resolve().parents[1] / 'src/arena/span030_engine.py'
        source_text = real.read_text()
        start = source_text.index('        begin = time.monotonic()', source_text.index('def infer_role'))
        end = source_text.index('        validate_probabilities(probs', start)
        actual_loop = textwrap.dedent(source_text[start:end])
        scope = dict(time=time, CpuPrefetch=CpuPrefetch, loader=source(), hashlib=hashlib,
            json=json, torch=SimpleNamespace(cat=torch.cat, no_grad=torch.no_grad,
                cuda=SimpleNamespace(amp=SimpleNamespace(autocast=nullcontext), synchronize=synchronize)),
            config=SimpleNamespace(max_position_embeddings=8192), model=Model(),
            BlockDiagonalCausalMask=SimpleNamespace(from_seqlens=lambda sizes: tuple(sizes)),
            to_device=device, freq0=None, freq1=None, event=lambda *args, **kwargs: None,
            role='m0', name='uniform', outs=[], lengths=[], token_hash=hashlib.sha256(),
            packed_tokens=[], batch_groups=[], host_tensor_hash=hashlib.sha256(),
            previous=None, forward_wall_seconds=0.)
        exec(compile(actual_loop, str(real)+'[actual-forward-loop]', 'exec'), scope)
        ids = torch.tensor([11., 22., 33., 44.])
        expected = torch.stack([ids/100, -ids/100, ids*0], dim=1).softmax(-1).numpy()
        np.testing.assert_array_equal(scope['probs'], expected)
        self.assertEqual(scope['batch_groups'], [[[3], [2]], [[1, 4]]])
        self.assertEqual(scope['lengths'], [3, 2, 1, 4])
        self.assertEqual([v[0] for v in calls], ['first', 'second', 'first', 'second', 'first', 'second'])
        self.assertEqual(len(set(producer_owners)), 1)
        self.assertNotEqual(producer_owners[0], owner)
        self.assertFalse(scope['prepared'].thread.is_alive())
        self.assertEqual(scope['prepared'].metrics['prepared_batches'], 2)
        self.assertLessEqual(scope['forward_wall_seconds'], scope['seconds'])


if __name__ == '__main__':
    unittest.main()
