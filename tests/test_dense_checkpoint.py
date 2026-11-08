from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file
from arena.dense_checkpoint import expand_checkpoint


class DenseCheckpointTests(unittest.TestCase):
    def test_streamed_conversion_preserves_head_and_independently_known_levels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);source=root/'source';source.mkdir()
            head = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]], dtype=np.float16)
            tensors = {'score.weight': head, 'layer.qweight': np.zeros((1,4),np.int32),
                       'layer.qzeros': np.zeros((1,1),np.int32), 'layer.scales': np.ones((1,4),np.float16),
                       'layer.g_idx': np.zeros(4,np.int32)}
            save_file(tensors, str(source/'original.safetensors'))
            (source/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{k:'original.safetensors' for k in tensors}}))
            result=expand_checkpoint(source,root/'dense',max_shard_bytes=32)
            self.assertEqual(result['tensor_keys'],2)
            self.assertEqual(result['total_tensor_bytes'],head.nbytes+32)
            output=json.loads((root/'dense/model.safetensors.index.json').read_text())['weight_map']
            for key, expected in [('score.weight',head),('layer.weight',-np.ones((4,4),np.float16))]:
                with safe_open(root/'dense'/output[key],framework='np') as f:
                    np.testing.assert_array_equal(f.get_tensor(key),expected)


if __name__ == '__main__': unittest.main()
