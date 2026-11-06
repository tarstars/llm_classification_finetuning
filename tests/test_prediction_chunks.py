import tempfile
import unittest
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from arena.prediction_chunks import ChunkStore


class PredictionChunksTest(unittest.TestCase):
    def test_resume_contract_and_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=ChunkStore(tmp,{'batch':8});p=np.array([[.2,.3,.5],[.4,.4,.2]])
            store.write(0,[3,5],p,p,seconds=2)
            resumed=ChunkStore(tmp,{'batch':8});np.testing.assert_array_equal(resumed.read(0,[3,5])['original'],p)
            with self.assertRaises(AssertionError):ChunkStore(tmp,{'batch':4})
            with self.assertRaises(AssertionError):resumed.read(0,[5,3])
            with self.assertRaises(AssertionError):resumed.assemble([0],[3,5,7])
            with self.assertRaises(AssertionError):resumed.write(0,[3,5],p,p)
            raw=resumed.assemble([0],[3,5]);np.testing.assert_array_equal(raw['ids'],[3,5])
            (Path(tmp)/'0000.npz').write_bytes(b'corruption')
            with self.assertRaises(AssertionError):resumed.read(0,[3,5])

    def test_uncommitted_temporary_is_not_a_cache_hit(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'0000.npz.tmp').write_bytes(b'interrupted')
            store=ChunkStore(tmp,{'batch':8});self.assertIsNone(store.read(0,[1]))
            p=np.array([[.2,.3,.5]]);store.write(0,[1],p,p)
            calls=0
            if ChunkStore(tmp,{'batch':8}).read(0,[1]) is None:calls+=1
            self.assertEqual(calls,0)


if __name__=='__main__':unittest.main()
