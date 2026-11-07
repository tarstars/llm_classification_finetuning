import unittest
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd

from arena.span010_order import ensemble
from arena.span013_order import complete_gemma_order
from arena.span013_pipeline import orchestrate


class GemmaOnlyOrder(unittest.TestCase):
    def test_aligns_both_swapped_predictions_and_preserves_unrouted_rows(self):
        gemma=np.array([[.8,.1,.1],[.4,.3,.3]],dtype=np.float32)
        llama_swapped=np.array([[.2,.6,.2],[.2,.5,.3]],dtype=np.float32)
        reverse_gemma=np.array([[.4,.5,.1]],dtype=np.float32)
        result=complete_gemma_order(gemma,llama_swapped,np.array([0]),reverse_gemma)
        np.testing.assert_allclose(result[0],[.625,.225,.15],atol=1e-7)
        np.testing.assert_array_equal(result[1],ensemble(gemma,llama_swapped)[1])
        empty=complete_gemma_order(gemma,llama_swapped,np.array([],dtype=int),np.empty((0,3)))
        np.testing.assert_array_equal(empty,ensemble(gemma,llama_swapped))

    def test_rejects_invalid_route_or_unmatched_reverse_rows(self):
        base=np.array([[.5,.3,.2],[.2,.3,.5]])
        for route in [np.array([0,0]),np.array([2]),np.array([.5])]:
            with self.assertRaises(ValueError):complete_gemma_order(base,base,route,base[:len(route)])
        with self.assertRaises(ValueError):complete_gemma_order(base,base,np.array([0]),base)

    def test_actual_pipeline_three_roles_full_warmup_and_hidden_without_train(self):
        for qualify in [True,False]:
            with self.subTest(qualify=qualify),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);raw=root/'raw';raw.mkdir();out=root/'out';calls=[]
                def frame(n,offset):return pd.DataFrame(dict(id=np.arange(n)+offset,
                    prompt=['["q"]']*n,response_a=['["a"]']*n,response_b=['["b"]']*n))
                test=frame(101,10000);test.to_csv(raw/'test.csv',index=False)
                if qualify:frame(1000,0).assign(winner_model_a=1).to_csv(raw/'train.csv',index=False)
                def worker(command,check):
                    role=command[command.index('--role')+1];names=command[-1].split(',');calls.append((role,names))
                    for name in names:
                        data=pd.read_parquet(out/(name+'.parquet'))
                        self.assertEqual(data.columns.tolist(),['id','prompt','response_a','response_b'])
                        if name=='warmup':self.assertEqual(len(data),80)
                        if role=='m0_reverse' and name!='warmup':self.assertTrue(name.startswith('routed_'))
                        p={'m0':[.75,.15,.1],'m3':[.4,.3,.3],'m0_reverse':[.25,.6,.15]}[role]
                        np.savez_compressed(out/(role+'_'+name+'.npz'),ids=data.id.to_numpy(),
                            probabilities=np.tile(p,(len(data),1)),lengths=np.full(len(data),512))
                    return SimpleNamespace(returncode=0)
                with patch('arena.span013_pipeline.subprocess.run',side_effect=worker):
                    orchestrate(SimpleNamespace(raw=str(raw),out=str(out),assets='unused',qualify=qualify))
                self.assertEqual([r for r,n in calls],['m0','m3','m0_reverse'])
                if not qualify:self.assertEqual(calls[-1][1],['warmup','routed_test'])
                q=json.loads((out/'qualification.json').read_text())
                self.assertFalse(q['uses_targets']);self.assertEqual(q['warmup_rows'],80)
                submission=pd.read_csv(root/'submission.csv');np.testing.assert_array_equal(submission.id,test.id)
                np.testing.assert_allclose(submission.iloc[:,1:],np.load(out/'ensemble_test.npz')['probabilities'])
                self.assertFalse(list(out.glob('*.parquet')))


if __name__=='__main__':unittest.main()
