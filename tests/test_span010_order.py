import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from arena.span010_order import route_indices,complete_orders,orchestrate


class OrderCompletion(unittest.TestCase):
    def test_route_respects_both_caps_is_permutation_equivariant_and_ignores_equal_experts(self):
        ids=np.arange(100,200);a=np.tile([.8,.1,.1],(100,1));b=np.tile([.1,.8,.1],(100,1))
        lengths=np.c_[np.arange(1,101)*40,np.arange(1,101)*50]
        selected,js,work=route_indices(ids,a,b,lengths)
        self.assertLessEqual(len(selected),20)
        self.assertLessEqual(work[selected].sum(),.1*work.sum())
        order=np.random.default_rng(42).permutation(100)
        perm,_,_=route_indices(ids[order],a[order],b[order],lengths[order])
        self.assertEqual(set(ids[selected]),set(ids[order][perm]))
        empty,_,_=route_indices(ids,a,a,lengths)
        self.assertEqual(len(empty),0)

    def test_four_predictions_are_aligned_and_unrouted_predictions_preserved(self):
        a=np.array([[.8,.1,.1],[.2,.3,.5]])
        b=np.array([[.2,.7,.1],[.4,.1,.5]])
        ar=np.array([[.15,.75,.1]]);br=np.array([[.65,.25,.1]])
        actual=complete_orders(a,b,np.array([0]),ar,br)
        np.testing.assert_allclose(actual,[[.725,.175,.1],[.15,.35,.5]])
        for indices in [[0,0],[-1],[2]]:
            with self.assertRaises(ValueError):
                complete_orders(a,b,np.array(indices),np.zeros((len(indices),3)),np.zeros((len(indices),3)))

    def test_real_orchestration_public_and_hidden_with_cpu_worker_fixture(self):
        # Exercise the actual deployment orchestration, file/ID/swap/CSV route,
        # and public-only profiling using a deterministic substitute worker.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=root/'raw';raw.mkdir()
            frame=pd.DataFrame(dict(id=np.arange(1000),prompt=['["question"]']*1000,
                response_a=['["a"]']*1000,response_b=['["b"]']*1000))
            frame.to_csv(raw/'train.csv',index=False);frame.iloc[:100].to_csv(raw/'test.csv',index=False)
            calls=[]
            def worker(command,check):
                role=command[command.index('--role')+1];names=command[-1].split(',');calls.append((role,names))
                folder=Path(command[command.index('--out')+1]);summary=dict(cold_seconds=1,cohorts={})
                for name in names:
                    data=pd.read_parquet(folder/(name+'.parquet'))
                    vector={'m0':[.8,.1,.1],'m3':[.5,.4,.1],
                            'm0_reverse':[.2,.7,.1],'m3_reverse':[.6,.3,.1]}[role]
                    probs=np.tile(vector,(len(data),1))
                    np.savez_compressed(folder/(role+'_'+name+'.npz'),ids=data.id.to_numpy(),
                        probabilities=probs,lengths=np.full(len(data),512))
                    summary['cohorts'][name]=dict(rows=len(data),seconds=len(data)*.01,
                        tokens=len(data)*512,max_tokens=512 if len(data) else 0)
                (folder/(role+'_summary.json')).write_text(json.dumps(summary))
                return SimpleNamespace(returncode=0)
            for qualify in [True,False]:
                calls.clear();out=root/('public' if qualify else 'hidden')
                args=SimpleNamespace(out=str(out),raw=str(raw),assets='unused',qualify=qualify)
                with patch('arena.span010_order.subprocess.run',side_effect=worker):orchestrate(args)
                q=json.loads((out/'qualification.json').read_text())
                self.assertTrue(q['complete'])
                if qualify:
                    self.assertTrue(q['runtime_gate_passed'])
                    selection=json.loads((out/'profile_selection.json').read_text())
                    self.assertEqual(len(selection['uniform_ids']),512)
                    self.assertFalse(set(selection['uniform_ids']) & set(selection['control_ids']))
                else:
                    self.assertFalse((out/'profile_selection.json').exists())
                    self.assertEqual([names for role,names in calls if 'reverse' in role],
                                     [['warmup','routed_test']]*2)
                routed=np.load(out/'routing_test.npz')['selected_indices']
                result=np.load(out/'ensemble_test.npz')['probabilities']
                self.assertEqual(len(routed),10)
                np.testing.assert_allclose(result[routed],np.tile([.625,.275,.1],(10,1)))
                self.assertEqual(pd.read_csv(root/'submission.csv').shape,(100,4))
                self.assertFalse(list(out.glob('*.parquet')))


if __name__=='__main__':unittest.main()
