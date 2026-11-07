import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from arena.span012_profile import orchestrate_profile,decide_packing,gpu_snapshot,PROFILE_NAMES


class MatchedPacking(unittest.TestCase):
    def test_same_role_process_executes_full_warmup_and_aba_on_identical_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=root/'raw';raw.mkdir()
            frame=pd.DataFrame(dict(id=np.arange(1000),prompt=['["q"]']*1000,
                response_a=['["a"]']*1000,response_b=['["b"]']*1000,winner_model_a=np.ones(1000)))
            frame.to_csv(raw/'train.csv',index=False);calls=[]
            def worker(command,check):
                role=command[command.index('--role')+1];calls.append(role)
                self.assertEqual(command[-1],','.join(PROFILE_NAMES))
                folder=Path(command[command.index('--out')+1]);reference=None
                for name in PROFILE_NAMES:
                    data=pd.read_parquet(folder/(name+'.parquet'))
                    self.assertEqual(data.columns.tolist(),['id','prompt','response_a','response_b'])
                    self.assertEqual(len(data),80 if name.startswith('warmup_') else 256)
                    if name=='profile_8192_first':reference=data.id.to_numpy()
                    if name.startswith('profile_'):np.testing.assert_array_equal(data.id,reference)
                    np.savez_compressed(folder/(role+'_'+name+'.npz'),ids=data.id.to_numpy(),
                        probabilities=np.tile([.6,.2,.2],(len(data),1)),lengths=np.full(len(data),512))
                return SimpleNamespace(returncode=0)
            args=SimpleNamespace(raw=str(raw),out=str(root/'out'),assets='unused')
            with patch('arena.span012_profile.subprocess.run',side_effect=worker):orchestrate_profile(args)
            self.assertEqual(calls,['m0','m3'])
            self.assertFalse(list((root/'out').glob('*.parquet')))
            result=json.loads((root/'out/profile_complete.json').read_text())
            self.assertEqual(result['pack_sequence'],[8192,6144,8192]);self.assertFalse(result['uses_targets'])

    def test_selection_rejects_numeric_failure_unstable_bracket_and_small_speed_change(self):
        def summary(a,b,c):return dict(cohorts={n:dict(seconds=v) for n,v in
            zip(['profile_8192_first','profile_6144','profile_8192_last'],[a,b,c])})
        s={'m0':summary(140,130,142),'m3':summary(104,96,106)}
        d={r:{n:0. for n in ['profile_8192_first','profile_6144','profile_8192_last']} for r in s}
        self.assertEqual(decide_packing(s,d)['selected_packing'],6144)
        d['m0']['profile_6144']=.00002
        self.assertEqual(decide_packing(s,d)['selected_packing'],8192)
        d['m0']['profile_6144']=0;s['m0']=summary(140,130,170)
        self.assertEqual(decide_packing(s,d)['selected_packing'],8192)
        s['m0']=summary(140,139,142);s['m3']=summary(104,104,106)
        self.assertEqual(decide_packing(s,d)['selected_packing'],8192)
        d['m0']['profile_8192_first']=.00002
        with self.assertRaises(ValueError):decide_packing(s,d)

    def test_xml_telemetry_reports_both_devices_and_unavailable_query(self):
        gpu='<gpu><product_name>Tesla T4</product_name><clocks><sm_clock>1590 MHz</sm_clock><mem_clock>5001 MHz</mem_clock></clocks><utilization><gpu_util>90 %</gpu_util></utilization></gpu>'
        success=SimpleNamespace(returncode=0,stdout='<nvidia_smi_log>'+gpu*2+'</nvidia_smi_log>')
        with patch('arena.span012_profile.subprocess.run',return_value=success):
            result=gpu_snapshot();self.assertTrue(result['available']);self.assertEqual(result['devices'][0]['sm_clock'],'1590 MHz')
        with patch('arena.span012_profile.subprocess.run',return_value=SimpleNamespace(returncode=3)):
            self.assertFalse(gpu_snapshot()['available'])


if __name__=='__main__':unittest.main()
