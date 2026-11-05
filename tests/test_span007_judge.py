from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from arena import span007_judge


class ConcurrencyIsolation(unittest.TestCase):
    def test_client_pool_changes_and_server_configuration_stays_fixed(self):
        config={'slots':16};observed=[]
        def score(frame,model,folder,both):
            observed.append(model.config['slots']);self.assertEqual(config['slots'],16)
            return [],.01
        def fake_run(settings,exe,raw,out,sample,plan):
            self.assertEqual(settings['slots'],16)
            self.assertEqual([c['cache_prompt'] for c in plan['diagnostic_configurations']],[False,False])
            for name in ['serial','parallel']:
                model=SimpleNamespace(config=settings)
                span007_judge.judge.score_frame([],model,Path(out)/name/'initial',False)
                self.assertIs(model.config,settings)
            case={'passes':{'initial':{'seconds_per_row':1.,'rows':64,'seconds':64.,'prediction_sha256':'x','requests_sha256':'y'}},
                  'server_load_seconds':[1.,1.],'repeatability_passed':False,'comparisons':{},'sampled_device_used_peak_mib':[2000,2000]}
            return {'configurations':{'serial':case,'parallel':case}}
        with tempfile.TemporaryDirectory() as directory,patch.object(span007_judge.judge,'score_frame',side_effect=score),patch.object(span007_judge.span_repeatability,'run',side_effect=fake_run):
            result=span007_judge.run(config,None,None,Path(directory),{}, {})
        self.assertEqual(observed,[1,16]);self.assertFalse(result['affordable_stable_recipe'])


if __name__=='__main__':unittest.main()
