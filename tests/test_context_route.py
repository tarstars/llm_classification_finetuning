from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from arena.context_route import changed_mask,competition_rerun


class RouteTests(unittest.TestCase):
    def test_route_compares_content_and_both_views(self):
        self.assertEqual(changed_mask([[1],[2],[3]],[[1],[2],[3]],[[1],[9],[3]],[[1],[2],[4]]).tolist(),[False,True,True])
    def test_hidden_inputs_and_platform_flag_skip_confirmation(self):
        self.assertFalse(competition_rerun({},'visible','visible'))
        self.assertTrue(competition_rerun({},'hidden','visible'))
        self.assertTrue(competition_rerun({'KAGGLE_IS_COMPETITION_RERUN':'true'},'visible','visible'))


if __name__=='__main__':unittest.main()
