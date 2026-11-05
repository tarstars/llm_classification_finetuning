from pathlib import Path
import sys
import tempfile
import unittest
import pandas as pd
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from arena.data import CLASSES
from arena.joint_data import read_competition,split_from_ids


class TargetIsolation(unittest.TestCase):
    def test_h49_selected_labels_do_not_depend_on_confirmation_votes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            frame=pd.DataFrame({'id':[1,2,3,4],'prompt':['["one"]','["two"]','["three"]','["four"]'],
                                'response_a':['["a"]']*4,'response_b':['["b"]']*4,
                                CLASSES[0]:[1,0,1,1],CLASSES[1]:[0,1,0,0],CLASSES[2]:[0,0,0,0]})
            frame.iloc[:1][['id','prompt','response_a','response_b']].to_csv(root/'test.csv',index=False)
            manifest={'dev':[3],'confirmation':[4]}
            def selected():
                frame.to_csv(root/'train.csv',index=False)
                parsed,_=read_competition(root);parts=split_from_ids(parsed,manifest)
                return [parsed.iloc[parts[k]][['id','prompt','response_a','response_b',*CLASSES]].reset_index(drop=True) for k in ['train','dev']]
            before=selected();frame.loc[frame.id==4,CLASSES]=[0,0,1];after=selected()
            for a,b in zip(before,after):pd.testing.assert_frame_equal(a,b)


if __name__=='__main__':unittest.main()
