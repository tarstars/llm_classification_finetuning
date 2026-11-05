from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from arena.prediction import InferenceCollator


class PredictionContracts(unittest.TestCase):
    def test_inference_never_forwards_dataset_labels(self):
        class Tokenizer:
            def pad(self,rows,**kwargs):
                assert all('labels' not in row for row in rows)
                return rows
        collator=InferenceCollator(Tokenizer())
        result=collator([{'input_ids':[1,2],'attention_mask':[1,1],'labels':2}])
        self.assertEqual(result,[{'input_ids':[1,2],'attention_mask':[1,1]}])


if __name__=='__main__':unittest.main()
