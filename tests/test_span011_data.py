from datetime import datetime
import copy
import json
from pathlib import Path
import tempfile
import unittest

from arena.span011_data import parse_pair,pair_key,Groups,split_selected,write_inference_inputs


def message(role,text):return dict(role=role,content=[dict(type='text',text=text,image=None)])


class LaterVoteData(unittest.TestCase):
    def row(self):
        return dict(id='uuid1',evaluation_session_id='session1',language='en',evaluation_order=1,
            timestamp=datetime(2025,5,1),conversation_a=[message('user','Question?'),message('assistant','Answer A')],
            conversation_b=[message('user','Question?'),message('assistant','Answer B')],winner='model_a',model_a='private-model')

    def test_pair_conversion_is_label_blind_and_preserves_order(self):
        row=self.row();a,reason=parse_pair(row);self.assertIsNone(reason)
        other=dict(row,winner='both_bad',model_a='different',model_b='different')
        self.assertEqual(parse_pair(other)[0],a)
        self.assertEqual(json.loads(a['response_a']),['Answer A'])
        self.assertEqual(json.loads(a['response_b']),['Answer B'])
        self.assertLess(a['id'],2**63)
        self.assertNotIn('winner',a)

    def test_rejects_old_mismatched_multiturn_and_nontext_histories(self):
        for modify,reason in [
            (lambda r:r.update(timestamp=datetime(2024,8,5)),'before_cutoff'),
            (lambda r:r.update(evaluation_order=2),'evaluation_order'),
            (lambda r:r['conversation_b'][0]['content'][0].update(text='Different'),'mismatched_user'),
            (lambda r:r['conversation_a'].append(message('user','More')),'turn_count'),
            (lambda r:r['conversation_a'][1]['content'][0].update(type='image',image='image-data'),'nontext_or_empty')]:
            row=copy.deepcopy(self.row());modify(row)
            self.assertEqual(parse_pair(row),(None,reason))

    def test_duplicate_key_is_swap_invariant_but_retains_turn_boundaries(self):
        a=pair_key([' Q '],['A'],['B'])
        self.assertEqual(a,pair_key(['q'],['b'],['a']))
        self.assertNotEqual(pair_key(['a','b'],['x','y'],['z','t']),pair_key(['a b'],['x y'],['z t']))

    def test_connected_groups_and_split_are_order_and_label_independent(self):
        first,second=Groups(),Groups();edges=[('p:a','s:1'),('p:b','s:1'),('p:b','s:2')]
        for x,y in edges:first.join(x,y)
        for x,y in reversed(edges):second.join(x,y)
        self.assertEqual(first.find('s:2'),second.find('s:2'))
        records=[dict(source_uuid=str(i),group_id='g'+str(i),length_bin=i%2,winner=i%3) for i in range(8)]
        a=split_selected(records,[2,2]);b=split_selected([dict(r,winner=9) for r in reversed(records)],[2,2])
        self.assertEqual({r['source_uuid']:r['split'] for r in a},{r['source_uuid']:r['split'] for r in b})
        self.assertEqual(sum(r['split']=='learning' for r in a),4)

    def test_gpu_parquet_is_plain_and_excludes_labels_and_identities(self):
        import pyarrow.parquet as pq
        row,_=parse_pair(self.row());row.update(winner='model_a',model_a='secret-identity')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'inputs.parquet';write_inference_inputs(path,[row])
            table=pq.read_table(path)
            self.assertEqual(table.column_names,['id','prompt','response_a','response_b'])
            self.assertIsNone(table.schema.metadata)
            self.assertEqual(table.to_pylist()[0]['id'],row['id'])
            self.assertEqual(json.loads(table.to_pylist()[0]['response_a']),['Answer A'])


if __name__=='__main__':unittest.main()
