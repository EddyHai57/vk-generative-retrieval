import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT))


def module():
    from analysis import ur001_derivation
    return ur001_derivation


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':')).encode()


class DerivationTests(unittest.TestCase):
    def fixture(self):
        def r(u,i,like=False):return {'user_id':u,'item_id':i,'like':like}
        return [(0,[r(1,7),r(1,9,True),r(2,8,True),r(3,7)]+[r(1,7)]*53),
                (1,[r(1,8,True),r(1,8,True),r(2,7)])],(25,[r(1,100,True),r(1,7,True),r(2,9,True),r(3,7,True),r(4,9,True)])

    def test_original_oracle_bitwise_and_train_end_history(self):
        m=module()
        from vkgr.retrieval_data import prefix_examples
        from vkgr.data_audit import build_queries,approved_policy
        from vkgr.history_v2 import build_observed,overlay_examples
        parts,validation=self.fixture()
        catalog=sorted({r['item_id'] for _,rows in parts for r in rows})
        examples=list(prefix_examples(parts))
        queries=build_queries(parts,validation,approved_policy(),set(catalog))['queries']
        old=build_observed(parts,examples,queries,catalog,max_len=50)
        expected=overlay_examples(examples,old['train'],50,set(catalog))
        emitted=[];new=m.derive(parts,validation,emitted.append)
        self.assertEqual(canonical(emitted),canonical(expected))
        self.assertEqual(canonical(new['validation']),canonical(old['validation']))
        self.assertEqual(canonical(new['slices']),canonical(old['slices']))
        self.assertEqual(new['validation'][0]['target'],100)
        self.assertTrue(new['validation'][0]['oov'])
        self.assertEqual(new['catalog'],catalog)
        self.assertEqual(new['supported'],[8,9])
        self.assertEqual(new['counts']['no_history'],2)
        self.assertTrue(all(p[0]<=24 for q in new['validation'] for p in q['history_positions']))

    def test_invalid_order_and_feedback_stop(self):
        m=module();parts,validation=self.fixture()
        for bad in [[(1,parts[0][1]),(0,parts[1][1])],[(26,parts[0][1])],[(0,[{'user_id':1,'item_id':2,'like':None}])]]:
            with self.assertRaises(ValueError):m.derive(bad,validation,lambda x:None)
        with self.assertRaises(ValueError):m.derive(parts,(26,validation[1]),lambda x:None)

    def test_parquet_rowgroup_parallel_order_and_single_file_gate(self):
        m=module()
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();folder=root/'train';folder.mkdir()
            p=folder/'week_00.parquet'
            rows=[{'user_id':i%3,'item_id':i,'like':i%2==0} for i in range(29)]
            pq.write_table(pa.Table.from_pylist(rows),p,row_group_size=4)
            baseline=list(m.parquet_rows(p,workers=1))
            self.assertEqual(baseline,rows)
            for workers in (2,4):self.assertEqual(canonical(list(m.parquet_rows(p,workers))),canonical(baseline))
            self.assertEqual(m.week_files(folder,[0]),[p])
            (folder/'week_00-copy.parquet').write_bytes(p.read_bytes())
            with self.assertRaises(ValueError):m.week_files(folder,[0])


if __name__=='__main__':unittest.main()
