import csv
import importlib
from pathlib import Path
import tempfile
import unittest

import duckdb
import numpy as np


class FoundationDataTests(unittest.TestCase):
    def implementation(self):
        try:
            return importlib.import_module('lsvd.data')
        except ModuleNotFoundError:
            self.fail('lsvd.data implementation is not available')

    def fixture(self, directory):
        raw=directory/'raw';freeze=directory/'freeze'
        (raw/'train').mkdir(parents=True);(raw/'validation').mkdir();(raw/'metadata').mkdir();freeze.mkdir()
        c=duckdb.connect()
        c.execute('''CREATE TABLE items AS SELECT * FROM(VALUES
        (101,11,10),(102,12,10),(103,13,10),(104,14,20),
        (105,15,20),(106,16,10),(107,17,10),(108,18,20)) t(item_id,author_id,duration)''')
        c.execute(f"COPY items TO '{raw}/metadata/items_metadata.parquet' (FORMAT PARQUET)")
        c.execute('''CREATE TABLE events(user_id UINTEGER,item_id UINTEGER,place UTINYINT,platform UTINYINT,
        agent UTINYINT,timespent UTINYINT,"like" BOOLEAN,dislike BOOLEAN,share BOOLEAN,
        bookmark BOOLEAN,click_on_author BOOLEAN,open_comments BOOLEAN)''')
        c.execute('''INSERT INTO events VALUES
        (10,101,0,0,0,10,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE),
        (20,104,0,1,1,20,TRUE,TRUE,FALSE,FALSE,FALSE,FALSE),
        (10,102,1,0,0,6,FALSE,FALSE,TRUE,FALSE,FALSE,FALSE),
        (10,103,2,0,0,2,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE),
        (20,105,2,1,1,255,FALSE,FALSE,FALSE,TRUE,TRUE,TRUE)''')
        c.execute(f"COPY events TO '{raw}/train/week_00.parquet' (FORMAT PARQUET)")
        c.execute('DELETE FROM events')
        c.execute('''INSERT INTO events VALUES
        (10,106,1,0,0,10,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE),
        (30,107,0,0,0,1,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE),
        (20,108,2,1,1,20,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE)''')
        c.execute(f"COPY events TO '{raw}/validation/week_25.parquet' (FORMAT PARQUET)")
        c.close()
        fields=['min_d','q25','need25','cut25_20261105','q50','need50','cut50_20261105']
        with (freeze/'training_thresholds_and_hash_cutoffs.csv').open('x',newline='') as handle:
            writer=csv.writer(handle);writer.writerow(fields)
            writer.writerow(['10.0','10.0','1.0',str(2**64-1)+'.0','5.0','1.0',str(2**64-1)+'.0'])
            writer.writerow(['20.0','15.0','1.0',str(2**64-1)+'.0','10.0','1.0',str(2**64-1)+'.0'])
        (freeze/'label_select.sql').write_text('''SELECT e.*,
        (NOT e.dislike AND (e.timespent>f.q25 OR (e.timespent=f.q25 AND f.need25>0 AND hash(e.user_id,e.item_id,20261105::UBIGINT)<=f.cut25_20261105))) flat25,
        (NOT e.dislike AND (e.timespent>f.q50 OR (e.timespent=f.q50 AND f.need50>0 AND hash(e.user_id,e.item_id,20261105::UBIGINT)<=f.cut50_20261105))) flat50 FROM events e''')
        return raw,freeze

    def test_popularity_equal_boundary_belongs_to_hotter_layer(self):
        module=self.implementation()
        self.assertEqual(module.popularity_tiers(np.array([0,5,16,17,100]),16,100).tolist(),[0,1,2,2,3])
        self.assertEqual(module.popularity_tiers(np.array([0,1,2]),1,1).tolist(),[0,3,3])

    def test_extracts_exact_frozen_sql_and_rejects_missing_alias(self):
        module=self.implementation()
        with tempfile.TemporaryDirectory() as temp:
            _,freeze=self.fixture(Path(temp))
            result=module.frozen_expressions(freeze/'label_select.sql')
            self.assertIn('<=f.cut25_20261105',result['main_positive'])
            self.assertIn('NOT e.dislike',result['secondary_positive'])
            invalid=Path(temp)/'invalid.sql';invalid.write_text('SELECT 1')
            with self.assertRaises(ValueError):module.frozen_expressions(invalid)

    def test_real_entry_builds_labels_sequences_views_and_arrays(self):
        module=self.implementation()
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp);raw,freeze=self.fixture(directory);output=directory/'dataset'
            manifest=module.build_foundation(raw,freeze,output,train_weeks=(0,),memory_limit='256MB',threads=1)
            self.assertEqual(manifest['train_rows'],5)
            self.assertEqual(manifest['validation_rows'],3)
            c=duckdb.connect()
            master=c.execute(f"SELECT * FROM read_parquet('{output}/master/*/*.parquet',hive_partitioning=true) ORDER BY week,ordinal").fetchnumpy()
            by_item={int(item):index for index,item in enumerate(master['item_id'])}
            for item,main,secondary,target,weight in [(101,1,1,1,1),(102,0,1,0,.25),(103,0,0,0,0),(104,0,0,0,0),(105,1,1,0,0),(106,1,1,1,0)]:
                index=by_item[item]
                self.assertEqual(int(master['main_positive'][index]),main)
                self.assertEqual(int(master['secondary_positive'][index]),secondary)
                self.assertEqual(int(master['target_main'][index]),target)
                self.assertEqual(float(master['train_weight'][index]),weight)
                self.assertEqual(int(master['user_block'][index]),int(master['user_id'][index])%32)
            self.assertTrue(master['quick_skip'][by_item[103]])
            self.assertTrue(master['like_positive'][by_item[104]])
            self.assertTrue(master['over_duration'][by_item[105]])
            self.assertTrue(master['at_cap'][by_item[105]])
            self.assertGreater(master['watch_ratio'][by_item[105]],1)
            sequence=c.execute(f"SELECT item_ids,weeks,ordinals,main_positive FROM read_parquet('{output}/sequences/*/*.parquet') WHERE user_id=10").fetchone()
            self.assertEqual(sequence[0],[101,102,103]);self.assertEqual(sequence[1],[0,0,0]);self.assertEqual(sequence[2],[0,2,3])
            queries=c.execute(f"SELECT user_id,history_count,last_item_id FROM read_parquet('{output}/queries.parquet') ORDER BY user_id").fetchall()
            self.assertEqual(queries,[(10,3,103),(20,2,105),(30,0,-1)])
            targets=c.execute(f"SELECT user_id,item_id,place,train_seen,popularity_tier FROM read_parquet('{output}/targets.parquet') ORDER BY user_id").fetchall()
            self.assertEqual(targets,[(10,106,1,False,0),(20,108,2,False,0)])
            q=np.load(output/'arrays/queries.npz');t=np.load(output/'arrays/targets.npz');h=np.load(output/'arrays/history.npz')
            self.assertEqual(q['user_id'].tolist(),[10,20,30]);self.assertEqual(t['offsets'].tolist(),[0,1,2,2])
            self.assertEqual(h['user_id'].tolist(),[10,20]);self.assertEqual(np.load(output/'arrays/history_items.npy').tolist(),[101,102,103,104,105])
            with self.assertRaises(FileExistsError):module.build_foundation(raw,freeze,output,train_weeks=(0,),memory_limit='256MB',threads=1)
            c.close()

    def test_week26_is_rejected_before_any_input_is_opened(self):
        module=self.implementation()
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)
            with self.assertRaises(ValueError):module.build_foundation(directory/'none',directory/'none',directory/'out',train_weeks=(26,))
            self.assertFalse((directory/'out').exists())

    def test_disk_budget_is_checked_before_writing_a_week(self):
        module=self.implementation()
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)
            raw,freeze=self.fixture(directory)
            output=directory/'dataset'
            with self.assertRaises(RuntimeError):
                module.build_foundation(raw,freeze,output,train_weeks=(0,),memory_limit='256MB',threads=1,disk_limit=0)
            self.assertFalse(list((output/'master').rglob('*.parquet')))
            self.assertFalse((output/'foundation.json').exists())
            self.assertTrue((output/'failure.json').exists())

    def test_sequence_query_orders_one_struct_list_for_all_fields(self):
        module=self.implementation()
        self.assertTrue(hasattr(module,'sequence_select_sql'))
        sql=module.sequence_select_sql('events','user_block=0')
        self.assertEqual(sql.count('ORDER BY week,ordinal'),1)
        self.assertIn('struct_pack(',sql)
        self.assertIn('list_transform(',sql)

    def test_recovery_reuses_complete_master_without_recalculating_labels(self):
        module=self.implementation()
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)
            raw,freeze=self.fixture(directory)
            first=directory/'first'
            second=directory/'second'
            module.build_foundation(raw,freeze,first,train_weeks=(0,),memory_limit='256MB',threads=1)
            original={p:p.read_bytes() for p in (first/'master').rglob('*.parquet')}
            result=module.build_foundation(raw,freeze,second,train_weeks=(0,),memory_limit='256MB',threads=1,reuse_master=first/'master')
            self.assertEqual(result['train_rows'],5)
            self.assertEqual(result['master_source'],str(first/'master'))
            self.assertFalse((second/'master').exists())
            self.assertEqual(original,{p:p.read_bytes() for p in original})
            self.assertNotIn('hash(e.user_id',(second/'executed.sql').read_text())


if __name__=='__main__':unittest.main()
