"""Independent item-level, shared statistics and beam/shard oracles."""
import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from vkgr.data_audit import canonical_bytes
try:
    from vkgr import generative_evaluation as ev
except ImportError:
    ev=None

def write(path,value):
    path.write_bytes(canonical_bytes(value));return hashlib.sha256(path.read_bytes()).hexdigest()

def queries():
    return [{'user_id':u,'target':t,'history':[20],'oov':t==999} for u,t in [(1,10),(2,20),(3,70),(4,110),(1,999),(2,40),(3,100)]]

def stats_bundle(root):
    catalog=list(range(10,111,10))
    binding={'raw_records':[{'role':'synthetic_train','sha256':'1'*64,'size_bytes':100}],
             'raw_producer':{'synthetic':True,'source_sha256':'2'*64},'prepared_sha':'3'*64,
             'catalog_sha':hashlib.sha256(canonical_bytes(catalog)).hexdigest(),
             'consumer':{'synthetic':True,'source_sha256':'4'*64}}
    stats={'schema_version':1,'popularity_thresholds':{'median':5,'p90':9},
           'item_like_counts':{str(item):i for i,item in enumerate(catalog)},
           'user_like_counts':{'1':1,'2':5,'3':20,'4':29},'user_buckets':[1,5,20],
           'definition':'train_like_counts_only_no_holdout_boundaries'}
    manifest={'schema_version':1,'task_id':'P4E','kind':'slice_statistics','binding':binding,
              'statistics_ref':'train-slices.json','statistics_sha':write(root/'train-slices.json',stats),
              'threshold_definition':'nearest_rank_positive_item_event_counts_median_p90',
              'thresholds':{'median':5,'p90':9},
              'item_groups':['unsupported0','positive_le_median','positive_gt_median_le_p90','positive_gt_p90'],
              'user_groups':['1','2-5','6-20','gt20'],'holdout_used':False}
    path=root/'slice-statistics-manifest.json';pin=write(path,manifest)
    return path,pin,binding,catalog,stats,manifest

class StatisticsMetricTests(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(ev,'G2 evaluation module missing')
    def test_complete_shared_counts_thresholds_and_unknown_not_zero(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            path,pin,binding,catalog,_,_=stats_bundle(Path(d))
            stats=ev.read_slice_statistics(path,pin,binding,catalog,queries())
            self.assertEqual(stats.thresholds,(5,9));self.assertEqual(stats.item_counts[10],0)
            self.assertEqual(stats.user_counts[1],1) # Single-like user is preserved.
            with self.assertRaises(TypeError):stats.item_counts[10]=1
            with self.assertRaises(ValueError):ev.read_slice_statistics(path,'0'*64,binding,catalog,queries())
            with self.assertRaises(ValueError):ev.read_slice_statistics(path,pin,binding,catalog,queries()+[{'user_id':9,'target':20,'history':[20],'oov':False}])
    def test_rehashed_stats_missing_ids_bool_negative_threshold_and_producer_reject(self):
        for mode in ('missing_item','missing_user','bool','negative','threshold','producer','role','holdout'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory(dir='/private/tmp') as d:
                root=Path(d);path,_,binding,catalog,stats,manifest=stats_bundle(root)
                if mode=='missing_item':del stats['item_like_counts']['10']
                if mode=='missing_user':del stats['user_like_counts']['1']
                if mode=='bool':stats['item_like_counts']['20']=True
                if mode=='negative':stats['user_like_counts']['1']=-1
                if mode=='threshold':stats['popularity_thresholds']['median']=6;manifest['thresholds']['median']=6
                if mode=='producer':manifest['binding']=copy.deepcopy(binding);manifest['binding']['raw_producer']['source_sha256']='f'*64
                if mode=='role':manifest['statistics_ref']='../test-queries.json'
                if mode=='holdout':manifest['holdout_used']=True
                manifest['statistics_sha']=write(root/'train-slices.json',stats);pin=write(path,manifest)
                with self.assertRaises(ValueError):ev.read_slice_statistics(path,pin,binding,catalog,queries())
    def test_item_metrics_full_oov_shared_slices_empty_null_and_coverage(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            path,pin,binding,catalog,_,_=stats_bundle(Path(d));stats=ev.read_slice_statistics(path,pin,binding,catalog,queries())
            predictions=[[20,10],[20],[20,70],[110],[20],[20],[100]]
            result=ev.aggregate_items(predictions,queries(),catalog,slice_definition=stats)
            self.assertEqual(result['metrics']['all_eligible']['count'],7)
            self.assertAlmostEqual(result['metrics']['all_eligible']['recall@50'],5/7)
            self.assertAlmostEqual(result['metrics']['all_eligible']['ndcg@50'],(3+2/math.log2(3))/7)
            self.assertEqual(result['metrics']['in_catalog']['count'],6);self.assertEqual(result['metrics']['oov']['count'],1)
            self.assertEqual(result['metrics']['oov']['recall@10'],0)
            self.assertEqual([result['user_history'][k]['count'] for k in ['1','2-5','6-20','gt20']],[2,2,2,1])
            self.assertEqual([result['popularity'][k]['count'] for k in ['unsupported0','positive_le_median','positive_gt_median_le_p90','positive_gt_p90']],[1,2,2,1])
            self.assertAlmostEqual(result['coverage'],5/11)
            subset=ev.aggregate_items(predictions[:1],queries()[:1],catalog,slice_definition=stats)
            self.assertEqual(subset['popularity']['positive_gt_p90']['count'],0)
            self.assertIsNone(subset['popularity']['positive_gt_p90']['recall@50'])
            with self.assertRaises(ValueError):ev.aggregate_items([[20,20]],queries()[:1],catalog,slice_definition=stats)

class BeamOracleTests(unittest.TestCase):
    def test_full_vocab_scores_ties_and_exhaustive_beam_agree(self):
        import torch
        from test_generative_training import codec
        from vkgr.tiger import constrained_beam
        c=codec()
        def score(paths):
            # Dominant PAD remains in the denominator; prefix-dependent terms
            # make legal-only normalization produce a different item ordering.
            rows=[]
            for path in paths:
                row=torch.zeros(c.vocab_size,dtype=torch.float64)
                row[0]=8 if path and path[0]==0 else 1
                rows.append(row)
            return torch.stack(rows)
        expected=[]
        for item in c.catalog.ids:
            path=tuple(c.catalog.encode(int(item)));total=0.
            for position in range(4):
                values=score([path[:position]])[0].tolist()
                total+=values[c.offsets[position]+path[position]]-math.log(sum(math.exp(x) for x in values))
            expected.append((int(item),total))
        expected.sort(key=lambda pair:(-pair[1],pair[0]))
        self.assertTrue(hasattr(ev,'oracle_rank'),'exhaustive G2 oracle missing')
        self.assertEqual(ev.oracle_rank(score,c),[item for item,_ in expected])
        self.assertEqual(constrained_beam(c.catalog,c,score,100)[0],ev.oracle_rank(score,c))
        tied=lambda paths:torch.zeros(len(paths),c.vocab_size,dtype=torch.float64)
        self.assertEqual(ev.oracle_rank(tied,c),[10,20,30,40])
        self.assertEqual(constrained_beam(c.catalog,c,tied,1)[0],[10])
    def test_oracle_refuses_nonfinite_and_unbounded_catalog(self):
        import torch
        from test_generative_training import codec
        self.assertTrue(hasattr(ev,'oracle_rank'),'exhaustive G2 oracle missing')
        c=codec()
        with self.assertRaises(ValueError):ev.oracle_rank(lambda p:torch.full((len(p),c.vocab_size),float('nan')),c)
        with self.assertRaises(ValueError):ev.oracle_rank(lambda p:torch.zeros(len(p),1),c)

class ShardTests(unittest.TestCase):
    def test_real_decode_three_three_one_resume_no_missing_or_duplicate(self):
        self.assertTrue(hasattr(ev,'evaluate_queries'),'resumable evaluation missing')
        import torch
        from vkgr import generative_experiment as ge,generative_training as gt
        from test_generative_training import config
        root=Path(__file__).resolve().parents[1];fixture=root/'tests/fixtures/g2_synthetic.json';pin=hashlib.sha256(fixture.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            out=Path(d);context=ge.prepare_synthetic(root,root/'artifacts/generative-contract-manifest.json','40863ebbe96f9e2e1da8f53d916e68ead1fa098b41ab80237481e53a22e2506d',fixture,pin,out/'prepared')
            torch.manual_seed(42);model=gt.make_model(context.codec,config(),device='cpu')
            first=ev.evaluate_queries(model,context,output=out/'first',checkpoint_sha256='a'*64,shard_size=3,stop_after_shards=1)
            self.assertFalse(first['manifest']['complete']);self.assertEqual(first['manifest']['completed_queries'],3)
            parent_bytes={p.name:p.read_bytes() for p in (out/'first').iterdir()}
            resumed=ev.evaluate_queries(model,context,output=out/'resumed',checkpoint_sha256='a'*64,shard_size=3,resume_manifest=first['manifest_path'],resume_sha256=first['manifest_sha256'])
            continuous=ev.evaluate_queries(model,context,output=out/'continuous',checkpoint_sha256='a'*64,shard_size=3)
            self.assertEqual([(s['start'],s['end']) for s in resumed['manifest']['shards']],[(0,3),(3,6),(6,7)])
            self.assertEqual(resumed['aggregate'],continuous['aggregate'])
            self.assertEqual(resumed['manifest']['computation_sha256'],continuous['manifest']['computation_sha256'])
            self.assertEqual(resumed['aggregate']['metrics']['all_eligible']['count'],7)
            self.assertEqual({p.name:p.read_bytes() for p in (out/'first').iterdir()},parent_bytes)
            self.assertEqual(resumed['observations']['latency']['count'],7)
            self.assertEqual(resumed['observations']['vram'],'not_applicable')
            with self.assertRaises(ValueError):ev.evaluate_queries(model,context,output=out/'wrong-checkpoint',checkpoint_sha256='b'*64,shard_size=3,resume_manifest=first['manifest_path'],resume_sha256=first['manifest_sha256'])
            with self.assertRaises(ValueError):ev.evaluate_queries(model,context,output=out/'wrong-head',checkpoint_sha256='a'*64,shard_size=3,resume_manifest=first['manifest_path'],resume_sha256='0'*64)
            self.assertFalse((out/'wrong-checkpoint').exists());self.assertFalse((out/'wrong-head').exists())
            # A caller-provided new manifest hash cannot legitimize repeated range.
            bad=copy.deepcopy(first['manifest']);bad['shards'][0]['start']=1;bad_path=out/'first'/'forged.json';bad_sha=write(bad_path,bad)
            with self.assertRaises(ValueError):ev.evaluate_queries(model,context,output=out/'bad-range',checkpoint_sha256='a'*64,shard_size=3,resume_manifest=bad_path,resume_sha256=bad_sha)
            shard=Path(first['manifest']['shards'][0]['path']);shard.write_bytes(b'{}')
            with self.assertRaises(ValueError):ev.evaluate_queries(model,context,output=out/'bad-shard',checkpoint_sha256='a'*64,shard_size=3,resume_manifest=first['manifest_path'],resume_sha256=first['manifest_sha256'])

if __name__=='__main__':unittest.main()
