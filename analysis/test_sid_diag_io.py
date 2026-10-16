"""Synthetic IO guards; no server/data access."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from analysis import sid_diag as d

class IOTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name).resolve()
    def tearDown(self): self.tmp.cleanup()
    def artifact(self, name, payload):
        p = self.root/name; p.write_bytes(payload)
        return {'path': str(p), 'sha256': hashlib.sha256(payload).hexdigest()}
    def test_missing_historical_receipts_stay_unknown_not_verified(self):
        missing={'terminal':{'status':'succeeded'},'verified':False,'verify':{'jobs':[]}}
        self.assertEqual(d.unaccepted_legacy({'g-g4-rand-s42':missing}),{'g-g4-rand-s42':'UNKNOWN: archived verifier receipt not found; no predictions read'})
        self.assertFalse(missing['verified'])
        u={'status':'UNKNOWN','reason':'missing raw verified'}
        report=d.render_report({'inventory':{'threshold_source':{}},'A':{},'B':{'random':{'42':u}},'C':{'random':{'42':u}},'C_delta_vs_same_seed_T3':{},'D':{},'decomposition':{}})
        self.assertIn('UNKNOWN',report)
        self.assertEqual(d.mean_tree({'42':u,'2024':u,'2025':u})['status'],'UNKNOWN')
        with self.assertRaises(ValueError):d.unaccepted_legacy({'g-tok-t3-bhkm-b256-s42':missing})
        with self.assertRaises(ValueError):d.unaccepted_legacy({'g-g4-rand-s42':{**missing,'verify':{'jobs':[{'terminal':{'status':'failed'}}]}}})
        self.assertEqual(d.unaccepted_legacy({'g-tok-t3-bhkm-b256-s42':{'verified':True}}),{})
    def test_prepared_manifest_registered_role(self):
        self.assertEqual(d.prepared_metadata_path({'prepared':'/fixture/prepared'}),Path('/fixture/prepared/prepare-manifest.json'))
    def test_hash_and_scope(self):
        ref = self.artifact('input.json', b'{}')
        self.assertEqual(d.read_json(ref), {})
        with self.assertRaises(ValueError): d.read_json({**ref, 'sha256': '0'*64})
        for name in ('test.json', 'checkpoint.json', 'embedding.json'):
            ref = self.artifact(name, b'{}')
            with self.assertRaises(ValueError): d.read_json(ref)
        link = self.root/'link.json'; link.symlink_to(self.root/'input.json')
        with self.assertRaises(ValueError): d.read_json({'path':str(link),'sha256':'0'*64})
    def test_strict_npz(self):
        p = self.root/'mapping.npz'; np.savez(p, ids=np.array([5,7]), sids=np.array([[1,0,0,0],[0,0,0,0]]))
        ref = {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        self.assertEqual(d.load_mapping(ref, [5,7]), {5:(1,0,0,0),7:(0,0,0,0)})
        np.savez(p, ids=np.array([5,5]), sids=np.array([[1,0,0,0],[0,0,0,0]]))
        ref['sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
        with self.assertRaises(ValueError): d.load_mapping(ref,[5,7])
        np.savez(p, ids=np.array([5,7]), sids=np.array([[1,0,0,0],[0,0,0,0]]), weights=np.ones(2))
        ref['sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
        with self.assertRaises(ValueError): d.load_mapping(ref,[5,7])
    def test_train_scope(self):
        rows = [{'path':f'subsamples/ur0.01_ir0.01/train/week_{w:02d}.parquet','sha256':'a'*64,'size_bytes':1} for w in range(25)]
        self.assertEqual(len(d.validate_train_records(rows)),25)
        rows[-1]['path'] = 'subsamples/ur0.01_ir0.01/train/week_26.parquet'
        with self.assertRaises(ValueError): d.validate_train_records(rows)
    def test_full_observed_histories(self):
        result = d.collect_histories([(0,[{'user_id':1,'item_id':5},{'user_id':2,'item_id':7},{'user_id':1,'item_id':5}]),(24,[{'user_id':1,'item_id':7}])], [5,7])
        self.assertEqual(result, [[5,5,7],[7]])
        with self.assertRaises(ValueError): d.collect_histories([(25,[])],[5,7])
        with self.assertRaises(ValueError): d.collect_histories([(1,[]),(0,[])],[5,7])
    def test_prediction_identity(self):
        qs=[{'target':5,'oov':False}]
        with self.assertRaises(ValueError): d.validate_predictions([[7,7]],qs,{5,7},2)
        with self.assertRaises(ValueError): d.validate_predictions([[7,9]],qs,{5,7},2)
        self.assertEqual(d.validate_predictions([[5,7]],qs,{5,7},2),[[5,7]])
    def test_p_archive_reuses_selection_hash_identity_checks(self):
        import math
        rid='fixture'; rd=self.root/'runs'/rid/'out'; rd.mkdir(parents=True)
        pred=list(range(50)); pp=rd/'predictions-epoch-001.json'
        pp.write_bytes(d.archive.canonical([pred])); psha=hashlib.sha256(pp.read_bytes()).hexdigest()
        hist={'binding':{'query_sha':'q','catalog_sha':'c'},'catalog_sha256':'catalog'}
        manifest={'selection':{'best_epoch':1,'best_metric':[1.,1.],'epoch':2},
                  'status':'complete','best':{'epoch':1,'metric':[1.,1.]},
                  'binding':{'inputs':{'history_manifest_sha256':d.archive.HISTORY_PIN,'query_sha':'q','catalog_sha':'c'},'catalog_sha256':'catalog'},
                  'evaluations':[{'epoch':1,'predictions_ref':pp.name,'predictions_sha256':psha}]}
        mp=rd/'manifest.json';mp.write_bytes(d.archive.canonical(manifest))
        state={'verified':True,'terminal':{'status':'succeeded'},'manifest_sha256':hashlib.sha256(mp.read_bytes()).hexdigest()}
        qs=[{'target':0,'oov':False}]
        with patch.object(d.archive,'ROOT',self.root), patch.object(d.archive,'N',1):
            rows,_=d.load_predictions(rid,qs,hist,state,list(range(50)))
            self.assertEqual(rows,[pred])
            pp.write_bytes(d.archive.canonical([list(reversed(pred))]))
            with self.assertRaises(ValueError): d.load_predictions(rid,qs,hist,state,list(range(50)))
            with self.assertRaises(ValueError): d.load_predictions(rid,qs,hist,{**state,'verified':False},list(range(50)))
        self.assertEqual(d.archive.read.__name__,'read')

    def test_mapping_order_and_random_determinism(self):
        base={i:(i%2,0,0,i//2) for i in range(12)}
        a=d.random_mapping(base)
        self.assertEqual(a,d.random_mapping(base))
        self.assertEqual(list(a),list(base))
        self.assertEqual(len(set(a.values())),len(a))
        from collections import Counter
        self.assertEqual(Counter(x[:3] for x in a.values()),Counter(x[:3] for x in base.values()))
        p=self.root/'mapping.npz';np.savez(p,ids=np.array([7,5]),sids=np.zeros((2,4),dtype=np.uint32))
        ref={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        with self.assertRaises(ValueError): d.load_mapping(ref,[5,7])

    def json_ref(self,name,value):
        return self.artifact(name,d.archive.canonical(value))

    def nested_source(self):
        source=self.json_ref('bundle.json',{'artifacts':{'mapping.npz':'f'*64}})
        spec=self.json_ref('spec.json',{'sid_path':source['path'],'sid_sha256':source['sha256']})
        prepared=self.json_ref('prepared.json',{'kind':'real_prepared','spec_path':spec['path'],'spec_sha256':spec['sha256']})
        return source,prepared

    def claimed(self,rid,argv):
        folder=self.root/'queue/claimed';folder.mkdir(parents=True,exist_ok=True)
        path=folder/('001-'+rid+'.json');path.write_bytes(d.archive.canonical({'run_id':rid,'argv':argv}))
        return path

    def test_nested_b16_source_cache_each_run_and_claim_identity(self):
        source,prepared=self.nested_source(); cache={}
        for rid in ('g-hist-v2-l50-s42','g-hist-v2-l50-s2024'):
            self.claimed(rid,['--prepared',prepared['path'],'--prepared-sha',prepared['sha256']])
            resolved=d.registered_sid_source(self.root,rid,cache)
            self.assertEqual(resolved,source)
        self.assertEqual(len(cache),2)
        with self.assertRaises(ValueError): d.registered_sid_source(self.root,'g-hist-v2-l50-s2025',cache)
        bad=self.claimed('bad',['--prepared',prepared['path'],'--prepared-sha','0'*64])
        with self.assertRaises(ValueError): d.registered_sid_source(self.root,'bad',cache)
        bad.write_bytes(d.archive.canonical({'run_id':'different','argv':['--sid-manifest',source['path'],'--sid-sha',source['sha256']]}))
        with self.assertRaises(ValueError): d.registered_sid_source(self.root,'bad',cache)

    def test_nested_b16_verification_uses_same_source_resolver(self):
        source,prepared=self.nested_source();rid='g-hist-v2-l50-s42'
        self.claimed(rid,['--prepared',prepared['path'],'--prepared-sha',prepared['sha256']])
        plan={'root':str(self.root),'bundles':{'B16':{'path':str(self.root/'mapping.npz'),'sha256':'f'*64,'source_manifest':source}},'runs':{'B16':{'42':rid}}}
        inventory={'run_input_bindings':{rid:{'binding':{}}}}
        d.verify_bundle_bindings(plan,inventory)

    def test_a1_fixed_train_control_provenance_and_unverified_rejected(self):
        history={'binding':{'query_sha':'q','catalog_sha':'file-catalog'},'catalog_sha256':'catalog'}
        plan={'root':str(self.root),'history':{'sha256':d.archive.HISTORY_PIN},'train_audit':{'sha256':'a'*64}}
        binding={'prepared_sha':'a'*64,'catalog_sha256':'catalog','inputs':{'history_manifest_sha256':d.archive.HISTORY_PIN,'query_sha':'q','catalog_sha':'file-catalog'}}
        manifest={'binding':binding};state={'verified':True,'terminal':{'status':'succeeded'},'manifest_path':'manifest','manifest_sha256':'m'}
        with patch.object(d.archive,'inspect_run',return_value=state),patch.object(d.archive,'read',return_value=(manifest,'m')):
            self.assertEqual(d.assert_train_source_control(plan,history)['run_id'],'p-seqm1-mean-s42')
            with self.assertRaises(ValueError): d.assert_train_source_control({**plan,'train_audit':{'sha256':'b'*64}},history)
            binding['inputs']['query_sha']='different'
            with self.assertRaises(ValueError): d.assert_train_source_control(plan,history)
        with patch.object(d.archive,'inspect_run',return_value={**state,'verified':False}):
            with self.assertRaises(ValueError): d.assert_train_source_control(plan,history)

    def test_preflight_runs_binding_guards(self):
        inventory={'read_artifacts_sha256':{}}
        with patch.object(d,'_plan_inventory',return_value=inventory),patch.object(d,'assert_metadata_bindings',side_effect=ValueError('unbound')) as guard:
            with self.assertRaisesRegex(ValueError,'unbound'): d.collect_inventory({'schema_version':1})
            guard.assert_called_once()

    def test_preflight_stat_does_not_hash_npz(self):
        ref=self.artifact('mapping.npz', b'not an npz')
        self.assertEqual(d.stat_ref(ref)['size_bytes'],10)

if __name__=='__main__': unittest.main()
