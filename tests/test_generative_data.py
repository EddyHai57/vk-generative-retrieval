"""G1 independent oracles; all input bundles below are synthetic only."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from vkgr import generative_data as gd
from vkgr.data_audit import canonical_bytes, file_sha256

ROOT = Path(__file__).resolve().parents[1]
CAPSULE = ROOT / 'provenance/g0-v1'
G0_SHA = 'fbcb5c6dc41e9144420a79cd1d26926b93c269a84adf7dc837ca4c1217781b59'

def write(path, value):
    path.write_bytes(canonical_bytes(value))
    return file_sha256(path)

class CapsuleTests(unittest.TestCase):
    def test_capsule_fixed_pins_and_inventory_rehash_cannot_replace_history(self):
        pin = file_sha256(CAPSULE / 'inventory.json')
        value = gd.validate_legacy_capsule(CAPSULE, pin)
        self.assertEqual(len(value['members']), 37)
        self.assertEqual(sum(x['bytes'] for x in value['members']), 427387)
        with tempfile.TemporaryDirectory(dir='/private/tmp') as directory:
            root = Path(directory) / 'capsule'
            shutil.copytree(CAPSULE, root)
            path = root / 'src/vkgr/tiger.py'
            path.write_bytes(path.read_bytes() + b'\n# substituted\n')
            inv = json.loads((root / 'inventory.json').read_text())
            row = next(x for x in inv['members'] if x['path'] == 'src/vkgr/tiger.py')
            row['sha256'] = file_sha256(path); row['bytes'] = path.stat().st_size
            new_pin = write(root / 'inventory.json', inv)
            with self.assertRaisesRegex(ValueError, 'historical|anchor'):
                gd.validate_legacy_capsule(root, new_pin)

    def test_capsule_missing_extra_duplicate_escape_and_symlink_reject(self):
        for mode in ['missing', 'extra', 'duplicate', 'escape', 'absolute', 'symlink', 'parent_symlink']:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(dir='/private/tmp') as directory:
                root = Path(directory) / 'capsule'; shutil.copytree(CAPSULE, root)
                inv = json.loads((root / 'inventory.json').read_text())
                if mode == 'missing': (root / 'src/vkgr/tiger.py').unlink()
                if mode == 'extra': (root / 'extra.txt').write_text('not allowed')
                if mode == 'duplicate': inv['members'].append(copy.deepcopy(inv['members'][0]))
                if mode == 'escape': inv['members'][0]['path'] = '../outside'
                if mode == 'absolute': inv['members'][0]['path'] = '/private/tmp/outside'
                if mode == 'symlink':
                    p = root / 'src/vkgr/tiger.py'; p.unlink(); p.symlink_to(CAPSULE / 'src/vkgr/tiger.py')
                if mode == 'parent_symlink':
                    shutil.rmtree(root / 'src'); (root / 'src').symlink_to(CAPSULE / 'src', target_is_directory=True)
                pin = write(root / 'inventory.json', inv)
                with self.assertRaises((ValueError, OSError)):
                    gd.validate_legacy_capsule(root, pin)

    def test_original_g0_replay_imports_capsule_in_fresh_process(self):
        env = dict(os.environ, PYTHONPATH='src', OMP_NUM_THREADS='1')
        code = 'from pathlib import Path; import vkgr.generative_foundation as g; assert Path(g.__file__).resolve()==Path("src/vkgr/generative_foundation.py").resolve(); g.verify_manifest("artifacts/g0-foundation-manifest.json", "' + G0_SHA + '")'
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=CAPSULE, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(file_sha256(ROOT / 'artifacts/g0-foundation-manifest.json'), G0_SHA)

class ContractTests(unittest.TestCase):
    def test_contract_identity_and_typed_fields_reject_even_after_rehash(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as directory:
            path = Path(directory) / 'contract.json'
            original = gd.build_contract(ROOT)
            pin = write(path, original)
            contract = gd.load_contract(ROOT, path, pin)
            self.assertEqual(contract['test_gate'], 'closed')
            self.assertNotEqual(contract['consumer']['source_sha256'], contract['historical_producers']['p4b_real']['source_fingerprint'])
            self.assertNotIn('artifacts/generative-contract-manifest.json', contract['consumer']['source_members'])
            with self.assertRaisesRegex(ValueError, 'hash'):
                gd.load_contract(ROOT, path, '0' * 64)
            edits = [lambda d: d.update(schema_version=True),
                     lambda d: d.update(schema_version=1.0),
                     lambda d: d['inputs']['prepared'].update(manifest_sha256='0'*64),
                     lambda d: d['historical_producers']['p4b_real'].update(source_fingerprint='0'*64),
                     lambda d: d['legacy_capsule'].update(g0_manifest_sha256='0'*64),
                     lambda d: d.update(test_gate='open'),
                     lambda d: d.update(unexpected=0)]
            for edit in edits:
                bad = copy.deepcopy(original); edit(bad); pin = write(path, bad)
                with self.assertRaises(ValueError): gd.load_contract(ROOT, path, pin)
            duplicate = canonical_bytes(original).decode().replace('"schema_version":1', '"schema_version":1,"schema_version":1')
            path.write_text(duplicate)
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                gd.load_contract(ROOT, path, file_sha256(path))

    def test_new_consumer_source_drift_does_not_change_historical_capsule(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as directory:
            root = Path(directory) / 'root'; root.mkdir()
            original = gd.build_contract(ROOT)
            for name in set(original['consumer']['source_members']) | {x['path'] for x in json.loads((CAPSULE/'inventory.json').read_text())['members']}:
                dest = root/name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/name, dest)
            shutil.copytree(CAPSULE, root/'provenance/g0-v1', dirs_exist_ok=True)
            manifest = root/'contract.json'; pin = write(manifest, original)
            with patch.object(gd, '_git', return_value={k:original['consumer'][k] for k in ('git_base','git_dirty')}):
                gd.load_contract(root, manifest, pin)
                source = root/'src/vkgr/generative_data.py'
                source.write_bytes(source.read_bytes()+b'\n# different consumer\n')
                with self.assertRaisesRegex(ValueError, 'consumer|contract'):
                    gd.load_contract(root, manifest, pin)
            self.assertEqual(gd.validate_legacy_capsule(root/'provenance/g0-v1', original['legacy_capsule']['inventory_sha256'])['anchors']['g0_manifest_sha256'], G0_SHA)

def prepared_fixture(directory, mutate=None):
    """Toy consumer context, NOT load_contract acceptance or real producer evidence."""
    from vkgr.retrieval_data import prefix_examples, write_prepared
    catalog = [10, 20, 30]
    items = [10, 20, 10] + [20, 10]*26
    rows = list(prefix_examples([(0, [{'user_id':1,'item_id':i,'like':True} for i in items])]))
    if mutate: mutate(rows)
    features = np.zeros((4,64), dtype=np.float32); features[1:,0] = 1
    # Fabricated historical fields are only a synthetic unit fixture.
    producer = {'source_fingerprint':'a'*64, 'lock_sha256':'b'*64}
    binding = {'consumer':producer,'input_records':[],'query_pins':[], 'legacy_producer':{},
               'legacy_training_examples_sha256':'c'*64,'test_gate':'closed_hash_only',
               'resources_start':{'ram':0.9,'vram':0.9,'disk':0.9},
               'resources_before_write':{'ram':0.9,'vram':0.9,'disk':0.9}, 'reconciliation':{}}
    output = Path(directory)/'prepared'
    manifest = write_prepared(output, catalog, features, rows, binding, [10,20])
    context = {'test_gate':'closed','inputs':{'prepared':{
        'manifest_sha256':file_sha256(output/'prepare-manifest.json'), 'artifacts':manifest['artifacts'],
        'examples':len(rows), 'catalog_items':3, 'binding_checks':binding}}}
    return output, gd._Contract(context, ROOT), rows

class PreparedTests(unittest.TestCase):
    def test_every_prefix_repeats_last50_fullpast_and_support(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as directory:
            out, context, rows = prepared_fixture(directory)
            loaded = gd.load_prepared(out/'prepare-manifest.json', context)
            self.assertEqual(rows[0]['history'], [10]); self.assertEqual(rows[0]['target'],20)
            self.assertEqual(rows[1]['history'], [10,20]); self.assertEqual(rows[1]['target'],10)
            self.assertEqual(rows[-1]['history'], ([10,20,10]+[20,10]*26)[:-1][-50:])
            self.assertEqual(rows[-1]['past_likes'],[10,20])
            self.assertEqual(loaded['catalog'].tolist(),[10,20,30])
            self.assertEqual(loaded['example_count'],54)
            self.assertEqual(loaded['support'],[10,20])
            self.assertEqual(loaded['features'].dtype,np.float32)
            self.assertFalse(loaded['features'].flags.writeable)

    def test_manifest_pin_precedes_any_artifact_open_and_context_mutation_rejects(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as directory:
            out, context, _ = prepared_fixture(directory)
            path = out/'prepare-manifest.json'; path.write_bytes(path.read_bytes()+b' ')
            with patch.object(gd, '_json', side_effect=AssertionError('must reject hash before JSON/artifacts')):
                with self.assertRaisesRegex(ValueError, 'hash'): gd.load_prepared(path, context)
            context['test_gate']='open'
            with self.assertRaisesRegex(ValueError,'context'): gd.load_prepared(path,context)

    def test_prepared_rehashed_invalid_rows_ids_support_and_producer_rejected(self):
        edits = [lambda r:r[0].update(example_id=True), lambda r:r[0].update(target=True),
                 lambda r:r[0]['history_positions'][0].__setitem__(1, r[0]['target_position'][1]),
                 lambda r:r[2]['history'].__setitem__(0,20),
                 lambda r:r[2].update(past_likes=[10]),
                 lambda r:r[2]['history_positions'].__setitem__(0,[0,1])]
        for edit in edits:
            with self.subTest(edit=edit), tempfile.TemporaryDirectory(dir='/private/tmp') as directory:
                out, context, rows = prepared_fixture(directory)
                edit(rows)
                (out/'examples.jsonl').write_bytes(b''.join(canonical_bytes(r) for r in rows))
                manifest=json.loads((out/'prepare-manifest.json').read_text())
                manifest['artifacts']['examples.jsonl']=file_sha256(out/'examples.jsonl')
                pin=write(out/'prepare-manifest.json',manifest)
                spec=copy.deepcopy(dict(context));spec['inputs']['prepared']['manifest_sha256']=pin
                spec['inputs']['prepared']['artifacts']=manifest['artifacts']
                with self.assertRaises(ValueError):gd.load_prepared(out/'prepare-manifest.json',gd._Contract(spec,ROOT))
        for mode in ['producer','idmap_bool','support_bool','schema_bool']:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(dir='/private/tmp') as directory:
                out,context,_=prepared_fixture(directory);manifest=json.loads((out/'prepare-manifest.json').read_text())
                if mode=='producer':manifest['binding']['consumer']['source_fingerprint']='d'*64
                if mode=='idmap_bool':
                    value=json.loads((out/'idmap.json').read_text());value['pad']=False;write(out/'idmap.json',value)
                    manifest['artifacts']['idmap.json']=file_sha256(out/'idmap.json')
                if mode=='support_bool':
                    write(out/'support.json',[True,20]);manifest['artifacts']['support.json']=file_sha256(out/'support.json')
                if mode=='schema_bool':manifest['schema_version']=True
                pin=write(out/'prepare-manifest.json',manifest);spec=copy.deepcopy(dict(context));spec['inputs']['prepared']['manifest_sha256']=pin;spec['inputs']['prepared']['artifacts']=manifest['artifacts']
                with self.assertRaises(ValueError):gd.load_prepared(out/'prepare-manifest.json',gd._Contract(spec,ROOT))

def sid_fixture(directory, prepared, context):
    from vkgr import sid_run
    from vkgr.sid import fit
    from vkgr.data_audit import producer_identity
    features = np.array(prepared['features'][1:], copy=True)
    model = fit(prepared['catalog'], features)
    evidence = {'frozen_manifest_sha256':sid_run.FROZEN_SHA,'inputs':[], 'feature_zero_norm':0}
    out=Path(directory)/'sid'; manifest=sid_run.write_artifacts(out,model,evidence,features)
    manifest.update({'producer':producer_identity(),'resource_minimum':{'ram_available_fraction':0.9,'disk_available_fraction':0.9},
                     'resource_samples':1,'elapsed_seconds':0.0,'rss_unit':'KiB_on_Linux_bytes_on_macOS',
                     'started_at_utc':'2026-10-08T00:00:00+00:00', 'claim_boundary':'synthetic_metadata_not_real_resource_measurement'})
    pin=write(out/'run_manifest.json',manifest)
    value=copy.deepcopy(dict(context));value['historical_producers']={'p5':{
        'source_fingerprint':manifest['producer']['source_fingerprint'], 'git_base':manifest['producer']['git_commit'],
        'python_major_minor':manifest['producer']['python_major_minor'],'dependency_versions':manifest['producer']['dependencies']}}
    value['inputs']['sid']={'manifest_sha256':pin,'artifacts':manifest['artifacts'],'effective_config':manifest['effective_config'],
                            'catalog_items':3,'feature_array_sha256':manifest['feature_array_sha256'],
                            'evidence_inputs':[], 'frozen_manifest_sha256':sid_run.FROZEN_SHA,'evidence_checks':{'feature_zero_norm':0}}
    return out,gd._Contract(value,ROOT)

class SIDTests(unittest.TestCase):
    def test_real_static_loader_on_synthetic_bundle_full_catalog_roundtrip(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as directory:
            out,context,_=prepared_fixture(directory);prepared=gd.load_prepared(out/'prepare-manifest.json',context)
            sid,context=sid_fixture(directory,prepared,context)
            codec=gd.load_sid(sid/'run_manifest.json',context,prepared)
            self.assertEqual(codec.catalog.ids.tolist(),[10,20,30])
            for item in [10,20,30]:self.assertEqual(codec.catalog.decode(codec.catalog.encode(item)),item)
            for path in [(16,0,0,0),(0,0,0,1),(0,0,0,True),(0,0,0)]:
                with self.assertRaises(ValueError):codec.catalog.decode(path)
            audit=gd.audit_sid(codec,prepared)
            self.assertEqual(audit['sid']['roundtrip_items'],3)
            self.assertEqual(audit['resources'],'not_measured')

    def test_sid_rehashed_catalog_suffix_digit_collision_and_feature_mismatch_reject(self):
        from vkgr.sid_run import feature_digest
        for mode in ['catalog','suffix','digit','collision','feature','producer','evidence']:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(dir='/private/tmp') as directory:
                out,context,_=prepared_fixture(directory);prepared=gd.load_prepared(out/'prepare-manifest.json',context)
                sid,context=sid_fixture(directory,prepared,context);manifest=json.loads((sid/'run_manifest.json').read_text())
                if mode in ('catalog','suffix','digit','collision'):
                    with np.load(sid/'mapping.npz') as data:ids=data['ids'];sids=data['sids']
                    if mode=='catalog':ids[0]=11
                    if mode=='suffix':sids[0,3]=2
                    if mode=='digit':sids[0,0]=16
                    if mode=='collision':sids[1]=sids[0]
                    np.savez(sid/'mapping.npz',ids=ids,sids=sids);manifest['artifacts']['mapping.npz']=file_sha256(sid/'mapping.npz')
                if mode=='feature':
                    features=np.load(sid/'normalized-features.npy');features[0,0]=-1
                    np.save(sid/'normalized-features.npy',features,allow_pickle=False)
                    manifest['feature_array_sha256']=feature_digest(features);manifest['artifacts']['normalized-features.npy']=file_sha256(sid/'normalized-features.npy')
                if mode=='producer':manifest['producer']['source_fingerprint']='d'*64
                if mode=='evidence':manifest['evidence']['inputs']=[{'role':'test','sha256':'d'*64,'path':'ignored'}]
                pin=write(sid/'run_manifest.json',manifest)
                value=copy.deepcopy(dict(context));value['inputs']['sid']['manifest_sha256']=pin;value['inputs']['sid']['artifacts']=manifest['artifacts']
                if mode=='feature':value['inputs']['sid']['feature_array_sha256']=manifest['feature_array_sha256']
                with self.assertRaises(ValueError):gd.load_sid(sid/'run_manifest.json',gd._Contract(value,ROOT),prepared)

    def test_content_distance_only_train_support_and_collision_suffix_oracle(self):
        from vkgr.sid import SIDCatalog
        from vkgr.tiger import SIDTokens
        with tempfile.TemporaryDirectory(dir='/private/tmp') as directory:
            out,context,_=prepared_fixture(directory);prepared=gd.load_prepared(out/'prepare-manifest.json',context)
            value=dict(prepared);features=np.array(value['features']);features[2]=0;features[2,1]=1;features[3]=0;features[3,0]=-1;features.setflags(write=False);value['features']=features
            prepared=gd._Prepared(value)
            catalog=SIDCatalog(np.array([10,20,30],np.uint32),np.array([[0,0,0,0],[0,0,0,1],[0,0,0,2]],np.uint32),3,3,[])
            codec=SIDTokens(catalog)
            result=gd.audit_sid(codec,prepared)
            self.assertEqual(result['content_distance']['pair_count'],1)
            self.assertEqual(result['content_distance']['mean_squared_l2'],2.0)
            self.assertEqual(result['content_distance']['supported_items'],2)
            self.assertEqual(result['sid']['semantic_collision_items'],3)
            self.assertEqual(catalog.decode([0,0,0,1]),20)

class ValidationTests(unittest.TestCase):
    def test_first_oov_query_preserves_population_and_shared_item_denominator(self):
        from vkgr.baselines import metrics
        from vkgr.data_audit import build_queries
        train=[(0,[{'user_id':1,'item_id':10,'like':True},{'user_id':1,'item_id':20,'like':True},{'user_id':1,'item_id':10,'like':True},
                   {'user_id':2,'item_id':20,'like':True}])]
        holdout=(25,[{'user_id':1,'item_id':999,'like':True},{'user_id':1,'item_id':10,'like':True},
                     {'user_id':2,'item_id':20,'like':True},{'user_id':3,'item_id':10,'like':True}])
        rows=build_queries(train,holdout,{'min_history':1,'max_history':50},{10,20})['queries']
        with tempfile.TemporaryDirectory(dir='/private/tmp') as directory:
            path=Path(directory)/'validation-queries.json';pin=write(path,rows)
            context=gd._Contract({'test_gate':'closed','inputs':{'validation':{'role':'validation-queries.json','artifact_sha256':pin},
                       'test_declaration':{'role':'test-queries.json','artifact_sha256':'a'*64,'read_allowed':False}}},ROOT)
            loaded=gd.load_validation(path,context,np.array([10,20],np.uint32))
            self.assertEqual([q['user_id'] for q in loaded],[1,2])
            self.assertEqual(loaded[0]['target'],999);self.assertEqual(loaded[0]['history'],[10,20,10])
            result=metrics([[10,20],[20]],loaded,[10,20],(10,20,50))
            self.assertEqual(result['all']['count'],2);self.assertEqual(result['all']['recall@50'],0.5)
            self.assertEqual(result['oov']['count'],1)

    def test_test_or_unknown_role_and_test_hash_rejected_before_file_open(self):
        inputs={'validation':{'role':'validation-queries.json','artifact_sha256':'a'*64},
                'test_declaration':{'role':'test-queries.json','artifact_sha256':'b'*64,'read_allowed':False}}
        for name in ['test-queries.json','unknown.json']:
            with self.subTest(name=name),patch.object(Path,'open',side_effect=AssertionError('test file opened')),patch.object(gd,'_checked_file',side_effect=AssertionError('test file hashed')):
                with self.assertRaisesRegex(ValueError,'validation|test'):
                    gd.load_validation(Path('/private/tmp')/name,gd._Contract({'test_gate':'closed','inputs':inputs},ROOT),np.array([10],np.uint32))
        for edit in [lambda d:d['validation'].update(role='test-queries.json'),
                     lambda d:d['validation'].update(artifact_sha256='b'*64)]:
            bad=copy.deepcopy(inputs);edit(bad)
            with patch.object(Path,'open',side_effect=AssertionError('file opened')):
                with self.assertRaises(ValueError):gd.load_validation(Path('/private/tmp/validation-queries.json'),gd._Contract({'test_gate':'closed','inputs':bad},ROOT),np.array([10],np.uint32))

    def test_validation_typed_schema_order_and_oov_flags_reject(self):
        row={'user_id':1,'history':[10,20,10],'history_positions':[[0,0],[0,1],[0,2]],'target':999,'target_position':[25,0],'oov':True}
        edits=[lambda q:q.update(user_id=True),lambda q:q.update(target=True),lambda q:q.update(oov=1),
               lambda q:q['target_position'].__setitem__(0,25.0),lambda q:q['target_position'].__setitem__(0,True),
               lambda q:q['target_position'].__setitem__(0,26),lambda q:q['history_positions'].__setitem__(1,[0,0]),
               lambda q:q['history_positions'].__setitem__(1,[25,1]),lambda q:q.update(history=[]),lambda q:q['history'].__setitem__(0,True)]
        with tempfile.TemporaryDirectory(dir='/private/tmp') as directory:
            path=Path(directory)/'validation-queries.json'
            for edit in edits:
                bad=copy.deepcopy(row);edit(bad);pin=write(path,[bad])
                context=gd._Contract({'test_gate':'closed','inputs':{'validation':{'role':'validation-queries.json','artifact_sha256':pin},'test_declaration':{'artifact_sha256':'a'*64}}},ROOT)
                with self.assertRaises(ValueError):gd.load_validation(path,context,np.array([10,20],np.uint32))
            pin=write(path,[row,row]);context=gd._Contract({'test_gate':'closed','inputs':{'validation':{'role':'validation-queries.json','artifact_sha256':pin},'test_declaration':{'artifact_sha256':'a'*64}}},ROOT)
            with self.assertRaises(ValueError):gd.load_validation(path,context,np.array([10,20],np.uint32))

class TargetTests(unittest.TestCase):
    def codec(self):
        from vkgr.sid import SIDCatalog
        from vkgr.tiger import SIDTokens
        return SIDTokens(SIDCatalog(np.array([10,20,30],np.uint32),np.array([[0,0,0,0],[0,0,0,1],[1,0,0,0]],np.uint32),3,3,[]))

    def test_teacher_forcing_literal_four_namespaces_padding_and_no_negative_mask(self):
        codec=self.codec()
        source,decoder,labels=gd.sid_targets([{'history':[10,20],'target':20,'past_likes':[10,20]}, {'history':[30],'target':10}],codec)
        self.assertEqual(source.tolist(),[[2,5,8,11,2,5,8,12],[3,5,8,11,0,0,0,0]])
        self.assertEqual(decoder.tolist(),[[1,2,5,8],[1,2,5,8]])
        self.assertEqual(labels.tolist(),[[2,5,8,12],[2,5,8,11]])
        self.assertEqual(codec.vocab_size,13)
        for position in range(4):
            self.assertTrue(((labels[:,position]>=codec.offsets[position]) & (labels[:,position]<codec.offsets[position]+codec.catalog.bases[position])).all())
        source,_,_=gd.sid_targets([{'history':[10]*50,'target':20}],codec)
        self.assertEqual(source.shape,(1,200))

    def test_target_history_oov_bool_empty_too_long_and_duplicate_sid_rejected(self):
        codec=self.codec()
        cases=[[],[{'history':[],'target':10}],[{'history':[999],'target':10}],[{'history':[True],'target':10}],
               [{'history':[10],'target':True}],[{'history':[10],'target':999}],[{'history':[10]*51,'target':20}], [{'history':[10]}]]
        for rows in cases:
            with self.subTest(rows=rows),self.assertRaises(ValueError):gd.sid_targets(rows,codec)
        codec.catalog.sids[1]=codec.catalog.sids[0]
        with self.assertRaises(ValueError):gd.sid_targets([{'history':[10],'target':20}],codec)

class CrossInputIdentityTests(unittest.TestCase):
    def test_sid_rejects_prepared_context_from_different_frozen_input(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as directory:
            out,context,_=prepared_fixture(directory);prepared=gd.load_prepared(out/'prepare-manifest.json',context)
            sid,context=sid_fixture(directory,prepared,context)
            value=copy.deepcopy(dict(context));value['inputs']['prepared']['manifest_sha256']='e'*64
            with self.assertRaisesRegex(ValueError,'prepared.*identity|prepared.*binding'):
                gd.load_sid(sid/'run_manifest.json',gd._Contract(value,ROOT),prepared)

class CodecCostTests(unittest.TestCase):
    def test_canonical_validation_uses_grouped_scan_not_per_leaf_full_catalog_scan(self):
        from vkgr.sid import SIDCatalog
        from vkgr.tiger import SIDTokens
        n=120
        ids=np.arange(10,10+n,dtype=np.uint32)
        sids=np.array([[i//256,(i//16)%16,i%16,0] for i in range(n)],np.uint32)
        codec=SIDTokens(SIDCatalog(ids,sids,16,3,[]))
        all_fn=np.all;full_scans=[]
        def observe(value,*args,**kwargs):
            if getattr(value,'shape',None)==(n,3):full_scans.append(1)
            return all_fn(value,*args,**kwargs)
        with patch.object(np,'all',side_effect=observe):gd._check_codec(codec)
        self.assertLessEqual(len(full_scans),1,'must not scan full catalog once per semantic leaf')

    def test_loaded_readonly_codec_batch_fast_path_rejects_replaced_identity(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as directory:
            out,context,_=prepared_fixture(directory);prepared=gd.load_prepared(out/'prepare-manifest.json',context)
            sid,context=sid_fixture(directory,prepared,context);codec=gd.load_sid(sid/'run_manifest.json',context,prepared)
            with patch.object(gd,'_check_codec',side_effect=AssertionError('full catalog audit in batch hot path')):
                _,_,labels=gd.sid_targets([{'history':[10],'target':20}],codec)
                self.assertEqual(labels.shape,(1,4))
                codec.catalog.sids=codec.catalog.sids.copy()
                with self.assertRaisesRegex(ValueError,'identity|readonly'):
                    gd.sid_targets([{'history':[10],'target':20}],codec)

class CodecLayoutTests(unittest.TestCase):
    def test_readonly_stride_mutation_rejected_before_token_generation(self):
        for name in ['ids','sids','keys','key_items']:
            with self.subTest(array=name), tempfile.TemporaryDirectory(dir='/private/tmp') as directory:
                out,context,_=prepared_fixture(directory);prepared=gd.load_prepared(out/'prepare-manifest.json',context)
                sid,context=sid_fixture(directory,prepared,context);codec=gd.load_sid(sid/'run_manifest.json',context,prepared)
                value=getattr(codec.catalog,name)
                with self.assertRaises(ValueError):value.setflags(write=True)
                value.strides=(0,*value.strides[1:])
                with self.assertRaisesRegex(ValueError,'identity|readonly'):
                    gd.sid_targets([{'history':[10],'target':20}],codec)
