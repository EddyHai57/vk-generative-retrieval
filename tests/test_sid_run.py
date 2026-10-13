import unittest,tempfile,json,hashlib
from pathlib import Path
import numpy as np
from vkgr import sid_run
from vkgr.sid import fit


def fixture_bundle(output,model,evidence,features):
    from vkgr.data_audit import producer_identity,canonical_bytes
    value=sid_run.write_artifacts(output,model,evidence,features)
    value.update({'producer':producer_identity(),'resource_minimum':{'ram_available_fraction':0.9,'disk_available_fraction':0.9},'resource_samples':1,'elapsed_seconds':0.0,'rss_unit':'KiB_on_Linux_bytes_on_macOS','started_at_utc':'2026-10-06T20:00:00+00:00'})
    (output/'run_manifest.json').write_bytes(canonical_bytes(value))
    return value

class SIDArtifactTests(unittest.TestCase):
    def test_static_hash_pin_and_prefix_consistency(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'sid';features=np.zeros((5,64),np.float32);features[:,0]=1
            model=fit([8,2,9,4,1],features)
            manifest=fixture_bundle(output,model,{'frozen_manifest_sha256':sid_run.FROZEN_SHA},features)
            pin=hashlib.sha256((output/'run_manifest.json').read_bytes()).hexdigest()
            loaded=sid_run.load_static(output/'run_manifest.json',pin)
            self.assertEqual(loaded.decode(loaded.encode(8)),8)
            with self.assertRaises(FileExistsError):sid_run.write_artifacts(output,model,{},features)
            path=output/'mapping.npz';data=bytearray(path.read_bytes());data[-1]^=1;path.write_bytes(data)
            with self.assertRaises(ValueError):sid_run.load_static(output/'run_manifest.json',pin)
            manifest['artifacts']['mapping.npz']=hashlib.sha256(path.read_bytes()).hexdigest()
            (output/'run_manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):sid_run.load_static(output/'run_manifest.json',pin)
    def test_config_and_unreachable_artifact_reject(self):
        self.assertEqual(sid_run.validate_config(dict(sid_run.CONFIG)),sid_run.CONFIG)
        for key,value in [('depth',2),('seed',True),('branching',8),('unknown',0)]:
            bad=dict(sid_run.CONFIG);bad[key]=value
            with self.assertRaises(ValueError):sid_run.validate_config(bad)
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'sid';features=np.zeros((2,64),np.float32)
            fixture_bundle(out,fit([1,2],features),{'frozen_manifest_sha256':sid_run.FROZEN_SHA},features)
            with np.load(out/'constraint.npz') as data:keys=data['keys'];items=data['items']
            items=items[::-1];np.savez(out/'constraint.npz',keys=keys,items=items)
            value=json.loads((out/'run_manifest.json').read_text());value['artifacts']['constraint.npz']=hashlib.sha256((out/'constraint.npz').read_bytes()).hexdigest()
            (out/'run_manifest.json').write_text(json.dumps(value))
            pin=hashlib.sha256((out/'run_manifest.json').read_bytes()).hexdigest()
            with self.assertRaises(ValueError):sid_run.load_static(out/'run_manifest.json',pin)

class SIDRunTests(unittest.TestCase):
    def test_frozen_tamper_before_raw_feature_read(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'frozen.json';path.write_bytes((root/'artifacts/p3-data-contract-manifest.json').read_bytes()+b' ')
            with self.assertRaisesRegex(ValueError,'hash'):sid_run.load_train_features(path,Path(directory)/'absent',Path(directory)/'absent-p4.json')
    def test_run_replay_refuses_changed_centers_after_rehash(self):
        from unittest.mock import patch
        features=np.zeros((3,64),np.float32);features[:,0]=1
        inputs={'catalog':[1,2,3],'features':features,'evidence':{'frozen_manifest_sha256':sid_run.FROZEN_SHA,'inputs':[]}}
        with tempfile.TemporaryDirectory() as directory,patch.object(sid_run,'load_train_features',return_value=inputs),patch.object(sid_run,'sample_headroom',return_value={'ram_available_fraction':0.9,'disk_available_fraction':0.9},create=True):
            root=Path(directory);config=root/'config.json';config.write_text(json.dumps(sid_run.CONFIG));out=root/'run'
            sid_run.run('freeze','raw','p4',config,out)
            sid_run.verify_run(out/'run_manifest.json')
            data=json.loads((out/'codebook.json').read_text());data[0]['centers'][0][0]=0.5;(out/'codebook.json').write_text(json.dumps(data))
            manifest=json.loads((out/'run_manifest.json').read_text());manifest['artifacts']['codebook.json']=hashlib.sha256((out/'codebook.json').read_bytes()).hexdigest();(out/'run_manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'replay'):sid_run.verify_run(out/'run_manifest.json')
    def test_resource_gate_rejects_low_headroom(self):
        with self.assertRaises(ValueError):sid_run.check_headroom({'ram_available_fraction':0.19,'disk_available_fraction':0.9})
        with self.assertRaises(ValueError):sid_run.check_headroom({'ram_available_fraction':0.9,'disk_available_fraction':0.19})
        sid_run.check_headroom({'ram_available_fraction':0.2,'disk_available_fraction':0.2})

class SIDResourcePublicationTests(unittest.TestCase):
    def test_final_low_headroom_never_publishes_success_manifest(self):
        from unittest.mock import patch
        features=np.zeros((1,64),np.float32)
        inputs={'catalog':[1],'features':features,'evidence':{'frozen_manifest_sha256':sid_run.FROZEN_SHA,'inputs':[]}}
        samples=iter([{'ram_available_fraction':0.9,'disk_available_fraction':0.9},{'ram_available_fraction':0.9,'disk_available_fraction':0.9},{'ram_available_fraction':0.19,'disk_available_fraction':0.9}])
        with tempfile.TemporaryDirectory() as directory,patch.object(sid_run,'load_train_features',return_value=inputs),patch.object(sid_run,'sample_headroom',side_effect=lambda root:next(samples)):
            root=Path(directory);config=root/'config.json';config.write_text(json.dumps(sid_run.CONFIG));out=root/'run'
            with self.assertRaises(ValueError):sid_run.run('freeze','raw','p4',config,out)
            self.assertFalse((out/'run_manifest.json').exists())
    def test_monitor_sampling_error_propagates_and_never_publishes(self):
        import threading,time
        from unittest.mock import patch
        features=np.zeros((1,64),np.float32)
        inputs={'catalog':[1],'features':features,'evidence':{'frozen_manifest_sha256':sid_run.FROZEN_SHA,'inputs':[]}}
        def sampler(root):
            if threading.current_thread() is not threading.main_thread():raise OSError('synthetic sampling failure')
            return {'ram_available_fraction':0.9,'disk_available_fraction':0.9}
        def loader(*args):time.sleep(0.35);return inputs
        with tempfile.TemporaryDirectory() as directory,patch.object(sid_run,'load_train_features',side_effect=loader),patch.object(sid_run,'sample_headroom',side_effect=sampler):
            root=Path(directory);config=root/'config.json';config.write_text(json.dumps(sid_run.CONFIG));out=root/'run'
            with self.assertRaisesRegex(ValueError,'sampling'):sid_run.run('freeze','raw','p4',config,out)
            self.assertFalse((out/'run_manifest.json').exists())
    def test_static_consumer_rejects_missing_resource_producer_evidence(self):
        from unittest.mock import patch
        features=np.zeros((1,64),np.float32);inputs={'catalog':[1],'features':features,'evidence':{'frozen_manifest_sha256':sid_run.FROZEN_SHA,'inputs':[]}}
        with tempfile.TemporaryDirectory() as directory,patch.object(sid_run,'load_train_features',return_value=inputs),patch.object(sid_run,'sample_headroom',return_value={'ram_available_fraction':0.9,'disk_available_fraction':0.9}):
            root=Path(directory);config=root/'config.json';config.write_text(json.dumps(sid_run.CONFIG));out=root/'run'
            sid_run.run('freeze','raw','p4',config,out)
            path=out/'run_manifest.json';value=json.loads(path.read_text());del value['producer'];del value['resource_minimum'];path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):sid_run.load_static(path,hashlib.sha256(path.read_bytes()).hexdigest())
