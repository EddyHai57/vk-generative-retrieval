import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from analysis import sgns_gpu_subset as subject


class SubsetInputs(unittest.TestCase):
    def test_fixed_records_no_holdout(self):
        records = subject.fixed_records()
        self.assertEqual(len(records), 26)
        self.assertEqual([r['path'] for r in records[:25]],
                         [f'subsamples/ur0.01_ir0.01/train/week_{w:02d}.parquet' for w in range(25)])
        self.assertEqual(records[-1]['path'], 'metadata/item_embeddings.npz')
        self.assertFalse(any('/test/' in r['path'] or '/validation/' in r['path'] for r in records))

    def test_physical_order_and_all_events(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp).resolve()/'week_00.parquet'
            rows = [{'user_id': 3, 'item_id': 9}, {'user_id': 1, 'item_id': 7},
                    {'user_id': 3, 'item_id': 7}, {'user_id': 3, 'item_id': 9}] + [{'user_id':3,'item_id':9}]*70
            pq.write_table(pa.Table.from_pylist(rows), p, row_group_size=2)
            histories, catalog = subject.load_sequences([p])
            self.assertEqual(histories, {3: [9, 7, 9]+[9]*70, 1: [7]})
            self.assertEqual(catalog, [7, 9])

    def test_checked_input_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)/'x'; p.write_bytes(b'original')
            with self.assertRaises(ValueError):
                subject.checked_file(p, {'size':8, 'sha256':'0'*64})

    def test_real_cli(self):
        run = subprocess.run([sys.executable, '-B', '-m', 'analysis.sgns_gpu_subset', 'dry'],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        value = json.loads(run.stdout)
        self.assertFalse(value['training'])
        self.assertEqual(value['files'], 26)
        invalid = subprocess.run([sys.executable, '-B', '-m', 'analysis.sgns_gpu_subset', 'bad'],
                                 capture_output=True, text=True)
        self.assertNotEqual(invalid.returncode, 0)


class FitBinding(unittest.TestCase):
    def test_device_is_only_scientific_argument(self):
        from unittest.mock import patch
        import numpy as np
        vectors = np.zeros((3,64), dtype=np.float32)
        marker = {'health':{'passed': True}}
        bundle = ([1,2,3], {7:[1,2,3]}, vectors)
        for device in ('cpu', 'cuda'):
            with patch('vkgr.sid_coll_sgns.fit_sgns', return_value=(vectors, marker)) as original:
                result = subject.original_fit(bundle, device)
                self.assertIs(result[0], vectors)
                self.assertEqual(original.call_args.kwargs, {'device':device,'engineering':False})
                self.assertEqual(original.call_args.args[0], bundle[0])
                self.assertEqual(original.call_args.args[1], bundle[1])
                np.testing.assert_array_equal(original.call_args.args[2], vectors)

    def test_cuda_failure_never_falls_back(self):
        from unittest.mock import patch
        with patch('vkgr.sid_coll_sgns.fit_sgns', side_effect=RuntimeError('unsupported cuda kernel')) as original:
            with self.assertRaisesRegex(RuntimeError,'unsupported cuda kernel'):
                subject.original_fit(([1,2,3],{},None), 'cuda')
            self.assertEqual(original.call_count, 1)

    def test_bound_bundle_rejects_manifest_drift_before_payload(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'bundle';p.mkdir();(p/'manifest.json').write_text('{}')
            with self.assertRaises(ValueError):
                subject.read_bundle(p, '0'*64, {})

class MemoryAndSourceBinding(unittest.TestCase):
    def test_content_loaded_after_hash_must_equal_pinned_bytes(self):
        import hashlib
        import numpy as np
        from unittest.mock import patch
        from analysis.ur001_derive_cli import encoded
        from vkgr.sid_coll_sgns import _sequence_digest
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve(); ids=list(range(1,120416))
            good=np.ones((120416,64),dtype=np.float32);good[0]=0
            (root/'catalog.json').write_bytes(encoded(ids))
            np.save(root/'content.npy',good,allow_pickle=False)
            (root/'histories.jsonl').write_bytes(encoded([7,[1,2,3]]))
            artifacts={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                       for name in ('catalog.json','content.npy','histories.jsonl')}
            value={'schema':'sgns-subset-physical-input-v1','status':'prepared','engineering':False,
                   'source':{},'records':subject.fixed_records(),'artifacts':artifacts,
                   'train_sequence_sha256':_sequence_digest({7:[1,2,3]}),'users':1,'events':3}
            (root/'manifest.json').write_bytes(encoded(value))
            pin=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()
            with patch.object(subject,'CATALOG_SHA',artifacts['catalog.json']), \
                 patch.object(subject,'CONTENT_SHA',artifacts['content.npy']):
                loaded,_=subject.read_bundle(root,pin,{})
                np.testing.assert_array_equal(loaded[2],good[1:])
                with self.assertRaisesRegex(ValueError,'manifest drift'):
                    subject.read_bundle(root,'0'*64,{})
            bad=good.copy();bad[3,7]=99
            with patch.object(subject,'CATALOG_SHA',artifacts['catalog.json']), \
                 patch.object(subject,'CONTENT_SHA',artifacts['content.npy']), \
                 patch('numpy.load',return_value=bad):
                with self.assertRaisesRegex(ValueError,'content|payload'):
                    subject.read_bundle(root,pin,{})

    def test_preflight_rechecks_source_after_reading(self):
        import hashlib
        import io
        import numpy as np
        from unittest.mock import patch
        from analysis.ur001_derive_cli import encoded
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve(); np.savez(root/'features.npz',item_id=[1],embedding=[1])
            ids=list(range(1,120416));content=np.ones((120416,64),dtype=np.float32);content[0]=0
            buffer=io.BytesIO();np.save(buffer,content,allow_pickle=False)
            binding={'source':{'commit':'before'},'runtime':{'runtime':{}},'records':[{'size':(root/'features.npz').stat().st_size,'sha256':hashlib.sha256((root/'features.npz').read_bytes()).hexdigest()}]}
            with patch.object(subject,'inspect',return_value=([root/'features.npz'],binding)), \
                 patch.object(subject,'load_sequences',return_value=({7:[1,2,3]},ids)), \
                 patch('vkgr.retrieval_data.content_join',return_value=content), \
                 patch.object(subject,'CATALOG_SHA',hashlib.sha256(encoded(ids)).hexdigest()), \
                 patch.object(subject,'CONTENT_SHA',hashlib.sha256(buffer.getbuffer()).hexdigest()), \
                 patch('analysis.ur001_derive_cli.source_identity',return_value={'commit':'after'}), \
                 patch('analysis.new_instance_prepare.runtime',return_value={'runtime':{}}):
                with self.assertRaisesRegex(ValueError,'source|runtime'):
                    subject.prepare(root,root,root/'lock',root/'new-output',readonly=True)

class RawSnapshot(unittest.TestCase):
    def test_physical_reader_rejects_actual_bytes_not_matching_pin(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp).resolve()/'week_00.parquet'
            pq.write_table(pa.Table.from_pylist([{'user_id':1,'item_id':2}]),p)
            with self.assertRaisesRegex(ValueError,'SHA|size'):
                subject.load_sequences([p],records=[{'size':p.stat().st_size,'sha256':'0'*64}])

class OriginalAuditVerification(unittest.TestCase):
    def test_original_verifier_body_bytes(self):
        import inspect
        import hashlib
        from vkgr.sgns_subset_verify import verify_c1_audit
        self.assertEqual(hashlib.sha256(inspect.getsource(verify_c1_audit).encode()).hexdigest(),
                         'fa9dc9d186d2810177d8a9382b00acfc03424647e40cdeaa0e09b941678ed143')

    def test_fit_verification_does_not_relax_formal_mode(self):
        from unittest.mock import patch
        with patch('vkgr.sgns_subset_verify.verify_c1_audit',return_value={'status':'verified'}) as verifier:
            receipt=subject.verify_original_fit(([1,2,3],{7:[1,2,3]},None),None,{})
            self.assertEqual(receipt['status'],'verified')
            self.assertEqual(verifier.call_args.kwargs,{'engineering':False})

class PersistedOutputBinding(unittest.TestCase):
    def test_output_changed_after_verify_is_not_signed(self):
        from unittest.mock import patch
        import numpy as np
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve(); output=root/'new-fit'
            vectors=np.ones((3,64),dtype=np.float32)
            inputs=([1,2,3],{7:[1,2,3]},vectors)
            def verify_then_change(*args):
                with (output/'vectors.npy').open('wb') as f:
                    np.save(f,np.zeros_like(vectors),allow_pickle=False)
                return {'status':'verified'}
            with patch('analysis.ur001_derive_cli.source_identity',return_value={}), \
                 patch('analysis.new_instance_prepare.runtime',return_value={}), \
                 patch('analysis.ur001_derive_cli.guards'), \
                 patch.object(subject,'read_bundle',return_value=(inputs,{})), \
                 patch.object(subject,'original_fit',return_value=(vectors,{})), \
                 patch.object(subject,'verify_original_fit',side_effect=verify_then_change):
                with self.assertRaisesRegex(ValueError,'output|artifact'):
                    subject.fit(root/'bundle','0'*64,'cpu',root/'lock',output)

if __name__ == '__main__': unittest.main()
