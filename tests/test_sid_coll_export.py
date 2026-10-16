import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from vkgr import sid_coll_export as ex, sid_v2, sid_tok_audit as au
from vkgr.data_audit import canonical_bytes,file_sha256

class ExportTests(unittest.TestCase):
    def test_metadata_no_checkpoint_read_and_wrong_source_seed_last_sha_rejected(self):
        ids=np.arange(4,dtype='uint32');c=np.ones((5,64),dtype='float32');c[0]=0
        support=np.array([False,True,False,True,False])
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            root=Path(d);(root/'src/vkgr').mkdir(parents=True);(root/'src/vkgr/__init__.py').write_text('')
            files={'src/vkgr/__init__.py':file_sha256(root/'src/vkgr/__init__.py')}
            lock={'schema':'coll-c2-source-lock-v1','source_commit':ex.TEACHER_COMMIT,'source_files':files,'checkpoint_sha256':'a'*64,
              'catalog_sha256':sid_v2.catalog_digest(ids),'supported_sha256':hashlib.sha256(support.tobytes()).hexdigest(),'train_sequence_sha256':'d'*64}
            binding={'catalog_sha256':ex._legacy_digest(ids.astype(int).tolist()),'support_sha256':ex._legacy_digest([0,2]),
              'content_sha256':hashlib.sha256(c.tobytes()).hexdigest(),'source':{'files':files}}
            value={'schema':'p-seqm1-train-manifest-v1','kind':'p_seqm1_train','status':'complete','config':{'arm':'mean','seed':42},
              'binding':binding,'best':{'path':'best.pt','sha256':'a'*64,'epoch':3,'step':4}}
            def call(v=value,l=lock):
                (root/'manifest.json').write_bytes(canonical_bytes(v))
                return ex.metadata_preflight(root,root/'manifest.json',file_sha256(root/'manifest.json'),l,ids,c,support,engineering=True)
            with patch.object(ex,'_git',return_value=ex.TEACHER_COMMIT),patch('torch.load',side_effect=AssertionError('metadata must not load checkpoint')):
                self.assertEqual(call()['status'],'metadata_verified')
                self.assertFalse((root/'best.pt').exists())
                for v in [{**value,'config':{'arm':'mean','seed':2024}},{**value,'best':{**value['best'],'path':'last.pt'}},
                          {**value,'best':{**value['best'],'sha256':'f'*64}},{**value,'binding':{**binding,'support_sha256':'f'*64}}]:
                    with self.assertRaises(ValueError):call(v)
                with self.assertRaises(ValueError):call(l={**lock,'source_commit':'f'*40})
                (root/'src/vkgr/__init__.py').write_text('tampered')
                with self.assertRaises(ValueError):call()

    def test_verify_replays_export_without_checkpoint(self):
        ids=np.arange(4,dtype='uint32');c=np.ones((5,64),dtype='float32');c[0]=0
        support=np.array([False,True,False,True,False]);p=np.zeros((4,64),dtype='float32');p[0,0]=2
        x,audit=ex.compose_c2(p,c[1:],support[1:])
        lock={'checkpoint_sha256':'a'*64,'source_files':{'src/vkgr/teacher.py':'b'*64},'supported_sha256':hashlib.sha256(support.tobytes()).hexdigest(),'train_sequence_sha256':'d'*64}
        m={'schema':'coll-input-produced-v1','status':'produced','variant':'C2','dimension':128,'engineering':True,
          'catalog_sha256':sid_v2.catalog_digest(ids),'vectors_sha256':au.feature_digest(x),'content_feature_sha256':au.feature_digest(c[1:]),
          'train_sequence_sha256':'d'*64,'script_sha256':'c'*64,'commit':'e'*40,'checkpoint_sha256':'a'*64,
          'source_files':{'src/vkgr/sid_coll_export.py':'c'*64},'teacher_source_commit':ex.TEACHER_COMMIT,'teacher_source_files':lock['source_files'],'audit':audit}
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            out=Path(d)
            for name,array in [('p-vectors.npy',p),('vectors.npy',x),('supported.npy',support)]:np.save(out/name,array)
            m['artifacts']={n:file_sha256(out/n) for n in ('p-vectors.npy','vectors.npy','supported.npy')}
            (out/'producer-manifest.json').write_bytes(canonical_bytes(m));sha=file_sha256(out/'producer-manifest.json')
            with patch('torch.load',side_effect=AssertionError('verify must not load any checkpoint')):
                result=ex.verify_c2_export(out,sha,c,ids,lock,engineering=True)
                self.assertEqual(result['status'],'verified');self.assertEqual(result['input_manifest']['status'],'accepted')
                with self.assertRaises(ValueError):ex.verify_c2_export(out,sha,c,ids,{**lock,'checkpoint_sha256':'f'*64},engineering=True)
                np.save(out/'vectors.npy',x*0)
                with self.assertRaises(ValueError):ex.verify_c2_export(out,sha,c,ids,lock,engineering=True)
