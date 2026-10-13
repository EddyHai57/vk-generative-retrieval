import unittest
from pathlib import Path
import importlib
class DataTests(unittest.TestCase):
 def module(self):
  self.assertIsNotNone(importlib.util.find_spec('vkgr.retrieval_data'), 'retrieval data behavior missing')
  return importlib.import_module('vkgr.retrieval_data')
 def test_catalog_pad_and_frozen_identity(self):
  m=self.module(); self.assertEqual(m.catalog_map([0,5,9]),{0:1,5:2,9:3})
  for bad in ([5,0],[1,1],[-1],[True]):
   with self.assertRaises(ValueError):m.catalog_map(bad)
  pins=m.validate_contract(Path(__file__).resolve().parents[1]);self.assertEqual(pins['protocol_version'],'traditional-like-v1')
 def test_prefix_identity_full_past_and_legacy_parity(self):
  m=self.module();self.assertTrue(hasattr(m,'prefix_examples'),'prefix stream missing')
  rows=[{'user_id':7,'item_id':i,'like':True} for i in range(53)]
  rows.insert(1,{'user_id':7,'item_id':99,'like':False})
  examples=list(m.prefix_examples([(0,rows)]))
  self.assertEqual(len(examples),52);last=examples[-1]
  self.assertEqual(last['history'],list(range(2,52)));self.assertEqual(last['past_likes'],list(range(52)))
  self.assertEqual(last['target_position'],[0,53]);self.assertEqual(last['example_id'],51)
  from vkgr.data_audit import training_examples
  old=training_examples([(0,rows)],{'min_history':1,'max_history':50})
  self.assertEqual([{k:v for k,v in r.items() if k not in ('past_likes','example_id')} for r in examples],old)
  with self.assertRaises(ValueError):list(m.prefix_examples([(25,rows)]))
  with self.assertRaises(ValueError):list(m.prefix_examples([(0,[{'user_id':1,'item_id':2,'like':None}])]))
 def test_requested_split_only_and_feature_join(self):
  m=self.module();self.assertTrue(hasattr(m,'read_split'),'split reader missing')
  import tempfile,json,numpy as np
  from vkgr.data_audit import file_sha256
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'validation-queries.json';p.write_text(json.dumps([{'user_id':1,'history':[5],'history_positions':[[0,0]],'target':9,'target_position':[25,0],'oov':False}]))
   self.assertEqual(m.read_split(tmp,'validation',file_sha256(p),[5,9])[0]['target'],9)
   with self.assertRaises(ValueError):m.read_split(tmp,'test','bad',[5,9])
  ids=np.array([9,5],dtype=np.uint32);f=np.zeros((2,64),dtype=np.float16);f[0,0]=3;f[0,1]=4
  joined=m.content_join([5,9],ids,f);self.assertEqual(joined.shape,(3,64));self.assertEqual(float(joined[2,0]),np.float32(.6));self.assertTrue(np.array_equal(joined[0],np.zeros(64)))
  with self.assertRaises(ValueError):m.content_join([5,8],ids,f)
  f[0,0]=np.nan
  with self.assertRaises(ValueError):m.content_join([5,9],ids,f)
 def test_exclusive_prepared_artifacts_and_hash_verification(self):
  m=self.module();self.assertTrue(hasattr(m,'write_prepared'),'prepared artifact behavior missing')
  import tempfile,numpy as np
  rows=list(m.prefix_examples([(0,[{'user_id':1,'item_id':i,'like':True} for i in [5,9,5]])]))
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'prepare';manifest=m.write_prepared(path,[5,9],np.zeros((3,64),dtype=np.float32),rows,{'consumer':'synthetic'},[5,9])
   m.verify_prepared(path);self.assertEqual(manifest['examples'],2)
   with self.assertRaises(FileExistsError):m.write_prepared(path,[5,9],np.zeros((3,64),dtype=np.float32),rows,{},[5,9])
   with (path/'examples.jsonl').open('ab') as stream:stream.write(b'{}\n')
   with self.assertRaises(ValueError):m.verify_prepared(path)
 def test_single_like_user_item_supported_without_example(self):
  m=self.module();import inspect,tempfile,numpy as np
  self.assertIn('supported_items',inspect.signature(m.write_prepared).parameters,'full train positive support missing')
  parts=[(0,[{'user_id':1,'item_id':5,'like':True},{'user_id':1,'item_id':9,'like':True},{'user_id':2,'item_id':12,'like':True}])]
  rows=list(m.prefix_examples(parts));self.assertEqual(len(rows),1)
  with tempfile.TemporaryDirectory() as tmp:
   out=Path(tmp)/'prepared';m.write_prepared(out,[5,9,12],np.zeros((4,64),np.float32),rows,{},supported_items=[5,9,12])
   import json
   self.assertEqual(json.loads((out/'idmap.json').read_text())['supported_indices'],[1,2,3]);m.verify_prepared(out)
