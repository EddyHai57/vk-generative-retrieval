"""Only SID input adapts; frozen A1 model/train/decode implementations stay unchanged."""
import importlib
import json
from pathlib import Path
import unittest
from unittest import mock
import numpy as np
from vkgr import generative_g6_run as old
from vkgr import sid_v2
from vkgr import sid_depth
from g6_fixtures import balanced_codes
ROOT=Path(__file__).parents[1]
LABEL='SID_BHKM_B1024D3_S42'
class ConfirmationChecks(unittest.TestCase):
 def new(self):return importlib.import_module('vkgr.generative_g6_tok2_run')
 def test_only_sid_changes_against_each_seed_control(self):
  n=self.new()
  for seed in (42,2024,2025):
   base=json.loads((ROOT/f'configs/gtok01b/g-gtok-a1-t3-b256d3-s{seed}.json').read_text());new=json.loads((ROOT/f'configs/gtok02a1/g-gtok02-a1-b1024d3-s{seed}.json').read_text())
   self.assertEqual({k for k in base if base[k]!=new[k]},{'sid'});self.assertEqual(new['sid'],LABEL);n.validate_config(new,False)
   with self.assertRaises(ValueError):old.validate_config(new,False)
 def test_formal_registry_accepts_only_selected_new_variant(self):
  n=self.new();c=json.loads((ROOT/'configs/gtok01b/g-gtok-a1-t3-b256d3-s42.json').read_text())
  for bad in ('SID_BHKM_B512D3_S42','SID_BHKM_B256D2_S42','SID_V2S_REPLAY_BHKM_B16D3'):
   with self.assertRaises(ValueError):n.validate_config({**c,'sid':bad},False)
 def test_real_loader_keeps_engineering_false_and_binds_depth3_codec(self):
  n=self.new();ids=np.arange(1,1031,dtype=np.uint32);codes=balanced_codes(len(ids),1024,3).astype(np.uint32);sids=sid_depth.assign_suffix(ids,codes);codec=sid_v2.build_codec(ids,sids,1024)
  loaded=sid_v2.LoadedSID(codec,{'content_feature_sha256':'fixture'},{'sid_version':LABEL,'branching':1024,'depth':3,'engineering':False})
  with mock.patch.object(n,'_content_rows',return_value=(ids,np.zeros((len(ids),64),dtype=np.float32))),mock.patch.object(n.sid_v2s,'load_bundle',return_value=loaded) as fn:
   value=n._load_sid_sweep('manifest','pin','content','pin',ids,False)
   self.assertIs(value.info['engineering'],False);self.assertEqual(value.codec.depth,3);self.assertEqual(value.codec.vocab_size,3075)
   self.assertIs(fn.call_args.kwargs['engineering'],False);self.assertTrue(np.array_equal(value.codec.catalog.sids,sids))
 def test_sweep_loader_refuses_depth2(self):
  n=self.new();loaded=mock.Mock();loaded.info={'depth':2}
  with mock.patch.object(n,'_content_rows',return_value=([],[])),mock.patch.object(n.sid_v2s,'load_bundle',return_value=loaded):
   with self.assertRaises(ValueError):n._load_sid_sweep('m','s','c','s',[],True)
 def test_source_members_include_new_contract_and_runner(self):
  n=self.new();self.assertIn('src/vkgr/sid_v2s.py',n.G6_MEMBERS);self.assertIn('src/vkgr/generative_g6_tok2_run.py',n.G6_MEMBERS)
  self.assertIs(n.make_model_arch_d,old.make_model_arch_d);self.assertIs(n.train_batch_d,old.train_batch_d)
 def test_manifest_source_carries_public_target_and_pin_hash(self):
  n=self.new()
  import tempfile,hashlib
  with tempfile.TemporaryDirectory() as temp:
   p=Path(temp)/'pin.json';p.write_text(json.dumps({'schema':'vkgr-pinned-v1','base_commit':'0'*40,'target_commit':'a'*40}))
   value=n.read_deployment_ref(p);self.assertEqual(value['target_commit'],'a'*40);self.assertEqual(value['pinned_manifest_sha256'],hashlib.sha256(p.read_bytes()).hexdigest())
   p.write_text(json.dumps({'schema':'vkgr-pinned-v1','base_commit':'0'*40,'target_commit':'wrong'}))
   with self.assertRaises(ValueError):n.read_deployment_ref(p)
if __name__=='__main__':unittest.main()
