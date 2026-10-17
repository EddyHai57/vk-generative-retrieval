import unittest
from analysis import profile_readout as p
class ProfileReadoutTests(unittest.TestCase):
 def test_registered_runs_are_exact_nine(self):
  self.assertEqual(len(p.registered()),9)
  self.assertIn('p-profile-u3-s2025',p.registered())
  self.assertNotIn('p-profile-u2-s2026',p.registered())
 def test_new_raw_verifier_identity_is_required(self):
  rid='p-profile-u2-s42';pin='a'*64;r={'status':'verified','run_id':rid+'-verify','train_run_id':rid,'manifest_sha256':pin}
  p.raw_verified([r],rid,pin)
  for patch in ({'run_id':'other'},{'train_run_id':'other'},{'manifest_sha256':'b'*64},{'status':'complete'}):
   with self.assertRaises(ValueError):p.raw_verified([{**r,**patch}],rid,pin)
  with self.assertRaises(ValueError):p.raw_verified([{'status':'verified'}],rid,pin)
 def test_frozen_four_rows_boundaries(self):
  self.assertEqual(p.classify(-6,3,.001),1);self.assertEqual(p.classify(6,2,.049),2)
  self.assertEqual(p.classify(6,2,.05),3);self.assertEqual(p.classify(6,1,.01),3)
  self.assertEqual(p.classify(5.999,3,.001),4)
 def test_h50_slices_and_empty_denominator(self):
  x=p.slices([1,50,50],[1,1,0],[0,0,1]);self.assertEqual(x['H50_underfull']['delta_hits'],1);self.assertEqual(x['H50_full']['delta_hits'],0)
  self.assertIsNone(p.slices([50],[1],[0])['H50_underfull']['new_recall'])
  for args in [([51],[1],[0]),([1],[2],[0]),([1,50],[1],[0])]:
   with self.assertRaises(ValueError):p.slices(*args)
 def test_latency_packing_is_paired_and_does_not_draw_rng(self):
  import torch
  g=torch.Generator().manual_seed(7);h=torch.ones(2,4,dtype=torch.long);s=torch.zeros(2,4,6,dtype=torch.long);before=g.get_state().clone();packed=p.pack_side(h,s)
  self.assertEqual(packed.shape,(2,4,9));self.assertTrue(torch.equal(s,packed[...,:6]));self.assertTrue(torch.equal(before,g.get_state()))
if __name__=='__main__':unittest.main()

class TimingImportTests(unittest.TestCase):
 def test_preflight_pins_model_import_root_before_loading_timing_source(self):
  import sys,types
  from unittest.mock import patch
  from analysis import postbatch_latency
  req={'profile_source':{'root':'/fixture/profile-source'},'lengths':'fixture','lengths_sha256':'a'*64,'bench_source':{}}
  def checked(_):
   self.assertEqual(sys.path[0],'/fixture/profile-source/src')
   return types.SimpleNamespace(load_lengths=lambda *a:{'lengths':[1,50],'histogram':{'1':1,'50':1}})
  with patch.object(sys,'path',list(sys.path)),patch.object(postbatch_latency,'checked_source',side_effect=checked):
   self.assertEqual(p.latency(req,True)['status'],'latency_preflight_ok')

class SourceOriginTests(unittest.TestCase):
 def test_cached_foreign_model_package_is_rejected(self):
  import sys,types
  from unittest.mock import patch
  with patch.dict(sys.modules,{'vkgr.fixture':types.SimpleNamespace(__file__='/fixture/old/src/vkgr/fixture.py')}):
   with self.assertRaises(ValueError):p.model_origins('/fixture/fixed')
  with patch.dict(sys.modules,{'vkgr.fixture':types.SimpleNamespace(__file__='/fixture/fixed/src/vkgr/fixture.py')}):
   self.assertEqual(p.model_origins('/fixture/fixed')['vkgr.fixture'],'vkgr/fixture.py')
