"""Registered C+D+E: tests guard real composition, replay and leakage seams."""
import copy
import importlib
import sys
import tempfile
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).parent))
import test_feat02_model as tm
import test_feat04b_model as old_tests
from vkgr import retrieval_feat04b as old
from vkgr import retrieval_feat04b_run as old_run

class ComboChecks(unittest.TestCase):
 def module(self):return importlib.import_module('vkgr.retrieval_feat04combo')
 def runner(self):return importlib.import_module('vkgr.retrieval_feat04combo_run')
 def build(self,seed=42):
  m=self.module();return m.make_feat04b_model(m.E_FEATURE,tm.CONTENT,tm.SUPPORTED,50,seed,old_tests.cd_buckets())
 def test_candidate_four_terms_supported_only_and_history_stays_e(self):
  m=self.build();h,s=tm.model_inputs();e=old.make_feat04b_model(old.E_FEATURE,tm.CONTENT,tm.SUPPORTED,50,42)
  for n,v in e.state_dict().items():self.assertTrue(torch.equal(m.state_dict()[n],v),n)
  with torch.no_grad():
   m.behavior_beta.copy_(torch.linspace(-.4,.6,6));e.behavior_beta.copy_(m.behavior_beta)
   m.recency_logit_pos.copy_(torch.linspace(-.3,.8,7));e.recency_logit_pos.copy_(m.recency_logit_pos)
   for i,(name,_,_) in enumerate(self.module().STAT_LAYOUT[self.module().E_FEATURE]):getattr(m,name).weight[1:].fill_(.1*(i+1))
  self.assertTrue(torch.equal(m.users(h,s),e.users(h,s)))
  ids=torch.arange(301);terms=sum(getattr(m,t)(getattr(m,b)[ids]) for t,b,_ in self.module().STAT_LAYOUT[self.module().E_FEATURE])
  self.assertTrue(torch.equal(m.candidate_items(ids),m.items(ids)+terms*m.supported[ids].unsqueeze(-1)))
  self.assertTrue(torch.equal(m.history_items(ids),e.items(ids)))
  self.assertEqual(float(m.candidate_items(ids)[~m.supported[ids]].detach().abs().sum()),0.)
 def test_stats_only_candidate_gradients_and_immutable_four_buffers(self):
  m=self.build();h,s=tm.model_inputs();m.users(h,s).sum().backward()
  for t,_,_ in self.module().STAT_LAYOUT[self.module().E_FEATURE]:self.assertIsNone(getattr(m,t).weight.grad)
  m.zero_grad(set_to_none=True);m.candidate_items(torch.tensor([1,2,3])).sum().backward()
  for t,b,_ in self.module().STAT_LAYOUT[self.module().E_FEATURE]:
   self.assertGreater(float(getattr(m,t).weight.grad.abs().sum()),0.)
   state=copy.deepcopy(m.state_dict());state[b][1]=0
   with self.assertRaises(ValueError):m.load_state_dict(state)
 def test_formula_parameter_count_and_spec_union(self):
  mod=self.module();m=self.build();self.assertEqual(mod.parameter_report4b(m),mod.expected_param_counts4b(mod.E_FEATURE));self.assertEqual(len(m.stat_buffers),4)
  self.assertEqual(mod.expected_param_counts4b(mod.E_FEATURE)['total'],old.expected_param_counts4b(old.E_FEATURE)['total']+sum((len(getattr(old,e))+2)*64 for _,_,e in mod.STAT_LAYOUT[mod.E_FEATURE]))
  self.assertNotEqual(mod.bucket_spec_sha256(mod.E_FEATURE),old.bucket_spec_sha256(old.E_FEATURE))
 def test_config_refuses_other_variants_seeds_and_budget_changes(self):
  r=self.runner();c=r.default_config();r.validate_config(c)
  for v in ({'feature':old.E_FEATURE},{'seed':7},{'temperature':.15},{'max_epochs':99},{'learning_rate':.0005},{'task_id':'FEAT-04B'}):
   with self.assertRaises(ValueError):r.validate_config({**c,**v})
 def test_preflight_checks_all_four_buckets_and_target_independence(self):
  r=self.runner();d=r.synthetic_data4b();c={**r.default_config(),'max_epochs':4,'min_epochs':2,'batch_size':16}
  v=r.preflight_check(d,c);self.assertEqual(v['status'],'preflight_ok');self.assertEqual(set(v['stat_buckets']),{'xb','fb','rb','lb'})
 def test_replay_retains_both_beta_and_stat_diagnostics_and_rejects_tamper(self):
  r=self.runner();d=r.synthetic_data4b();c={**r.default_config(),'max_epochs':4,'min_epochs':2,'batch_size':16}
  with tempfile.TemporaryDirectory() as temp:
   p=Path(temp)/'run';v=r.run_session(p,d,c);self.assertEqual(r.verify_run(p/'manifest.json',v['manifest_sha256'],d,c)['status'],'verified')
   self.assertIn('beta_bit_train_counts',v['data_diagnostics']);self.assertIn('logq_redundancy',v['data_diagnostics'])
   diag=v['evaluations'][v['selection']['best_epoch']-1]['diagnostics'];self.assertEqual(set(diag['behavior_beta']),set(old.BEHAVIOR_BITS));self.assertEqual(set(diag['top50_stat_profile']),{'slots','expo','first','ratio','like'})
   import json
   manifest=json.loads((p/'manifest.json').read_text());manifest['evaluations'][0]['diagnostics']['top50_stat_profile']['slots']+=1
   pin=r.b1._write_json_atomic(p/'tampered.json',manifest)
   with self.assertRaises(ValueError):r.verify_run(p/'tampered.json',pin,d,c)
 def test_cli_has_no_new_stats_preparation_or_test_command(self):
  parser=self.runner().build_parser()
  choices=next(a.choices for a in parser._actions if hasattr(a,'choices') and isinstance(a.choices,dict))
  self.assertEqual(set(choices),{'train','verify','preflight','synthetic-train','synthetic-verify'})
 def test_old_source_bytes_are_untouched(self):
  import hashlib
  root=Path(__file__).parents[1]
  pins={'src/vkgr/retrieval_feat04b.py': '7313d585fa3c468f4f5fe1c3c279e4a6e89be540c10af6ae943e8f79717147fd', 'src/vkgr/retrieval_feat04b_run.py': '36a6b037bd8ed0b84234a5d9325c012028fad182995c0594e98ad3d0fffe0116', 'configs/feat04b/pinned-files.json': '05627b7cbb7ddddfef809c8710335ba1b4b6bdf9952a4c82f350a208a259e6ed'}
  for rel,pin in pins.items():self.assertEqual(hashlib.sha256((root/rel).read_bytes()).hexdigest(),pin)
if __name__=='__main__':unittest.main()
