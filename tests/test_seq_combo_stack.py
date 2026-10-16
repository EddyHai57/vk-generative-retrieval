"""Stack seam oracles: shared RNG, candidate-only stats, pooling and replay."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import torch
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1]
class StackTests(unittest.TestCase):
 def modules(self):
  self.assertTrue((ROOT/'src/vkgr/retrieval_seq_combo_stack.py').exists(),'stack model missing')
  self.assertTrue((ROOT/'src/vkgr/retrieval_seq_combo_stack_run.py').exists(),'stack runner missing')
  return importlib.import_module('vkgr.retrieval_seq_combo_stack'),importlib.import_module('vkgr.retrieval_seq_combo_stack_run')
 def fixture(self):
  m,r=self.modules();d=r.synthetic_data();c={**r.default_config(),'batch_size':16,'max_epochs':4,'min_epochs':2};model=r._model(d,c,'cpu').eval()
  qs=r.fr._view(d,50)[1];h,s=r.fr._tensors2(qs[:3],d['idmap'],50,c['feature']);return m,r,d,c,model,h,s
 def test_original_m1_blobs_exact(self):
  for f in ('retrieval_seqm1.py','retrieval_seqm1_run.py'):
   p=ROOT/'src/vkgr'/f;self.assertTrue(p.exists(),'pinned dependency missing')
   self.assertEqual(p.read_bytes(),subprocess.check_output(['git','show','04d4ede:src/vkgr/'+f],cwd=ROOT))
 def test_shared_initialization_and_zero_gate(self):
  m,r,d,c,model,h,s=self.fixture()
  from vkgr import retrieval_seqm1 as old
  from vkgr.training import seed_all
  seed_all(c['seed']);control=old.make_seqm1_model(d['content'],d['supported_indices'],50,'mean').eval()
  for k,v in control.state_dict().items():self.assertTrue(torch.equal(v,model.state_dict()[k]),k)
  self.assertTrue(torch.equal(control.users(h,s),model.users(h,s)))
 def test_candidate_only_four_terms_and_immutable_buffers(self):
  m,r,d,c,model,h,s=self.fixture();ids=torch.arange(len(d['content']));base=model.items(ids).detach().clone();u=model.users(h,s).detach().clone();br=model.parts(h,s).r.detach().clone()
  with torch.no_grad():
   for j,n in enumerate(('expo_table','first_table','ratio_stat_table','like_table'),1):getattr(model,n).weight.fill_(j);getattr(model,n).weight[0].zero_()
  self.assertTrue(torch.equal(model.candidate_items(ids),base+10*model.supported.unsqueeze(-1)))
  self.assertTrue(torch.equal(u,model.users(h,s)));self.assertTrue(torch.equal(br,model.parts(h,s).r))
  state=copy.deepcopy(model.state_dict());state['xb'][1]+=1
  with self.assertRaises(ValueError):model.load_state_dict(state)
 def test_manual_beta_pool_and_branch_independent(self):
  m,r,d,c,model,h,s=self.fixture();h=h[:1,:2].clone();h[:]=1;s=s[:1,:2].clone();s.zero_();s[...,1]=10;s[...,2]=20;s[...,3]=4;s[0,1,0]=1
  with torch.no_grad():model.behavior_beta.zero_();model.behavior_beta[0]=torch.log(torch.tensor(3.));model.gate.fill_(.5)
  e=model.event_vectors(h,s,model._check(h,s));self.assertTrue(torch.allclose(model.anchor_vector(h,s),(e[:,0]+3*e[:,1])/4,atol=1e-7,rtol=0))
  br=model.parts(h,s).r.clone();flipped=s.clone();flipped[...,0]^=63
  self.assertTrue(torch.equal(br,model.parts(h,flipped).r))
  with torch.no_grad():model.recency_logit_pos.copy_(torch.arange(7,dtype=torch.float32))
  self.assertTrue(torch.equal(br,model.parts(h,s).r))
 def test_config_target_and_engineering_flag(self):
  m,r,d,c,model,h,s=self.fixture();config=r.default_config()
  for bad in ({'arm':'bi'},{'seed':7},{'temperature':.15},{'feature':'f4_combo_c_d_e'},{'max_epochs':101}):
   with self.assertRaises(ValueError):r.validate_config({**config,**bad})
  with self.assertRaises(ValueError):r.preflight_check(d,config,engineering=False)
  self.assertTrue(r.preflight_check(d,config,engineering=True)['engineering'])
  qs=r.fr._view(d,50)[1][:3];other=copy.deepcopy(qs)
  for q in other:q.update(target=d['catalog'][-1],target_position=[30,999],past_likes=999,oov=True)
  a=r.fr._tensors2(qs,d['idmap'],50,c['feature']);b=r.fr._tensors2(other,d['idmap'],50,c['feature'])
  with torch.no_grad():model.gate.fill_(.5);model.behavior_beta.fill_(.1)
  self.assertTrue(torch.equal(model.users(*a),model.users(*b)))
 def test_new_checkpoint_replay_and_binding_identity(self):
  m,r,d,c,model,h,s=self.fixture()
  with tempfile.TemporaryDirectory() as tmp:
   out=Path(tmp)/'run';v=r.run_session(out,d,c);self.assertEqual(r.verify_run(out/'manifest.json',v['manifest_sha256'],d,c)['status'],'verified')
   v=json.loads((out/'manifest.json').read_text());self.assertIn('branch_spec_sha256',v['binding']);self.assertIn('behavior_beta',v['evaluations'][0]['diagnostics']);v['binding']['stat_cutoff_week']=25
   payload=json.dumps(v).encode();(out/'manifest.json').write_bytes(payload)
   with self.assertRaises(ValueError):r.verify_run(out/'manifest.json',hashlib.sha256(payload).hexdigest(),d,c)
 def test_private_data_contract_closed_by_default_and_hash_drift(self):
  _,r=self.modules()
  self.assertTrue(hasattr(r,'real_inputs'),'private data loader missing')
  from unittest import mock
  with mock.patch.dict('os.environ',{},clear=True):
   with self.assertRaises(ValueError):r.real_inputs({})
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);(root/'docs').mkdir();(root/'docs/p3c_validation.md').write_text('wrong contract')
   with self.assertRaises(ValueError):r.real_inputs({},data_root=root)

 def test_training_reaches_new_pooling_stats_and_branch_after_gate_opens(self):
  m,r,d,c,model,h,s=self.fixture()
  from vkgr.retrieval_training import SamplerStats
  from vkgr.retrieval_negatives import target_frequency
  import random
  rows=r.fr._view(d,50)[0];batch=rows[:16];model.train()
  with torch.no_grad():model.gate.fill_(.2)
  optimizer=r.make_optimizer(model,c)
  r.train_step(model,optimizer,batch,{**d,'rows':rows},SamplerStats(),0,c,random.Random(0),None,target_frequency(rows))
  for name in ('gate','behavior_beta','recency_logit_pos','recency_logit_gap','expo_table.weight','first_table.weight','ratio_stat_table.weight','like_table.weight','layer.linear1.weight'):
   grad=dict(model.named_parameters())[name].grad
   self.assertIsNotNone(grad,name);self.assertTrue(torch.isfinite(grad).all(),name);self.assertGreater(float(grad.abs().sum()),0,name)
 def test_original_config_fields_and_missing_stats_rejected(self):
  _,r,d,c,model,h,s=self.fixture()
  for seed in (42,2024,2025):
   p=ROOT/f'configs/seq_combo_stack/p-seq-combo-stack-s{seed}.json';new=json.loads(p.read_text());r.validate_config(new)
   old=json.loads(subprocess.check_output(['git','show',f'04d4ede:configs/seqm1/p-seqm1-mean-s{seed}.json'],cwd=ROOT))
   self.assertEqual({k:v for k,v in new.items() if k not in ('schema','task_id')},{k:v for k,v in old.items() if k not in ('schema','task_id')})
  bad={k:v for k,v in d.items() if k!='stat_buckets'}
  with self.assertRaises((ValueError,KeyError)):r._model(bad,c,'cpu')
 def test_private_parameters_pin_is_checked_before_loading(self):
  _,r=self.modules();self.assertTrue(hasattr(r,'private_data_parameters'),'pinned private parameters reader missing')
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'parameters.json';p.write_text(json.dumps({'data_root':str(Path(tmp)/'private')}));pin=hashlib.sha256(p.read_bytes()).hexdigest()
   self.assertEqual(r.private_data_parameters(p,pin),str(Path(tmp)/'private'))
   p.write_text(json.dumps({'data_root':'relative'}))
   with self.assertRaises(ValueError):r.private_data_parameters(p,hashlib.sha256(p.read_bytes()).hexdigest())
   with self.assertRaises(ValueError):r.private_data_parameters(p,pin)
 def test_recency_position_and_gap_manual_weights(self):
  _,r,d,c,model,h,s=self.fixture();h=h[:1,:2].clone();h[0]=torch.tensor([1,2]);s=s[:1,:2].clone();s.zero_();s[...,1]=10;s[...,2]=20;s[0,:,3]=torch.tensor([4,8])
  # back indices1,0 => position ids2,1; gaps4,0 => gap ids4,1.
  # position first log2 and gap first log3 give relative weights6:1.
  with torch.no_grad():
   model.recency_logit_pos.zero_();model.recency_logit_gap.zero_();model.behavior_beta.zero_()
   model.recency_logit_pos[1]=torch.log(torch.tensor(2.));model.recency_logit_gap[3]=torch.log(torch.tensor(3.))
  e=model.event_vectors(h,s,model._check(h,s))
  self.assertTrue(torch.allclose(model.anchor_vector(h,s),(6*e[:,0]+e[:,1])/7,atol=1e-7,rtol=0))
 def test_private_root_binding_separate_from_clean_code_root(self):
  _,r=self.modules()
  from unittest import mock
  original_root=r.b1.ROOT
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);(root/'docs').mkdir();doc=root/'docs/p3c_validation.md';doc.write_text('synthetic-contract');(root/'artifacts').mkdir();manifest=root/'artifacts/p3d-retrieval-contract-manifest.json';manifest.write_text('{}')
   pins={'full_query_pins':[{'role':'validation-queries.json','sha256':'a'*64}],'catalog_sha256':'b'*64}
   def contract(where):
    if Path(where)!=root:raise ValueError('wrong data root')
    return pins
   def load(prepared,holdout,prepared_sha,source,query_sha):
    if (prepared,holdout,prepared_sha,query_sha)!=('prepared','holdout',r.b1.PREPARED_SHA,'a'*64):raise ValueError('scientific pin drift')
    return {'rows':[]}
   with mock.patch.object(r,'DATA_CONTRACT_SHA',hashlib.sha256(doc.read_bytes()).hexdigest()),mock.patch('vkgr.retrieval_data.validate_contract',side_effect=contract),mock.patch('vkgr.retrieval_experiment.load_training_inputs',side_effect=load):
    d,_,binding=r.real_inputs({'prepared':'prepared','holdout':'holdout'},data_root=root)
   self.assertFalse(d['synthetic']);self.assertEqual(binding['data_root'],str(root));self.assertEqual(binding['query_sha'],'a'*64);self.assertEqual(binding['data_contract_manifest_sha256'],hashlib.sha256(b'{}').hexdigest());self.assertEqual(r.b1.ROOT,original_root)
if __name__=='__main__':unittest.main()
