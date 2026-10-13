import unittest,importlib
class TrainingTests(unittest.TestCase):
 def module(self):
  self.assertIsNotNone(importlib.util.find_spec('vkgr.retrieval_training'),'training behavior missing')
  return importlib.import_module('vkgr.retrieval_training')
 def test_tail_boundary_and_cursor_resume(self):
  m=self.module();c=m.EpochCursor(5,42);a=c.next(3);state=c.state_dict();b=c.next(3)
  self.assertEqual(len(b),2);self.assertEqual(sorted(a+b),list(range(5)));self.assertEqual(c.epoch,0)
  d=m.EpochCursor(5,99);d.load_state_dict(state);self.assertEqual(d.next(3),b)
  self.assertEqual(d.next(3),c.next(3));self.assertEqual(c.epoch,1)
  boundary=m.EpochCursor(5,0);boundary.load_state_dict(d.state_dict());self.assertEqual(boundary.next(3),d.next(3))
  with self.assertRaises(ValueError):m.EpochCursor(0,1)
 def test_deduplicated_columns_and_full_past_mask(self):
  m=self.module();self.assertTrue(hasattr(m,'candidate_batch'),'candidate mask missing')
  rows=[{'target':3,'past_likes':[1,3]},{'target':2,'past_likes':[1]},{'target':3,'past_likes':[2]}]
  columns,labels,mask=m.candidate_batch(rows,{1:1,2:2,3:3})
  self.assertEqual(columns.tolist(),[2,3]);self.assertEqual(labels.tolist(),[1,0,1]);self.assertEqual(mask.tolist(),[[True,True],[True,True],[False,True]])
  with self.assertRaises(ValueError):m.candidate_batch([{'target':0,'past_likes':[]}],{1:1})
 def test_hand_calculated_ce_and_skip_denominator(self):
  m=self.module();self.assertTrue(hasattr(m,'masked_ce'),'loss missing')
  import torch,math
  user=torch.tensor([[1.,0.],[0.,1.],[1.,0.]],requires_grad=True);item=torch.eye(2)
  loss,skips=m.masked_ce(user,item,torch.tensor([0,1,0]),torch.tensor([[True,True],[True,True],[True,False]]),1.)
  self.assertAlmostEqual(float(loss.detach()),math.log1p(math.exp(-1)),places=6);self.assertEqual(skips,1);loss.backward();self.assertTrue(torch.isfinite(user.grad).all())
  stats=m.SamplerStats();stats.record(0,100,1);stats.check(0)
  stats.record(0,1,1)
  with self.assertRaises(ValueError):stats.check(0)
  # A small incomplete batch must not become its own denominator.
  other=m.SamplerStats();other.record(0,1,1);other.record(0,199,0);other.check(0)
 def test_id_mean_padding_unsupported_and_parameter_update(self):
  m=self.module();self.assertTrue(hasattr(m,'IDMean'),'model adapter missing')
  import torch
  torch.manual_seed(4);model=m.IDMean(4,[1,2,3]);opt=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=1e-4)
  self.assertEqual(tuple(model.embedding.weight.shape),(5,64))
  self.assertTrue(torch.equal(model.items(torch.tensor([0,4])),torch.zeros(2,64)))
  self.assertTrue(torch.allclose(model.users(torch.tensor([[1,2,0]])),model.items(torch.tensor([1,2])).mean(0,keepdim=True)))
  rows=[{'history':[1],'target':2,'past_likes':[1]},{'history':[1],'target':3,'past_likes':[1]}]
  before=model.embedding.weight.detach().clone();stats=m.SamplerStats();result=m.train_step(model,opt,rows,{1:1,2:2,3:3,4:4},stats,0,.15)
  self.assertTrue(result['loss']>0);self.assertFalse(torch.equal(before,model.embedding.weight))
  self.assertTrue(torch.equal(model.items(torch.tensor([0,4])),torch.zeros(2,64)))
  with self.assertRaises(ValueError):model.users(torch.tensor([[0,0]]))
 def test_versioned_checkpoint_restore_and_tamper(self):
  m=self.module();self.assertTrue(hasattr(m,'save_checkpoint'),'checkpoint behavior missing')
  import torch,tempfile
  from pathlib import Path
  from vkgr.training import seed_all
  seed_all(42);model=m.IDMean(3,[1,2,3]);opt=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=1e-4);cursor=m.EpochCursor(2,42);stats=m.SamplerStats()
  rows=[{'history':[1],'target':2,'past_likes':[1]},{'history':[1],'target':3,'past_likes':[1]}]
  cursor.next(2);m.train_step(model,opt,rows,{1:1,2:2,3:3},stats,0,.15)
  binding={'source':'abc','config':'def','inputs':'ghi'}
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'step-1.pt';sha=m.save_checkpoint(path,model,opt,cursor,stats,1,binding,{'steps':1,'max_steps':3})
   expected_batch=cursor.next(2);expected=m.train_step(model,opt,rows,{1:1,2:2,3:3},stats,1,.15)
   actual=m.IDMean(3,[1,2,3]);o=torch.optim.AdamW(actual.parameters(),lr=.01,weight_decay=1e-4);c=m.EpochCursor(2,0);s=m.SamplerStats()
   m.load_checkpoint(path,sha,actual,o,c,s,binding);self.assertEqual(c.next(2),expected_batch)
   got=m.train_step(actual,o,rows,{1:1,2:2,3:3},s,1,.15);self.assertEqual(got,expected)
   for key,value in model.state_dict().items():self.assertTrue(torch.equal(value,actual.state_dict()[key]))
   with self.assertRaises(FileExistsError):m.save_checkpoint(path,actual,o,c,s,2,binding,{'steps':2})
   with self.assertRaises(ValueError):m.load_checkpoint(path,sha,actual,o,c,s,{'source':'wrong'})
   with path.open('ab') as f:f.write(b'x')
   with self.assertRaises(ValueError):m.load_checkpoint(path,sha,actual,o,c,s,binding)
 def test_all_skipped_stops_before_optimizer(self):
  m=self.module();import torch
  model=m.IDMean(2,[1,2]);opt=torch.optim.AdamW(model.parameters());before=model.embedding.weight.detach().clone()
  rows=[{'history':[1],'target':2,'past_likes':[1]},{'history':[1],'target':2,'past_likes':[1]}]
  with self.assertRaises(ValueError):m.train_step(model,opt,rows,{1:1,2:2},m.SamplerStats(),0,.15)
  self.assertTrue(torch.equal(before,model.embedding.weight))
