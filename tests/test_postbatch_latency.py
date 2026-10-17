import unittest,torch
from analysis import postbatch_latency as p
class LatencyTests(unittest.TestCase):
 def test_fixed_formal_scope(self):
  self.assertEqual(p.scope(),'model.users() only; not end-to-end')
 def test_factory_mean_bi_stack(self):
  for name in ('mean','bi','stack'):
   model=p.make_model(name,64); model.eval()
   histories=torch.tensor([[1,2,0]])
   side=torch.zeros(1,3,6,dtype=torch.long);side[0,:2,0]=1;side[0,:2,1:3]=2;side[0,1,3]=1
   self.assertEqual(tuple(model.users(histories,side).shape),(1,64))
 def test_factory_rejects_unregistered_arm(self):
  with self.assertRaises(ValueError):p.make_model('extra',64)
if __name__=='__main__':unittest.main()
