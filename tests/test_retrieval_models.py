"""Independent hand fixtures for core two-tower behavior."""
import unittest
import torch

class CoreModelsTests(unittest.TestCase):
 def test_mean_uses_only_valid_items_and_suppresses_unsupported_id(self):
  try: import vkgr.retrieval_models as module
  except ModuleNotFoundError: self.fail("missing four-architecture module")
  self.assertTrue(hasattr(module,'TwoTower'),'missing four-architecture model')
  model=module.TwoTower('id_mean',torch.zeros(5,64),[1,2])
  with torch.no_grad():
   model.embedding.weight.fill_(9);model.embedding.weight[1].fill_(2);model.embedding.weight[2].fill_(4)
  torch.testing.assert_close(model.users(torch.tensor([[1,2,0],[2,1,0]])),torch.full((2,64),3.))
  torch.testing.assert_close(model.items(torch.tensor([0,3,4])),torch.zeros(3,64))
  model.items(torch.tensor([1,3])).sum().backward()
  self.assertEqual(float(model.embedding.weight.grad[3].abs().sum()),0.)
  self.assertGreater(float(model.embedding.weight.grad[1].abs().sum()),0.)
 def test_content_is_frozen_bias_free_and_usable_for_unsupported_items(self):
  from vkgr.retrieval_models import TwoTower
  features=torch.zeros(5,64);features[1,0]=1;features[3,1]=1
  import vkgr.retrieval_models as module
  self.assertIn('id_content_mean',getattr(module,'MODEL_ORDER',()))
  model=TwoTower('id_content_mean',features,[1,2])
  with torch.no_grad():
   model.embedding.weight.zero_();model.content_projection.weight.copy_(torch.eye(64))
  torch.testing.assert_close(model.items(torch.tensor([3])),features[3:4])
  torch.testing.assert_close(model.items(torch.tensor([0,4])),torch.zeros(2,64))
  model.items(torch.tensor([1,3])).sum().backward()
  self.assertIsNone(model.content.grad)
  self.assertGreater(float(model.content_projection.weight.grad.abs().sum()),0.)
  torch.testing.assert_close(model.users(torch.tensor([[1,3,0]])),(features[1:2]+features[3:4])/2)
 def test_ordinal_weights_follow_valid_event_distance_not_pad_width(self):
  import vkgr.retrieval_models as module
  self.assertIn('ordinal_content_mean',module.MODEL_ORDER)
  model=module.TwoTower('ordinal_content_mean',torch.zeros(5,64),[1,2,3])
  with torch.no_grad():
   model.embedding.weight.zero_();model.embedding.weight[1,0]=1;model.embedding.weight[2,1]=1;model.embedding.weight[3,2]=1
  expected=torch.zeros(1,64);expected[0,:3]=torch.tensor([.64,.8,1.])/2.44
  torch.testing.assert_close(model.users(torch.tensor([[1,2,3]])),expected)
  torch.testing.assert_close(model.users(torch.tensor([[0,1,0,2,3,0]])),expected)
  self.assertFalse(torch.equal(model.users(torch.tensor([[3,2,1]])),expected))
 def test_transformer_order_response_and_pad_invariance(self):
  import vkgr.retrieval_models as module
  self.assertIn('content_transformer',module.MODEL_ORDER)
  torch.manual_seed(42)
  model=module.TwoTower('content_transformer',torch.zeros(5,64),[1,2,3]).eval()
  plain=model.users(torch.tensor([[1,2,3]]));padded=model.users(torch.tensor([[0,1,0,2,3,0]]))
  torch.testing.assert_close(plain,padded,atol=1e-6,rtol=1e-6)
  self.assertGreater(float((plain-model.users(torch.tensor([[3,2,1]]))).abs().max()),1e-4)
  plain.sum().backward()
  self.assertGreater(float(model.positions.weight.grad[:3].abs().sum()),0.)
  self.assertEqual(float(model.positions.weight.grad[3:].abs().sum()),0.)
  self.assertIsNone(model.user_projection.bias)
 def test_each_architecture_learns_synthetic_pairs_and_changes_parameters(self):
  from vkgr.retrieval_models import TwoTower,MODEL_ORDER
  from vkgr.retrieval_training import SamplerStats,train_step
  torch.set_num_threads(1)
  rows=[{'history':[i],'target':i+4,'past_likes':[i]} for i in range(1,5)];idmap={i:i for i in range(1,9)}
  for architecture in MODEL_ORDER:
   with self.subTest(architecture=architecture):
    torch.manual_seed(42);features=torch.zeros(9,64)
    for i in range(1,9):features[i,i]=1
    model=TwoTower(architecture,features,list(range(1,9)));optimizer=torch.optim.AdamW(model.parameters(),lr=.03,weight_decay=1e-4);stats=SamplerStats()
    before=model.embedding.weight.detach().clone();losses=[]
    for step in range(35):losses.append(train_step(model,optimizer,rows,idmap,stats,0,.15)['loss'])
    self.assertLess(sum(losses[-5:])/5,sum(losses[:5])/5*.3)
    self.assertGreater(float((model.embedding.weight.detach()-before).abs().sum()),.1)
    self.assertEqual(stats.skipped,0)

 def test_invalid_empty_nonfinite_and_overlong_histories_are_rejected(self):
  from vkgr.retrieval_models import TwoTower,MODEL_ORDER
  for architecture in MODEL_ORDER:
   model=TwoTower(architecture,torch.zeros(5,64),[1,2])
   for history in (torch.zeros(2,3,dtype=torch.long),torch.ones(1,51,dtype=torch.long),torch.tensor([[9]]),torch.tensor([[1.]])):
    with self.assertRaises(ValueError):model.users(history)
  bad=torch.zeros(5,64);bad[3,4]=float('nan')
  with self.assertRaises(ValueError):TwoTower('id_mean',bad,[1])
 def test_checkpoint_cannot_replace_train_support_or_frozen_content(self):
  from vkgr.retrieval_models import TwoTower
  model=TwoTower('id_content_mean',torch.zeros(5,64),[1,2])
  state={k:v.clone() for k,v in model.state_dict().items()};state['supported'][3]=True
  with self.assertRaises(ValueError):model.load_state_dict(state)
  state={k:v.clone() for k,v in model.state_dict().items()};state['content'][3,0]=1.
  with self.assertRaises(ValueError):model.load_state_dict(state)
