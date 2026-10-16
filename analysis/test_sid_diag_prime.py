import random
import unittest
from analysis import sid_diag_prime as d
class PrimeTests(unittest.TestCase):
 def test_singleton_and_unknown(self):
  self.assertEqual(d.cooccurrence_prime({1:(0,0,0,0),2:(1,0,0,0)},[[1,2]])['prefix_by_k']['1']['lift'],'STRUCT_NA')
  m={1:(0,0,0,0),2:(0,1,0,0),3:(1,0,0,0)}
  row=d.cooccurrence_prime(m,[[1,2]],random_pairs=[(0,2)])['prefix_by_k']['1']
  self.assertIsNone(row['lift']);self.assertEqual(row['status'],'UNKNOWN')
 def test_repeats_no_bridge_pad(self):
  m={1:(0,1,0,0),2:(0,1,0,1),3:(0,2,0,0)}
  v=d.cooccurrence_prime(m,[[1,1,2],[3]])
  self.assertEqual((v['n_adjacent_events'],v['n_same_item_excluded'],v['n_pairs']),(2,1,1))
  self.assertEqual(v['prefix_by_k']['3']['cooccurrence_matches'],1)
  with self.assertRaises(ValueError):d.cooccurrence_prime(m,[[1,9,2]])
 def test_sample_exact_ascending(self):
  m={3:(1,0,0,0),1:(0,0,0,0),2:(0,0,0,1)};h=[[1,2,1,2,3,1,2,1,2,1,2]]
  v=d.cooccurrence_prime(m,h);rng=random.Random(0)
  pairs=[rng.sample(range(3),2) for _ in range(10)]
  expected=sum((a==2)==(b==2) for a,b in pairs)
  self.assertEqual(v['prefix_by_k']['1']['random_matches'],expected)
  self.assertEqual(v,d.cooccurrence_prime(dict(reversed(list(m.items()))),h))
  with self.assertRaises(ValueError):d.cooccurrence_prime(m,h,random_pairs=[(0,0)]*10)
 def test_pair_repetition_permitted_no_cross_user_bridge(self):
  m={1:(0,0,0,0),2:(0,0,0,1)}
  v=d.cooccurrence_prime(m,[[1,2,1],[2]],random_pairs=[(0,1),(0,1)])
  self.assertEqual(v['n_pairs'],2);self.assertEqual(v['n_random_pairs'],2)
  with self.assertRaises(ValueError):d.cooccurrence_prime(m,[[1]],seed=True)
