import unittest
from analysis import profile_diagnostic as p
class DiagnosticTests(unittest.TestCase):
 def test_overlap_set_and_order_are_distinct(self):
  a=list(range(1,51));b=list(range(2,52));x=p.compare([a,b,a[::-1]],[a,a,a],[1,51,1])
  self.assertEqual(x['overlap_histogram']['50'],2);self.assertEqual(x['overlap_histogram']['49'],1)
  self.assertEqual(x['identical_order_queries'],1);self.assertEqual(x['identical_set_queries'],2)
  self.assertEqual(x['only_new']['hit_side_rank_histogram']['50'],1)
  self.assertEqual(x['only_new']['miss_side_exact_rank'],'UNKNOWN (>50)')
  self.assertIsNone(x['only_new']['miss_side_rank_51_55_count'])
 def test_only_control_rank_and_oov_zero(self):
  a=list(range(1,51));b=list(range(2,52));x=p.compare([b,b],[a,a],[1,999])
  self.assertEqual(x['only_control']['queries'],1);self.assertEqual(x['only_control']['hit_side_rank_histogram']['1'],1)
  self.assertEqual(x['only_new']['queries'],0);self.assertIsNone(x['only_new']['hit_side_share_rank45_50'])
 def test_prediction_contract_fails_closed(self):
  a=list(range(1,51))
  for new,ctl,tgt in [([a[:-1]],[a],[1]),([[1]*50],[a],[1]),([a],[a],[])]:
   with self.assertRaises(ValueError):p.compare(new,ctl,tgt)
 def test_bucket_people_and_query_grains(self):
  out=p.buckets([1,1,2],[[1,0,9],[1,0,9],[4,2,0]])
  self.assertEqual(out['queries'],3);self.assertEqual(out['distinct_users'],2)
  self.assertEqual(out['fields']['N_exp']['query_counts']['1'],2);self.assertEqual(out['fields']['N_exp']['user_counts']['1'],1)
  self.assertEqual(out['fields']['H50_like_rate']['user_counts']['9'],1)
  self.assertFalse(out['fields']['N_exp']['constant_on_validation'])
  with self.assertRaises(ValueError):p.buckets([1,1],[[1,0,9],[2,0,9]])
  with self.assertRaises(ValueError):p.buckets([1],[[17,0,0]])
if __name__=='__main__':unittest.main()
