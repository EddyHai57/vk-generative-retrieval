"""Hand-computed pure S1 diagnostics; no files, tensors or server data."""
import unittest
from analysis import sid_coll_diagnostics as d
class DiagnosticsTests(unittest.TestCase):
 def test_structure_and_distinct_prefix_struct_na(self):
  m={1:(0,0,0,0),2:(1,0,0,0)}
  value=d.mapping_diagnostics(m,[[1,1,2],[2]])
  self.assertEqual(set(value),{'A','D','Dprime'})
  self.assertEqual(value['A']['n_items'],2)
  self.assertEqual(value['A']['layers'][0]['branching'],256)
  self.assertEqual(value['Dprime']['n_pairs'],1)
  self.assertEqual(value['Dprime']['prefix_by_k']['1']['lift'],'STRUCT_NA')
  self.assertEqual(value['D']['random_algorithm'],'python_random_Random_randrange_independent_with_replacement')
 def test_zero_denominator_and_oov_fail_closed(self):
  m={1:(0,0,0,0),2:(0,0,0,1)}
  v=d.mapping_diagnostics(m,[[1],[2]])
  self.assertIsNone(v['D']['prefix_by_k']['1']['lift'])
  self.assertIsNone(v['Dprime']['prefix_by_k']['1']['lift'])
  self.assertEqual(v['Dprime']['prefix_by_k']['1']['status'],'UNKNOWN')
  with self.assertRaises(ValueError):d.mapping_diagnostics(m,[[1,999,2]])
  with self.assertRaises(ValueError):d.mapping_diagnostics({1:(0,0,0)},[[1]])
 def test_popularity_delta_and_oov_denominator(self):
  m={1:(0,0,0,0),2:(0,0,0,1),3:(1,0,0,0),4:(1,1,0,0)}
  targets=[1,2,3,4,999];preds=[[1],[1],[3],[4],[1]]
  slices={'popularity_thresholds':{'median':1,'p90':2},'item_like_counts':{'1':0,'2':1,'3':2,'4':3}}
  reference={'bins':{b:{'hits':0} for b in ('oov','unsupported','low','mid','high')}}
  v=d.prediction_diagnostics(targets,preds,m,slices,reference)
  self.assertEqual(v['B']['n_in_catalog'],4);self.assertEqual(v['B']['n_oov'],1)
  self.assertEqual(v['B']['exact_item']['top50_hits'],3)
  self.assertEqual(v['C_delta_vs_T3'],{'oov':0,'unsupported':1,'low':0,'mid':1,'high':1})
  self.assertEqual(v['C']['bins']['oov']['hits'],0)
  self.assertEqual(v['B']['prefix_by_k']['3']['top1_hits'],4)
  v=d.prediction_diagnostics([999],[[1]],m,slices,reference)
  self.assertIsNone(v['B']['exact_item']['top50_rate'])
  with self.assertRaises(ValueError):d.prediction_diagnostics([1],[[999]],m,slices,reference)

 def test_reusable_history_generator_and_unknown_reference(self):
  m={1:(0,0,0,0),2:(0,0,0,1)}
  v=d.mapping_diagnostics(m,(iter(row) for row in [[1,2,1],[2]]))
  self.assertEqual(v['D']['n_pairs'],2);self.assertEqual(v['Dprime']['n_pairs'],2)
  self.assertEqual(v['D']['prefix_by_k']['1']['lift'],1)
  self.assertEqual(v['Dprime']['prefix_by_k']['1']['lift'],1)
  s={'popularity_thresholds':{'median':0,'p90':0},'item_like_counts':{}}
  self.assertEqual(d.prediction_diagnostics([],[],m,s,None)['C_delta_vs_T3']['status'],'UNKNOWN')
