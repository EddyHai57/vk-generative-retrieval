import copy, hashlib, json, tempfile, unittest
from pathlib import Path
from analysis import coll_sid_diagnostics_read as r

def fixture():
 s={'n':2,'top1_hits':1,'top50_hits':1,'top1_rate':.5,'top50_rate':.5}
 return {'B':{'n_total':3,'n_in_catalog':2,'n_oov':1,'prefix_by_k':{str(k):dict(s) for k in (1,2,3)},'exact_item':dict(s),'full_sid':dict(s),'full_sid_is_item_equivalent':True},'C':{'thresholds':{'median':1.,'p90':11.},'bins':{k:{'n':n,'hits':h,'rate':h/n if n else None} for k,n,h in [('high',1,1),('mid',0,0),('low',1,0),('unsupported',0,0),('oov',1,0)]},'supplemental_longtail_unsupported':{'n':1,'hits':0,'rate':0.}},'C_delta_vs_T3':{k:0 for k in r.BINS}}
class Tests(unittest.TestCase):
 def test_hand_count_and_delta(self):
  p=fixture();r.validate_prediction(p);ref=copy.deepcopy(p['C']);ref['bins']['high'].update(hits=0,rate=0.)
  self.assertEqual(r.compare_c(p,ref)['high'],1)
 def test_mutations(self):
  for f in [lambda p:p['B'].update(n_in_catalog=3),lambda p:p['C']['bins']['high'].update(rate=.4),lambda p:p['C']['bins']['oov'].update(hits=1,rate=1.)]:
   p=fixture();f(p)
   with self.assertRaises(ValueError):r.validate_prediction(p)
 def test_struct_na_and_zero(self):
  x={'n':2,'cooccurrence_matches':0,'random_matches':0,'cooccurrence_rate':0.,'random_rate':0.,'lift':'STRUCT_NA','status':'STRUCT_NA','max_prefix_group':1}
  r.validate_lift_row(x,2,True);x['lift']=0
  with self.assertRaises(ValueError):r.validate_lift_row(x,2,True)
  x.pop('status');x.pop('max_prefix_group');x['lift']=None;r.validate_lift_row(x,2,False)
 def test_mapping_hand_counts_and_array_rejection(self):
  leaf={'count':2,'avg_items':1.,'max_items':1,'collision_items':0,'collision_pairs':0,'collided_leaves':0}
  a={'n_items':2,'semantic_depth':3,'effective_layers':1,'layers':[],'leaves':leaf,'full_code_leaves':dict(leaf)}
  for k in (1,2,3):
   a['layers'].append({'k':k,'branching':256,'distinct_codes':2 if k==1 else 1,'utilization':(2 if k==1 else 1)/256,'normalized_entropy':.125 if k==1 else 0.,'parents':1 if k==1 else 2,'nodes':2,'avg_children':2. if k==1 else 1.,'avg_items_per_node':1.})
  d={'seed':0,'random_algorithm':'frozen','catalog_order':'mapping_insertion_order','catalog_size':2,'history_oov_policy':'fail_closed_no_skips_no_bridging','n_adjacent_events':2,'n_same_item_excluded':1,'n_pairs':1,'n_random_pairs':1,'prefix_by_k':{str(k):{'n':1,'cooccurrence_matches':0,'random_matches':0,'cooccurrence_rate':0.,'random_rate':0.,'lift':None} for k in (1,2,3)}}
  prime=copy.deepcopy(d);prime['stored_prefix_positions']=[1,2,3]
  for row in prime['prefix_by_k'].values():row.update(max_prefix_group=1,status='STRUCT_NA',lift='STRUCT_NA')
  v={'A':a,'D':d,'Dprime':prime,'physical_sources':{'catalog':'a'*64}}
  r.validate_mapping(v)
  v['A']['leaves']['predictions']=[1]
  with self.assertRaises(ValueError):r.validate_mapping(v)
 def test_file_guards(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t).resolve()/'runs'/'sid-coll-c1-verify42-v1'/'out'/'prediction-diagnostics.json';p.parent.mkdir(parents=True);p.write_text(json.dumps(fixture()))
   ref={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'run_id':p.parents[1].name,'role':'prediction','variant':'C1','seed':42}
   self.assertEqual(r.read_aggregate(ref)['B']['n_oov'],1)
   for bad in [dict(ref,sha256='0'*64),dict(ref,run_id='other'),dict(ref,path=str(p.with_name('predictions.json')))]:
    with self.assertRaises(ValueError):r.read_aggregate(bad)
   v=fixture();v['predictions']=[1,2];p.write_text(json.dumps(v));ref['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
   with self.assertRaises(ValueError):r.read_aggregate(ref)
if __name__=='__main__':unittest.main()
