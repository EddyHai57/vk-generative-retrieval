import importlib.util, unittest, math

class BaselineTests(unittest.TestCase):
 def module(self):
  self.assertIsNotNone(importlib.util.find_spec('vkgr.retrieval_baselines'), 'P4C baseline behavior missing')
  from vkgr import retrieval_baselines
  return retrieval_baselines
 def test_pop_all_events_single_like_and_repetition(self):
  b=self.module()
  events=[(1,10,True),(1,10,True),(1,20,True),(2,20,True),(3,30,True),(4,40,False)]
  graph=b.train_graph([10,20,30,40],events)
  self.assertEqual(graph['popularity'],[2,2,1,0])
  self.assertEqual(graph['degrees'],[1,2,1,0])
  self.assertEqual(b.pop_rank(graph),[10,20,30,40])
 def test_cosine_binary_single_degree_no_self_edge(self):
  b=self.module();self.assertTrue(hasattr(b,'item_neighbors'),'ItemCF behavior missing')
  g=b.train_graph([10,20,30,40],[(1,10,True),(1,10,True),(1,20,True),(2,20,True),(3,30,True)])
  n=b.item_neighbors(g,200)
  self.assertEqual(n[10],[(20,1/math.sqrt(2))]);self.assertEqual(n[20],[(10,1/math.sqrt(2))])
  self.assertEqual(n[30],[]);self.assertEqual(n[40],[])
 def test_top_neighbors_stable_raw_id_tie(self):
  b=self.module();self.assertTrue(hasattr(b,'item_neighbors'),'ItemCF behavior missing')
  g=b.train_graph([10,20,30],[(1,10,True),(1,30,True),(2,10,True),(2,20,True)])
  self.assertEqual(b.item_neighbors(g,1)[10],[(20,1/math.sqrt(2))])
 def test_history_repetition_float64_and_pop_fill(self):
  b=self.module();self.assertTrue(hasattr(b,'itemcf_rank'),'ItemCF scoring missing')
  n={10:[(20,.5),(30,.25)],20:[(30,.5)],30:[],40:[]}
  ranked,fill=b.itemcf_rank([10,20,10],n,[40,10,30,20],4)
  self.assertEqual(ranked,[20,30,40,10]);self.assertEqual(fill,2)
  self.assertEqual(b.itemcf_scores([10,20,10],n),{20:1.,30:1.})
  self.assertEqual(b.itemcf_rank([40],n,[40,10,30,20],3),([40,10,30],3))
 def test_content_exact_block_parity_zero_tie_oov(self):
  b=self.module();self.assertTrue(hasattr(b,'evaluate_baselines'),'shared baseline evaluation missing')
  import numpy as np
  g=b.train_graph([10,20,30,40],[(1,10,True),(1,20,True),(2,40,True)])
  features=np.zeros((4,64),dtype=np.float32);features[0,0]=1;features[1,1]=1;features[2,0]=-1
  queries=[{'history':[10,30],'target':10,'oov':False},{'history':[20],'target':99,'oov':True}]
  result,pred=b.evaluate_baselines(g,b.item_neighbors(g),features,queries,ks=(1,2,4),query_block=1,item_block=1)
  self.assertEqual(pred['content'][0],[10,20,30,40]);self.assertEqual(result['content']['all']['count'],2)
  self.assertEqual(result['content']['all']['recall@1'],.5);self.assertEqual(result['content']['oov']['recall@4'],0.)
  from vkgr.baselines import brute_topk,history_centroids
  centroids,_=history_centroids([q['history'] for q in queries],g['catalog'],features)
  self.assertEqual(pred['content'],brute_topk(centroids,features,g['catalog'],4).tolist())
  _,other=b.evaluate_baselines(g,b.item_neighbors(g),features,queries,ks=(1,2,4),query_block=2,item_block=3)
  self.assertEqual(pred,other)
 def test_persist_verify_shared_inputs_and_tamper_rejection(self):
  b=self.module();self.assertTrue(hasattr(b,'write_run'),'run proof behavior missing')
  import numpy as np,tempfile,json
  from pathlib import Path
  g=b.train_graph([10,20,30],[(1,10,True),(1,20,True),(2,30,True)]);n=b.item_neighbors(g)
  features=np.zeros((3,64),dtype=np.float32);features[:,0]=1
  queries=[{'user_id':1,'history':[10],'target':20,'oov':False},{'user_id':2,'history':[30],'target':99,'oov':True}]
  binding={'producer_source':'old-producer','consumer_source':'new-consumer','catalog_sha256':'fixed-catalog','query_sha256':'fixed-queries','test_gate':'closed'}
  with tempfile.TemporaryDirectory() as tmp:
   output=Path(tmp)/'run';value=b.write_run(output,g,n,features,queries,binding,ks=(1,2,3))
   self.assertEqual(value['binding'],binding)
   self.assertEqual(b.verify_run(output,queries,g['catalog'],binding,g,features)['status'],'success')
   with self.assertRaises(FileExistsError):b.write_run(output,g,n,features,queries,binding,ks=(1,2,3))
   with self.assertRaises(ValueError):b.verify_run(output,queries[::-1],g['catalog'],binding,g,features)
   with self.assertRaises(ValueError):b.verify_run(output,queries,g['catalog'],{**binding,'producer_source':'forged'},g,features)
   with (output/'predictions.json').open('a') as stream:stream.write(' ')
   with self.assertRaises(ValueError):b.verify_run(output,queries,g['catalog'],binding,g,features)
 def test_missing_pinned_prepare_rejected_without_own_source_substitution(self):
  from vkgr import retrieval
  self.assertTrue(hasattr(retrieval,'load_baseline_inputs'),'static prepared consumer missing')
  import tempfile
  with tempfile.TemporaryDirectory() as tmp:
   with self.assertRaises((ValueError,FileNotFoundError)):retrieval.load_baseline_inputs(tmp,tmp,tmp,tmp,tmp)
 def test_cli_baseline_commands_and_test_gate_closed(self):
  import subprocess,sys,tempfile
  proc=subprocess.run([sys.executable,'-B','-m','vkgr.retrieval','baseline-pilot','--help'],capture_output=True,text=True)
  self.assertEqual(proc.returncode,0,'P4C pilot command missing: '+proc.stderr)
  with tempfile.TemporaryDirectory() as tmp:
   proc=subprocess.run([sys.executable,'-B','-m','vkgr.retrieval','baselines','--synthetic','--output',tmp+'/new'],capture_output=True,text=True)
   self.assertEqual(proc.returncode,0,proc.stderr)
   proc=subprocess.run([sys.executable,'-B','-m','vkgr.retrieval','verify-baselines','--synthetic','--manifest',tmp+'/new/baseline-manifest.json'],capture_output=True,text=True)
   self.assertEqual(proc.returncode,0,proc.stderr)
   proc=subprocess.run([sys.executable,'-B','-m','vkgr.retrieval','baselines','--synthetic','--split','test','--output',tmp+'/test'],capture_output=True,text=True)
   self.assertNotEqual(proc.returncode,0)
   self.assertFalse(__import__('pathlib').Path(tmp+'/test').exists())
 def test_pilot_counts_potential_pair_work_including_heavy_user(self):
  b=self.module();self.assertTrue(hasattr(b,'graph_pilot'),'bounded graph pilot missing')
  g=b.train_graph([10,20,30,40],[(1,10,True),(1,20,True),(1,30,True),(2,20,True),(3,40,True)])
  stats=b.graph_pilot(g)
  self.assertEqual(stats['binary_edges'],5);self.assertEqual(stats['cooccurrence_updates_including_self'],11)
  self.assertEqual(stats['max_user_degree'],3);self.assertEqual(stats['max_item_degree'],2)
  self.assertEqual(stats['pilot_items'],4);self.assertEqual(stats['pilot_pair_updates_including_self'],11)
 def test_rehashed_wrong_content_or_graph_is_not_input_replay(self):
  b=self.module()
  import numpy as np,tempfile,json
  from pathlib import Path
  from vkgr.data_audit import canonical_bytes,file_sha256
  from vkgr.baseline_run import evaluate_predictions
  g=b.train_graph([10,20,30],[(1,10,True),(1,20,True),(2,30,True)]);n=b.item_neighbors(g)
  features=np.zeros((3,64),dtype=np.float32);features[0,0]=1;features[1,1]=1;features[2,0]=-1
  q=[{'user_id':1,'history':[10],'target':10,'oov':False}];binding={'test_gate':'closed'}
  for mutation in ('content','graph','counts'):
   with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp:
    out=Path(tmp)/'run';b.write_run(out,g,n,features,q,binding,ks=(1,2,3))
    pred=json.loads((out/'predictions.json').read_text());result=json.loads((out/'metrics.json').read_text());proof=json.loads((out/'graph.json').read_text());manifest=json.loads((out/'baseline-manifest.json').read_text())
    if mutation=='content':pred['content']=[[30,20,10]];method='content'
    elif mutation=='graph':
     proof['neighbors']={str(i):[] for i in [10,20,30]};proof['degrees']=[0,0,0];pred['itemcf']=[[10,20,30]];method='itemcf'
    else:proof['popularity']=[0,0,0];method='popularity'
    result[method]=evaluate_predictions(pred[method],q,g['catalog'],(1,2,3))
    for name,value in (('predictions.json',pred),('metrics.json',result),('graph.json',proof)):
     (out/name).write_bytes(canonical_bytes(value));manifest['artifacts'][name]=file_sha256(out/name)
    (out/'baseline-manifest.json').write_bytes(canonical_bytes(manifest))
    with self.assertRaises(ValueError):b.verify_run(out,q,g['catalog'],binding,g,features)
 def test_final_manifest_rejects_conflicting_artifact_hashes(self):
  import tempfile,numpy as np
  from pathlib import Path
  from types import SimpleNamespace
  from unittest.mock import patch
  from vkgr import retrieval
  from vkgr.data_audit import canonical_bytes,file_sha256
  b=self.module();g,features,q,base=retrieval.synthetic_baseline_inputs()
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);pilot=root/'pilot';pilot.mkdir();proof={'catalog':g['catalog'],'users':{str(u):v for u,v in g['users'].items()},'popularity':g['popularity']}
   (pilot/'incidence.json').write_bytes(canonical_bytes(proof));pm={'kind':'baseline_pilot','status':'success','binding':base,'incidence_sha256':file_sha256(pilot/'incidence.json')};(pilot/'pilot-manifest.json').write_bytes(canonical_bytes(pm))
   bound={**base,'pilot_manifest_sha256':file_sha256(pilot/'pilot-manifest.json')};out=root/'run';initial=b.write_run(out,g,b.item_neighbors(g),features,q,bound)
   regression={'membership_history_target_equal':True,'predictions_at20_equal':{'popularity':True,'content':True},'metrics_at10_20_equal':{'popularity':True,'content':True}};(out/'legacy-regression.json').write_bytes(canonical_bytes(regression))
   final={**initial,'artifacts':{**initial['artifacts'],'legacy-regression.json':file_sha256(out/'legacy-regression.json')}};final['artifacts']['metrics.json']='0'*64;(out/'baseline-final-manifest.json').write_bytes(canonical_bytes(final))
   args=SimpleNamespace(synthetic=False,manifest=str(out/'baseline-final-manifest.json'),pilot=str(pilot/'pilot-manifest.json'),train_directory=None,holdout_directory=None,raw_root=None,prepared=None,legacy_validation=None)
   with patch.object(retrieval,'resource_snapshot',return_value={'ram':.8,'vram':.8,'disk':.8}),patch.object(retrieval,'load_baseline_inputs',return_value=({'catalog':g['catalog']},features,q,{},base)):
    with self.assertRaises(ValueError):retrieval.verify_baselines(args)
 def test_real_verify_resource_gate_precedes_input_loading(self):
  import tempfile
  from pathlib import Path
  from types import SimpleNamespace
  from unittest.mock import patch
  from vkgr import retrieval
  from vkgr.data_audit import canonical_bytes
  args=SimpleNamespace(synthetic=False,train_directory=None,holdout_directory=None,raw_root=None,prepared=None,legacy_validation=None)
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'baseline-final-manifest.json';path.write_bytes(canonical_bytes({'kind':'baselines'}));args.manifest=str(path)
   with patch.object(retrieval,'resource_snapshot',side_effect=ValueError('resource gate')),patch.object(retrieval,'load_baseline_inputs',side_effect=ValueError('loader sentinel')) as load:
    with self.assertRaises(ValueError):retrieval.verify_baselines(args)
    self.assertFalse(load.called,'real verifier loaded inputs before resource gate')
