"""Hand-checked search budget and validation-best semantics."""
import unittest

class ExperimentTests(unittest.TestCase):
 def module(self):
  try: import vkgr.retrieval_experiment as module
  except ModuleNotFoundError: self.fail('missing formal experiment module')
  return module
 def test_validation_selection_keeps_ties_and_stops_after_three_bad_epochs(self):
  m=self.module();self.assertTrue(hasattr(m,'ValidationSelection'))
  s=m.ValidationSelection()
  self.assertTrue(s.observe(1,.1,.05));self.assertFalse(s.observe(2,.1,.05))
  self.assertTrue(s.observe(3,.1,.06));self.assertEqual(s.best_epoch,3)
  self.assertFalse(s.observe(4,.09,.9));self.assertFalse(s.observe(5,.1,.06));self.assertFalse(s.stopped)
  self.assertFalse(s.observe(6,.1,.05));self.assertTrue(s.stopped)
  self.assertEqual(s.best_epoch,3)
 def test_finite_config_order_and_validation_ties_choose_first(self):
  m=self.module();self.assertTrue(hasattr(m,'core_configs'))
  configs=m.core_configs()
  self.assertEqual(len(configs),16)
  self.assertEqual([(x['learning_rate'],x['temperature']) for x in configs[:4]],[(.001,.07),(.001,.15),(.0003,.07),(.0003,.15)])
  self.assertEqual([configs[i]['architecture'] for i in (0,4,8,12)],['id_mean','id_content_mean','ordinal_content_mean','content_transformer'])
  results=[{'config':c,'metric':[.2,.1]} for c in configs]
  self.assertEqual(m.select_config(results)['config'],configs[0])
  results[1]['metric']=[.2,.11]
  self.assertEqual(m.select_config(results)['config'],configs[1])
 def test_export_exact_ranking_retains_zero_unsupported_and_oov_denominators(self):
  import torch,numpy as np
  m=self.module();self.assertTrue(hasattr(m,'evaluate_vectors'))
  users=np.zeros((2,64),dtype=np.float32);items=np.zeros((4,64),dtype=np.float32)
  users[:,0]=1;items[1,0]=1;items[2,0]=-1
  queries=[{'target':10,'oov':False},{'target':99,'oov':True}]
  result=m.evaluate_vectors(users,items,[10,20,30,40],queries,{20,30},ks=(1,2,4))
  self.assertEqual(result['predictions'][0],[20,10,40,30])
  self.assertEqual(result['metrics']['all']['count'],2)
  self.assertEqual(result['metrics']['all']['recall@2'],.5)
  self.assertEqual(result['support_metrics']['unsupported']['count'],1)
  self.assertEqual(result['support_metrics']['oov']['count'],1)
  self.assertEqual(result['zero_item_vectors'],2)
 def test_checkpoint_restores_mid_epoch_short_tail_optimizer_rng_and_selection(self):
  import torch,tempfile,random,numpy as np
  from pathlib import Path
  from vkgr.retrieval_models import TwoTower
  from vkgr.retrieval_training import EpochCursor,SamplerStats,train_step
  m=self.module();self.assertTrue(hasattr(m,'save_state'))
  def setup():
   torch.manual_seed(7);random.seed(7);np.random.seed(7)
   model=TwoTower('content_transformer',torch.zeros(11,64),list(range(1,11)))
   return model,torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4),EpochCursor(5,7),SamplerStats(),m.ValidationSelection()
  rows=[{'history':[i],'target':i+5,'past_likes':[i]} for i in range(1,6)];idmap={i:i for i in range(1,11)}
  model,opt,cursor,stats,selection=setup();batch=cursor.next(3);train_step(model,opt,[rows[i] for i in batch],idmap,stats,0,.15)
  selection.observe(1,.2,.1)
  binding={'config':{'seed':7},'input_sha256':'fixture','source':'fixture'}
  with tempfile.TemporaryDirectory() as directory:
   path=Path(directory)/'state.pt';digest=m.save_state(path,model,opt,cursor,stats,1,binding,selection,{'max_epochs':30})
   batch=cursor.next(3);self.assertEqual(len(batch),2)
   expected=train_step(model,opt,[rows[i] for i in batch],idmap,stats,0,.15);wanted={k:v.clone() for k,v in model.state_dict().items()};next_rng=torch.rand(8)
   a,b,c,d,e=setup();state=m.restore_state(path,digest,a,b,c,d,binding)
   self.assertEqual(state['selection']['best_epoch'],1)
   self.assertEqual(c.next(3),batch)
   actual=train_step(a,b,[rows[i] for i in batch],idmap,d,0,.15)
   self.assertEqual(actual,expected)
   for k,v in wanted.items():torch.testing.assert_close(a.state_dict()[k],v,rtol=0,atol=0)
   torch.testing.assert_close(torch.rand(8),next_rng,rtol=0,atol=0)
   self.assertEqual(c.next(3).__len__(),3);self.assertEqual(c.epoch,1)
 def test_formal_session_consumes_real_rows_and_resume_does_not_reset_epoch_budget(self):
  import torch,tempfile
  from pathlib import Path
  m=self.module();self.assertTrue(hasattr(m,'run_session'))
  torch.set_num_threads(1)
  rows=[{'history':[i],'target':i+5,'past_likes':[i]} for i in range(1,6)]
  config={'architecture':'id_mean','learning_rate':.001,'temperature':.15,'seed':42,'batch_size':3}
  evaluate=lambda model,epoch:{'metrics':{'all':{'recall@50':.2,'ndcg@50':.1}},'epoch':epoch}
  binding={'origin':'independent_synthetic_session_test','config':config,'input':'five_explicit_rows'}
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);a=m.run_session(root/'continuous',rows,torch.zeros(11,64),list(range(1,11)),{i:i for i in range(1,11)},config,binding,evaluate,'cpu')
   first=m.run_session(root/'resumed',rows,torch.zeros(11,64),list(range(1,11)),{i:i for i in range(1,11)},config,binding,evaluate,'cpu',until_step=1)
   self.assertEqual(first['status'],'segment_complete');self.assertEqual(first['completed_epochs'],0)
   b=m.run_session(root/'resumed',rows,torch.zeros(11,64),list(range(1,11)),{i:i for i in range(1,11)},config,binding,evaluate,'cpu',resume=root/'resumed'/first['manifest_name'])
   self.assertEqual((a['steps'],a['completed_epochs'],a['stop_reason']),(8,4,'patience'))
   self.assertEqual(len(a.get('epoch_checkpoints',[])),4)
   self.assertEqual(b['records'],a['records']);self.assertEqual(b['evaluations'],a['evaluations']);self.assertEqual(b['best']['epoch'],1)
   sa=torch.load(root/'continuous'/a['last']['path'],weights_only=True);sb=torch.load(root/'resumed'/b['last']['path'],weights_only=True)
   for k,v in sa['model'].items():torch.testing.assert_close(v,sb['model'][k],rtol=0,atol=0)
   self.assertEqual(sb['budget']['max_epochs'],30)
   self.assertEqual(sa['cursor'],sb['cursor'])
   self.assertEqual(sa['stats'],sb['stats'])
 def test_export_normalizes_float32_without_losing_zero_rows(self):
  import torch,numpy as np
  from vkgr.retrieval_models import TwoTower
  m=self.module();self.assertTrue(hasattr(m,'export_vectors'))
  model=TwoTower('id_mean',torch.zeros(5,64),[1,2])
  with torch.no_grad():
   model.embedding.weight.zero_();model.embedding.weight[1,0]=3;model.embedding.weight[2,1]=4
  users,items=m.export_vectors(model,[10,20,30,40],[{'history':[10,20]},{'history':[30]}],{10:1,20:2,30:3,40:4},2)
  expected=np.zeros((2,64),dtype=np.float32);expected[0,:2]=[.6,.8]
  np.testing.assert_array_equal(users,expected)
  self.assertEqual(items.dtype,np.float32);self.assertEqual(items[0,0],1.)
  np.testing.assert_array_equal(items[2:],np.zeros((2,64),dtype=np.float32))
 def test_loader_binds_prepared_producer_and_reads_only_validation(self):
  import tempfile,torch,numpy as np
  from pathlib import Path
  from vkgr.retrieval_data import write_prepared
  from vkgr.data_audit import canonical_bytes,file_sha256
  m=self.module();self.assertTrue(hasattr(m,'load_training_inputs'))
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);prepared=root/'prepared';holdout=root/'holdout';holdout.mkdir()
   row={'example_id':0,'user_id':1,'history':[10],'history_positions':[[0,0]],'target':20,'target_position':[0,1],'past_likes':[10]}
   producer={'source_fingerprint':'fixture'}
   value=write_prepared(prepared,[10,20],np.zeros((3,64),dtype=np.float32),[row],{'consumer':producer},[10,20])
   queries=[{'user_id':1,'history':[10],'history_positions':[[0,0]],'target':99,'target_position':[25,0],'oov':True}]
   (holdout/'validation-queries.json').write_bytes(canonical_bytes(queries))
   # Deliberately no test file: loading it would fail.
   actual=m.load_training_inputs(prepared,holdout,file_sha256(prepared/'prepare-manifest.json'),'fixture',file_sha256(holdout/'validation-queries.json'))
   self.assertEqual(actual['rows'],[row]);self.assertEqual(actual['queries'],queries);self.assertEqual(actual['supported_indices'],[1,2]);self.assertEqual(actual['support'],{10,20})
   with self.assertRaises(ValueError):m.load_training_inputs(prepared,holdout,file_sha256(prepared/'prepare-manifest.json'),'wrong',file_sha256(holdout/'validation-queries.json'))
 def test_runtime_configuration_rejects_budget_or_precision_changes(self):
  m=self.module();self.assertTrue(hasattr(m,'validate_runtime'))
  valid={'schema_version':1,'task_id':'P4D','architecture':'id_mean','learning_rate':.001,'temperature':.07,'seed':42,'batch_size':512,'max_epochs':30,'patience':3,'precision':'fp32'}
  self.assertEqual(m.validate_runtime(valid),valid)
  for key,bad in [('batch_size',64),('max_epochs',31),('precision','fp16'),('learning_rate',.01),('seed',1),('patience',4),('schema_version',True),('max_epochs',30.),('batch_size',512.)]:
   with self.assertRaises(ValueError):m.validate_runtime({**valid,key:bad})
  with self.assertRaises(ValueError):m.validate_runtime({**valid,'uniform_negatives':128})
 def test_resume_rejects_manifest_best_metric_rewrite(self):
  import torch,tempfile,json
  from pathlib import Path
  m=self.module();torch.set_num_threads(1)
  rows=[{'history':[i],'target':i+5,'past_likes':[i]} for i in range(1,6)]
  config={'architecture':'id_mean','learning_rate':.001,'temperature':.15,'seed':42,'batch_size':3};binding={'config':config}
  evaluate=lambda model,epoch:{'metrics':{'all':{'recall@50':.2,'ndcg@50':.1}}}
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory)/'run';first=m.run_session(root,rows,torch.zeros(11,64),list(range(1,11)),{i:i for i in range(1,11)},config,binding,evaluate,'cpu',until_step=2)
   path=root/first['manifest_name'];value=json.loads(path.read_text());value['best']['metric']=[.9,.9];path.write_text(json.dumps(value))
   with self.assertRaises(ValueError):m.run_session(root,rows,torch.zeros(11,64),list(range(1,11)),{i:i for i in range(1,11)},config,binding,evaluate,'cpu',resume=path)
 def test_shared_batch_requires_every_model_and_preserves_failed_attempts(self):
  m=self.module();self.assertTrue(hasattr(m,'choose_shared_batch'))
  calls=[]
  def measure(architecture,batch):
   calls.append((architecture,batch))
   return {'resources_minimum':{'ram':.5,'disk':.8,'vram':.19 if batch==512 and architecture=='content_transformer' else .4},'measured_peak_reserved':123}
  result=m.choose_shared_batch(measure)
  self.assertEqual(result['batch_size'],256);self.assertEqual(len(calls),8)
  self.assertEqual(len(result['attempts']),2);self.assertFalse(result['attempts'][0]['all_pass'])
  with self.assertRaises(ValueError):m.choose_shared_batch(lambda a,b:{'resources_minimum':{'ram':.1,'disk':.8,'vram':.9}})
 def test_determinism_requires_explicit_cuda_workspace_and_disables_tf32(self):
  import os,torch
  m=self.module();self.assertTrue(hasattr(m,'configure_precision'))
  prior=os.environ.get('CUBLAS_WORKSPACE_CONFIG')
  try:
   os.environ.pop('CUBLAS_WORKSPACE_CONFIG',None)
   with self.assertRaises(ValueError):m.configure_precision('cuda')
   os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
   result=m.configure_precision('cpu')
   self.assertTrue(result['deterministic_algorithms']);self.assertFalse(result['tf32_matmul']);self.assertFalse(result['tf32_cudnn'])
   self.assertEqual(result['precision'],'fp32')
  finally:
   if prior is None:os.environ.pop('CUBLAS_WORKSPACE_CONFIG',None)
   else:os.environ['CUBLAS_WORKSPACE_CONFIG']=prior
 def test_epoch_artifacts_bind_vectors_predictions_and_metrics_for_replay(self):
  import torch,tempfile
  from pathlib import Path
  from vkgr.retrieval_models import TwoTower
  m=self.module();self.assertTrue(hasattr(m,'persist_evaluation'))
  model=TwoTower('id_mean',torch.zeros(5,64),[1,2])
  with torch.no_grad():model.embedding.weight.zero_();model.embedding.weight[1,0]=1;model.embedding.weight[2,1]=1
  queries=[{'history':[10],'target':10,'oov':False},{'history':[20],'target':99,'oov':True}]
  with tempfile.TemporaryDirectory() as directory:
   output=Path(directory)/'epoch-1';result=m.persist_evaluation(output,model,[10,20,30,40],queries,{10:1,20:2,30:3,40:4},{10,20},2)
   self.assertNotIn('predictions',result);self.assertEqual(len(result['artifacts']),4)
   self.assertEqual(result['metrics']['all']['recall@50'],.5)
   m.verify_evaluation(output,result,[10,20,30,40],queries,{10,20})
   (output/'predictions.json').write_text('[]')
   with self.assertRaises(ValueError):m.verify_evaluation(output,result,[10,20,30,40],queries,{10,20})
 def test_gpu_pilot_refuses_cpu_capacity_guesses(self):
  import torch,tempfile
  m=self.module();self.assertTrue(hasattr(m,'measure_gpu_model'))
  with tempfile.TemporaryDirectory() as directory:
   with self.assertRaises(ValueError):m.measure_gpu_model('id_mean',128,{'content':torch.zeros(5,64),'supported_indices':[1,2],'catalog':[10,20,30,40],'idmap':{10:1,20:2,30:3,40:4},'queries':[{'history':[10]}]},directory,'cpu')
 def test_cli_plan_is_finite_and_refuses_test_split(self):
  import io,json,contextlib,tempfile
  from pathlib import Path
  m=self.module();self.assertTrue(hasattr(m,'main'))
  with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()) as stdout:
   output=Path(directory)/'plan';self.assertEqual(m.main(['plan','--output',str(output)]),0)
   plan=json.loads((output/'run-plan.json').read_text())
   self.assertEqual(len(plan['tuning_runs']),16);self.assertEqual(plan['maximum_formal_runs'],24)
   self.assertEqual(plan['additional_seeds'],[2024,2025]);self.assertEqual(plan['test_gate'],'closed')
  with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):m.main(['pilot','--split','test'])
 def test_checkpoint_reexport_rejects_synchronously_rehashed_vectors(self):
  import torch,tempfile,numpy as np
  from pathlib import Path
  from vkgr.retrieval_models import TwoTower
  from vkgr.retrieval_training import EpochCursor,SamplerStats
  from vkgr.data_audit import file_sha256
  m=self.module();self.assertTrue(hasattr(m,'verify_checkpoint_evaluation'))
  data={'content':torch.zeros(5,64),'supported_indices':[1,2],'catalog':[10,20,30,40],'queries':[{'history':[10],'target':10,'oov':False}],'idmap':{10:1,20:2,30:3,40:4},'support':{10,20}}
  model=TwoTower('id_mean',data['content'],[1,2]);opt=torch.optim.AdamW(model.parameters(),lr=.001);selection=m.ValidationSelection();selection.observe(1,1.,1.)
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);out=root/'epoch-1';result=m.persist_evaluation(out,model,data['catalog'],data['queries'],data['idmap'],data['support'],2)
   path=root/'checkpoint.pt';digest=m.save_state(path,model,opt,EpochCursor(2,42),SamplerStats(),1,{'config':'fixture'},selection,{'max_epochs':30})
   cp={'path':path.name,'sha256':digest,'epoch':1,'step':1}
   m.verify_checkpoint_evaluation(root,cp,out,result,data,{'architecture':'id_mean','batch_size':2},{'config':'fixture'},'cpu')
   users=np.load(out/'user-vectors.npy');users*=.5
   with (out/'user-vectors.npy').open('wb') as stream:np.save(stream,users)
   result['artifacts']['user-vectors.npy']=file_sha256(out/'user-vectors.npy')
   m.verify_evaluation(out,result,data['catalog'],data['queries'],data['support'])
   with self.assertRaises(ValueError):m.verify_checkpoint_evaluation(root,cp,out,result,data,{'architecture':'id_mean','batch_size':2},{'config':'fixture'},'cpu')
 def test_selection_freezes_all_four_configs_and_seed42_best_architecture(self):
  m=self.module();self.assertTrue(hasattr(m,'freeze_selection'))
  results=[]
  for config in m.core_configs():results.append({'config':config,'metric':[.2,.1],'manifest_sha256':'synthetic-ref','best':{'epoch':1}})
  result=m.freeze_selection(results,{'input':'fixture'})
  self.assertEqual(result['best_architecture'],'id_mean');self.assertEqual(len(result['selected_configs']),4)
  self.assertEqual(result['selected_configs']['id_mean']['config']['temperature'],.07)
  results[13]['metric']=[.2,.11]
  self.assertEqual(m.freeze_selection(results,{})['best_architecture'],'content_transformer')
  with self.assertRaises(ValueError):m.freeze_selection(results[:-1],{})
  with self.assertRaises(ValueError):m.freeze_selection(results+[results[0]],{})
 def test_draw_provenance_recomputes_order_masks_and_skip_denominators(self):
  import torch,tempfile,copy
  from pathlib import Path
  m=self.module();self.assertTrue(hasattr(m,'verify_draw_provenance'))
  config={'architecture':'id_mean','learning_rate':.001,'temperature':.15,'seed':42,'batch_size':3}
  data={'rows':[{'history':[i],'target':i+5,'past_likes':[i]} for i in range(1,6)],'idmap':{i:i for i in range(1,11)}}
  with tempfile.TemporaryDirectory() as directory:
   value=m.run_session(Path(directory)/'run',data['rows'],torch.zeros(11,64),list(range(1,11)),data['idmap'],config,{'config':config},lambda model,epoch:{'metrics':{'all':{'recall@50':.2,'ndcg@50':.1}}},'cpu',until_step=3)
   m.verify_draw_provenance(value,data)
   wrong=copy.deepcopy(value);wrong['records'][0]['example_indices']=wrong['records'][0]['example_indices'][::-1]
   with self.assertRaises(ValueError):m.verify_draw_provenance(wrong,data)
   wrong=copy.deepcopy(value);wrong['records'][0]['skipped']=1
   with self.assertRaises(ValueError):m.verify_draw_provenance(wrong,data)

 def test_strictly_improving_run_still_stops_at_thirty_epochs(self):
  m=self.module();selection=m.ValidationSelection()
  for epoch in range(1,31):self.assertTrue(selection.observe(epoch,epoch/100,.1))
  self.assertTrue(selection.stopped);self.assertEqual(selection.best_epoch,30)
  with self.assertRaises(ValueError):selection.observe(31,.4,.1)
 def test_cli_reads_string_json_paths_then_reports_expected_gate_failure(self):
  import tempfile,io,contextlib,json
  from pathlib import Path
  from unittest.mock import patch
  m=self.module()
  def cli(args):
   try:return m.main(args)
   except AttributeError as error:self.fail('CLI path wiring raised AttributeError: '+str(error))
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);run_list=root/'runs.json';run_list.write_text('[]')
   with contextlib.redirect_stderr(io.StringIO()):self.assertEqual(cli(['select','--run-list',str(run_list),'--output',str(root/'selection.json')]),2)
   config=root/'runtime.json';config.write_text('{}')
   args=['train','--runtime-config',str(config),'--pilot',str(root/'pilot.json'),'--pilot-sha256','fixture','--prepared','fixture','--holdout-directory','fixture','--output',str(root/'run')]
   # Only the external CUDA/real-input boundary is substituted; JSON/gates are real.
   with patch.object(m,'real_inputs',return_value=({}, {}, lambda: None)),contextlib.redirect_stderr(io.StringIO()):self.assertEqual(cli(args),2)
 def test_complete_cli_selection_rejects_mixed_shared_batch_or_pilot(self):
  import torch,tempfile,json,io,contextlib
  from pathlib import Path
  from types import SimpleNamespace
  m=self.module();torch.set_num_threads(1)
  rows=[{'history':[i],'target':i+5,'past_likes':[i]} for i in range(1,6)]
  evaluate=lambda model,epoch:{'metrics':{'all':{'recall@50':.2,'ndcg@50':.1}}}
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);paths=[]
   for j,core in enumerate(m.core_configs()):
    config={**core,'schema_version':1,'task_id':'P4D','batch_size':128 if j<8 else 256,'max_epochs':30,'patience':3,'precision':'fp32'}
    binding={'config':config,'consumer':'explicit_synthetic_cli_fixture','pilot_manifest_sha256':'pilot-A' if j<8 else 'pilot-B'}
    run=m.run_session(root/f'run-{j}',rows,torch.zeros(11,64),list(range(1,11)),{i:i for i in range(1,11)},config,binding,evaluate,'cpu');paths.append(str(root/f'run-{j}'/run['manifest_name']))
   run_list=root/'run-list.json';run_list.write_text(json.dumps(paths))
   with self.assertRaises(ValueError):m.select_core(SimpleNamespace(run_list=run_list,output=root/'selection.json'))
 def test_additional_seeds_require_the_same_selected_batch_and_pilot(self):
  m=self.module();self.assertTrue(hasattr(m,'require_selected_configuration'))
  config={'architecture':'id_mean','learning_rate':.001,'temperature':.07,'seed':2024,'batch_size':128}
  inputs={'consumer':'fixture'};selection={'kind':'core_selection','input_binding':{**inputs,'shared_batch_size':128,'pilot_manifest_sha256':'pilot-A'},'selected_configs':{'id_mean':{'config':{'architecture':'id_mean','learning_rate':.001,'temperature':.07,'seed':42}}}}
  m.require_selected_configuration(selection,inputs,config,'pilot-A')
  with self.assertRaises(ValueError):m.require_selected_configuration(selection,inputs,{**config,'batch_size':256},'pilot-A')
  with self.assertRaises(ValueError):m.require_selected_configuration(selection,inputs,config,'pilot-B')
 def test_pilot_never_retries_nonfinite_or_determinism_failures_as_capacity(self):
  import torch
  m=self.module();calls=[]
  def measure(model,batch):
   calls.append(batch)
   if batch==512:return {'failure':'ValueError: nonfinite pilot loss','failure_kind':'gate','resources_minimum':{'ram':.9,'vram':.9,'disk':.9}}
   return {'resources_minimum':{'ram':.9,'vram':.9,'disk':.9}}
  with self.assertRaises(ValueError):m.choose_shared_batch(measure)
  self.assertNotIn(256,calls)
  self.assertTrue(hasattr(m,'pilot_failure_kind'))
  self.assertEqual(m.pilot_failure_kind(torch.cuda.OutOfMemoryError('fixture')),'capacity')
  self.assertEqual(m.pilot_failure_kind(ValueError('resource headroom below 20%')),'gate')
  self.assertEqual(m.pilot_failure_kind(m.ResourceCapacityError('measured below minimum')),'capacity')
  self.assertEqual(m.pilot_failure_kind(ValueError('nonfinite pilot loss')),'gate')
  self.assertEqual(m.pilot_failure_kind(RuntimeError('deterministic implementation unavailable')),'gate')
 def test_pilot_resource_context_holds_exports_ranks_and_score_block_without_metrics(self):
  import torch,tempfile,inspect
  from pathlib import Path
  from vkgr.retrieval_models import TwoTower
  m=self.module();self.assertTrue(hasattr(m,'resource_export_context'))
  data={'content':torch.zeros(5,64),'supported_indices':[1,2],'catalog':[10,20,30,40],'queries':[{'history':[10],'target':99,'oov':True},{'history':[20],'target':10,'oov':False}],'idmap':{10:1,20:2,30:3,40:4}}
  model=TwoTower('id_mean',data['content'],[1,2]);observed=[]
  def check():
   local=inspect.currentframe().f_back.f_locals
   if all(key in local for key in ('users','items','rankings','score_block')):observed.append((local['users'].shape,local['items'].shape,local['rankings'].shape,local['score_block'].shape))
   if all(key in local for key in ('norms','normalized','rows')):observed.append('normalize_live')
  with tempfile.TemporaryDirectory() as directory:
   result=m.resource_export_context(Path(directory)/'context',model,data,2,check)
   self.assertIn(((2,64),(4,64),(2,4),(2,4)),observed);self.assertIn('normalize_live',observed)
   self.assertNotIn('metrics',result);self.assertEqual(len(result['artifacts']),3)
