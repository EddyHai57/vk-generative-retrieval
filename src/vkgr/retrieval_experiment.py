"""Finite P4D core search; validation-best differs from synthetic loss-best."""
import math

class ValidationSelection:
 def __init__(self):
  self.epoch=0;self.best_epoch=None;self.best_metric=None;self.bad_epochs=0
 @property
 def stopped(self):return self.bad_epochs>=3 or self.epoch>=30
 def observe(self,epoch,recall,ndcg):
  if self.stopped or type(epoch)is not int or epoch!=self.epoch+1 or any(not math.isfinite(v) or not 0<=v<=1 for v in (recall,ndcg)):raise ValueError('invalid validation budget/metrics')
  self.epoch=epoch;metric=(recall,ndcg)
  improved=self.best_metric is None or metric>self.best_metric
  if improved:self.best_metric=metric;self.best_epoch=epoch;self.bad_epochs=0
  else:self.bad_epochs+=1
  return improved

def core_configs():
 from .retrieval_models import MODEL_ORDER
 return [{'architecture':model,'learning_rate':lr,'temperature':temperature,'seed':42} for model in MODEL_ORDER for lr in (.001,.0003) for temperature in (.07,.15)]

def select_config(results):
 if not results:raise ValueError('empty successful configurations')
 return max(results,key=lambda r:tuple(r['metric']))

def evaluate_vectors(users,items,catalog,queries,support,ks=(10,20,50),check=None):
 import numpy as np,time
 from .baselines import blockwise_topk,metrics
 started=time.monotonic()
 if check:check()
 predictions=[]
 for start in range(0,len(users),32):
  if check:check()
  predictions.extend(blockwise_topk(users[start:start+32],items,catalog,max(ks)).tolist())
 result=metrics(predictions,queries,catalog,ks);slices={}
 for name in ('supported','unsupported','oov'):
  positions=[j for j,q in enumerate(queries) if ('oov' if q['oov'] else 'supported' if q['target'] in support else 'unsupported')==name]
  slices[name]=metrics([predictions[j] for j in positions],[queries[j] for j in positions],catalog,ks)['all']
 return {'predictions':predictions,'metrics':result,'support_metrics':slices,'zero_item_vectors':int(np.count_nonzero(np.linalg.norm(items,axis=1)==0)),'zero_user_vectors':int(np.count_nonzero(np.linalg.norm(users,axis=1)==0)),'exact_seconds':time.monotonic()-started}

def save_state(path,model,optimizer,cursor,stats,step,binding,selection,budget):
 from .retrieval_training import save_checkpoint
 return save_checkpoint(path,model,optimizer,cursor,stats,step,binding,{**budget,'selection':dict(selection.__dict__)})

def restore_state(path,digest,model,optimizer,cursor,stats,binding):
 from .retrieval_training import load_checkpoint
 state=load_checkpoint(path,digest,model,optimizer,cursor,stats,binding)
 selected=state['budget']['selection']
 if set(selected)!={'epoch','best_epoch','best_metric','bad_epochs'} or not 0<=selected['epoch']<=30 or not 0<=selected['bad_epochs']<=3:raise ValueError('selection budget mismatch')
 state['selection']=selected
 return state

def run_session(output,rows,content,supported,idmap,config,binding,evaluate,device,until_step=None,resume=None,check=None):
 """One fixed config/seed, segmented without resetting its epoch budget."""
 import torch,time
 from pathlib import Path
 from .training import seed_all
 from .retrieval_models import TwoTower
 from .retrieval_training import EpochCursor,SamplerStats,train_step
 from .manifest import read_json
 from .data_audit import canonical_bytes
 output=Path(output);seed_all(config['seed'])
 model=TwoTower(config['architecture'],content,supported).to(device)
 optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=1e-4)
 cursor=EpochCursor(len(rows),config['seed']);stats=SamplerStats();selection=ValidationSelection()
 step=0;records=[];evaluations=[];epoch_checkpoints=[];best=None;elapsed=0.;evaluation_seconds=0.
 if resume:
  prior=verify_session(resume,binding)
  if Path(resume).parent.resolve()!=output.resolve() or prior['binding']!=binding or prior['max_epochs']!=30:raise ValueError('resume bindings/budget mismatch')
  state=restore_state(output/prior['last']['path'],prior['last']['sha256'],model,optimizer,cursor,stats,binding)
  if state['budget']['max_epochs']!=30 or state['step']!=prior['steps'] or state['budget']['records']!=prior['records'] or state['budget']['evaluations']!=prior['evaluations']:raise ValueError('resume manifest/checkpoint state mismatch')
  epoch_checkpoints=prior['epoch_checkpoints']
  selection.__dict__.update(state['selection']);step=state['step'];records=prior['records'];evaluations=prior['evaluations'];best=prior['best'];elapsed=prior['training_seconds'];evaluation_seconds=prior['evaluation_seconds']
  if selection.stopped or (until_step is not None and until_step<=step):raise ValueError('completed or exhausted segment')
 else:output.mkdir(parents=True,exist_ok=False)
 if until_step is not None and (type(until_step)is not int or until_step<1):raise ValueError('invalid segment limit')
 clock=time.monotonic();segment_eval=0.
 while not selection.stopped:
  if check:check()
  indices=cursor.next(config['batch_size'])
  record=train_step(model,optimizer,[rows[i] for i in indices],idmap,stats,cursor.epoch,config['temperature'])
  step+=1;records.append({**record,'step':step,'epoch':cursor.epoch+1,'example_indices':indices})
  improved=False
  if cursor.offset==cursor.size:
   before=time.monotonic();result=evaluate(model,cursor.epoch+1);duration=time.monotonic()-before;segment_eval+=duration
   primary=result['metrics']['all'];improved=selection.observe(cursor.epoch+1,primary['recall@50'],primary['ndcg@50'])
   evaluations.append({'epoch':cursor.epoch+1,'result':result})
  cutoff=until_step is not None and step>=until_step
  if cursor.offset==cursor.size or cutoff or step%50==0:
   if check:check()
   path=output/f'checkpoint-step-{step}.pt'
   logical_last={'path':path.name,'step':step,'epoch':cursor.epoch+1}
   logical_best={**logical_last,'metric':list(selection.best_metric)} if improved else ({k:v for k,v in best.items() if k!='sha256'} if best else None)
   budget={'max_epochs':30,'epoch_checkpoints':list(epoch_checkpoints),'records':records,'evaluations':evaluations,'best':logical_best,'training_seconds':elapsed+time.monotonic()-clock-segment_eval,'evaluation_seconds':evaluation_seconds+segment_eval}
   digest=save_state(path,model,optimizer,cursor,stats,step,binding,selection,budget)
   last={'path':path.name,'sha256':digest,'step':step,'epoch':cursor.epoch+1}
   if cursor.offset==cursor.size:epoch_checkpoints.append(dict(last))
   if improved:best={**last,'metric':list(selection.best_metric)}
   reason='patience' if selection.bad_epochs>=3 else 'max_epochs' if selection.epoch>=30 else 'segment_limit' if cutoff else 'checkpoint'
   value={'schema_version':1,'task_id':'P4D','kind':'core_train','status':'complete' if selection.stopped else 'segment_complete' if cutoff else 'running','binding':binding,'max_epochs':30,'steps':step,'completed_epochs':selection.epoch,'epoch_checkpoints':epoch_checkpoints,'selection':dict(selection.__dict__),'records':records,'evaluations':evaluations,'last':last,'best':best,'stop_reason':reason,'training_seconds':budget['training_seconds'],'evaluation_seconds':budget['evaluation_seconds'],'stats':{'drawn':stats.drawn,'skipped':stats.skipped,'epochs':stats.epochs},'manifest_name':f'manifest-step-{step}.json'}
   with (output/value['manifest_name']).open('xb') as stream:stream.write(canonical_bytes(value))
   if cutoff or selection.stopped:return value
 raise ValueError('unexpected exhausted session')

def export_vectors(model,catalog,queries,idmap,batch_size,check=None):
 import torch,numpy as np
 from .baselines import normalize
 model.eval();device=next(model.parameters()).device
 item_rows=[];user_rows=[]
 with torch.no_grad():
  for start in range(0,len(catalog),batch_size):
   if check:check()
   item_rows.append(model.items(torch.tensor([idmap[i] for i in catalog[start:start+batch_size]],device=device)).cpu().numpy())
  for start in range(0,len(queries),batch_size):
   if check:check()
   batch=queries[start:start+batch_size];histories=torch.zeros((len(batch),50),dtype=torch.long,device=device)
   for j,q in enumerate(batch):histories[j,:len(q['history'])]=torch.tensor([idmap[i] for i in q['history']],device=device)
   user_rows.append(model.users(histories).cpu().numpy())
 raw_users=np.concatenate(user_rows);raw_items=np.concatenate(item_rows)
 if check:check()
 users=normalize_export(raw_users,check);items=normalize_export(raw_items,check)
 return users,items

def load_training_inputs(prepared,holdout,prepared_sha,producer_source,validation_sha,check=None):
 import json,torch,numpy as np
 from pathlib import Path
 from .retrieval_data import checked,verify_prepared,catalog_map,read_split
 from .manifest import read_json
 if check:check()
 prepared=Path(prepared);checked(prepared/'prepare-manifest.json',prepared_sha)
 manifest=verify_prepared(prepared)
 if manifest['binding']['consumer']['source_fingerprint']!=producer_source:raise ValueError('prepared producer mismatch')
 if check:check()
 catalog=read_json(prepared/'catalog.json');idmap=catalog_map(catalog)
 with (prepared/'examples.jsonl').open() as stream:rows=[json.loads(line) for line in stream]
 content=torch.from_numpy(np.load(prepared/'content.npy',allow_pickle=False))
 queries=read_split(holdout,'validation',validation_sha,catalog)
 support=set(read_json(prepared/'support.json'));supported_indices=read_json(prepared/'idmap.json')['supported_indices']
 if check:check()
 return {'rows':rows,'content':content,'catalog':catalog,'idmap':idmap,'queries':queries,'support':support,'supported_indices':supported_indices,'prepared':manifest,'prepared_sha256':prepared_sha,'query_sha256':validation_sha}

def validate_runtime(value):
 from .retrieval_models import MODEL_ORDER
 fields={'schema_version','task_id','architecture','learning_rate','temperature','seed','batch_size','max_epochs','patience','precision'}
 if set(value)!=fields or any(type(value[k])is not int for k in ('schema_version','seed','batch_size','max_epochs','patience')) or value['schema_version']!=1 or value['task_id']!='P4D' or value['architecture'] not in MODEL_ORDER or value['learning_rate'] not in (.001,.0003) or value['temperature'] not in (.07,.15) or type(value['seed'])is not int or value['seed'] not in (42,2024,2025) or type(value['batch_size'])is not int or value['batch_size'] not in (512,256,128) or value['max_epochs']!=30 or value['patience']!=3 or value['precision']!='fp32':raise ValueError('unregistered runtime configuration')
 return value

def verify_session(path,binding=None):
 import torch
 from pathlib import Path
 from .manifest import read_json
 from .retrieval_data import checked
 from .data_audit import canonical_bytes
 path=Path(path);value=read_json(path)
 if value.get('task_id')!='P4D' or value.get('kind')!='core_train' or value.get('schema_version')!=1 or value['max_epochs']!=30 or (binding is not None and value['binding']!=binding):raise ValueError('session binding mismatch')
 replay=ValidationSelection()
 for rec in value['evaluations']:
  metrics=rec['result']['metrics']['all'];replay.observe(rec['epoch'],metrics['recall@50'],metrics['ndcg@50'])
 if canonical_bytes(replay.__dict__)!=canonical_bytes(value['selection']) or replay.epoch!=value['completed_epochs'] or value['steps']!=len(value['records']):raise ValueError('selection/consumed budget mismatch')
 if len(value['epoch_checkpoints'])!=len(value['evaluations']) or [c['epoch'] for c in value['epoch_checkpoints']]!=list(range(1,replay.epoch+1)):raise ValueError('epoch checkpoint ledger mismatch')
 for checkpoint in value['epoch_checkpoints']:
  if Path(checkpoint['path']).name!=checkpoint['path']:raise ValueError('epoch checkpoint path escape')
  epoch_state=torch.load(checked(path.parent/checkpoint['path'],checkpoint['sha256']),map_location='cpu',weights_only=True)
  if epoch_state['binding']!=value['binding'] or epoch_state['step']!=checkpoint['step'] or epoch_state['budget']['selection']['epoch']!=checkpoint['epoch']:raise ValueError('epoch checkpoint state mismatch')
 for role in ('last','best'):
  rec=value[role]
  if rec is None:
   if role=='best' and replay.best_epoch is None:continue
   raise ValueError('missing checkpoint role')
  if Path(rec['path']).name!=rec['path']:raise ValueError('checkpoint path escape')
  state=torch.load(checked(path.parent/rec['path'],rec['sha256']),map_location='cpu',weights_only=True)
  if state['binding']!=value['binding'] or state['step']!=rec['step']:raise ValueError('checkpoint binding mismatch')
  if role=='best' and (rec['epoch']!=replay.best_epoch or tuple(rec['metric'])!=replay.best_metric or state['budget']['selection']['best_epoch']!=rec['epoch'] or tuple(state['budget']['selection']['best_metric'])!=tuple(rec['metric'])):raise ValueError('best checkpoint/metric mismatch')
  if role=='last':
   prior_epochs=value['epoch_checkpoints'][:-1] if value['epoch_checkpoints'] and value['epoch_checkpoints'][-1]['step']==state['step'] else value['epoch_checkpoints']
   if state['budget']['epoch_checkpoints']!=prior_epochs:raise ValueError('epoch checkpoint chain mismatch')
   logical_best={k:v for k,v in value['best'].items() if k!='sha256'} if value['best'] else None
   if state['budget']['best']!=logical_best or canonical_bytes(state['budget']['selection'])!=canonical_bytes(value['selection']) or state['budget']['records']!=value['records'] or state['budget']['evaluations']!=value['evaluations'] or canonical_bytes(state['stats'])!=canonical_bytes(value['stats']):raise ValueError('last checkpoint/manifest mismatch')
 return value

def choose_shared_batch(measure):
 from .retrieval_models import MODEL_ORDER
 attempts=[]
 for batch in (512,256,128):
  measurements={}
  for architecture in MODEL_ORDER:
   measurements[architecture]=measure(architecture,batch)
   if measurements[architecture].get('failure') and measurements[architecture].get('failure_kind')!='capacity':raise ValueError('pilot non-capacity stop gate')
  passed=all(set(rec.get('resources_minimum',{}))=={'ram','vram','disk'} and all(math.isfinite(v) and .2<=v<=1 for v in rec['resources_minimum'].values()) and not rec.get('failure') for rec in measurements.values())
  attempts.append({'batch_size':batch,'models':measurements,'all_pass':passed})
  if passed:return {'batch_size':batch,'attempts':attempts}
 raise ValueError('no shared batch passes resource gate')

def configure_precision(device):
 import os,torch
 if device=='cuda' and os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('CUDA workspace must be set before interpreter')
 torch.use_deterministic_algorithms(True)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.set_float32_matmul_precision('highest');torch.set_num_threads(4)
 # The standard math attention implementation avoids fused-kernel nondeterminism.
 torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
 return {'precision':'fp32','deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32,'cublas_workspace':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),'flash_sdp':torch.backends.cuda.flash_sdp_enabled(),'mem_efficient_sdp':torch.backends.cuda.mem_efficient_sdp_enabled(),'math_sdp':torch.backends.cuda.math_sdp_enabled(),'threads':torch.get_num_threads()}

def persist_evaluation(output,model,catalog,queries,idmap,support,batch_size,check=None):
 import numpy as np,time
 from pathlib import Path
 from .data_audit import canonical_bytes,file_sha256
 output=Path(output);output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
 users,items=export_vectors(model,catalog,queries,idmap,batch_size,check)
 result=evaluate_vectors(users,items,catalog,queries,support,check=check)
 for name,array in (('user-vectors.npy',users),('item-vectors.npy',items)):
  with (output/name).open('xb') as stream:np.save(stream,array,allow_pickle=False)
 predictions=result.pop('predictions')
 with (output/'predictions.json').open('xb') as stream:stream.write(canonical_bytes(predictions))
 with (output/'metrics.json').open('xb') as stream:stream.write(canonical_bytes({k:v for k,v in result.items() if k not in ('exact_seconds',)}))
 result['artifacts']={name:file_sha256(output/name) for name in ('user-vectors.npy','item-vectors.npy','predictions.json','metrics.json')}
 result['export_and_exact_seconds']=time.monotonic()-started
 if check:check()
 return result

def verify_evaluation(output,result,catalog,queries,support,check=None):
 import numpy as np
 from pathlib import Path
 from .retrieval_data import checked
 from .manifest import read_json
 from .data_audit import canonical_bytes
 output=Path(output)
 if set(result['artifacts'])!={'user-vectors.npy','item-vectors.npy','predictions.json','metrics.json'}:raise ValueError('epoch artifact set mismatch')
 for name,digest in result['artifacts'].items():checked(output/name,digest)
 users=np.load(output/'user-vectors.npy',allow_pickle=False);items=np.load(output/'item-vectors.npy',allow_pickle=False)
 if users.dtype!=np.float32 or items.dtype!=np.float32 or users.shape!=(len(queries),64) or items.shape!=(len(catalog),64):raise ValueError('export shape/dtype mismatch')
 replay=evaluate_vectors(users,items,catalog,queries,support,check=check)
 if read_json(output/'predictions.json')!=replay['predictions']:raise ValueError('prediction replay mismatch')
 metrics=read_json(output/'metrics.json')
 for key in ('metrics','support_metrics','zero_item_vectors','zero_user_vectors'):
  if canonical_bytes(result[key])!=canonical_bytes(replay[key]) or canonical_bytes(metrics[key])!=canonical_bytes(replay[key]):raise ValueError('metric replay mismatch')
 return replay

def measure_gpu_model(architecture,batch,data,output,device='cuda'):
 """Resource-only synthetic worst prefix; never selection metrics."""
 import torch,time,gc
 from .retrieval_models import TwoTower
 from .retrieval import resource_snapshot
 from .training import seed_all
 if device!='cuda' or not torch.cuda.is_available():raise ValueError('actual CUDA pilot required')
 started=time.monotonic();snapshots=[];model=None;optimizer=None
 def check():
  try:sample=capacity_snapshot(output)
  except ResourceCapacityError as error:
   snapshots.append(error.resources);raise
  snapshots.append(sample);return sample
 try:
  seed_all(42);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();check()
  model=TwoTower(architecture,data['content'],data['supported_indices']).cuda()
  optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
  indices=data['supported_indices'][:batch]
  if len(indices)!=batch:raise ValueError('insufficient distinct support for representative batch')
  targets=torch.tensor(indices,device='cuda');histories=targets[(torch.arange(batch,device='cuda').unsqueeze(1)+torch.arange(50,device='cuda').unsqueeze(0)+1)%batch]
  for _ in range(3):
   model.train();optimizer.zero_grad(set_to_none=True)
   users=torch.nn.functional.normalize(model.users(histories),dim=1);items=torch.nn.functional.normalize(model.items(targets),dim=1)
   loss=torch.nn.functional.cross_entropy(users@items.T/.07,torch.arange(batch,device='cuda'))
   if not torch.isfinite(loss):raise ValueError('nonfinite pilot loss')
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();torch.cuda.synchronize();check()
  export_context=resource_export_context(__import__('pathlib').Path(output)/f'export-{batch}-{architecture}',model,data,batch,check)
  torch.cuda.synchronize();check();free,total=torch.cuda.mem_get_info()
  return {'resources_minimum':{key:min(s[key] for s in snapshots) for key in ('ram','vram','disk')},'measured_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'measured_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'free_cuda_bytes':free,'total_cuda_bytes':total,'elapsed_seconds':time.monotonic()-started,'history_width':50,'optimizer_steps':3,'export_context':export_context,'scope':'resource-only synthetic worst-prefix forward/backward/AdamW plus full validation exports and maximum normal exact query/item blocks; no quality metrics'}
 except (RuntimeError,ValueError) as error:
  return {'failure':type(error).__name__+': '+str(error),'failure_kind':pilot_failure_kind(error),'resources_minimum':{key:min(s[key] for s in snapshots) for key in ('ram','vram','disk')} if snapshots else {},'measured_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'measured_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'elapsed_seconds':time.monotonic()-started}
 finally:
  del model,optimizer;gc.collect();torch.cuda.empty_cache()

def write_plan(output):
 from pathlib import Path
 from .data_audit import canonical_bytes
 output=Path(output);output.mkdir(parents=True,exist_ok=False)
 value={'schema_version':1,'task_id':'P4D','kind':'core_run_plan','tuning_runs':core_configs(),'additional_seeds':[2024,2025],'additional_role':'selected config for each architecture only; all seeds reported','maximum_formal_runs':24,'max_epochs_per_run':30,'patience':3,'model_order':[c['architecture'] for c in core_configs()[::4]],'test_gate':'closed'}
 with (output/'run-plan.json').open('xb') as stream:stream.write(canonical_bytes(value))
 return value

def main(argv=None):
 import argparse,sys
 from .data_audit import canonical_bytes
 parser=argparse.ArgumentParser(description='P4D fixed core experiments; test gate closed')
 commands=parser.add_subparsers(dest='command',required=True)
 p=commands.add_parser('plan');p.add_argument('--output',required=True)
 for command in ('pilot','train','verify'):
  p=commands.add_parser(command);p.add_argument('--split',choices=['validation'],default='validation')
  for name in ('prepared','holdout-directory','output'):p.add_argument('--'+name,required=True)
  if command=='train':
   for name in ('runtime-config','pilot','pilot-sha256'):p.add_argument('--'+name,required=True)
   p.add_argument('--until-step',type=int);p.add_argument('--resume');p.add_argument('--selection');p.add_argument('--selection-sha256')
  if command=='verify':
   p.add_argument('--manifest',required=True);p.add_argument('--manifest-sha256',required=True)
 p=commands.add_parser('select')
 for name in ('run-list','output'):p.add_argument('--'+name,required=True)
 args=parser.parse_args(argv)
 try:
  if args.command=='plan':value=write_plan(args.output)
  elif args.command=='pilot':value=gpu_pilot(args)
  elif args.command=='train':value=formal_train(args)
  elif args.command=='verify':value=formal_verify(args)
  else:value=select_core(args)
  print(canonical_bytes({'task_id':'P4D','kind':value['kind'],'status':value.get('status','success')}).decode(),end='');return 0
 except (OSError,ValueError,KeyError,TypeError,RuntimeError):
  print('ERROR P4D contract/input/state/resource gate failed; preserve evidence',file=sys.stderr);return 2



def architecture_spec(architecture):
 return {'architecture':architecture,'dimension':64,'dtype':'float32','id_initialization':'torch Embedding default N(0,1), PAD row zero','id_support':'full-train likes; unsupported output fixed zero','content_projection':'Linear(64,64,bias=False) default U(-1/sqrt(64),1/sqrt(64))' if architecture!='id_mean' else None,'content':'immutable official normalized float32 64D buffer' if architecture!='id_mean' else None,'ordinal_decay':.8 if architecture=='ordinal_content_mean' else None,'transformer':{'layers':1,'heads':4,'feedforward':256,'activation':'relu','dropout':.1,'layer_norm_eps':1e-5,'norm_first':False,'internal_bias':True,'batch_first':True,'mask':'past-only prefix bidirectional; PAD key mask; valid output mean','positions':'Embedding(50,64) N(0,1), ordinal count of valid events','user_projection':'Linear(64,64,bias=False) default initialization'} if architecture=='content_transformer' else None,'mean_user_projection':False,'export':'NumPy float32 norm axis=1 keepdims; zero remains zero','exact':'float64 fixed-pair einsum optimize=False on exported float32, raw item-ID ties'}

def real_inputs(args):
 from pathlib import Path
 from .retrieval import ROOT,PREPARED_SHA,PREPARED_SOURCE,resource_snapshot,consumer_identity
 from .retrieval_data import validate_contract
 from .data_audit import file_sha256
 from .manifest import read_json
 import torch
 if not torch.cuda.is_available():raise ValueError('existing CUDA environment required')
 samples=[]
 def check():
  sample=resource_snapshot(Path(args.output).parent);samples.append(sample);return sample
 check.samples=samples
 precision=configure_precision('cuda');check();pins=validate_contract(ROOT)
 query_pin=next(r for r in pins['full_query_pins'] if r['role']=='validation-queries.json')
 data=load_training_inputs(args.prepared,args.holdout_directory,PREPARED_SHA,PREPARED_SOURCE,query_pin['sha256'],check)
 if data['prepared']['artifacts']['catalog.json']!=pins['catalog_sha256'] or data['prepared']['binding']['query_pins']!=pins['full_query_pins'] or data['prepared']['binding']['legacy_producer']!=pins['legacy_p3_producer']:raise ValueError('prepared/contract lineage mismatch')
 consumer=consumer_identity();origin=ROOT/'cache/p4d-source-entry.json'
 if origin.exists():
  entry=read_json(origin)
  if entry['source_fingerprint']!=consumer['source_fingerprint']:raise ValueError('P4D transport source mismatch')
  consumer['source_transport_origin']=entry
 binding={'consumer':consumer,'prepared_manifest_sha256':PREPARED_SHA,'prepared_producer':data['prepared']['binding']['consumer'],'prepared_artifacts':data['prepared']['artifacts'],'validation_query_sha256':query_pin['sha256'],'catalog_sha256':pins['catalog_sha256'],'contract_manifest_sha256':file_sha256(ROOT/'artifacts/p3d-retrieval-contract-manifest.json'),'test_gate':'closed_hash_only','precision':precision}
 return data,binding,check

def gpu_pilot(args):
 from pathlib import Path
 from .data_audit import canonical_bytes
 output=Path(args.output)
 if output.exists():raise FileExistsError('exclusive pilot output required')
 data,binding,check=real_inputs(args);output.mkdir(parents=True,exist_ok=False)
 ledger=[]
 def measure(model,batch):
  result=measure_gpu_model(model,batch,data,output)
  ledger.append({'architecture':model,'batch_size':batch,**result})
  with (output/f'measure-{batch}-{model}.json').open('xb') as stream:stream.write(canonical_bytes(ledger[-1]))
  return result
 try:chosen=choose_shared_batch(measure)
 except (RuntimeError,ValueError):
  with (output/'pilot-failure.json').open('xb') as stream:stream.write(canonical_bytes({'task_id':'P4D','kind':'resource_pilot','status':'failed','binding':binding,'measurements':ledger}))
  raise
 value={'schema_version':1,'task_id':'P4D','kind':'resource_pilot','status':'success','binding':binding,**chosen,'resources_end':check(),'selection_metrics_read':False}
 with (output/'pilot-manifest.json').open('xb') as stream:stream.write(canonical_bytes(value))
 return value

def formal_train(args):
 import torch
 from pathlib import Path
 from .data_audit import canonical_bytes,file_sha256
 from .manifest import read_json
 from .retrieval_data import checked
 data,input_binding,check=real_inputs(args)
 config=validate_runtime(read_json(Path(args.runtime_config)));pilot=read_json(checked(args.pilot,args.pilot_sha256))
 if pilot.get('kind')!='resource_pilot' or pilot.get('status')!='success' or pilot['binding']!=input_binding or config['batch_size']!=pilot['batch_size'] or not pilot['attempts'][-1]['all_pass']:raise ValueError('frozen pilot/runtime mismatch')
 if config['seed']!=42:
  if not args.selection or not args.selection_sha256:raise ValueError('additional seed requires frozen selected config')
  selection=read_json(checked(args.selection,args.selection_sha256))
  require_selected_configuration(selection,input_binding,config,args.pilot_sha256)
 binding={**input_binding,'config':config,'runtime_config_sha256':file_sha256(args.runtime_config),'pilot_manifest_sha256':args.pilot_sha256,'architecture':architecture_spec(config['architecture']),'selection_sha256':args.selection_sha256 if config['seed']!=42 else None}
 output=Path(args.output)
 def evaluate(model,epoch):
  result=persist_evaluation(output/f'epoch-{epoch}',model,data['catalog'],data['queries'],data['idmap'],data['support'],config['batch_size'],check)
  result['directory']=f'epoch-{epoch}'
  return result
 try:
  torch.cuda.reset_peak_memory_stats()
  value=run_session(output,data['rows'],data['content'],data['supported_indices'],data['idmap'],config,binding,evaluate,'cuda',args.until_step,args.resume,check)
 except (OSError,ValueError,RuntimeError) as error:
  if output.exists():
   import time
   with (output/f'failure-{time.time_ns()}.json').open('xb') as stream:stream.write(canonical_bytes({'task_id':'P4D','status':'failed','binding':binding,'failure':type(error).__name__+': '+str(error)}))
  raise
 # A separate immutable completion record binds final resource measurements.
 value={**value,'resources_end':check(),'resources_minimum':{key:min(sample[key] for sample in check.samples) for key in ('ram','vram','disk')},'resource_samples':len(check.samples),'measured_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'measured_peak_reserved_bytes':torch.cuda.max_memory_reserved()}
 with (output/f'resource-step-{value["steps"]}.json').open('xb') as stream:stream.write(canonical_bytes(value))
 return value

def formal_verify(args):
 import torch,numpy as np
 from pathlib import Path
 from .retrieval_models import TwoTower
 from .retrieval_data import checked
 from .data_audit import canonical_bytes
 data,input_binding,check=real_inputs(args);checked(args.manifest,args.manifest_sha256);value=verify_session(args.manifest)
 expected_cursor=verify_draw_provenance(value,data)
 actual_last=torch.load(checked(Path(args.manifest).parent/value['last']['path'],value['last']['sha256']),map_location='cpu',weights_only=True)
 if canonical_bytes(expected_cursor)!=canonical_bytes(actual_last['cursor']):raise ValueError('draw replay/cursor checkpoint mismatch')
 for key,expected in input_binding.items():
  if value['binding'].get(key)!=expected:raise ValueError('formal runtime/input mismatch')
 config=validate_runtime(value['binding']['config']);root=Path(args.manifest).parent
 for evaluation in value['evaluations']:
  result=evaluation['result']
  if result['directory']!=f'epoch-{evaluation["epoch"]}':raise ValueError('epoch path mismatch')
  checkpoint=value['epoch_checkpoints'][evaluation['epoch']-1]
  verify_checkpoint_evaluation(root,checkpoint,root/result['directory'],result,data,config,value['binding'],'cuda',check)
 check();return {**value,'verification':'all epoch checkpoint bitwise vector reexport + all exact ranks/metrics + sampler/order/mask replay'}

def verify_checkpoint_evaluation(root,checkpoint,output,result,data,config,binding,device,check=None):
 import torch,numpy as np
 from pathlib import Path
 from .retrieval_models import TwoTower
 from .retrieval_data import checked
 root=Path(root);output=Path(output)
 if Path(checkpoint['path']).name!=checkpoint['path']:raise ValueError('checkpoint path escape')
 state=torch.load(checked(root/checkpoint['path'],checkpoint['sha256']),map_location='cpu',weights_only=True)
 if state['binding']!=binding or state['step']!=checkpoint['step'] or state['budget']['selection']['epoch']!=checkpoint['epoch']:raise ValueError('epoch checkpoint binding mismatch')
 model=TwoTower(config['architecture'],data['content'],data['supported_indices']).to(device);model.load_state_dict(state['model'])
 users,items=export_vectors(model,data['catalog'],data['queries'],data['idmap'],config['batch_size'],check)
 if not np.array_equal(users,np.load(output/'user-vectors.npy',allow_pickle=False)) or not np.array_equal(items,np.load(output/'item-vectors.npy',allow_pickle=False)):raise ValueError('checkpoint vector replay mismatch')
 return verify_evaluation(output,result,data['catalog'],data['queries'],data['support'],check)

def freeze_selection(results,input_binding):
 from .retrieval_models import MODEL_ORDER
 expected=core_configs()
 if len(results)!=16 or [r['config'] for r in results]!=expected:raise ValueError('complete fixed seed42 enumeration required')
 chosen={model:select_config([r for r in results if r['config']['architecture']==model]) for model in MODEL_ORDER}
 best=select_config(list(chosen.values()))
 return {'schema_version':1,'task_id':'P4D','kind':'core_selection','status':'success','input_binding':input_binding,'selected_configs':chosen,'best_architecture':best['config']['architecture'],'representative_seed':42,'all_tuning_runs':results,'additional_seeds':[2024,2025],'test_gate':'closed'}

def select_core(args):
 from pathlib import Path
 from .data_audit import canonical_bytes,file_sha256
 from .manifest import read_json
 records=read_json(Path(args.run_list))
 if not isinstance(records,list) or len(records)!=16:raise ValueError('complete run list required')
 results=[];binding=None
 for path in records:
  run=verify_session(path)
  if run['status']!='complete' or run['binding']['config']['seed']!=42:raise ValueError('complete seed42 run required')
  current={key:value for key,value in run['binding'].items() if key not in ('config','runtime_config_sha256','architecture','selection_sha256')}
  current['shared_batch_size']=run['binding']['config']['batch_size']
  if binding is None:binding=current
  if current!=binding:raise ValueError('mixed input/consumer sources')
  config={k:run['binding']['config'][k] for k in ('architecture','learning_rate','temperature','seed')}
  results.append({'config':config,'metric':run['best']['metric'],'manifest_sha256':file_sha256(path),'best':run['best'],'checkpoint_logical_ref':Path(path).parent.name+'/'+run['best']['path']})
 value=freeze_selection(results,binding)
 with Path(args.output).open('xb') as stream:stream.write(canonical_bytes(value))
 return value



def verify_draw_provenance(value,data):
 from .retrieval_training import EpochCursor,candidate_batch,SamplerStats
 from .data_audit import canonical_bytes
 config=value['binding']['config'];cursor=EpochCursor(len(data['rows']),config['seed']);stats=SamplerStats()
 for step,record in enumerate(value['records'],1):
  indices=cursor.next(config['batch_size'])
  if record['step']!=step or record['epoch']!=cursor.epoch+1 or record['example_indices']!=indices or record['drawn']!=len(indices):raise ValueError('draw order/budget mismatch')
  columns,labels,mask=candidate_batch([data['rows'][i] for i in indices],data['idmap'])
  skipped=int((mask.sum(1)<=1).sum())
  if record['skipped']!=skipped:raise ValueError('past-positive mask/skip mismatch')
  stats.record(cursor.epoch,len(indices),skipped);stats.check(cursor.epoch)
  if skipped!=len(indices) and (record['loss'] is None or not math.isfinite(record['loss']) or not math.isfinite(record['gradient_norm'])):raise ValueError('invalid training record')
 actual={'drawn':stats.drawn,'skipped':stats.skipped,'epochs':stats.epochs}
 if canonical_bytes(value['stats'])!=canonical_bytes(actual):raise ValueError('skip denominator reconciliation mismatch')
 return cursor.state_dict()



def require_selected_configuration(selection,input_binding,config,pilot_sha):
 expected={**input_binding,'shared_batch_size':config['batch_size'],'pilot_manifest_sha256':pilot_sha}
 if selection.get('kind')!='core_selection' or selection['input_binding']!=expected:raise ValueError('selected shared batch/pilot/input mismatch')
 chosen=selection['selected_configs'][config['architecture']]['config']
 if any(config[k]!=chosen[k] for k in ('architecture','learning_rate','temperature')):raise ValueError('additional seed selected configuration mismatch')



def pilot_failure_kind(error):
 import torch
 return 'capacity' if isinstance(error,(torch.cuda.OutOfMemoryError,ResourceCapacityError)) else 'gate'



class ResourceCapacityError(ValueError):
 """Actual finite resource fractions below the preregistered minimum."""
 def __init__(self,message,resources=None):
  super().__init__(message);self.resources=resources

def capacity_snapshot(path):
 import torch,shutil,sys
 from pathlib import Path
 if sys.platform!='linux' or not torch.cuda.is_available():raise ValueError('actual Linux CUDA measurements required')
 memory={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
 ram=memory['MemAvailable']/memory['MemTotal'];limit=Path('/sys/fs/cgroup/memory.max');current=Path('/sys/fs/cgroup/memory.current')
 if limit.exists() and limit.read_text().strip()!='max':ram=min(ram,1-int(current.read_text())/int(limit.read_text()))
 free,total=torch.cuda.mem_get_info();disk=shutil.disk_usage(path)
 result={'ram':ram,'vram':free/total,'disk':disk.free/disk.total}
 if any(not math.isfinite(value) or not 0<=value<=1 for value in result.values()):raise ValueError('invalid/nonfinite resource measurements')
 if any(value<.2 for value in result.values()):raise ResourceCapacityError('measured resource headroom below minimum',result)
 return result

def normalize_export(rows,check=None):
 import numpy as np
 if rows.dtype!=np.float32 or rows.ndim!=2 or not np.isfinite(rows).all():raise ValueError('invalid exported vectors')
 norms=np.linalg.norm(rows,axis=1,keepdims=True)
 normalized=np.divide(rows,norms,out=np.zeros_like(rows),where=norms!=0)
 if check:check()  # rows, norms and normalized remain alive during measurement.
 return normalized

def resource_export_context(output,model,data,batch,check):
 import numpy as np,resource
 from pathlib import Path
 from .baselines import blockwise_topk
 from .data_audit import file_sha256
 output=Path(output);output.mkdir(parents=True,exist_ok=False)
 users,items=export_vectors(model,data['catalog'],data['queries'],data['idmap'],batch,check)
 # Full exports remain alive; only the largest ordinary exact query block is ranked.
 query_count=min(32,len(users));item_count=min(4096,len(items))
 score_block=np.einsum('qd,id->qi',users[:query_count],items[:item_count],dtype=np.float64,optimize=False)
 rankings=blockwise_topk(users[:query_count],items,data['catalog'],50)
 check()
 for name,array in (('user-vectors.npy',users),('item-vectors.npy',items),('resource-rankings.npy',rankings)):
  with (output/name).open('xb') as stream:np.save(stream,array,allow_pickle=False)
 check()
 names=('user-vectors.npy','item-vectors.npy','resource-rankings.npy')
 return {'artifacts':{name:file_sha256(output/name) for name in names},'artifact_bytes':sum((output/name).stat().st_size for name in names),'full_export_users':len(users),'full_export_items':len(items),'ranked_resource_queries':query_count,'score_block_shape':list(score_block.shape),'process_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'scope':'full exports plus maximum normal query block/full catalog exact ranking and live float64 scoring block; resource only, no targets or metrics consumed'}

if __name__=='__main__':raise SystemExit(main())
