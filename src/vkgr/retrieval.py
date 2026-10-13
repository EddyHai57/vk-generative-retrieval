"""P4B audit/prepare and bounded synthetic training CLI; test gate closed."""
import argparse
import json
import sys
from pathlib import Path
from .data_audit import canonical_bytes,file_sha256
from .manifest import read_json,source_fingerprint,git_context
from .retrieval_data import validate_contract,checked
ROOT=Path(__file__).resolve().parents[2]

def consumer_identity():
 import platform
 from importlib.metadata import version
 try:commit,dirty=git_context()
 except __import__("subprocess").CalledProcessError:commit,dirty=None,None
 origin_path=ROOT/"cache/p4b-source-entry.json"
 origin=read_json(origin_path) if origin_path.exists() else None
 if origin is not None and origin["source_fingerprint"]!=source_fingerprint():raise ValueError("source transport mismatch")
 return {'git_commit':commit,'git_dirty':dirty,'source_transport_origin':origin,'source_fingerprint':source_fingerprint(),'lock_sha256':file_sha256(ROOT/'uv.lock'),'config_sha256':file_sha256(ROOT/'configs/traditional.yaml'),'python':platform.python_version(),'dependencies':{name:version(name) for name in ('numpy','torch','pyarrow','PyYAML')}}

def synthetic_train(output,max_steps,until_step,resume=None):
 import torch,time
 from datetime import datetime,timezone
 from .retrieval_training import IDMean,EpochCursor,SamplerStats,train_step,save_checkpoint,load_checkpoint
 from .training import seed_all
 if type(max_steps)is not int or not 1<=max_steps<=200 or not 1<=until_step<=max_steps:raise ValueError('bounded synthetic steps required')
 validate_contract(ROOT);output=Path(output).resolve()
 rows=[{'history':[i],'target':i+5,'past_likes':[i]} for i in range(1,6)];idmap={i:i for i in range(1,11)}
 binding={'consumer':consumer_identity(),'protocol':'traditional-like-v1','origin':'hand_authored_synthetic_not_formal_experiment','input_sha256':__import__('hashlib').sha256(canonical_bytes(rows)).hexdigest(),'max_steps':max_steps,'seed':42,'batch_size':3,'lr':.03,'temperature':.15,'optimizer':'AdamW','weight_decay':1e-4,'gradient_clip':1.}
 seed_all(42);torch.set_num_threads(1);model=IDMean(10,list(range(1,11)));opt=torch.optim.AdamW(model.parameters(),lr=.03,weight_decay=1e-4);cursor=EpochCursor(5,42);stats=SamplerStats();records=[];step=0;best=None;started=datetime.now(timezone.utc).isoformat();elapsed=0.
 if resume:
  prior=verify_train(resume)
  if Path(resume).resolve().parent!=output or prior['binding']!=binding:raise ValueError('resume output/config/budget binding mismatch')
  state=load_checkpoint(output/prior['last']['path'],prior['last']['sha256'],model,opt,cursor,stats,binding)
  step=state['step'];records=prior['records'];best=prior['best'];started=state['budget']['started_at_utc'];elapsed=state['budget']['elapsed_seconds']
  if step>=until_step or state['budget']['max_steps']!=max_steps:raise ValueError('invalid resumed budget')
 else:output.mkdir(parents=True,exist_ok=False)
 clock=time.monotonic()
 while step<until_step:
  batch=cursor.next(3);record=train_step(model,opt,[rows[i] for i in batch],idmap,stats,cursor.epoch,.15);step+=1;record.update({'step':step,'epoch':cursor.epoch,'batch':batch});records.append(record)
  path=output/f'checkpoint-step-{step}.pt';budget={'steps':step,'max_steps':max_steps,'started_at_utc':started,'elapsed_seconds':elapsed+time.monotonic()-clock}
  digest=save_checkpoint(path,model,opt,cursor,stats,step,binding,budget);last={'path':path.name,'sha256':digest,'step':step}
  if best is None or record['loss']<best['loss']:best={**last,'loss':record['loss']}
 value={'schema_version':1,'task_id':'P4B','kind':'synthetic_train','status':'success','binding':binding,'step':step,'records':records,'last':last,'best':best,'best_role':'minimum_training_loss_synthetic_only','stats':{'epochs':{str(k):v for k,v in stats.epochs.items()},'drawn':stats.drawn,'skipped':stats.skipped},'budget':budget}
 with (output/f'manifest-step-{step}.json').open('xb') as stream:stream.write(canonical_bytes(value))
 return value

def verify_train(path):
 import torch
 path=Path(path).resolve();value=read_json(path);validate_contract(ROOT)
 if value.get('schema_version')!=1 or value.get('task_id')!='P4B' or value.get('kind')!='synthetic_train' or value.get('status')!='success' or value['binding']['consumer']!=consumer_identity():raise ValueError('training manifest/runtime mismatch')
 if value['step']!=len(value['records']) or value['budget']['steps']!=value['step'] or value['step']>value['binding']['max_steps']:raise ValueError('consumed budget mismatch')
 for role in ('best','last'):
  rec=value[role]
  if Path(rec['path']).name!=rec['path']:raise ValueError('checkpoint path escape')
  state=torch.load(checked(path.parent/rec['path'],rec['sha256']),map_location='cpu',weights_only=True)
  if state['binding']!=value['binding'] or state['step']!=rec['step']:raise ValueError('checkpoint manifest mismatch')
  if role=='last' and (state['budget']!=value['budget'] or state['step']!=value['step'] or state['stats']['drawn']!=value['stats']['drawn'] or state['stats']['skipped']!=value['stats']['skipped']):raise ValueError('checkpoint sampler/budget mismatch')
 return value

def require_resources(fractions):
 import math
 if set(fractions)!={'ram','vram','disk'} or any(not isinstance(v,(int,float)) or not math.isfinite(v) or not .2<=v<=1 for v in fractions.values()):raise ValueError('resource headroom below 20%')

def resource_snapshot(path):
 import shutil,subprocess
 if sys.platform!='linux':raise ValueError('real prepare is server-only Linux')
 memory={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
 ram=memory['MemAvailable']/memory['MemTotal'];limit=Path('/sys/fs/cgroup/memory.max');current=Path('/sys/fs/cgroup/memory.current')
 if limit.exists() and limit.read_text().strip()!='max':ram=min(ram,1-int(current.read_text())/int(limit.read_text()))
 usage=shutil.disk_usage(path)
 result=subprocess.run(['nvidia-smi','--query-gpu=memory.free,memory.total','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True)
 pairs=[list(map(float,line.split(','))) for line in result.stdout.strip().splitlines()];fractions={'ram':ram,'vram':min(free/total for free,total in pairs),'disk':usage.free/usage.total};require_resources(fractions);return fractions

def audit_inputs(train_directory,holdout_directory,raw_root):
 from .data_audit import validate_frozen_manifest
 from .retrieval_data import catalog_map
 train_directory=Path(train_directory).resolve();holdout_directory=Path(holdout_directory).resolve();raw_root=Path(raw_root).resolve();pins=validate_contract(ROOT)
 legacy_path=checked(train_directory/'frozen-contract-manifest.json','3b042b7635972e9a255c0b2cd8de0a7b7a472a08a0d074eb3133068d527c1e0f')
 frozen=validate_frozen_manifest(legacy_path,'3b042b7635972e9a255c0b2cd8de0a7b7a472a08a0d074eb3133068d527c1e0f');evidence=frozen['train_evidence']
 train_path=checked(train_directory/'train-audit.json',evidence['train_audit_sha256']);train=read_json(train_path)
 if train['producer']!=pins['legacy_p3_producer'] or train['producer']!=evidence['producer'] or train['policy_sha256']!=pins['legacy_policy_sha256']:raise ValueError('legacy producer binding mismatch')
 catalog_path=checked(train_directory/'catalog.json',pins['catalog_sha256']);catalog=read_json(catalog_path);catalog_map(catalog)
 if len(catalog)!=train['catalog_items']:raise ValueError('catalog denominator mismatch')
 checked(train_directory/'training-examples.json',train['training_examples_sha256'])
 expected_paths=[f'subsamples/ur0.01_ir0.01/train/week_{w:02d}.parquet' for w in range(25)]
 if [r['path'] for r in train['train_files']]!=expected_paths:raise ValueError('pinned train partition/order mismatch')
 records=[]
 for rec in train['train_files']+[r for r in train['feature_files'] if r['path']=='metadata/item_embeddings.npz']:
  path=checked(raw_root/rec['path'],rec['sha256'])
  if path.stat().st_size!=rec['size_bytes']:raise ValueError('raw input size mismatch')
  records.append({'role':rec['path'],'sha256':rec['sha256'],'size_bytes':rec['size_bytes']})
 if len(records)!=26:raise ValueError('missing official content pin')
 # Bind both holdout bytes, but never decode test values or compute metrics.
 query_pins=pins['full_query_pins']
 for rec in query_pins:checked(holdout_directory/rec['role'],rec['sha256'])
 return {'train':train,'catalog':catalog,'input_records':records,'query_pins':query_pins,'legacy_producer':pins['legacy_p3_producer'],'legacy_training_examples_sha256':train['training_examples_sha256'],'consumer':consumer_identity(),'test_gate':'closed_hash_only'}

def prepare(train_directory,holdout_directory,raw_root,output):
 import numpy as np
 from .data_audit import parquet_rows
 from .retrieval_data import prefix_examples,content_join,write_prepared,verify_prepared
 output=Path(output).resolve()
 if output.exists():raise FileExistsError('exclusive prepare output required')
 snapshot=resource_snapshot(output.parent);inputs=audit_inputs(train_directory,holdout_directory,raw_root);raw_root=Path(raw_root)
 support=set();observed=set();positive_events=0
 def partition_rows(path):
  nonlocal positive_events
  resource_snapshot(output.parent)
  for row in parquet_rows(path):
   observed.add(row['item_id'])
   if row['like']:support.add(row['item_id']);positive_events+=1
   yield row
 # One streamed scan builds all prefixes plus all-event support, including single-like users.
 examples=list(prefix_examples((w,partition_rows(raw_root/r['path'])) for w,r in enumerate(inputs['train']['train_files'])))
 if sorted(observed)!=inputs['catalog'] or len(examples)!=inputs['train']['training_examples']:raise ValueError('train/catalog reconciliation mismatch')
 legacy=read_json(Path(train_directory)/'training-examples.json')
 if [{k:v for k,v in row.items() if k not in ('past_likes','example_id')} for row in examples]!=legacy:raise ValueError('every legacy target/history identity mismatch')
 if len(support)!=inputs['train']['audit']['like_items'] or positive_events!=inputs['train']['audit']['feedback']['like']['true']:raise ValueError('train support reconciliation mismatch')
 with np.load(raw_root/'metadata/item_embeddings.npz',allow_pickle=False) as source:features=content_join(inputs['catalog'],source['item_id'],source['embedding'])
 binding={k:v for k,v in inputs.items() if k not in ('train','catalog')};binding['resources_start']=snapshot;binding['resources_before_write']=resource_snapshot(output.parent);binding['reconciliation']={'every_legacy_example':True,'all_train_exposure_catalog':True,'full_train_like_support':True,'train_like_items':len(support),'train_like_events':positive_events}
 value=write_prepared(output,inputs['catalog'],features,examples,binding,sorted(support));verify_prepared(output);return value

PREPARED_SHA='7f3ccd70c02157542e94f41f39e62988fbd650b4be8976455ff20cfe1e4394d2'
PREPARED_SOURCE='5868c2c5db3df83bcd8f15a0235aba39a31ea4a6d532f31a4a69dfe1699fe6fe'

def load_baseline_inputs(train_directory,holdout_directory,raw_root,prepared,legacy_validation):
 import numpy as np
 from .retrieval_data import verify_prepared,read_split
 from .baselines import sample_queries
 prepared=Path(prepared).resolve();checked(prepared/'prepare-manifest.json',PREPARED_SHA)
 original=verify_prepared(prepared);inputs=audit_inputs(train_directory,holdout_directory,raw_root)
 if original['binding']['consumer']['source_fingerprint']!=PREPARED_SOURCE or original['binding']['input_records']!=inputs['input_records'] or original['binding']['query_pins']!=inputs['query_pins'] or original['artifacts']['catalog.json']!=file_sha256(Path(train_directory)/'catalog.json'):raise ValueError('prepared static producer/input mismatch')
 query_pin=next(r for r in inputs['query_pins'] if r['role']=='validation-queries.json')
 queries=read_split(holdout_directory,'validation',query_pin['sha256'],inputs['catalog'])
 holdout=read_json(checked(Path(holdout_directory)/'holdout-audit.json','f9e44a97d410488b7c132abaa27da66076b27917474bcc1adaf0875f46c6f85d'))
 if len(queries)!=holdout['validation']['eligible_queries']:raise ValueError('full query denominator mismatch')
 old=read_json(checked(Path(legacy_validation),'726965d6eae9049e27f8636fe39c0aa96dadb493311dd954f039367584bb2e5b'))
 selected=sample_queries(queries,'validation')
 if old['queries']!=selected:raise ValueError('legacy audit membership/history/target mismatch')
 features=np.load(checked(prepared/'content.npy',original['artifacts']['content.npy']),allow_pickle=False)[1:]
 binding={'consumer':baseline_identity(),'prepared_manifest_sha256':PREPARED_SHA,'prepared_producer':original['binding']['consumer'],'prepared_artifacts':original['artifacts'],'input_records':inputs['input_records'],'query_pins':inputs['query_pins'],'legacy_producer':inputs['legacy_producer'],'catalog_sha256':file_sha256(Path(train_directory)/'catalog.json'),'query_sha256':query_pin['sha256'],'legacy_validation_sha256':file_sha256(legacy_validation),'test_gate':'closed_hash_only'}
 return inputs,features,queries,old,binding

def baseline_identity():
 value=consumer_identity();path=ROOT/'cache/p4c-source-entry.json'
 if path.exists():
  origin=read_json(path)
  if origin['source_fingerprint']!=value['source_fingerprint']:raise ValueError('P4C transport mismatch')
  value['source_transport_origin']=origin
 return value

def synthetic_baseline_inputs():
 import numpy as np
 from .retrieval_baselines import train_graph
 catalog=[10,20,30,40];graph=train_graph(catalog,[(1,10,True),(1,10,True),(1,20,True),(2,20,True),(3,30,True)])
 features=np.zeros((4,64),dtype=np.float32);features[0,0]=1;features[1,1]=1;features[2,0]=-1
 queries=[{'user_id':1,'history':[10,20,10],'target':20,'oov':False},{'user_id':2,'history':[30],'target':99,'oov':True}]
 binding={'consumer':baseline_identity(),'origin':'hand_authored_synthetic_not_formal_experiment','test_gate':'closed'}
 return graph,features,queries,binding

def _train_baseline_graph(inputs,raw_root,check):
 import pyarrow.parquet as pq
 from .retrieval_baselines import train_graph
 def events():
  for rec in inputs['train']['train_files']:
   check()
   for batch in pq.ParquetFile(Path(raw_root)/rec['path']).iter_batches(batch_size=65536,columns=['user_id','item_id','like'],use_threads=False):
    for row in zip(*(batch.column(i).to_pylist() for i in range(3))):yield row
 graph=train_graph(inputs['catalog'],events())
 expected=inputs['train']['audit']
 if sum(graph['popularity'])!=expected['feedback']['like']['true'] or sum(d>0 for d in graph['degrees'])!=expected['like_items'] or len(graph['users'])!=expected['like_users']:raise ValueError('all-like graph reconciliation mismatch')
 return graph

def baseline_pilot(args):
 import time,numpy as np
 from .retrieval_baselines import graph_pilot
 from .baselines import history_centroids,blockwise_topk
 output=Path(args.output).resolve()
 if output.exists():raise FileExistsError('exclusive pilot output required')
 before=resource_snapshot(output.parent);inputs,features,queries,old,binding=load_baseline_inputs(args.train_directory,args.holdout_directory,args.raw_root,args.prepared,args.legacy_validation)
 output.mkdir(parents=True,exist_ok=False);check=lambda:resource_snapshot(output.parent)
 start=time.monotonic();graph=_train_baseline_graph(inputs,args.raw_root,check);fit_seconds=time.monotonic()-start
 stats=graph_pilot(graph,check);rows=queries[:32];start=time.monotonic();centroids,zero=history_centroids([q['history'] for q in rows],graph['catalog'],features)
 blockwise_topk(centroids,features,graph['catalog'],50);content_seconds=time.monotonic()-start
 # Persist only server-side binary incidence, not a model prefix reconstruction.
 proof={'catalog':graph['catalog'],'popularity':graph['popularity'],'users':{str(u):items for u,items in graph['users'].items()}}
 with (output/'incidence.json').open('xb') as stream:stream.write(canonical_bytes(proof))
 value={'schema_version':1,'task_id':'P4C','kind':'baseline_pilot','status':'success','binding':binding,'incidence_sha256':file_sha256(output/'incidence.json'),'graph':stats,'graph_input_scan_seconds':fit_seconds,'content_pilot_queries':len(rows),'content_pilot_seconds':content_seconds,'full_validation_queries':len(queries),'resources_start':before,'resources_end':check()}
 with (output/'pilot-manifest.json').open('xb') as stream:stream.write(canonical_bytes(value))
 return value

def _load_pilot_graph(pilot,binding):
 from .retrieval_data import catalog_map
 path=Path(pilot);value=read_json(path)
 if value.get('kind')!='baseline_pilot' or value.get('status')!='success' or value['binding']!=binding:raise ValueError('pilot binding mismatch')
 proof=read_json(checked(path.parent/'incidence.json',value['incidence_sha256']));catalog=proof['catalog'];catalog_map(catalog);users={int(u):items for u,items in proof['users'].items()};inverted={i:[] for i in catalog}
 for u,items in sorted(users.items()):
  if items!=sorted(set(items)) or not items or not 0<=u<2**32:raise ValueError('invalid pilot incidence')
  for i in items:
   if type(i)is not int or i not in inverted:raise ValueError('pilot catalog mismatch')
   inverted[i].append(u)
 if len(proof['popularity'])!=len(catalog) or any(type(x)is not int or x<0 for x in proof['popularity']):raise ValueError('invalid pilot popularity')
 return {**proof,'users':users,'inverted':inverted,'degrees':[len(inverted[i]) for i in catalog]},value

def run_baselines(args):
 from .retrieval_baselines import item_neighbors,write_run
 from .baselines import sample_queries,metrics
 output=Path(args.output).resolve()
 if output.exists():raise FileExistsError('exclusive baseline output required')
 if args.synthetic:
  graph,features,queries,binding=synthetic_baseline_inputs();return write_run(output,graph,item_neighbors(graph),features,queries,binding)
 before=resource_snapshot(output.parent);inputs,features,queries,old,binding=load_baseline_inputs(args.train_directory,args.holdout_directory,args.raw_root,args.prepared,args.legacy_validation)
 graph,pilot=_load_pilot_graph(args.pilot,binding)
 if graph['catalog']!=inputs['catalog']:raise ValueError('pilot catalog conflict')
 check=lambda:resource_snapshot(output.parent);neighbors=item_neighbors(graph,check=check)
 binding={**binding,'pilot_manifest_sha256':file_sha256(args.pilot),'resources_start':before,'resources_before_evaluation':check()}
 value=write_run(output,graph,neighbors,features,queries,binding,check=check)
 prediction=read_json(output/'predictions.json');selected=sample_queries(queries,'validation');lookup={q['user_id']:i for i,q in enumerate(queries)};regression={}
 for method,key in (('popularity','popularity_predictions'),('content','content_predictions')):
  actual=[prediction[method][lookup[q['user_id']]][:20] for q in selected]
  if actual!=old[key] or metrics(actual,selected,graph['catalog'],(10,20))!=metrics(old[key],old['queries'],graph['catalog'],(10,20)):raise ValueError('legacy validation regression mismatch')
  regression[method]=True
 with (output/'legacy-regression.json').open('xb') as stream:stream.write(canonical_bytes({'membership_history_target_equal':True,'predictions_at20_equal':regression,'metrics_at10_20_equal':regression}))
 value['artifacts']['legacy-regression.json']=file_sha256(output/'legacy-regression.json');value['resources_end']=check()
 # Preserve first manifest, finalize under a new filename; incomplete failures remain inspectable.
 with (output/'baseline-final-manifest.json').open('xb') as stream:stream.write(canonical_bytes(value))
 return value

def verify_baselines(args):
 from .retrieval_baselines import verify_run
 path=Path(args.manifest).resolve();manifest=read_json(path)
 if args.synthetic:
  graph,features,queries,binding=synthetic_baseline_inputs();return verify_run(path.parent,queries,graph['catalog'],binding,graph,features)
 import time
 started=time.monotonic();resources=[resource_snapshot(path.parent)]
 inputs,features,queries,old,binding=load_baseline_inputs(args.train_directory,args.holdout_directory,args.raw_root,args.prepared,args.legacy_validation)
 if path.name!='baseline-final-manifest.json' or manifest['binding']['consumer']!=binding['consumer']:raise ValueError('final baseline manifest required')
 for k,v in binding.items():
  if manifest['binding'].get(k)!=v:raise ValueError('final baseline binding mismatch')
 pilot_path=Path(args.pilot);checked(pilot_path,manifest['binding']['pilot_manifest_sha256']);_load_pilot_graph(pilot_path,binding)
 initial=read_json(path.parent/'baseline-manifest.json')
 if any(manifest[k]!=initial[k] for k in initial if k not in ('artifacts',)):raise ValueError('initial/final manifest mismatch')
 if set(manifest['artifacts'])!=set(initial['artifacts'])|{'legacy-regression.json'} or any(manifest['artifacts'][k]!=v for k,v in initial['artifacts'].items()):raise ValueError('initial/final artifact bindings conflict')
 for name,digest in manifest['artifacts'].items():checked(path.parent/name,digest)
 def check():
  sample=resource_snapshot(path.parent);resources.append(sample);return sample
 actual_graph=_train_baseline_graph(inputs,args.raw_root,check)
 pilot_graph,_=_load_pilot_graph(pilot_path,binding)
 for key in ('catalog','popularity','users','degrees'):
  if canonical_bytes(actual_graph[key])!=canonical_bytes(pilot_graph[key]):raise ValueError('pilot incidence differs from pinned raw train')
 verify_run(path.parent,queries,inputs['catalog'],manifest['binding'],actual_graph,features,check)
 from .baselines import sample_queries
 predictions=read_json(path.parent/'predictions.json');positions={q['user_id']:i for i,q in enumerate(queries)}
 for method,key in (('popularity','popularity_predictions'),('content','content_predictions')):
  if [predictions[method][positions[q['user_id']]][:20] for q in sample_queries(queries,'validation')]!=old[key]:raise ValueError('legacy regression replay mismatch')
 regression=read_json(path.parent/'legacy-regression.json')
 if regression!={'membership_history_target_equal':True,'predictions_at20_equal':{'popularity':True,'content':True},'metrics_at10_20_equal':{'popularity':True,'content':True}}:raise ValueError('legacy proof mismatch')
 check()
 return {**manifest,'verification':{'elapsed_seconds':time.monotonic()-started,'scope':'input audit + raw train rescan + graph rebuild + full three-route replay + metrics/legacy proof','resource_samples':len(resources),'resources_minimum':{key:min(r[key] for r in resources) for key in resources[0]}}}

def main(argv=None):
 parser=argparse.ArgumentParser(description='P4B reusable pipeline; bounded synthetic train only')
 sub=parser.add_subparsers(dest='command',required=True)
 p=sub.add_parser('train');p.add_argument('--synthetic',action='store_true',required=True);p.add_argument('--output',required=True);p.add_argument('--max-steps',type=int,default=20);p.add_argument('--until-step',type=int,default=20);p.add_argument('--resume')
 p=sub.add_parser('verify');p.add_argument('--manifest',required=True)
 for command in ('audit','prepare'):
  p=sub.add_parser(command)
  for name in ('train-directory','holdout-directory','raw-root'):p.add_argument('--'+name,required=True)
  if command=='prepare':p.add_argument('--output',required=True)
 for command in ('baseline-pilot','baselines','verify-baselines'):
  p=sub.add_parser(command)
  p.add_argument('--synthetic',action='store_true');p.add_argument('--split',choices=['validation'],default='validation')
  for name in ('train-directory','holdout-directory','raw-root','prepared','legacy-validation','pilot'):p.add_argument('--'+name)
  if command=='verify-baselines':p.add_argument('--manifest',required=True)
  else:p.add_argument('--output',required=True)
 args=parser.parse_args(argv)
 try:
  if args.command in ('baseline-pilot','baselines','verify-baselines'):
   if not args.synthetic and any(getattr(args,name) is None for name in ('train_directory','holdout_directory','raw_root','prepared','legacy_validation')):raise ValueError('pinned baseline input paths required')
   if not args.synthetic and args.command!='baseline-pilot' and args.pilot is None:raise ValueError('accepted pilot required')
   value={'baseline-pilot':baseline_pilot,'baselines':run_baselines,'verify-baselines':verify_baselines}[args.command](args)
  elif args.command=='train':value=synthetic_train(args.output,args.max_steps,args.until_step,args.resume)
  elif args.command=='audit':
   inputs=audit_inputs(args.train_directory,args.holdout_directory,args.raw_root);value={'kind':'audit','status':'success'}
  elif args.command=='prepare':value=prepare(args.train_directory,args.holdout_directory,args.raw_root,args.output)
  else:
   value=read_json(Path(args.manifest))
   if value.get('kind')=='prepare':
    from .retrieval_data import verify_prepared
    value=verify_prepared(Path(args.manifest).parent)
    if value['binding']['consumer']!=consumer_identity():raise ValueError('consumer mismatch')
    validate_contract(ROOT)
   else:value=verify_train(args.manifest)
  print(canonical_bytes({'task_id':value.get('task_id','P4B'),'kind':value['kind'],'status':'verified' if args.command.startswith('verify') else value['status'],**({'verification':value['verification']} if 'verification' in value else {})}).decode(),end='')
 except (OSError,ValueError,KeyError,TypeError,RuntimeError):
  print('ERROR P4B contract/input/state/output gate failed; preserve failure evidence',file=sys.stderr);return 2
 return 0
if __name__=='__main__':raise SystemExit(main())
