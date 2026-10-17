"""Fresh-seed confirmation operations; all deployment/runtime paths are explicit inputs.

No deployment or checkpoint reuse. Real preflight -> synthetic CUDA rehearsal and
full-shape peaks -> dry -> six unconditional trains, exactly one verifier each.
The model runner rechecks config/input/source/data/init hashes during verification.
"""
import argparse
import copy
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
SEEDS=(42,2024,2025)
ARMS=('u2','u3')
RIDS=tuple(f'p-profile-{arm}-s{seed}' for arm in ARMS for seed in SEEDS)
ENTRY='vkgr.retrieval_profile_run'

def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def new(p,v):
 p=Path(p);temporary=p.with_name('.'+p.name+'.'+str(time.time_ns())+'.tmp')
 try:
  with temporary.open('x') as f:json.dump(v,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
  os.link(temporary,p)
 finally:
  if temporary.exists():temporary.unlink()
def pinned(p,pin):
 from vkgr.retrieval_ablation_run import read_pinned
 return read_pinned(p,pin)
def source():
 from vkgr.retrieval_ablation_run import source_identity
 return {**source_identity(),'ops_sha256':sha(__file__),'config_files':{f'configs/profile/{rid}.json':sha(ROOT/f'configs/profile/{rid}.json') for rid in RIDS}}
def check_source(expected):
 current=source()
 if current!=expected or current['git_dirty'] or not current['git_commit']:raise ValueError('source/commit/config/ops identity mismatch or dirty deployment')
 return current

def registered_configs():
 from vkgr import retrieval_profile_run as run
 out={}
 if {p.name for p in (ROOT/'configs/profile').glob('*.json')}!={rid+'.json' for rid in RIDS}:raise ValueError('exactly six configs required')
 for rid in RIDS:
  path=ROOT/f'configs/profile/{rid}.json';cfg=run.validate_config(load(path))
  if rid!=f"p-profile-{cfg['arm']}-s{cfg['seed']}":raise ValueError('config filename binding mismatch')
  out[rid]=(path,cfg)
 return out

def preflight(inputs,inputs_sha,data_root,profile_manifest,profile_sha):
 from vkgr import retrieval_profile_run as run
 from vkgr.retrieval_seqm1_run import _model as original_model,default_config as original_config
 import torch
 torch.set_num_threads(1);src=source();check_source(src)
 spec=run.fr.validate_inputs(pinned(inputs,inputs_sha),run.FEATURE)
 data,_,binding=run.real_inputs(spec,data_root);data,extra=run.fr.apply_feat02(data,spec,50)
 rows,queries=run.fr._view(data,50);data={**data,'rows':rows,'queries':queries}
 data=run.load_profiles(profile_manifest,profile_sha,data,inputs_sha);rows,queries=data['rows'],data['queries']
 if data.get('synthetic') or (len(rows),len(queries),len(data['catalog']))!=(86785,2520,120415):raise ValueError('formal population mismatch')
 bound={**binding,**extra,'inputs_sha256':inputs_sha};entries={};checks={}
 for rid,(path,cfg) in registered_configs().items():
  model=run._model(data,cfg,'cpu');old=original_model(data,{**original_config(),'seed':cfg['seed']},'cpu');model.eval();old.eval()
  if any(not torch.equal(p,dict(old.named_parameters())[n]) for n,p in model.base.named_parameters()):raise ValueError('shared initialization mismatch')
  batch=rows[:512]+queries;count=0
  with torch.no_grad():
   for start in range(0,len(batch),512):
    part=batch[start:start+512];h,side=run._tensors2(part,data['idmap'],50,cfg['feature'])
    if not torch.equal(model.users(h,side),old.users(h,side[...,:6])):raise ValueError('zero profile forward mismatch')
    changed=[{**r,'target':-999,'target_feedback':63,'future_like':True} for r in part]
    hh,ss=run._tensors2(changed,data['idmap'],50,cfg['feature'])
    if not torch.equal(h,hh) or not torch.equal(side,ss):raise ValueError('target tensor leak')
    count+=len(part)
   for table in model.profile_tables:
    table.weight.copy_(torch.arange(table.num_embeddings,dtype=table.weight.dtype)[:,None].expand_as(table.weight)*.03125)
   for start in range(0,len(batch),512):
    part=batch[start:start+512];h,side=run._tensors2(part,data['idmap'],50,cfg['feature']);hh,ss=run._tensors2([{**r,'target':-999,'target_feedback':63,'future_like':True} for r in part],data['idmap'],50,cfg['feature'])
    if not torch.equal(model.users(h,side),model.users(hh,ss)):raise ValueError('nonzero profile target leakage')
   for table in model.profile_tables:table.weight.zero_()
  history_oracle(model);b=run.binding_for4(data,cfg,rows,queries,model,'cuda',bound)
  entries[rid]={'seed':cfg['seed'],'arm':cfg['arm'],'source_commit':src['git_commit'],'config_rel':path.relative_to(ROOT).as_posix(),'config_sha256':sha(path),'expected_binding_sha256':run.b1._digest(b),'data_hashes':{k:b[k] for k in ('rows_sha256','queries_sha256','catalog_sha256','support_sha256','content_sha256','side_sha256','weeks_sha256','profile_ids_sha256')}}
  checks[rid]={'zero_init_bitwise':True,'shared_parameters_bitwise':True,'nonzero_profile_target_invariance':True,'rows_checked':count,'profile_prepare_binding':data['profile_manifest_sha256'],'test_read':False}
  del model,old
 return {'status':'preflight_ok','engineering':False,'read_at_utc':now(),'source':src,'inputs_sha256':inputs_sha,'profile_manifest':str(profile_manifest),'profile_sha256':profile_sha,'entries':entries,'leakage_checks':checks,'data_root':str(Path(data_root).resolve()),'shape':{'content_rows':len(data['content']),'supported':len(data['support']),'catalog':len(data['catalog']),'queries':len(queries),'history':50,'batch':512},'test_read':False,'existing_checkpoint_read':False}



def history_oracle(model):
 """The arm label fixes the expected mask; test actual earlier hidden states."""
 import torch
 device=next(model.parameters()).device
 h=torch.tensor([[11,12,13,14]],device=device);s=torch.zeros(1,4,6,dtype=torch.long,device=device)
 s[0,:,0]=1;s[0,:,1]=torch.tensor([3,20,40,70],device=device);s[0,:,2]=90;s[0,:,3]=torch.tensor([0,1,2,3],device=device)
 changed=h.clone();changed[0,-1]=99;changed_side=s.clone();changed_side[0,-1,1]=254;model.eval()
 with torch.no_grad():same=torch.equal(model.parts(h,s).z[:,0],model.parts(changed,changed_side).z[:,0])
 if same!=(model.arm=='mean'):raise ValueError('history mask oracle mismatch')
 return True

def small_replay(run,out,device='cuda'):
 data=run.synthetic_data();result={}
 for arm in ARMS:
  cfg={**run.default_config(),'arm':arm,'batch_size':16,'max_epochs':3,'min_epochs':2}
  target=Path(out)/('replay-'+arm);value=run.run_session(target,data,cfg,device)
  result[arm]=run.verify_run(target/'manifest.json',value['manifest_sha256'],data,cfg,device)
  if result[arm]['status']!='verified':raise ValueError('new synthetic artifact verify failed')
 return result

def declared_vram(measured):
 if type(measured) not in (int,float) or not math.isfinite(measured) or measured<=0:raise ValueError('finite positive measured VRAM required')
 return math.ceil(measured*1.15/256)*256

def own_gpu_mib():
 result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
 values=[float(x.split(',')[1]) for x in result.stdout.splitlines() if x.split(',')[0].strip()==str(os.getpid())]
 if len(values)!=1:raise ValueError('unique own GPU PID measurement required')
 return values[0]

def full_shape_data(run,shape,h,s,profile_ids):
 import torch
 n=shape['catalog'];sup=shape['supported'];ncontent=shape['content_rows']
 if ncontent!=n+1 or shape['queries']!=2520:raise ValueError('real one-padding-row shape contract')
 if tuple(h.shape)!=(512,50) or int(h.min())<1 or int(h.max())>=ncontent:raise ValueError('synthetic item index contract')
 proto=run.synthetic_data()['queries'][0];queries=[]
 for j in range(2520):
  i=j%512;queries.append({**proto,'history':(h[i]-1).tolist(),'side':s[i,:,:3].tolist(),'history_positions':[[int(s[i,k,3]),k] for k in range(50)],'profile_ids':profile_ids[i].tolist(),'target':j%sup,'oov':False})
 return {'synthetic':True,'catalog':list(range(n)),'idmap':{i:i+1 for i in range(n)},'content':torch.zeros(ncontent,64),'supported_indices':list(range(1,sup+1)),'support':set(range(sup)),'queries':queries,'rows':queries,'profile_manifest_sha256':'synthetic_full_shape','profile_fit_sha256':'synthetic_full_shape'}


def full_shape(run,arm,shape):
 import gc
 import torch
 from php01_vram_smoke import synthetic_batch
 from vkgr.retrieval_negatives import masked_ce_logq
 gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
 context=own_gpu_mib();reserved0=torch.cuda.memory_reserved()/2**20
 n=shape['catalog'];sup=shape['supported'];ncontent=shape['content_rows']
 if ncontent!=n+1 or shape['queries']!=2520:raise ValueError('real shape contract')
 g=torch.Generator().manual_seed(7);h,s=synthetic_batch(g,512,sup,ncontent)
 h=torch.clamp(h,min=1,max=ncontent-1)
 profile_ids=torch.stack((torch.arange(512)%17,torch.arange(512)%17,torch.arange(512)%10),dim=-1)
 packed=torch.cat((s,profile_ids[:,None,:].expand(-1,50,-1)),dim=-1)
 data=full_shape_data(run,shape,h,s,profile_ids)
 cfg=run.default_config(arm);m=run._model(data,cfg,'cuda');opt=run.make_optimizer(m,cfg);history_oracle(m)
 h=h.cuda();packed=packed.cuda();targets=(torch.randperm(sup,generator=g)[:512]+1).cuda();labels=torch.arange(512,device='cuda');mask=torch.ones(512,512,dtype=torch.bool,device='cuda');logq=torch.full((512,),math.log(1/sup),device='cuda')
 m.eval()
 with torch.no_grad():zero=torch.equal(m.users(h,packed),m.base.users(h,packed[...,:6]))
 if not zero:raise ValueError('CUDA zero start mismatch')
 samples=[]
 def observe(stage):
  torch.cuda.synchronize();samples.append({'stage':stage,'own_process_mib':own_gpu_mib(),'max_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'max_reserved_mib':torch.cuda.max_memory_reserved()/2**20})
 for step in range(3):
  m.train();opt.zero_grad(set_to_none=True);loss,_=masked_ce_logq(m.users(h,packed),m.items(targets),labels,mask,.1,logq)
  if not torch.isfinite(loss):raise ValueError('nonfinite CUDA loss')
  loss.backward()
  if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in m.parameters()):raise ValueError('nonfinite CUDA gradient')
  torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step();observe('train-'+str(step+1))
 result,_=run.evaluate_model(m,data,queries,cfg);observe('full-catalog-evaluation')
 if len(result['predictions'])!=2520 or any(len(p)!=50 or len(set(p))!=50 for p in result['predictions']):raise ValueError('full shape exact top50')
 run.ablation_g0(m,data,queries,cfg,1,result['metrics']['all']['recall@50']);observe('full-shape-g0')
 measured=max(max(x['own_process_mib'],x['max_reserved_mib']+max(0,context-reserved0)) for x in samples)
 return {'full_shape':{'batch':512,'history':50,'catalog':n,'content_rows':ncontent,'supported':sup,'queries':2520},'zero_start':zero,'finite_gradients':True,'history_oracle':True,'samples':samples,'measured_vram_mib':measured,'declared_vram_mib':declared_vram(measured)}



def cuda_check(pf,out):
 from vkgr import retrieval_profile_run as run
 import torch
 check_source(pf['source'])
 if pf['status']!='preflight_ok' or pf['engineering'] is not False:raise ValueError('formal real preflight required')
 if not torch.cuda.is_available():raise ValueError('actual CUDA required')
 torch.set_num_threads(1);out=Path(out);out.mkdir(parents=True,exist_ok=False);replay=small_replay(run,out)
 arms={arm:{**full_shape(run,arm,pf['shape']),'synthetic_verify':replay[arm]} for arm in ARMS}
 return {'status':'cuda_engineering_verified','source':source(),'preflight_sha256':digest(pf),'read_at_utc':now(),'arms':arms,'test_read':False,'existing_checkpoint_read':False,'real_training_rows_used':False}

def make_plan(pf,cu,request):
 if pf.get('status')!='preflight_ok' or pf.get('engineering') is not False or cu.get('status')!='cuda_engineering_verified':raise ValueError('real preflight/CUDA proofs required')
 if not pf['source'].get('git_commit') or pf['source'].get('git_dirty') is not False or cu['source']!=pf['source'] or cu['preflight_sha256']!=digest(pf):raise ValueError('proof source/preflight identity mismatch')
 if any(p.get('test_read') is not False or p.get('existing_checkpoint_read') is not False for p in (pf,cu)) or cu.get('real_training_rows_used') is not False:raise ValueError('forbidden evidence scope')
 if set(pf['entries'])!=set(RIDS) or set(cu['arms'])!=set(ARMS):raise ValueError('all six configs and both CUDA arms required')
 for arm in ARMS:
  proof=cu['arms'][arm]
  if proof['full_shape']!=pf['shape'] or proof['synthetic_verify'].get('status')!='verified' or any(proof.get(k) is not True for k in ('zero_start','finite_gradients','history_oracle')):raise ValueError('incomplete CUDA proof')
  if proof['declared_vram_mib']!=declared_vram(proof['measured_vram_mib']):raise ValueError('VRAM declaration mismatch')
 if not Path(request['data_root']).is_absolute() or request['data_root']!=pf['data_root']:raise ValueError('private data channel differs from real preflight')
 root=Path(request['root']);runs=Path(request['runs_root']);inputs=request['inputs'];entries=[];data_hashes=None
 for rid in RIDS:
  e=pf['entries'][rid];arm=e['arm'];seed=e['seed']
  if rid!=f'p-profile-{arm}-s{seed}' or arm not in ARMS or type(seed) is not int or seed not in SEEDS or e['config_rel']!=f'configs/profile/{rid}.json':raise ValueError('config/arm/seed binding mismatch')
  if data_hashes is None:data_hashes=e['data_hashes']
  if e['data_hashes']!=data_hashes:raise ValueError('paired data identities differ')
  cfg=root/e['config_rel'];out=runs/rid/'out';j=copy.deepcopy(request['job_template'])
  j.update(run_id=rid,seed=seed,cwd=str(root),config=str(cfg),vram_mib=cu['arms'][arm]['declared_vram_mib'],pinned_manifest=request['pinned_manifest'],pinned_files=[])
  j['argv']=[request['python'],'-B','-m',ENTRY,'train','--inputs',inputs,'--inputs-sha',pf['inputs_sha256'],'--profile-manifest',pf['profile_manifest'],'--profile-sha',pf['profile_sha256'],'--config',str(cfg),'--config-sha',e['config_sha256'],'--output',str(out)]
  j['env']=[x for x in j.get('env',[]) if not x.startswith('VKGR_DATA_ROOT=')]+['VKGR_DATA_ROOT='+request['data_root']]
  j['inputs']=list(dict.fromkeys([*j.get('inputs',[]),inputs,str(cfg),pf['profile_manifest']]));entries.append({'job':j,'arm':arm,'expected':e})
 created=request['created_epoch']
 if type(created) not in (int,float) or not math.isfinite(created):raise ValueError('finite created epoch required')
 return {'schema':'profile-plan-v1','source':pf['source'],'created_epoch':created,'deadline_epoch':created+48*3600,'poll_seconds':900,'entries':entries,'preflight_sha256':digest(pf),'engineering_sha256':digest(cu),'runs_root':str(runs)}

def derive_verify(job,manifest_sha):
 if len(manifest_sha)!=64 or any(c not in '0123456789abcdef' for c in manifest_sha):raise ValueError('manifest sha256 required')
 v=copy.deepcopy(job);a=v['argv'];idx=a.index('--output');out=Path(a[idx+1]);del a[idx:idx+2];a[a.index(ENTRY)+1]='verify';a.extend(['--manifest',str(out/'manifest.json'),'--manifest-sha',manifest_sha]);v.update(run_id=job['run_id']+'-verify',disk_gib=1,inputs=[*job['inputs'],str(out/'manifest.json')]);return v

def where(q,rid):return {s:sorted((Path(q)/s).glob('*-'+rid+'.json')) for s in ('pending','claimed','rejected')}
def exists(q,runs,rid):return (Path(runs)/rid).exists() or any(where(q,rid).values())
def publish(q,runs,job):
 q=Path(q)
 with (q/'.profile-publish.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if exists(q,runs,job['run_id']):raise FileExistsError(job['run_id'])
  target=q/'pending'/f"{time.time_ns():020d}-{job['run_id']}.json";temporary=target.with_name('.'+target.name+'.tmp');new(temporary,job)
  try:os.link(temporary,target)
  finally:temporary.unlink()
  return target.name

def next_action(train,verify,verify_exists,rejected,verified=False):
 if rejected or train not in (None,'running','succeeded') or verify not in (None,'running','succeeded'):return 'ALERT'
 if train=='succeeded':
  if verify=='succeeded':return 'verified' if verified else 'ALERT'
  return 'wait' if verify_exists else 'verify'
 return 'wait'

def claimed(q,rid):
 files=where(q,rid)['claimed']
 if len(files)!=1:raise ValueError('unique claimed job required: '+rid)
 return load(files[0])

def inspect_entry(e,q,runs):
 from vkgr import retrieval_profile_run as run
 rid=e['job']['run_id'];runs=Path(runs);tp=runs/rid/'terminal.json';vp=runs/(rid+'-verify')/'terminal.json';t=load(tp)['status'] if tp.exists() else None;v=load(vp)['status'] if vp.exists() else None
 rejected=bool(where(q,rid)['rejected'] or where(q,rid+'-verify')['rejected']);verified=False;pin=None
 if t=='succeeded':
  if any(claimed(q,rid).get(k)!=v for k,v in e['job'].items()):raise ValueError('claimed train differs from frozen plan')
  mp=runs/rid/'out/manifest.json';m=load(mp);pin=sha(mp)
  if m['status']!='complete' or m['schema']!=run.MANIFEST_SCHEMA or m['kind']!=run.KIND or m['config']!=load(e['job']['config']) or run.b1._digest(m['binding'])!=e['expected']['expected_binding_sha256']:raise ValueError('manifest/config/source/data/init binding mismatch')
  if v=='succeeded':
   if any(claimed(q,rid+'-verify').get(k)!=v for k,v in derive_verify(e['job'],pin).items()):raise ValueError('claimed verifier differs from frozen plan')
   records=[]
   for line in (runs/(rid+'-verify')/'logs/stdout.log').read_text().splitlines():
    try:records.append(json.loads(line))
    except ValueError:pass
   verified=any(isinstance(x,dict) and x.get('status')=='verified' and x.get('run_id')==rid+'-verify' and x.get('train_run_id')==rid and x.get('manifest_sha256')==pin for x in records)
 action=next_action(t,v,exists(q,runs,rid+'-verify'),rejected,verified)
 result={'run_id':rid,'verify_run_id':rid+'-verify','train_status':t,'verify_status':v,'action':action,'verified':verified,'manifest_sha256':pin,'source_commit':e['expected'].get('source_commit'),'config_sha256':e['expected']['config_sha256'],'expected_binding_sha256':e['expected']['expected_binding_sha256'],'data_hashes':e['expected']['data_hashes']}
 if verified:
  metrics=m['selection']['best_metric'];result.update(best_epoch=m['best']['epoch'],stop_epoch=m['selection']['epoch'],R50=metrics[0],N50=metrics[1],hits=round(metrics[0]*2520),verify_stdout_sha256=sha(runs/(rid+'-verify')/'logs/stdout.log'))
 return result

def prepare(job,launcher_dir):
 sys.path.insert(0,str(launcher_dir));from vkgr_ops import launcher
 result=launcher.prepare(job)
 if len(json.dumps({'spec':job,'prepared':result},allow_nan=False).encode())>=48*1024-2048:raise ValueError('launcher metadata exceeds safe budget')
 return result

def dry(plan,queue,launcher_dir):
 check_source(plan['source'])
 if len(plan['entries'])!=6 or {e['job']['run_id'] for e in plan['entries']}!=set(RIDS):raise ValueError('six-run plan required')
 for e in plan['entries']:
  job=e['job']
  if sha(job['config'])!=e['expected']['config_sha256']:raise ValueError('config drift')
  for rid in (job['run_id'],job['run_id']+'-verify'):
   if exists(queue,plan['runs_root'],rid):raise FileExistsError(rid)
  prepare(job,launcher_dir)
 return {'status':'dry_ok','plan_sha256':digest(plan),'trains':6,'unique_verifiers':6}

def bound_published(q,runs,job):
 """A publication exception can occur after the queue link became visible."""
 records=[load(p) for files in where(q,job['run_id']).values() for p in files]
 return len(records)==1 and all(records[0].get(k)==v for k,v in job.items())

def publish_intents(plan,q,launcher_dir,state):
 """Only the detached driver calls this; every intent has exactly one attempt."""
 results=[];stopped=False
 for index,entry in enumerate(plan['entries']):
  job=entry['job'];result={'run_id':job['run_id'],'job_sha256':digest(job),'status':'not_attempted'}
  if not stopped:
   new(state/f'publication-intent-{index:02d}.json',{**result,'status':'attempting','utc':now()})
   try:
    prepare(job,launcher_dir);name=publish(q,plan['runs_root'],job);result.update(status='published',queue_file=name)
   except Exception as exc:
    stopped=True;observed=bound_published(q,plan['runs_root'],job)
    result.update(status='published' if observed else 'failed',error=type(exc).__name__+': '+str(exc),queue_visible=observed)
    new(state/'publication-ALERT.json',{'status':'ALERT','utc':now(),'failed_intent':result,'remaining_intents':'not attempted; no automatic retry'})
  new(state/f'publication-result-{index:02d}.json',result);results.append(result)
 new(state/'publication-final.json',{'utc':now(),'status':'ALERT' if stopped else 'published_not_accepted','runs':results})
 return {r['run_id']:r for r in results},stopped

def ready_handshake(plan,state):
 """Ready/release protects the pre-publication window; this is not job polling."""
 if os.getsid(os.getpid())!=os.getpid():raise ValueError('driver must own a detached session before publication')
 ready={'pid':os.getpid(),'ppid':os.getppid(),'launcher_pid':os.environ.get('VKGR_PROFILE_LAUNCHER_PID'),'plan_sha256':digest(plan),'deadline_epoch':plan['deadline_epoch'],'utc':now()};new(state/'driver-ready.json',ready)
 release=state/'publish-release.json';until=min(plan['deadline_epoch'],time.time()+30)
 while time.time()<until:
  if release.exists():
   value=load(release)
   if value!={'plan_sha256':digest(plan),'ready_sha256':sha(state/'driver-ready.json'),'driver_pid':os.getpid()}:raise ValueError('controller ready confirmation mismatch')
   return
  time.sleep(.05)
 raise TimeoutError('controller did not confirm ready; zero intents published')

def driver(plan,q,launcher_dir,state):
 launcher_pid=os.getpid()
 if os.fork():return
 os.setsid()
 until=time.time()+30
 while os.getppid()!=1 and time.time()<until:time.sleep(.05)
 if os.getppid()!=1:raise RuntimeError('driver did not become PPID1')
 state=Path(state);state.mkdir(parents=True,exist_ok=True)
 with (state/'driver.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);new(state/'driver-pid.json',{'pid':os.getpid(),'ppid':os.getppid(),'deadline_epoch':plan['deadline_epoch'],'poll_seconds':900})
  try:
   os.environ['VKGR_PROFILE_LAUNCHER_PID']=str(launcher_pid)
   check_source(plan['source']);ready_handshake(plan,state);check_source(plan['source'])
  except Exception as exc:
   new(state/'summary-final.json',{'utc':now(),'status':'ALERT','reason':'source/startup gate: '+type(exc).__name__+': '+str(exc),'published_intents':0});return
  publications,publication_failed=publish_intents(plan,q,launcher_dir,state)
  latched={}
  while time.time()<plan['deadline_epoch']:
   try:check_source(plan['source'])
   except Exception as exc:
    new(state/'summary-final.json',{'utc':now(),'status':'ALERT','reason':'source gate: '+type(exc).__name__+': '+str(exc)});return
   states=[]
   for entry in plan['entries']:
    rid=entry['job']['run_id'];publication=publications[rid]
    if rid in latched:states.append(latched[rid]);continue
    if publication['status']!='published':
     states.append({'run_id':rid,'action':'ALERT','verified':False,'publication_status':publication['status'],'reason':'unpublished intent; no automatic retry'});continue
    try:
     s=inspect_entry(entry,q,plan['runs_root'])
     if s['action']=='verify':
      job=derive_verify(entry['job'],s['manifest_sha256']);new(state/('verify-intent-'+rid+'.json'),{'status':'attempting','manifest_sha256':s['manifest_sha256']});prepare(job,launcher_dir);publish(q,plan['runs_root'],job);s=inspect_entry(entry,q,plan['runs_root'])
    except Exception as exc:s={'run_id':rid,'action':'ALERT','verified':False,'error':type(exc).__name__+': '+str(exc)}
    states.append(s)
    if s['action'] in ('verified','ALERT'):latched[rid]=s
   done=all(s['action'] in ('verified','ALERT') for s in states)
   status='verified' if all(s['verified'] for s in states) and not publication_failed else 'ALERT' if done else 'running'
   new(state/f'summary-{time.time_ns()}.json',{'utc':now(),'status':status,'runs':states,'publication_alert':publication_failed})
   if done:new(state/'summary-final.json',{'utc':now(),'status':status,'runs':states,'publication_alert':publication_failed});return
   time.sleep(min(900,max(0,plan['deadline_epoch']-time.time())))
  new(state/'summary-final.json',{'utc':now(),'status':'ALERT','reason':'original 48h deadline','publication_alert':publication_failed})

def start_submission(plan,plan_path,q,launcher_dir,state):
 """Controller persists the whole plan, starts the owner, confirms ready, releases."""
 state=Path(state);state.mkdir(parents=True,exist_ok=False);sealed=state/'submitted-plan.json';new(sealed,plan)
 proc=None
 try:
  with (state/'driver-launcher.log').open('xb') as log:
   proc=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'driver','--plan',str(sealed.resolve()),'--plan-sha',sha(sealed),'--queue',str(q),'--launcher-dir',str(launcher_dir),'--state-dir',str(state)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=os.environ.copy(),cwd=ROOT)
  until=min(plan['deadline_epoch'],time.time()+30);ready_path=state/'driver-ready.json'
  while time.time()<until:
   if proc.poll() not in (None,0):raise RuntimeError('detached driver launcher failed before ready')
   if ready_path.exists():
    ready=load(ready_path)
    if ready['launcher_pid']!=str(proc.pid) or ready['ppid']!=1 or ready['plan_sha256']!=digest(plan) or os.getsid(ready['pid'])!=ready['pid']:raise ValueError('detached driver ready identity mismatch')
    new(state/'publish-release.json',{'plan_sha256':digest(plan),'ready_sha256':sha(ready_path),'driver_pid':ready['pid']})
    return {'status':'driver_released_not_accepted','driver_launcher_pid':proc.pid,'driver_pid':ready['pid'],'train_intents':len(plan['entries']),'plan_sha256':digest(plan)}
   time.sleep(.05)
  raise TimeoutError('driver readiness deadline; zero controller publications')
 except Exception as exc:
  alert={'utc':now(),'status':'ALERT','reason':'controller startup: '+type(exc).__name__+': '+str(exc),'controller_publications':0}
  new(state/'controller-ALERT.json',alert)
  if proc is None:new(state/'summary-final.json',alert)
  raise

def _main(argv=None):
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
 pf=sub.add_parser('preflight');pf.add_argument('--inputs',required=True);pf.add_argument('--inputs-sha',required=True);pf.add_argument('--data-root',required=True);pf.add_argument('--output',required=True);pf.add_argument('--profile-manifest',required=True);pf.add_argument('--profile-sha',required=True)
 cu=sub.add_parser('cuda-check');cu.add_argument('--preflight',required=True);cu.add_argument('--preflight-sha',required=True);cu.add_argument('--output',required=True)
 pl=sub.add_parser('plan');pl.add_argument('--preflight',required=True);pl.add_argument('--preflight-sha',required=True);pl.add_argument('--engineering',required=True);pl.add_argument('--engineering-sha',required=True);pl.add_argument('--request',required=True);pl.add_argument('--output',required=True)
 for cmd in ('dry','submit','driver'):
  x=sub.add_parser(cmd);x.add_argument('--plan',required=True);x.add_argument('--plan-sha',required=True);x.add_argument('--queue',required=True);x.add_argument('--launcher-dir',required=True);x.add_argument('--state-dir',required=cmd!='dry')
 a=p.parse_args(argv)
 if a.command=='preflight':value=preflight(a.inputs,a.inputs_sha,a.data_root,a.profile_manifest,a.profile_sha);new(a.output,value)
 elif a.command=='cuda-check':value=cuda_check(pinned(a.preflight,a.preflight_sha),a.output);new(Path(a.output)/'engineering.json',value)
 elif a.command=='plan':
  pf=pinned(a.preflight,a.preflight_sha);check_source(pf['source']);request=load(a.request)
  if Path(request['root']).resolve()!=ROOT.resolve() or sha(request['inputs'])!=pf['inputs_sha256']:raise ValueError('request deployment/input mismatch')
  value=make_plan(pf,pinned(a.engineering,a.engineering_sha),request);new(a.output,value)
 else:
  plan=pinned(a.plan,a.plan_sha)
  if a.command=='driver':driver(plan,a.queue,a.launcher_dir,a.state_dir);return 0
  value=dry(plan,a.queue,a.launcher_dir)
  if a.command=='submit':
   if time.time()>=plan['deadline_epoch']:raise ValueError('original deadline expired')
   value=start_submission(plan,a.plan,a.queue,a.launcher_dir,a.state_dir)
 print(json.dumps(value,sort_keys=True,allow_nan=False));return 0
def main(argv=None):
 from vkgr.profile_io import input_guard
 args=list(sys.argv[1:] if argv is None else argv);owner=None
 if args and args[0]=='cuda-check' and '--output' in args:owner=Path(args[args.index('--output')+1])
 with input_guard(owner):return _main(argv)

if __name__=='__main__':raise SystemExit(main())
