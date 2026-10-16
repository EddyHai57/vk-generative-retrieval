"""SID-FIX S0 fixed-SHA deployment and finite one-CPU queue admission.
Only new own paths may be written. Default stdin-source mode is read-only.
"""
import datetime,fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
R=Path('/workspace/vkgr-infra01-root');Q=R/'queue'
DEP=Path('/workspace/vkgr-sidfix-s0-code-v5');OPS=Path('/workspace/vkgr-sidfix-s0-ops-v5')
SRC=Path('/workspace/vkgr-b1-src');PY='/workspace/vkgr-p6-20261006t2050/code-v1/.venv/bin/python'
MODES={'first_batch':'sid-fix-s0-first-v1','a1_b1024':'sid-fix-s0-a1b1024-v1'}
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def new(p,v):
 with Path(p).open('x') as f:json.dump(v,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
def where(q,rid):
 out={k:sorted(p.name for p in (q/k).glob('*-'+rid+'.json')) for k in ('pending','claimed','rejected')};out['reasons']={}
 for name in out['rejected']:
  for p in ((q/'rejected'/name).with_suffix('.reason'),q/'rejected'/(name+'.reason')):
   if p.exists():out['reasons'][p.name]=p.read_text()[:4000]
 return out
def exists(q,runs,rid):
 v=where(q,rid);return (runs/rid).exists() or any(v[k] for k in ('pending','claimed','rejected'))
def publish(q,runs,spec):
 with (q/'.sidfix-s0-publish.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX)
  if exists(q,runs,spec['run_id']):raise FileExistsError(spec['run_id'])
  name=f"{time.time_ns():020d}-{spec['run_id']}.json";tmp=q/'pending'/('.'+name+'.tmp')
  new(tmp,{'schema':'vkgr-queue-job-v1',**spec});os.link(tmp,q/'pending'/name);return name

def metadata_guard(spec,prepared):
 size=len(json.dumps({'spec':spec,'prepared':prepared},sort_keys=True,allow_nan=False).encode())
 if size>48*1024:raise ValueError('launch metadata >48KiB')
 return prepared or True

def identity():
 meta=load(OPS/'deployment.json')
 if subprocess.check_output(['git','-C',str(DEP),'rev-parse','HEAD'],text=True).strip()!=meta['commit']:raise ValueError('deployment commit changed')
 for rel,pin in meta['files'].items():
  if sha(DEP/rel)!=pin:raise ValueError('source changed: '+rel)
 return meta

def runner():
 identity();sys.dont_write_bytecode=True;sys.path.insert(0,str(DEP/'analysis'))
 import sid_diag
 return sid_diag

def inventory():
 return {'read_at_utc':now(),'queue':{rid:where(Q,rid) for rid in MODES.values()},'existing_runs':[rid for rid in MODES.values() if (R/'runs'/rid).exists()],
         'deployment_exists':DEP.exists(),'ops_exists':OPS.exists(),'checkpoint_read':False,'test_read':False}

def deploy(req):
 if DEP.exists() or OPS.exists():raise FileExistsError('new deployment roots already exist')
 if any(exists(Q,R/'runs',rid) for rid in MODES.values()):raise FileExistsError('existing diagnostic run')
 if not req['branch'].startswith('codex/'):raise ValueError('working branch required')
 subprocess.run(['git','-C',str(SRC),'fetch','-q','origin',req['branch']],check=True)
 actual=subprocess.check_output(['git','-C',str(SRC),'rev-parse','origin/'+req['branch']],text=True).strip()
 if actual!=req['commit']:raise ValueError('remote commit mismatch')
 subprocess.run(['git','-C',str(SRC),'worktree','add','-q','--detach',str(DEP),actual],check=True)
 for rel,pin in req['files'].items():
  if Path(rel).is_absolute() or '..' in Path(rel).parts or sha(DEP/rel)!=pin:raise ValueError('deployment hash mismatch')
 OPS.mkdir();new(OPS/'deployment.json',{'commit':actual,'files':req['files'],'created_at_utc':now()})
 return {'status':'deployed','commit':actual,'files_checked':len(req['files']),'read_at_utc':now()}

def preflight(req):
 mod=runner();mode=req.get('mode','first_batch')
 if mode not in MODES:raise ValueError('mode')
 plan=req.get('plan')
 if plan is None:raise ValueError('explicit metadata-discovered plan required')
 if plan['mode']!=mode:raise ValueError('plan mode mismatch')
 result=mod.collect_inventory(plan)
 # Missing historical receipts remain explicitly UNKNOWN; runner never reads their predictions.
 if mode=='a1_b1024' and result.get('unaccepted_runs'):raise ValueError('A1 confirm requires all three raw verified')
 return {'status':'preflight_ok','read_at_utc':now(),'plan':plan,'inventory':result}

def seal(req):
 runner();plan=req['plan'];mode=plan['mode']
 if mode not in MODES:raise ValueError('mode')
 # Repeat only metadata guards, no large-file scan or diagnostic work in mux.
 result=preflight({'mode':mode,'plan':plan})
 pp=OPS/(mode+'-plan.json');pf=OPS/(mode+'-preflight.json')
 if pp.exists() or pf.exists():raise FileExistsError('plan/preflight already sealed')
 new(pp,plan);new(pf,{'status':'preflight_ok','plan_sha256':sha(pp),'source_commit':identity()['commit'],'inventory':result['inventory'],'created_at_utc':now()})
 return {'status':'sealed','plan_sha256':sha(pp),'read_at_utc':now()}

def job(plan,plan_sha):
 mode=plan['mode']
 if mode not in MODES:raise ValueError('unregistered mode')
 rid=MODES[mode];pp=OPS/(mode+'-plan.json')
 return {'run_id':rid,'kind':'ops','cwd':str(DEP),'config':str(pp),'seed':0,
  'inputs':[str(pp),str(OPS/(mode+'-preflight.json')),str(OPS/'deployment.json')],
  'pinned_manifest':None,'pinned_files':[],'env':['OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1'],
  'cpu_cores':1,'ram_mib':12288,'vram_mib':0,'disk_gib':1,'wall_hours':4.,'stop_mode':'signal','no_admission':False,
  'argv':[PY,'-B','analysis/sid_diag.py','--plan',str(pp),'--plan-sha256',plan_sha,'--output',str(R/'runs'/rid/'out'/'diagnostic')]}

def prepared(req):
 identity();mode=req.get('mode','first_batch')
 if mode not in MODES:raise ValueError('mode')
 pp=OPS/(mode+'-plan.json');pf=load(OPS/(mode+'-preflight.json'))
 if sha(pp)!=pf['plan_sha256'] or pf['source_commit']!=identity()['commit']:raise ValueError('sealed binding drift')
 plan=load(pp);spec=job(plan,sha(pp))
 if exists(Q,R/'runs',spec['run_id']):raise FileExistsError(spec['run_id'])
 sys.path.insert(0,'/workspace/vkgr-infra03-ops/ops');from vkgr_ops import launcher
 result=launcher.prepare(spec);metadata_guard(spec,result)
 return spec,result

def status():
 result=inventory();result['runs']={}
 if DEP.exists():
  identity();plans=[mode for mode in MODES if (OPS/(mode+'-plan.json')).exists()]
  if not plans:result['discovery']=runner().collect_inventory()
  else:result['preflight']={mode:preflight({'mode':mode,'plan':load(OPS/(mode+'-plan.json'))}) for mode in plans}
 for mode,rid in MODES.items():
  rd=R/'runs'/rid;v={}
  for name in ('terminal.json','owner.json'):
   p=rd/name
   if p.exists():v[name]=load(p);v[name+'_sha256']=sha(p)
  p=rd/'out/diagnostic/result.json'
  if p.exists():v['result']=load(p);v['result_sha256']=sha(p)
  result['runs'][rid]=v
 return result

def request():
 if len(sys.argv)>1:return load(sys.argv[1])
 if sys.argv[0]=='-c':return json.load(sys.stdin)
 return {}

def main():
 req=request();action=req.get('action','status')
 if action=='deploy':out=deploy(req)
 elif action=='discover':
  out=runner().collect_inventory();out['read_at_utc']=now()
 elif action=='preflight':out=preflight(req)
 elif action=='seal':out=seal(req)
 elif action=='dry':
  spec,pf=prepared(req);out={'status':'dry_ok','job':spec,'prepared':pf,'read_at_utc':now()}
 elif action=='submit':
  spec,pf=prepared(req);name=publish(Q,R/'runs',spec);out={'action':'written','queue_name':name,'run_id':spec['run_id'],'read_at_utc':now()}
 elif action=='status':out=status()
 else:raise ValueError('unknown action')
 print(json.dumps(out,sort_keys=True,allow_nan=False))
if __name__=='__main__':main()
