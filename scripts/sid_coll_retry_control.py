"""Exclusive v2 deployment/seal/start for Eddy's single repaired PF authorization.
No refit/export/checkpoint loading; original failures and deadline remain intact.
"""
import datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
R=Path('/workspace/vkgr-infra01-root');OLD=Path('/workspace/vkgr-sid-coll-g-v1');OLDOPS=Path('/workspace/vkgr-sid-coll-ops-v1')
DEP=Path('/workspace/vkgr-sid-coll-g-v2');OPS=Path('/workspace/vkgr-sid-coll-ops-v2');SRC=Path('/workspace/vkgr-b1-src')
PY='/workspace/vkgr-p6-20261006t2050/code-v1/.venv/bin/python';OLDPIN='0f5c1029281ec7c65266a4c5ea802425e0ccede8298bb931b020d3ec48c7161a'
REUSED=('input_preflight','input','input_verify','sid_fit','sid_verify')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_bytes())
def new(p,v):
 with Path(p).open('x') as f:json.dump(v,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
def old_ops():
 sys.path.insert(0,'/workspace/vkgr-sid-coll-observer-v2/scripts')
 import sid_coll_ops
 return sid_coll_ops

def scientific_equal(old,newfiles):
 names=[n for n in old if n.startswith('src/') or n.startswith('configs/sid_coll/g-coll-') or n.startswith('configs/sid_coll/fit-coll-') or n=='scripts/sid_coll_job_entry.py']
 if not names or any(newfiles.get(n)!=old[n] for n in names):raise ValueError('scientific or frozen config bytes changed')
 return len(names)

def validate_retry_ids(old,newplan):
 if any(old[k]!=newplan[k] for k in ('created_epoch','deadline_epoch')) or set(old['branches'])!=set(newplan['branches']):raise ValueError('budget or branch changed')
 for label,branch in old['branches'].items():
  stages=newplan['branches'][label]['stages']
  if set(branch['stages'])!=set(stages):raise ValueError('arms/stages changed')
  for name,stage in branch['stages'].items():
   if name in REUSED:
    if stage!=stages[name]:raise ValueError('successful stage changed')
   else:
    rid=stage['job']['run_id'];expected=rid[:-3]+'-v2' if name=='preflight' else rid
    if stages[name]['job']['run_id']!=expected:raise ValueError('outside approved retry identity')

def inventory():
 op=old_ops();plan=op.load_plan(OLDOPS/'pipeline-plan.json',OLDPIN);reused={};ids=[];failed={}
 for branch in plan['branches'].values():
  accepted={}
  for name,stage in branch['stages'].items():
   rid=stage['job']['run_id']
   if name in REUSED:
    state,proof=op.inspect_stage(plan,stage,accepted)
    if not state['accepted']:raise ValueError('old input/SID not directly accepted')
    accepted[name]=proof;reused[rid]={'state':'ACCEPTED','receipt_ref':{'path':state['proof_path'],'sha256':state['proof_sha256']},'artifacts':proof.get('proofs',{})}
   else:
    if name=='preflight':
     failed[rid]=load(R/'runs'/rid/'terminal.json')
     if failed[rid]['status']!='failed':raise ValueError('not the approved failed PF')
     rid=rid[:-3]+'-v2'
    ids.append(rid)
 queue={rid:{k:[p.name for p in (R/'queue'/k).glob('*-'+rid+'.json')] for k in ('pending','claimed','rejected')} for rid in ids}
 pid=load(OLDOPS/'controller/driver-pid-v2.json')['pid']
 return {'read_at_utc':now(),'runs':reused,'failed_preserved':failed,'new_ids':ids,'queue':queue,'existing_new_runs':[r for r in ids if (R/'runs'/r).exists()],'old_driver_alive':(Path('/proc')/str(pid)).exists(),'new_dep_exists':DEP.exists(),'new_ops_exists':OPS.exists(),'original_deadline_epoch':plan['deadline_epoch'],'test_read':False,'checkpoint_read':False}

def identity():
 d=load(OPS/'deployment.json')
 if subprocess.check_output(['git','-C',str(DEP),'rev-parse','HEAD'],text=True).strip()!=d['commit']:raise ValueError('new source commit drift')
 for n,pin in d['files'].items():
  if sha(DEP/n)!=pin:raise ValueError('new deployed source changed')
 return d

def deploy(req):
 v=inventory()
 if v['old_driver_alive'] or v['new_dep_exists'] or v['new_ops_exists'] or v['existing_new_runs'] or any(any(q.values()) for q in v['queue'].values()):raise ValueError('recovery identity already occupied')
 scientific_equal(load(OLDOPS/'deployment.json')['files'],req['files'])
 subprocess.run(['git','-C',str(SRC),'fetch','-q','origin','codex/sid-coll-s1'],check=True)
 if subprocess.check_output(['git','-C',str(SRC),'rev-parse','origin/codex/sid-coll-s1'],text=True).strip()!=req['commit']:raise ValueError('pushed SHA mismatch')
 subprocess.run(['git','-C',str(SRC),'worktree','add','-q','--detach',str(DEP),req['commit']],check=True)
 for n,pin in req['files'].items():
  if Path(n).is_absolute() or '..' in Path(n).parts or sha(DEP/n)!=pin:raise ValueError('per-file source deployment mismatch')
 OPS.mkdir();new(OPS/'deployment.json',{'commit':req['commit'],'files':req['files'],'created_at_utc':now(),'recovery_authorization':'Eddy 2026-10-17 one preflight retry per C1/C2','original_plan_sha256':OLDPIN,'original_deadline_epoch':v['original_deadline_epoch']})
 return {'status':'deployed','read_at_utc':now(),'files_checked':len(req['files']),'deployment_ref':{'path':str(OPS/'deployment.json'),'sha256':sha(OPS/'deployment.json')},'reused':v['runs']}

def seal(req):
 identity();v=inventory()
 if v['old_driver_alive'] or any(any(q.values()) for q in v['queue'].values()) or v['existing_new_runs']:raise ValueError('new attempt identity occupied before seal')
 for rid,ref in req['reused_refs'].items():
  if v['runs'][rid]['receipt_ref']!=ref:raise ValueError('old successful receipt changed')
 p=req['documents']['pipeline-plan.json']
 validate_retry_ids(load(OLDOPS/'pipeline-plan.json'),p)
 if p['deadline_epoch']!=v['original_deadline_epoch'] or p['control_dir']!=str(OPS/'controller'):raise ValueError('deadline/control changed')
 for name,value in req['documents'].items():
  if '/' in name or name=='deployment.json':raise ValueError('exclusive document basename required')
  new(OPS/name,value)
 return {'status':'sealed','read_at_utc':now(),'plan_sha256':sha(OPS/'pipeline-plan.json')}

def run(req):
 identity();plan=OPS/'pipeline-plan.json'
 if sha(plan)!=req['plan_sha256']:raise ValueError('new frozen plan changed')
 env=os.environ.copy();env['PYTHONPATH']=str(DEP/'src')+':'+str(DEP)+':/workspace/vkgr-infra03-ops/ops';sys.path.insert(0,'/workspace/vkgr-infra03-ops/ops')
 if req['action']=='dry':
  op=old_ops();p=op.load_plan(plan,req['plan_sha256']);v=inventory();prepared={}
  for label,branch in p['branches'].items():
   prepared[label]={}
   for name,stage in branch['stages'].items():
    if name in REUSED:continue
    if any(c.startswith('@') for c in stage['job']['argv']):prepared[label][name]={'status':'DEFERRED_DYNAMIC_BINDING'}
    else:prepared[label][name]=op.default_prepare(stage['job'])
  result={'status':'dry_ok','prepared':prepared,'read_at_utc':now()};new(OPS/'dry-v2.json',result);return result
 out=OPS/'driver-start-v2.json'
 cp=subprocess.run([PY,'-B',str(DEP/'scripts/sid_coll_ops.py'),'start','--plan',str(plan),'--plan-sha',req['plan_sha256'],'--out',str(out)],env=env,cwd=DEP,capture_output=True,text=True,timeout=55)
 if cp.returncode:raise RuntimeError(cp.stdout+cp.stderr)
 return {'status':'started_unconfirmed','read_at_utc':now(),'result':load(out)}

def status():
 identity();op=old_ops();p=OPS/'pipeline-plan.json';plan=op.load_plan(p,sha(p));summary=op.tick(plan,submit=False)
 for branch in summary['branches'].values():
  for stage in branch.get('stages',{}).values():
   if stage.get('accepted') and stage.get('proof_path'):
    raw,_=op.read(stage['proof_path'],stage['proof_sha256']);stage['receipt_status']=raw['status'];stage['raw_proof_status']=raw.get('raw_proof',{}).get('status')
    ref=raw.get('proofs',{}).get('verification')
    if ref:stage['raw_verification'],_=op.read(ref['path'],ref['sha256'])
 driver=None;pf=OPS/'controller/driver-pid.json'
 if pf.exists():
  driver=load(pf);proc=Path('/proc')/str(driver['pid']);driver['alive']=proc.exists()
  if (proc/'stat').exists():driver['actual_ppid']=int((proc/'stat').read_text().rsplit(')',1)[1].split()[1])
 return {'read_at_utc':now(),'summary':summary,'driver':driver,'new_plan_sha256':sha(p),'original_plan_sha256':OLDPIN}
if __name__=='__main__':
 req=json.load(sys.stdin) if sys.argv[0]=='-c' else {'action':'inventory'};a=req['action']
 value=inventory() if a=='inventory' else deploy(req) if a=='deploy' else seal(req) if a=='seal' else run(req) if a in ('dry','start') else None
 if value is None:raise ValueError('unknown action')
 print(json.dumps(value,sort_keys=True,allow_nan=False))
