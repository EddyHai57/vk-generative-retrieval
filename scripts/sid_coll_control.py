"""S1 fixed-SHA deployment and metadata control. Stdin-source mode is read-only.
No checkpoint bytes, full embeddings or train partitions are opened here.
"""
import datetime,hashlib,io,json,os,subprocess,sys
from pathlib import Path
R=Path('/workspace/vkgr-infra01-root');Q=R/'queue'
SRC=Path('/workspace/vkgr-b1-src');DEP=Path('/workspace/vkgr-sid-coll-g-v1');OPS=Path('/workspace/vkgr-sid-coll-ops-v1')
TEACHER=Path('/workspace/vkgr-seqm1-p-v1')
PREP=Path('/workspace/vkgr-p4b-20261007t-pipeline-03/prepared-v3')
PY='/workspace/vkgr-p6-20261006t2050/code-v1/.venv/bin/python'
STAGES=('input_preflight','input','input_verify','sid_fit','sid_verify','preflight','cuda','dry','train42','verify42','train2024','verify2024','train2025','verify2025')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def new(p,v):
 with Path(p).open('x') as f:json.dump(v,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
def relative(v):
 if Path(v).is_absolute() or '..' in Path(v).parts:raise ValueError('relative nontraversing source required')
 return v
def ids():return ['sid-coll-'+v+'-'+s+'-v1' for v in ('c1','c2') for s in STAGES]+['sid-coll-s0-prime-v1','sid-coll-legacy-epochs-v1']
def where(rid):
 x={k:sorted(p.name for p in (Q/k).glob('*-'+rid+'.json')) for k in ('pending','claimed','rejected')};x['reasons']={}
 for n in x['rejected']:
  for p in ((Q/'rejected'/n).with_suffix('.reason'),Q/'rejected'/(n+'.reason')):
   if p.exists():x['reasons'][p.name]=p.read_text()[:4000]
 return x
def inventory():
 return {'read_at_utc':now(),'queue':{rid:where(rid) for rid in ids()},'existing_runs':[x for x in ids() if (R/'runs'/x).exists()],
 'deployment_exists':DEP.exists(),'ops_exists':OPS.exists(),'checkpoint_read':False,'test_read':False}
def git(*args):return subprocess.check_output(['git','-C',str(SRC),*args],text=True)
def identity():
 m=load(OPS/'deployment.json')
 if subprocess.check_output(['git','-C',str(DEP),'rev-parse','HEAD'],text=True).strip()!=m['commit']:raise ValueError('deployed source commit changed')
 if subprocess.check_output(['git','-C',str(DEP),'status','--porcelain'],text=True).strip():raise ValueError('deployed source dirty')
 for rel,pin in m['files'].items():
  if sha(DEP/relative(rel))!=pin:raise ValueError('source file changed: '+rel)
 return m

def deploy(req):
 if DEP.exists() or OPS.exists() or any(any(where(x)[k] for k in ('pending','claimed','rejected')) or (R/'runs'/x).exists() for x in ids()):raise FileExistsError('existing S1 identity')
 branch=req['branch'];commit=req['commit']
 if not branch.startswith('codex/') or len(commit)!=40:raise ValueError('working branch/fullSHA required')
 git('fetch','-q','origin',branch)
 if git('rev-parse','origin/'+branch).strip()!=commit:raise ValueError('pushed branch mismatch')
 git('worktree','add','-q','--detach',str(DEP),commit)
 for rel,pin in req['files'].items():
  if sha(DEP/relative(rel))!=pin:raise ValueError('deployed file differs')
 OPS.mkdir();new(OPS/'deployment.json',{'commit':commit,'files':req['files'],'created_at_utc':now()})
 identity();return {'status':'deployed','commit':commit,'read_at_utc':now(),'files_checked':len(req['files']),'deployment_ref':{'path':str(OPS/'deployment.json'),'sha256':sha(OPS/'deployment.json')}}

def teacher_metadata():
 """Manifest/source/support only. Never stat/hash/open an old checkpoint."""
 import numpy as np
 mp=R/'runs/p-seqm1-mean-s42/out/manifest.json';m=load(mp)
 expected='bdb0e39d6abb721ac8de78dbdd11ebf2da2614b0e25292c3154ca096ec8cdc6e'
 if sha(mp)!=expected or m['status']!='complete' or m['config']['seed']!=42 or m['config']['arm']!='mean':raise ValueError('teacher identity')
 commit=subprocess.check_output(['git','-C',str(TEACHER),'rev-parse','HEAD'],text=True).strip()
 if commit!='0ca61bc35ad68bd79b86103b54e470445cbab07f':raise ValueError('teacher source commit')
 files={str(p.relative_to(TEACHER)):sha(p) for p in sorted((TEACHER/'src/vkgr').glob('*.py'))}
 if any(m['binding']['source']['files'].get(k)!=v for k,v in files.items()):raise ValueError('teacher source ledger mismatch')
 catalog=load(PREP/'catalog.json');support=load(PREP/'support.json');idmap=load(PREP/'idmap.json');pm=load(PREP/'prepare-manifest.json')
 for name in ('catalog.json','support.json','idmap.json'):
  if sha(PREP/name)!=pm['artifacts'][name]:raise ValueError('prepared metadata hash')
 supported=set(support)
 if idmap['items']!=catalog or idmap['supported_indices']!=[i+1 for i,x in enumerate(catalog) if x in supported]:raise ValueError('support mapping identity')
 mask=np.zeros(len(catalog)+1,dtype=bool);mask[idmap['supported_indices']]=True
 best=m['best']
 if best['path']!='best.pt':raise ValueError('teacher best selection')
 lock={'schema':'coll-c2-source-lock-v1','source_commit':commit,'source_files':files,'checkpoint_sha256':best['sha256'],
       'catalog_sha256':hashlib.sha256((json.dumps(catalog,sort_keys=True,separators=(',',':'))+'\n').encode()).hexdigest(),
       'supported_sha256':hashlib.sha256(mask.tobytes()).hexdigest(),'train_sequence_sha256':'PENDING_TRAIN_SEQUENCE'}
 buf=io.BytesIO();np.save(buf,mask,allow_pickle=False)
 claimed=list((Q/'claimed').glob('*-g-tok-t3-bhkm-b256-s42.json'))
 if len(claimed)!=1:raise ValueError('unique original T3 template')
 job=load(claimed[0]);argv=job['argv']
 def flag(k):
  if argv.count(k)!=1:raise ValueError('single frozen template flag required')
  return argv[argv.index(k)+1]
 common={'g_prepared':{'path':flag('--prepared'),'sha256':flag('--prepared-sha')},
  'history':{'path':flag('--history-manifest'),'sha256':flag('--history-sha')},
  'content':{'path':flag('--content'),'sha256':flag('--content-sha')},
  'catalog':{'path':str(PREP/'catalog.json'),'sha256':sha(PREP/'catalog.json')},
  'slices':{'path':str(R/'runs/h2-prepare/out/train-slices.json'),'sha256':'6c534c35c01a79c43ed6387390e5753f33503365f3ea8c867e35a8657d62bd26'},
  's0_result':{'path':str(R/'runs/sid-fix-s0-first-v1/out/diagnostic/result.json'),'sha256':'a2e93d69bfb79af2c94e344aa5368e7604825777acc3ed6f2f6cde37c5db1924'}}
 for ref in common.values():
  if sha(ref['path'])!=ref['sha256']:raise ValueError('T3 real data pin mismatch')
 return {'read_at_utc':now(),'manifest':{'path':str(mp),'sha256':expected},'best':best,'lock':lock,
         'common':common,'template_job_sha256':sha(claimed[0]),
         'deployment_ref':{'path':str(OPS/'deployment.json'),'sha256':sha(OPS/'deployment.json')},
         'support_ref':{'path':str(OPS/'supported.npy'),'sha256':hashlib.sha256(buf.getvalue()).hexdigest()},
         'lock_template_ref':{'path':str(OPS/'teacher-lock.json'),'sha256':hashlib.sha256(json.dumps(lock,sort_keys=True,allow_nan=False).encode()).hexdigest()},
         'supported_items':int(mask.sum()),'catalog_items':len(catalog),'checkpoint_read':False,'test_read':False},mask

def metadata():return teacher_metadata()[0]
def seal(req):
 identity();m,mask=teacher_metadata()
 # Exclusive new derived metadata only. Original support/manifest remain untouched.
 if req['teacher_metadata']!= {k:v for k,v in m.items() if k!='read_at_utc'}:raise ValueError('metadata request drift')
 new(OPS/'teacher-lock.json',m['lock'])
 with (OPS/'supported.npy').open('xb') as f:
  import numpy as np
  np.save(f,mask,allow_pickle=False)
 for name,value in req['documents'].items():
  if '/' in name or name in ('deployment.json','teacher-lock.json','supported.npy'):raise ValueError('exclusive document basename required')
  new(OPS/name,value)
 return {'status':'sealed','read_at_utc':now(),'files':{p.name:sha(p) for p in OPS.iterdir() if p.is_file()}}

def run_ops(req):
 identity();p=OPS/req['plan_name']
 if '/' in req['plan_name'] or sha(p)!=req['plan_sha']:raise ValueError('sealed plan identity')
 env=os.environ.copy();env['PYTHONPATH']=str(DEP/'src')+':'+str(DEP)+':/workspace/vkgr-infra03-ops/ops'
 action=req['action'];out=OPS/req['output_name']
 if '/' in req['output_name'] or out.exists():raise ValueError('new output basename required')
 a=[PY,'-B',str(DEP/'scripts/sid_coll_ops.py'),action,'--plan',str(p),'--plan-sha',req['plan_sha'],'--out',str(out)]
 cp=subprocess.run(a,env=env,cwd=DEP,capture_output=True,text=True,timeout=55)
 if cp.returncode:raise RuntimeError(cp.stdout+'\n'+cp.stderr)
 return {'read_at_utc':now(),'output':load(out),'stdout':cp.stdout,'path':str(out),'sha256':sha(out)}

def aux_allowed(rid):
 if rid not in ('sid-coll-s0-prime-v1','sid-coll-legacy-epochs-v1'):raise ValueError('outside registered readonly auxiliaries')
 return rid

def aux(req):
 import fcntl
 identity();rid=aux_allowed(req['run_id']);jp=OPS/'aux-jobs.json'
 if sha(jp)!=req['jobs_sha256']:raise ValueError('sealed auxiliary jobs changed')
 spec=load(jp)[rid]
 if spec['run_id']!=rid or spec['schema']!='vkgr-queue-job-v1' or spec['cwd']!=str(DEP) or spec['vram_mib']!=0 or spec['cpu_cores']!=1:
  raise ValueError('readonly auxiliary queue identity/resource mismatch')
 for ref in req['input_refs']:
  if sha(ref['path'])!=ref['sha256']:raise ValueError('auxiliary input pin changed')
 sys.path.insert(0,'/workspace/vkgr-infra03-ops/ops')
 from vkgr_ops import launcher,scheduler
 prepared=launcher.prepare(spec)
 if len(json.dumps({'spec':spec,'prepared':prepared}).encode())>=48*1024-2048:raise ValueError('aux launch metadata exceeds envelope')
 result={'status':'aux_dry_ok','run_id':rid,'read_at_utc':now()}
 if req['action']=='aux-submit':
  with (OPS/'aux-publish.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   if (R/'runs'/rid).exists() or any(where(rid)[k] for k in ('pending','claimed','rejected')) or (OPS/('intent-'+rid+'.json')).exists():raise FileExistsError('never resubmit auxiliary')
   new(OPS/('intent-'+rid+'.json'),{'run_id':rid,'job_sha256':sha(jp),'created_at_utc':now()})
   result={'status':'submitted_unconfirmed','run_id':rid,'queue_path':str(scheduler.submit(R,spec)),'read_at_utc':now()}
 return result

def request():
 # Existing transport uses multiple argv[0] spellings for readonly stdin source.
 # Only the explicit python -c write route consumes a JSON request.
 return json.load(sys.stdin) if sys.argv[0]=='-c' else {'action':'inventory'}

def main():
 req=request()
 action=req.get('action','inventory')
 if sys.argv[0]!='-c' and action not in ('inventory','metadata'):raise ValueError('readonly mode')
 result=inventory() if action=='inventory' else metadata() if action=='metadata' else deploy(req) if action=='deploy' else seal(req) if action=='seal' else run_ops(req) if action in ('dry','tick','status','start') else aux(req) if action in ('aux-dry','aux-submit') else None
 if result is None:raise ValueError('unknown action')
 print(json.dumps(result,sort_keys=True,allow_nan=False))
if __name__=='__main__':main()
