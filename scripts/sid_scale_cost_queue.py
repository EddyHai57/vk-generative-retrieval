"""Metadata-only bridge for the one registered bounded synthetic cost job.
Never runs calibration inline. No retry, deletion, live budget mutation or model.
"""
import fcntl,hashlib,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
RID='sid-scale-cost-v1'
LIMIT=2*1024*1024

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def load(path):
 p=Path(path)
 if p.is_symlink() or not p.is_file() or p.stat().st_size>LIMIT:raise ValueError('bounded nonsymlink metadata required')
 value=json.loads(p.read_text())
 if type(value) is not dict:raise ValueError('metadata object required')
 return value

def new(path,value):
 with Path(path).open('x') as f:json.dump(value,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())

def absolute(value):
 p=Path(value)
 if not p.is_absolute() or '..' in p.parts or any(x.is_symlink() for x in (p,*p.parents)):raise ValueError('absolute nonsymlink own path required')
 return p

def runtime(req):
 p=absolute(req['deployment'])/'scripts/sid_scale_cost_v1.py'
 spec=importlib.util.spec_from_file_location('scale_cost_bound_runtime',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def identity(req):
 root=absolute(req['deployment']);actual=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
 if actual!=req['source_commit']:raise ValueError('fixed deployment commit differs')
 if subprocess.check_output(['git','-C',str(root),'status','--porcelain','--untracked-files=all'],text=True):raise ValueError('fixed deployment source is dirty')
 files=req.get('files')
 minimum={'scripts/sid_scale_cost_v1.py','scripts/sid_scale_cost_queue.py','scripts/sid_scale_cost_deploy.py'}
 if type(files) is not dict or not minimum<=set(files):raise ValueError('pinned runtime/queue/deploy bytes required before import')
 for name,pin in files.items():
  relative=Path(name)
  if relative.is_absolute() or '..' in relative.parts or (root/relative).is_symlink() or sha(root/relative)!=pin or hashlib.sha256(subprocess.check_output(['git','-C',str(root),'show',actual+':'+name])).hexdigest()!=pin:raise ValueError('fixed source bytes differ before runtime import')
 cost=runtime(req);source=cost.source_identity(False)
 required={**source['files'],**{name:sha(root/name) for name in ('scripts/sid_scale_cost_queue.py','scripts/sid_scale_cost_deploy.py')}}
 if req['files']!=required:raise ValueError('exact source/dependency/queue/deploy ledger required')
 for name,pin in required.items():
  if (root/name).is_symlink() or hashlib.sha256(subprocess.check_output(['git','-C',str(root),'show',actual+':'+name])).hexdigest()!=pin:raise ValueError('fixed Git source bytes differ')
 return source

def validate(req):
 cost=runtime(req);root=absolute(req['root']);dep=absolute(req['deployment']);ops=absolute(req['ops']);job=req['job'];cfg=ops/'benchmark-request.json';pinned=ops/'scheduler-pinned-cost-v1.json'
 fixed={'schema':'vkgr-queue-job-v1','kind':'ops','run_id':RID,'cwd':str(dep),'config':str(cfg),'seed':42,'cpu_cores':1,'ram_mib':18432,'vram_mib':0,'wall_hours':2.5,'disk_gib':1,'no_admission':False,'stop_mode':'signal','pinned_manifest':str(pinned),'pinned_files':[],'inputs':[str(cfg)]}
 if type(job) is not dict or set(job)!=set(fixed)|{'argv','env'} or any(type(job[k]) is not type(v) or job[k]!=v for k,v in fixed.items()):raise ValueError('one fixed CPU-only cost parent recipe required')
 expected=[sys.executable,'-B',str(dep/'scripts/sid_scale_cost_v1.py'),'--request',str(cfg),'--request-sha',sha(cfg),'--output',str(root/'runs'/RID/'out')]
 a=job['argv']
 if type(a) is not list or len(a)!=len(expected) or any(type(v) is not str for v in a) or not Path(a[0]).is_absolute() or Path(a[0]).resolve()!=Path(sys.executable).resolve() or a[1:]!=expected[1:]:raise ValueError('actual fixed parent interpreter/argv required')
 env=['PYTHONPATH='+str(dep/'src'),'PYTHONDONTWRITEBYTECODE=1',*[key+'=1' for key in cost.THREAD_ENV],'VKGR_PINNED_MANIFEST='+str(pinned)]
 if type(job['env']) is not list or len(job['env'])!=len(env) or set(job['env'])!=set(env):raise ValueError('fixed single-thread parent environment required')
 config=cost.validate_request(load(cfg))
 if config!=cost.default_request() or config['engineering'] is not False:raise ValueError('only frozen formal calibration points required')
 return job

def where(req):return {k:sorted((Path(req['root'])/'queue'/k).glob('*-'+RID+'.json')) for k in ('pending','claimed','rejected')}

def inventory(req):
 state=where(req);root=Path(req['root'])
 return {'run_id':RID,'queue':{k:[p.name for p in v] for k,v in state.items()},'run_exists':(root/'runs'/RID).exists(),'intent_exists':(Path(req['ops'])/('intent-'+RID+'.json')).exists()}

def exists(req):
 root=Path(req['root'])
 return (root/'runs'/RID).exists() or any(root.glob('queue/*/*-'+RID+'.json')) or any(root.glob('queue/*/*-'+RID+'.reason')) or any(RID in p.name for p in root.glob('queue/*/.*.tmp'))

def check_fresh(req):
 validate(req)
 if exists(req) or (Path(req['ops'])/('intent-'+RID+'.json')).exists():raise FileExistsError('never resubmit '+RID)

def prepare_envelope(req,job):
 sys.path.insert(0,str(absolute(req['scheduler_ops'])))
 from vkgr_ops import launcher
 prepared=launcher.prepare(job)
 if len(json.dumps({'spec':job,'prepared':prepared},allow_nan=False).encode())>=48*1024-2048:raise ValueError('launcher metadata envelope exceeded')
 return prepared

def publish(req,job):
 if exists(req):raise FileExistsError('existing cost job')
 pending=Path(req['root'])/'queue/pending';name=f'{time.time_ns():020d}-{RID}.json';temp=pending/('.'+name+'.tmp');new(temp,job);temp.rename(pending/name)

def dry(req,*,prepare=None):
 check_fresh(req);envelope=(prepare or prepare_envelope)(req,req['job']);proof={'status':'dry_ok','job_sha256':digest(req['job']),'request_file_sha256':sha(req['job']['config']),'run_id':RID,'envelope':envelope};new(Path(req['ops'])/('dry-'+RID+'.json'),proof);return proof

def submit(req,*,prepare=None,publisher=None):
 with (Path(req['ops'])/'publish.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);check_fresh(req);(prepare or prepare_envelope)(req,req['job'])
  new(Path(req['ops'])/('intent-'+RID+'.json'),{'run_id':RID,'job_sha256':digest(req['job'])})
  (publisher or publish)(req,req['job'])
 return {'status':'written_unconfirmed','run_id':RID}

def submit_checked(req,*,prepare=None,publisher=None):
 proof=load(Path(req['ops'])/('dry-'+RID+'.json'))
 if proof.get('job_sha256')!=digest(req['job']) or proof.get('request_file_sha256')!=sha(req['job']['config']):raise ValueError('original dry binding differs')
 return submit(req,prepare=prepare,publisher=publisher)


def accepted_report(req,source):
 validate(req);queue=where(req);base=Path(req['root'])/'runs'/RID
 if queue['rejected']:return {'status':'ALERT','reason':'queue rejected; no retry','inventory':inventory(req)}
 if not (base/'terminal.json').exists():return {'status':'not_terminal','inventory':inventory(req)}
 terminal=load(base/'terminal.json')
 if terminal.get('run_id')!=RID or terminal.get('status')!='succeeded' or type(terminal.get('exit_code')) is not int or terminal['exit_code']!=0 or terminal.get('wall_cap_exceeded') is not False:raise ValueError('original cost terminal not bound success')
 if len(queue['claimed'])!=1 or queue['pending'] or load(queue['claimed'][0])!=req['job']:raise ValueError('unique original claimed job differs')
 meta=load(base/'run.json');config=Path(req['job']['config'])
 if meta.get('run_id')!=RID or meta.get('argv')!=req['job']['argv'] or meta.get('cwd')!=req['deployment'] or meta.get('git',{}).get('commit')!=req['source_commit'] or meta['git'].get('dirty') is not False or meta.get('config',{}).get('sha256')!=sha(config) or meta['config'].get('path')!=str(config):raise ValueError('original run source/config/argv differs')
 cost=runtime(req);path=base/'out/report.json';report=load(path)
 binding={'run_id':RID,'run_manifest_sha256':sha(base/'run.json'),'request_file_sha256':sha(config),'source_commit':req['source_commit']}
 if report.get('binding')!=binding or report.get('schema')!='sid-scale-cost-report-v1' or report.get('engineering') is not False or report.get('request')!=cost.default_request() or report.get('source')!=source or report.get('producer')!=cost.producer() or report.get('fixed_l2')!=cost.check_l2_source():raise ValueError('aggregate source/recipe/job identity differs')
 if any(report.get(k) is not v for k,v in (('synthetic_only',True),('test_read',False),('checkpoint_written',False),('mapping_exported',False))):raise ValueError('aggregate synthetic/export contract differs')
 records=report.get('records');stages=[('bhkm',250000),('bhkm',300000),('rqvae',120415)]
 if type(records) is not list or len(records)!=3 or any(type(row) is not dict or (row.get('method'),row.get('n'))!=stage for row,stage in zip(records,stages)):raise ValueError('exact three calibration stages required')
 allowed={'COMPLETE','CENSORED_WALL','CENSORED_RSS','NOT_RUN_BUDGET','NOT_RUN_FAILURE'}
 if any(row.get('status') not in allowed for row in records):raise ValueError('failed/unknown calibration is not accepted')
 expected='COMPLETE' if all(row['status']=='COMPLETE' for row in records) else 'INCOMPLETE'
 if report.get('status')!=expected:raise ValueError('unfinished points cannot mean complete calibration')
 for row in records:
  if row['status']!='COMPLETE':continue
  data_sha=row.get('synthetic_features_sha256')
  if type(data_sha) is not str or len(data_sha)!=64 or set(data_sha)-set('0123456789abcdef'):raise ValueError('completed point actual data hash required')
  if row['method']=='rqvae':
   if type(row.get('steps')) is not int or row['steps']!=3000:raise ValueError('exact calibrated core step count required')
  else:
   units=cost.bhkm_work(row['n'])['units']
   if type(row.get('model_units')) is not int or row['model_units']!=units or type(row.get('actual_units')) is not int or not 0<row['actual_units']<=units:raise ValueError('balanced work/actual iteration units differ')
 if report.get('projection')!=cost.extrapolate(records):raise ValueError('projection differs from raw calibration records')
 log=base/'logs/stdout.log'
 if log.is_symlink() or log.stat().st_size>65536:raise ValueError('bounded original stdout required')
 expected={'status':'benchmark_reported','report_status':report['status'],'report_sha256':sha(path),'synthetic_only':True,**binding};found=False
 for line in log.read_text().splitlines():
  try:row=json.loads(line)
  except ValueError:continue
  if type(row) is dict and all(type(row.get(k)) is type(v) and row[k]==v for k,v in expected.items()):found=True
 if not found:raise ValueError('original benchmark_reported stdout/job/manifest absent')
 return {'status':'benchmark_receipt_accepted','calibration_status':report['status'],'run_id':RID,'report_sha256':sha(path),'stdout_sha256':sha(log),'run_manifest_sha256':binding['run_manifest_sha256'],'projection':report['projection'],'records':records}

def main():
 req=json.loads(sys.stdin.read(LIMIT+1));source=identity(req);action=req['action']
 if action=='dry':out=dry(req)
 elif action=='submit':out=submit_checked(req)
 elif action=='inventory':validate(req);out=inventory(req)
 elif action=='status':out=accepted_report(req,source)
 else:raise ValueError('metadata dry/submit/inventory/status only')
 print(json.dumps(out,sort_keys=True,allow_nan=False))
if __name__=='__main__':main()
