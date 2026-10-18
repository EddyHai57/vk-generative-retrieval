"""Finite non-model observer for one immutable S2 fit and its unique replay verify.
No training submission, retries, deletion or deadline extension. Poll >=900s.
"""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import sid_learn_ops as ops
import sid_learn_fit_queue as queue
FIT='sid-learn-fit-v1'
VERIFY='sid-learn-fit-verify-v1'

def records(path):
 p=Path(path)
 if p.stat().st_size>65536:raise ValueError('bounded raw fit/verify stdout required')
 result=[]
 for line in p.read_text().splitlines():
  try:row=json.loads(line)
  except ValueError:continue
  if type(row) is dict:result.append(row)
 return result

def observe(req,*,prepare=None,publisher=None):
 runs=Path(req['runs']);q=Path(req['root'])/'queue';fit=runs/FIT;vr=runs/VERIFY
 state={'status':'running','settled':False,'verified':False,'fit_run_id':FIT,'verify_run_id':VERIFY}
 queues={rid:ops.where(q,rid) for rid in (FIT,VERIFY)}
 if any(row['rejected'] for row in queues.values()):return dict(state,status='ALERT',settled=True,reason='queue rejection; no retry')
 terminal=fit/'terminal.json'
 if not terminal.exists():return state
 t=ops.load(terminal)
 if t.get('status')!='succeeded':return dict(state,status='ALERT',settled=True,reason='fit terminal failure; no retry')
 if t.get('run_id')!=FIT or t.get('exit_code')!=0 or t.get('wall_cap_exceeded') is not False:raise ValueError('fit terminal binding/cap differs')
 claimed=queues[FIT]['claimed']
 if len(claimed)!=1 or queues[FIT]['pending'] or ops.load(claimed[0])!=req['job']:raise ValueError('original fit must be unique claimed with exact job')
 meta=ops.load(fit/'run.json');a=req['job']['argv'];bundle=fit/'out/payload/run_manifest.json';manifest_sha=ops.sha(bundle);producer_sha=ops.sha(fit/'run.json')
 if meta.get('run_id')!=FIT or meta.get('argv')!=a or meta.get('cwd')!=req['deployment'] or (meta.get('git',{}).get('commit')!=req['source_commit'] or meta.get('git',{}).get('dirty') is not False) or meta.get('config',{}).get('sha256')!=ops.flag(a,'--config-sha'):raise ValueError('fit source/config/argv differs')
 expected={'status':'complete','engineering':False,'run_id':FIT,'run_manifest_sha256':producer_sha,'manifest_sha256':manifest_sha,'source_commit':req['source_commit'],'fit_config_file_sha256':ops.flag(a,'--config-sha'),'input_spec_file_sha256':ops.flag(a,'--inputs-sha')}
 if not any(all(type(row.get(k)) is type(v) and row[k]==v for k,v in expected.items()) for row in records(fit/'logs/stdout.log')):raise ValueError('original complete raw stdout/job/manifest absent')
 job=queue.derive_verify(req,manifest_sha,producer_sha)
 if not ops.exists(q,runs,VERIFY):
  derived={**req,'job':job}
  queue.submit(derived,prepare=prepare,publisher=publisher)
 else:
  actual=[ops.load(p) for rows in queues[VERIFY].values() for p in rows]
  if len(actual)!=1 or actual[0]!=job:raise ValueError('existing verify job differs; no retry')
 if not (vr/'terminal.json').exists():return dict(state,fit_manifest_sha256=manifest_sha)
 term=ops.load(vr/'terminal.json')
 if term.get('status')!='succeeded':return dict(state,status='ALERT',settled=True,reason='verify failure; no retry')
 if term.get('exit_code')!=0 or term.get('wall_cap_exceeded') is not False or len(queues[VERIFY]['claimed'])!=1 or queues[VERIFY]['pending']:raise ValueError('verify terminal/claim differs')
 proof=ops.load(vr/'out/verification.json')
 expected={'status':'verified','run_id':VERIFY,'run_manifest_sha256':ops.sha(vr/'run.json'),'manifest_sha256':manifest_sha,'source_commit':req['source_commit'],'producer_run_manifest_sha256':producer_sha}
 if any(type(proof.get(k)) is not type(v) or proof[k]!=v for k,v in expected.items()) or proof not in records(vr/'logs/stdout.log'):raise ValueError('original verified raw/job/manifest absent')
 return dict(state,status='verified',verified=True,settled=True,fit_manifest_sha256=manifest_sha,verification_sha256=ops.sha(vr/'out/verification.json'),verify_stdout_sha256=ops.sha(vr/'logs/stdout.log'))

def driver(req,request_sha):
 root=Path(req['ops'])
 if ops.sha(root/'fit-driver-request.json')!=request_sha or not ops.valid_deadline(req['deadline_epoch']):raise ValueError('sealed finite driver request required')
 if os.getppid()!=1:
  child=os.fork()
  if child:os._exit(0)
  os.setsid()
 until=time.monotonic()+5
 while os.getppid()!=1 and time.monotonic()<until:time.sleep(.01)
 if os.getppid()!=1 or os.getsid(os.getpid())!=os.getpid():raise ValueError('detached PPID1 driver required')
 with (root/'fit-driver.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  ops.identity(req)
  ops.new(root/'fit-driver-pid.json',{'pid':os.getpid(),'ppid':os.getppid(),'sid':os.getsid(os.getpid()),'deadline_epoch':req['deadline_epoch'],'poll_seconds':900,'request_sha256':request_sha})
  while time.time()<req['deadline_epoch']:
   try:
    ops.identity(req);state=observe(req)
   except Exception as exc:state={'status':'ALERT','settled':True,'reason':type(exc).__name__+': '+str(exc)}
   ops.new(root/f'fit-summary-{time.time_ns()}.json',state)
   if state['settled']:ops.new(root/'fit-summary-final.json',state);return state
   time.sleep(min(900,max(0,req['deadline_epoch']-time.time())))
  state={'status':'ALERT','settled':True,'reason':'original finite deadline expired; no job killed or retried'};ops.new(root/'fit-summary-final.json',state);return state

def check_launch(req):
 queue.validate(req)
 if req['job']['run_id']!=FIT or not ops.valid_deadline(req['deadline_epoch']):raise ValueError('only original fit finite continuation')
 state=ops.where(Path(req['root'])/'queue',FIT)
 if len(state['claimed'])!=1 or state['pending'] or state['rejected'] or ops.load(state['claimed'][0])!=req['job']:raise ValueError('original fit unique claimed identity required before driver launch')

def launch(req):
 ops.identity(req);check_launch(req)
 root=Path(req['ops']);path=root/'fit-driver-request.json';ops.new(path,req)
 with (root/'fit-driver.log').open('x') as log:
  process=subprocess.Popen([req['python'],'-B',str(Path(__file__).resolve()),'driver','--request',str(path),'--request-sha',ops.sha(path)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
 until=time.monotonic()+30
 while time.monotonic()<until:
  if (root/'fit-driver-pid.json').exists():
   pid=ops.load(root/'fit-driver-pid.json')
   if pid['ppid']!=1 or pid['request_sha256']!=ops.sha(path):raise ValueError('driver ready binding differs')
   return {'status':'fit_driver_ready','pid':pid['pid'],'deadline_epoch':req['deadline_epoch']}
  if process.poll() not in (None,0) or (root/'fit-summary-final.json').exists():raise RuntimeError('driver startup failed; no retry')
  time.sleep(.05)
 raise TimeoutError('driver readiness unknown; do not relaunch')

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=('launch','driver'));p.add_argument('--request',required=True);p.add_argument('--request-sha',required=True);a=p.parse_args()
 if ops.sha(a.request)!=a.request_sha:raise ValueError('fixed request bytes required')
 req=ops.load(a.request);answer=launch(req) if a.action=='launch' else driver(req,a.request_sha);print(json.dumps(answer,sort_keys=True))
if __name__=='__main__':main()
