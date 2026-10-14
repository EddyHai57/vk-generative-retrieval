"""Writer (stdin = JSON {'name','cwd','argv','env'}): start argv DETACHED (new session) in NEW dir
CFG work_dir/<name>/ (refuse if exists). Writes stdout.log/stderr.log/exit.json there only.
Wrapper records exit code + wall time. No interaction with any other job. (DEPLOY-01 deploy01_detach_remote.py)"""
import json,os,re,subprocess,sys,time
from pathlib import Path
spec=json.loads(sys.stdin.buffer.read())
if not re.fullmatch(r'[a-z0-9][a-z0-9._-]{0,63}',spec['name']):raise SystemExit('bad name')
W=Path(CFG['server']['work_dir']);W.mkdir(mode=0o700,exist_ok=True)
d=W/spec['name'];d.mkdir(mode=0o700)
(d/'spec.json').write_text(json.dumps(spec))
wrapper=r'''
import json,os,subprocess,sys,time
d=sys.argv[1];spec=json.load(open(d+'/spec.json'))
env=dict(os.environ);env.update(spec.get('env',{}))
t=time.time();m=time.monotonic()
with open(d+'/stdout.log','wb') as o,open(d+'/stderr.log','wb') as e,open(os.devnull,'rb') as n:
    rc=subprocess.run(spec['argv'],cwd=spec['cwd'],env=env,stdin=n,stdout=o,stderr=e).returncode
with open(d+'/exit.json.tmp','w') as f:json.dump({'exit_code':rc,'started_at':t,'ended_at':time.time(),'wall_seconds':time.monotonic()-m},f)
os.replace(d+'/exit.json.tmp',d+'/exit.json')
'''
(d/'wrapper.py').write_text(wrapper)
with open(os.devnull,'rb') as n,(d/'wrapper.log').open('xb') as o:
    c=subprocess.Popen([sys.executable,'-B',str(d/'wrapper.py'),str(d)],cwd=str(d),stdin=n,stdout=o,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
print(json.dumps({'dir':str(d),'pid':c.pid,'started_at':time.time()}))
