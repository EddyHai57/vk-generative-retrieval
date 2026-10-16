"""Finite read-only Linux filesystem completion event wait, no polling/timer job.
Only the two registered S0 terminal JSON files are read. No job mutation.
"""
import ctypes,datetime,hashlib,json,os,select,time
from pathlib import Path
R=Path('/workspace/vkgr-infra01-root/runs');RIDS=('sid-fix-s0-first-v1','sid-fix-s0-a1b1024-v1')
def snapshot(root,rids):
 if not rids or not set(rids)<=set(RIDS):raise ValueError('only registered S0 runs')
 out={}
 for rid in rids:
  p=Path(root)/rid/'terminal.json';value={}
  if p.exists():
   if p.is_symlink() or p.stat().st_size>65536:raise ValueError('unsafe terminal')
   b=p.read_bytes();value={'terminal':json.loads(b),'terminal_sha256':hashlib.sha256(b).hexdigest()}
  out[rid]=value
 return {'complete':all('terminal' in v for v in out.values()),'runs':out,'read_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
def wait_for_terminals(root,rids,wait_event,deadline,clock=time.monotonic):
 while True:
  state=snapshot(root,rids)
  if state['complete']:return state
  left=deadline-clock()
  if left<=0:return state
  wait_event(left)
def main():
 rids=[rid for rid in RIDS if (R/rid).is_dir()]
 if not rids:raise ValueError('no S0 run directory')
 libc=ctypes.CDLL(None,use_errno=True);fd=libc.inotify_init1(os.O_CLOEXEC|os.O_NONBLOCK)
 if fd<0:raise OSError(ctypes.get_errno(),'inotify_init1')
 try:
  for rid in rids:
   if libc.inotify_add_watch(fd,os.fsencode(R/rid),0x8|0x80)<0:raise OSError(ctypes.get_errno(),'inotify_add_watch')
  def event(left):
   ready,_,_=select.select([fd],[],[],left)
   if ready:os.read(fd,65536)
  out=wait_for_terminals(R,rids,event,time.monotonic()+900)
  print(json.dumps(out,sort_keys=True,allow_nan=False))
 finally:os.close(fd)
if __name__=='__main__':main()
