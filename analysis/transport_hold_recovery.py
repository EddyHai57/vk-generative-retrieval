"""Finite local SSH transport. Eddy approved stale-socket recovery; no unlink.
Called only by kiro_b_run.sh as hold_mux_n.py. No remote scientific mutation.
"""
import datetime
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import time

def recovery_allowed(owner_alive,master_alive,owned_socket,control_ok):
    return not owner_alive and not master_alive and owned_socket and not control_ok

def alive(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False

def main():
    r=Path(__file__).parent;n=int(sys.argv[1]);cur=f'formal{n:02d}';prev=f'formal{n-1:02d}'
    old=json.loads((r/f'ssh-control-state-{prev}.json').read_text());sock=Path(old['socket_path']);parent=sock.parent
    owned=not parent.is_symlink() and str(parent).startswith('/private/tmp/vkgr-main05-ssh-') and parent.stat().st_uid==os.getuid() and stat.S_IMODE(parent.stat().st_mode)==0o700
    if sock.exists():owned=owned and not sock.is_symlink() and sock.stat().st_uid==os.getuid() and stat.S_ISSOCK(sock.stat().st_mode)
    if not owned:raise ValueError('unsafe old local control path')
    live=[alive(old[k]) for k in ('owner_pid','master_pid')]
    check=subprocess.run(['ssh','-S',str(sock),'-O','check','vast-a6000'],capture_output=True,timeout=5)
    if not recovery_allowed(*live,owned,check.returncode==0):raise ValueError('previous owner/master or control still live; no duplicate master')
    if (r/f'ssh-control-state-{cur}.json').exists():raise ValueError('new state already exists')
    if any(r.glob('recovery01-launch*.transport.json')) and not (r/'direct-human-reconnect-20261013.json').exists():raise ValueError('fresh authentication not authorized')
    # Append a separate observation; never invent actual_master_exit or remove a stale socket.
    note={'previous':prev,'observed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'old_owner_alive':live[0],'old_master_alive':live[1],'old_control_check_exit':check.returncode,'old_socket_preserved':sock.exists(),'approval':'Eddy 2026-10-16 morning local transport repair','remote_job_change':False}
    with (r/f'ssh-control-recovery-{cur}.json').open('x') as f:json.dump(note,f)
    try:os.setsid()
    except PermissionError:pass
    directory=Path(tempfile.mkdtemp(prefix='vkgr-main05-ssh-',dir='/private/tmp'));os.chmod(directory,0o700);socket=directory/'ctl';log=directory/'ssh-diagnostic.log'
    started=time.monotonic();deadline=started+57600
    with log.open('xb') as diagnostic:
        os.chmod(log,0o600)
        process=subprocess.Popen(['ssh','-E',str(log),'-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ControlMaster=yes','-o','ControlPersist=no','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','-S',str(socket),'-N','vast-a6000'],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=diagnostic)
        exit_code=None
        try:
            bootstrap=time.monotonic()+30
            while not socket.exists():
                if process.poll() is not None or time.monotonic()>=bootstrap:raise RuntimeError('authenticated master bootstrap failed')
                time.sleep(.1)
            actual=subprocess.run(['ssh','-S',str(socket),'-O','check','vast-a6000'],capture_output=True,timeout=5)
            if actual.returncode or not stat.S_ISSOCK(socket.stat().st_mode):raise RuntimeError('master check failed')
            state={'socket_path':str(socket),'created_at':time.time(),'started_monotonic':started,'deadline_monotonic':deadline,'max_seconds':57600,'diagnostic_log_path':str(log),'master_pid':process.pid,'owner_pid':os.getpid(),'foreground_actual_wait':True,'new_server_resource':False,'server_config_change':False,'replaces':prev}
            with (r/f'ssh-control-state-{cur}.json').open('x') as f:json.dump(state,f)
            print(json.dumps({'status':'authenticated_foreground_transport_ready','master_pid':process.pid,'finite_seconds':57600}),flush=True)
            try:exit_code=process.wait(timeout=max(1,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:process.terminate();exit_code=process.wait(timeout=10)
        finally:
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
            if exit_code is not None:
                with (r/f'ssh-control-terminal-{cur}.json').open('x') as f:json.dump({'actual_master_exit':exit_code,'elapsed_seconds':time.monotonic()-started,'socket_exists':socket.exists(),'remote_job_state':'UNKNOWN_REQUIRES_REMOTE_RECEIPT'},f)
if __name__=='__main__':main()
