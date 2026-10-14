"""Writer (stdin = JSON {'tool': 'launch'|'submit' (default launch), 'launch_args':[...], 'tamper': null | {'path': rel-in-cwd}}).
Runs CFG ops_repo/ops/bin/vkgr-launch, or `vkgr-sched submit` (queue; admission waits in the scheduler), with python -I
and a scrubbed env (no VKGR_*, L4), --root CFG ops_root. Tokens '@sha:<abs path>' in launch_args are replaced by that
file's sha256 computed now.
If 'tamper' is set (launch only): append one byte to that pinned file, run launcher (expect exit 1 + no run dir), restore
original bytes and verify sha256. Writes only OPS_ROOT (launcher/queue) and, transiently, the tampered file.
(DEPLOY-01 deploy01_launch_remote.py + DEPLOY-02 deploy02_launch_remote.py submit mode)"""
import hashlib,json,os,subprocess,sys
from pathlib import Path
spec=json.loads(sys.stdin.buffer.read())
OPS=Path(CFG['server']['ops_repo']);ROOT=Path(CFG['server']['ops_root']);ROOT.mkdir(mode=0o755,exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
tool=spec.get('tool','launch')
if tool not in ('launch','submit'):raise SystemExit('tool must be launch|submit')
largs=[sha(a[5:]) if a.startswith('@sha:') else a for a in spec['launch_args']]
env={k:v for k,v in os.environ.items() if not k.startswith('VKGR_')}
cmd=[str(OPS/'ops/bin/vkgr-launch')] if tool=='launch' else [str(OPS/'ops/bin/vkgr-sched'),'submit']
args=[sys.executable,'-I',*cmd,'--root',str(ROOT),*largs]
run_id=largs[largs.index('--run-id')+1]
cwd=Path(largs[largs.index('--cwd')+1])
res={'tool':tool,'run_id':run_id,'resolved_args':largs}
t=spec.get('tamper')
if t:
    if tool!='launch':raise SystemExit('tamper is launch-only')
    f=cwd/t['path'];orig=f.read_bytes();osha=hashlib.sha256(orig).hexdigest()
    try:
        f.write_bytes(orig+b'\n')
        r=subprocess.run(args,capture_output=True,text=True,env=env,timeout=120)
    finally:
        f.write_bytes(orig)
    res.update({'tamper_path':t['path'],'tamper_exit':r.returncode,'tamper_stderr':r.stderr[-400:],'run_dir_exists_after':(ROOT/'runs'/run_id).exists(),'restored_sha_ok':hashlib.sha256(f.read_bytes()).hexdigest()==osha})
else:
    r=subprocess.run(args,capture_output=True,text=True,env=env,timeout=150)
    res.update({'exit':r.returncode,'stdout':r.stdout[-800:],'stderr':r.stderr[-800:],'run_dir_exists':(ROOT/'runs'/run_id).exists()})
    rj=ROOT/'runs'/run_id/'run.json'
    if rj.exists():
        v=json.loads(rj.read_text());res['run_json']={k:v.get(k) for k in ('kind','cwd','argv','env','git','overlay','resources','stop_mode','ops_source','admission')}
        if res['run_json'].get('git'):res['run_json']['git']={**v['git'],'dirty_files':len(v['git'].get('dirty_files') or [])}
        res['run_json_sha256']=sha(rj)
    pend=ROOT/'queue'/'pending'
    res['pending']=sorted(p.name for p in pend.glob('*.json')) if pend.exists() else []
print(json.dumps(res,indent=1))
