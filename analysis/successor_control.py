"""Versioned one-shot successor deployment/admission and paired latency job.
Default mode is read-only metadata; no retries, cleanup or scientific rule edits.
"""
import contextlib
import datetime
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
R=Path('/workspace/vkgr-infra01-root');Q=R/'queue';OPS=Path('/workspace/vkgr-successor-ops-v1');BENCH=Path('/workspace/vkgr-seqm1-latency-v2');REPO=Path('/workspace/vkgr-b1-src');PY='/workspace/vkgr-p6-20261006t2050/code-v1/.venv/bin/python'
RID='p-seqm1-latency-report-v1';VAL=R/'runs/h2-prepare/out/validation-l50.json';VAL_SHA='c703bbcc3e3004a58a2a14c2c7c2420b51b3b7ae0c47f6b284d587e67819ab1a'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def new(p,v):
 with Path(p).open('x') as f:json.dump(v,f,sort_keys=True)
def check_latency(d,device):
 if d.get('formal') is not True or d.get('device')!=device:raise ValueError('nonformal/device mismatch')
 for mode,b in [('offline',512),('online',1)]:
  for name in ('anchor','mean'):
   x=d['models'][name][mode+'_ms_per_query']
   if (x['n'],x['timed_calls'],x['warmup_calls'],x['batch'])!=(1000,1000,50,b):raise ValueError('latency call count mismatch')
  if d['models']['anchor'][mode+'_ms_per_query']['input_sha256']!=d['models']['mean'][mode+'_ms_per_query']['input_sha256']:raise ValueError('unpaired inputs')
def module():
 p=BENCH/'scripts/seqm1_latency_bench.py';meta=load(OPS/'deployment.json')
 if sha(p)!=meta['bench_script_sha256']:raise ValueError('benchmark source changed')
 sys.path.insert(0,str(BENCH/'src'));s=importlib.util.spec_from_file_location('latency',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def status():
 from gtok02_seed_batch import where
 out={'read_at_utc':now(),'queue':where(Q,RID),'new_submit':False}
 for name in ('terminal.json',):
  p=R/'runs'/RID/name
  if p.exists():out[name]=load(p)
 p=R/'runs'/RID/'out/report.json'
 if p.exists():
  d=load(p)
  for device in ('cpu','cuda'):check_latency(d['reports'][device],device)
  out['report']=d;out['report_sha256']=sha(p)
 return out

def latency_job(output):
 os.chdir(BENCH);m=module();output=Path(output);output.mkdir(parents=True,exist_ok=False);reports={}
 for device in ('cpu','cuda'):
  captured=io.StringIO()
  with contextlib.redirect_stdout(captured):m.main(['--device',device,'--lengths-json',str(VAL),'--lengths-sha',VAL_SHA])
  d=json.loads(captured.getvalue());check_latency(d,device);reports[device]=d
 result={'status':'latency_measured','reports':reports,'read_at_utc':now(),'validation_sha256':VAL_SHA,'source':load(OPS/'deployment.json'),'test_read':False,'checkpoint_read':False}
 new(output/'report.json',result);print(json.dumps({'status':'latency_measured','report_sha256':sha(output/'report.json'),'source_commit':result['source']['bench_commit']}))

def preflight():
 if sha(VAL)!=VAL_SHA:raise ValueError('validation pin mismatch')
 m=module();d=m.load_lengths(VAL,VAL_SHA,2520)
 return {'status':'latency_preflight_ok','read_at_utc':now(),'length_source':{k:v for k,v in d.items() if k!='lengths'},'test_read':False,'checkpoint_read':False}

def main():
 if '--latency-job' in sys.argv:latency_job(sys.argv[sys.argv.index('--latency-job')+1]);return
 req=json.loads(sys.stdin.read() or '{"action":"status"}');sys.path.insert(0,str(OPS/'code/analysis'));sys.path.insert(0,'/workspace/vkgr-infra03-ops/ops');action=req['action']
 if action=='deploy':
  if OPS.exists() or BENCH.exists():raise FileExistsError('deployment exists')
  for path,branch,commit,files in [(BENCH,'codex/m1-latency-real',req['bench_commit'],req['bench_files']),(OPS/'code','codex/phase2-readonly-tools',req['ops_commit'],req['ops_files'])]:
   subprocess.run(['git','-C',str(REPO),'fetch','-q','origin',branch],check=True);subprocess.run(['git','-C',str(REPO),'worktree','add','-q','--detach',str(path),commit],check=True)
   for name,pin in files.items():
    if sha(path/name)!=pin:raise ValueError('deployed file mismatch: '+name)
  new(OPS/'deployment.json',{'bench_commit':req['bench_commit'],'ops_commit':req['ops_commit'],'bench_script_sha256':sha(BENCH/'scripts/seqm1_latency_bench.py'),'files_checked':[len(req['bench_files']),len(req['ops_files'])],'utc':now()});out={'status':'deployed','metadata':load(OPS/'deployment.json')}
 elif action=='preflight':out=preflight()
 elif action in ('dry','submit'):
  from gtok02_seed_batch import publish,where
  if (R/'runs'/RID).exists() or any(where(Q,RID).values()):raise FileExistsError(RID)
  fs=list((Q/'claimed').glob('*-p-seqm1-mean-s42.json'))
  if len(fs)!=1:raise ValueError('unique P template required')
  d=load(fs[0]);d.update(run_id=RID,cwd=str(BENCH),config=None,inputs=[str(VAL)],pinned_manifest=None,pinned_files=[],disk_gib=1,wall_hours=2.,vram_mib=1024,argv=[PY,'-B',str(OPS/'code/analysis/successor_control.py'),'--latency-job',str(R/'runs'/RID/'out')])
  from vkgr_ops import launcher
  launcher.prepare(d);out={'status':'dry_ok','job':d}
  if action=='submit':out['written_not_accepted']=publish(Q,R/'runs',d)
 elif action=='status':out=status()
 else:raise ValueError('unknown action')
 print(json.dumps(out,sort_keys=True))
if __name__=='__main__':main()
