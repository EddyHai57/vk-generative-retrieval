"""Real launcher prepare -> metadata -> module dry; no real payload read."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
repo=Path(__file__).resolve().parents[2]
base=repo/'cache/sgns-subset-sid'/('launcher-'+str(time.time_ns()));base.mkdir(parents=True)
def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args],text=True,stderr=subprocess.PIPE).strip()
def clone(name,ref):
    path=base/name;subprocess.run(['git','clone','-q','--shared','--no-checkout',str(repo),str(path)],check=True,capture_output=True);git(path,'checkout','-q','--detach',ref);return path
code=clone('CODE',git(repo,'rev-parse','HEAD'));ops=clone('OPS','5840e6b686ec9651314a9e28a1a2c0befcc05158')
env={**os.environ,'PYTHONPATH':str(code/'src'),'PYTHONDONTWRITEBYTECODE':'1'}
script="from analysis.ur001_derive_cli import source_identity;import json;print(json.dumps(source_identity()))"
source=json.loads(subprocess.check_output([sys.executable,'-B','-c',script],cwd=code,env=env,text=True))
request={k:str(base/k) for k in ('lock','ops_root','ops_code','bundle','sgns_output','sgns_run','preflight_output','fit_output','verify_output')}
request.update(schema='gpu-sgns-subset-sid-request-v1',device='cpu',source=source,
               lock=str(code/'configs/new-instance/anchor-uv.lock'),ops_code=str(ops),bundle_sha='1'*64,sgns_receipt_sha='2'*64,cpu_request=None,cpu_request_sha=None)
path=base/'request.json';path.write_text(json.dumps(request,sort_keys=True));import hashlib;sha=hashlib.sha256(path.read_bytes()).hexdigest()
args=[sys.executable,'-B','-m','analysis.sgns_subset_sid','queue-dry','--stage','preflight','--request',str(path),'--request-sha',sha]
result=subprocess.run(args,cwd=code,env=env,text=True,capture_output=True);assert result.returncode==0,result.stderr
value=json.loads(result.stdout);assert value['metadata']['git']['commit']==source['commit'];assert value['metadata']['git']['dirty_files']==[]
assert value['spec']['argv'][3:5]==['analysis.sgns_subset_sid','preflight'];assert value['spec']['seed']==42
(base/'prepared.json').write_text(result.stdout)
dry=subprocess.run(value['spec']['argv'][:4]+['dry'],cwd=code,env=env,text=True,capture_output=True);assert dry.returncode==0,dry.stderr
assert json.loads(dry.stdout)['config']['branching']==48
(base/'dry.json').write_text(dry.stdout)
print(json.dumps({'status':'launcher_metadata_dry_verified','source':source['commit'],'evidence':str(base.relative_to(repo)),
                  'payload_read':False,'real_preflight':False,'submitted':False,'p3c_overlay':'not_applicable: SID stages do not read P3-D'},sort_keys=True))
