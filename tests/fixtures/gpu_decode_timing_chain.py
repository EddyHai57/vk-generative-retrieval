"""Transport-only synthetic source -> real launcher metadata -> module dry."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

repo=Path(__file__).resolve().parents[2]
base=repo/'cache/gpu-trie-decode'/('timing-chain-'+str(time.time_ns()));base.mkdir(parents=True)


def git(root,*args):
    return subprocess.check_output(['git','-C',str(root),*args],text=True,stderr=subprocess.PIPE).strip()


def clone(name,ref):
    root=base/name
    subprocess.run(['git','clone','--quiet','--no-checkout','--shared',str(repo),str(root)],check=True,capture_output=True)
    git(root,'checkout','--quiet','--detach',ref)
    return root


ops=clone('ops-code','5840e6b686ec9651314a9e28a1a2c0befcc05158')
runner=clone('deployment','HEAD')
paths=['analysis/gpu_decode_timing.py','src/vkgr/gpu_decode_session.py','tests/test_gpu_decode_timing.py']
for p in paths:shutil.copyfile(repo/p,runner/p)
git(runner,'add',*paths)
if git(runner,'diff','--cached','--name-only'):
    git(runner,'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','--quiet','-m','synthetic timing adapter fixture')
channel=base/'channel';channel.mkdir()
inputs=channel/'inputs.json';inputs.write_text(json.dumps({'synthetic_transport_only':True}))
lock=channel/'lock';lock.write_text('synthetic transport fixture; no runtime check claimed\n')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
request={'schema':'gpu-decode-timing-request-v1','source_commit':git(runner,'rev-parse','HEAD'),
         'lock':{'path':str(lock),'sha256':sha(lock)},'input_root':str(channel),'input_manifest_sha256':sha(inputs),
         'generation':{},'ops_root':str(base/'ops-root'),'output_root':str(base/'new-output')}
request_path=base/'request.json';request_path.write_text(json.dumps(request,sort_keys=True))
env={**os.environ,'PYTHONPATH':str(runner/'src'),'PYTHONDONTWRITEBYTECODE':'1'}
cmd=[sys.executable,'-B','-m','analysis.gpu_decode_timing']
prepared=subprocess.run(cmd+['prepare','--stage','preflight','--request',str(request_path),'--request-sha256',sha(request_path),'--ops-code',str(ops)],cwd=runner,env=env,text=True,capture_output=True)
assert prepared.returncode==0,prepared.stderr
value=json.loads(prepared.stdout)
assert value['status']=='prepared_dry' and value['no_submit'] is True
assert value['metadata']['git']['dirty'] is False
assert value['metadata']['git']['commit']==request['source_commit']
assert len(value['metadata']['inputs'])==4
assert value['spec']['vram_mib']==8192 and value['spec']['cpu_cores']==9
(base/'prepare.json').write_text(prepared.stdout)
dry=subprocess.run(cmd+['dry'],cwd=runner,env=env,text=True,capture_output=True)
assert dry.returncode==0,dry.stderr
assert json.loads(dry.stdout)['workers']==[1,2,4,8]
(base/'dry.json').write_text(dry.stdout)
bad=subprocess.run(cmd+['preflight','--request',str(request_path),'--request-sha256','0'*64,'--execute-on-new-instance'],cwd=runner,env=env,text=True,capture_output=True)
assert bad.returncode!=0 and 'request SHA differs' in bad.stderr
(base/'wrong-request.stderr').write_text(bad.stderr)
print(json.dumps({'status':'real_launcher_prepare_metadata_dry_verified','no_submit':True,
                  'synthetic_transport_only':True,'runtime_preflight':False,'data_contract_validated':False,
                  'p3c_contract':'not applicable: no P3-D reads','evidence_directory':str(base.relative_to(repo))}))
